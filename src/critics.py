"""Five specialist VLM critics + invocation against an OpenAI-compatible vLLM endpoint.

Implementation notes:

* JSON output uses ``response_format={"type": "json_schema", ...}`` driven by
  vLLM's xgrammar guided-decoding backend. Plain ``{"type": "json_object"}``
  has known reliability issues with Qwen3-VL (vLLM #18819) — emits stray
  backticks and prose. We pick the schema route and fall back to json_object,
  then to a regex JSON salvage path, on each failure tier.
* User messages have ``/no_think`` appended so Qwen3 returns the answer
  directly instead of producing a chain-of-thought before the JSON.
* ``response_format`` is passed via ``extra_body`` for endpoints that don't
  yet expose the SDK-level field.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

from .config import CrucibleConfig
from .lerobot_io import EpisodeBundle

logger = logging.getLogger(__name__)

CRITIC_NAMES = ["visual", "kinematic", "task", "strategy", "safety"]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
NO_THINK_SUFFIX = "\n\nRespond with valid JSON ONLY, no commentary, no markdown fences. /no_think"

CRITIC_VERDICT_VOCAB: dict[str, list[str]] = {
    "visual": ["EXCELLENT", "ACCEPTABLE", "MARGINAL", "REJECT"],
    "kinematic": ["SMOOTH", "ACCEPTABLE", "JERKY", "REJECT"],
    "task": ["COMPLETE", "PARTIAL", "FAILED", "UNCLEAR"],
    "strategy": ["EXEMPLARY", "GOOD", "MEDIOCRE", "POOR"],
    "safety": ["SAFE", "MINOR_CONCERN", "MODERATE_CONCERN", "UNSAFE"],
}


def _critic_schema(name: str) -> dict[str, Any]:
    """JSON schema enforced via xgrammar guided decoding for the named critic."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "verdict", "rationale", "evidence"],
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 10},
            "verdict": {"type": "string", "enum": CRITIC_VERDICT_VOCAB[name]},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 1200},
            "evidence": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["timestamp", "observation"],
                    "properties": {
                        "timestamp": {"type": "string"},
                        "observation": {"type": "string", "maxLength": 240},
                    },
                },
            },
        },
    }


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    p = PROMPTS_DIR / f"critic_{name}.txt"
    return p.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_aggregator_prompt() -> str:
    return (PROMPTS_DIR / "aggregator.txt").read_text(encoding="utf-8")


def encode_image_b64(img: Image.Image, max_dim: int) -> str:
    img = img.copy()
    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _frames_to_send(name: str, bundle: EpisodeBundle) -> tuple[list[Image.Image], list[float]]:
    """Different critics need different frame slices."""
    frames = bundle.sampled_frames
    timestamps = bundle.sample_timestamps
    if not frames:
        return [], []
    if name == "task":
        last = max(1, min(5, len(frames)))
        return frames[-last:], timestamps[-last:]
    if name == "kinematic":
        if len(frames) <= 5:
            return frames, timestamps
        idxs = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
        return [frames[i] for i in idxs], [timestamps[i] for i in idxs]
    return frames, timestamps


def build_user_message(name: str, bundle: EpisodeBundle, cfg: CrucibleConfig) -> list[dict]:
    frames, timestamps = _frames_to_send(name, bundle)
    text_block = (
        f"TASK: {bundle.task_description}\n\n"
        f"EPISODE INDEX: {bundle.episode_index}\n"
        f"DURATION: {bundle.duration_s:.1f}s @ {bundle.fps}fps\n"
        f"PRIMARY CAMERA: {bundle.primary_camera or 'unknown'}\n\n"
        f"TELEMETRY DIGEST:\n{bundle.telemetry_digest}\n\n"
        f"Frames sampled at timestamps (seconds): {[round(t, 2) for t in timestamps]}"
        f"{NO_THINK_SUFFIX}"
    )
    content: list[dict] = [{"type": "text", "text": text_block}]
    for img in frames:
        b64 = encode_image_b64(img, cfg.image_max_dim)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return content


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_loose(raw: str) -> dict:
    if not raw:
        return _parse_error_payload("(empty model response)")
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return _parse_error_payload(raw)


def _parse_error_payload(raw: str) -> dict:
    return {
        "score": 5.0,
        "verdict": "PARSE_ERROR",
        "rationale": (raw or "")[:500],
        "evidence": [],
    }


def _build_response_format(schema: dict[str, Any] | None, schema_name: str) -> dict[str, Any] | None:
    if schema is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": schema, "strict": True},
    }


async def _chat_once(
    client: AsyncOpenAI,
    cfg: CrucibleConfig,
    *,
    system: str,
    user_content: list[dict] | str,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, Any] | None,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    kwargs: dict[str, Any] = {
        "model": cfg.vlm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


async def _chat_with_retries(
    client: AsyncOpenAI,
    cfg: CrucibleConfig,
    *,
    system: str,
    user_content: list[dict] | str,
    max_tokens: int,
    temperature: float,
    schema: dict[str, Any] | None,
    schema_name: str,
) -> str:
    """Try json_schema first, then json_object, then unconstrained — with retries on each tier."""
    response_format_tiers: list[dict[str, Any] | None] = [
        _build_response_format(schema, schema_name),
        {"type": "json_object"},
        None,
    ]
    last_exc: Exception | None = None
    for tier_idx, fmt in enumerate(response_format_tiers):
        for attempt in range(max(1, cfg.request_retries + 1)):
            try:
                return await _chat_once(
                    client,
                    cfg,
                    system=system,
                    user_content=user_content,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=fmt,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "VLM call failed (tier %d attempt %d): %s",
                    tier_idx, attempt + 1, exc,
                )
                # If json_schema isn't supported, fall through to json_object immediately.
                if tier_idx == 0 and any(s in str(exc).lower() for s in ("schema", "response_format", "guided")):
                    break
                await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"VLM call exhausted all response-format tiers: {last_exc}")


async def run_critic(
    name: str,
    bundle: EpisodeBundle,
    cfg: CrucibleConfig,
    client: AsyncOpenAI,
) -> dict:
    system = load_prompt(name)
    user_content = build_user_message(name, bundle, cfg)
    try:
        raw = await _chat_with_retries(
            client,
            cfg,
            system=system,
            user_content=user_content,
            max_tokens=cfg.critic_max_tokens,
            temperature=cfg.critic_temperature,
            schema=_critic_schema(name),
            schema_name=f"critic_{name}_verdict",
        )
    except Exception as exc:
        return {**_parse_error_payload(str(exc)), "verdict": "ERROR"}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _extract_json_loose(raw)


async def run_all_critics(
    bundle: EpisodeBundle,
    cfg: CrucibleConfig,
    client: AsyncOpenAI,
) -> dict[str, dict]:
    if cfg.parallel_critics:
        results = await asyncio.gather(*[run_critic(n, bundle, cfg, client) for n in CRITIC_NAMES])
    else:
        results = []
        for n in CRITIC_NAMES:
            results.append(await run_critic(n, bundle, cfg, client))
    storage_keys = ["visual_quality", "kinematic_quality", "task_success", "strategy", "safety"]
    return dict(zip(storage_keys, results, strict=False))
