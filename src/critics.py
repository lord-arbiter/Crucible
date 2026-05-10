"""Five specialist VLM critics + a multi-provider transport layer.

Implementation notes:

* **Transport selection**. ``_get_transport(cfg)`` picks the right backend
  based on the config:
  1. **OpenAI-compat client** when ``cfg.vlm_endpoint`` is set. Standard
     path for self-hosted vLLM, Hyperbolic, Together AI, DashScope,
     OpenAI, Gemini's OpenAI-compat endpoint, etc.
  2. **LiteLLM** when ``cfg.vlm_endpoint`` is empty AND ``cfg.vlm_model``
     has a recognized provider prefix (``anthropic/``, ``gemini/``,
     ``bedrock/``, ``vertex_ai/``, ``cohere/``, etc.). Routes natively
     to 100+ providers via ``litellm.acompletion()``. Lazy-imports
     litellm so users without the ``[universal]`` extra aren't forced
     to install it.
  3. **OpenAI direct** when no endpoint set and the model id is a
     known OpenAI name (``gpt-4o``, ``gpt-4o-mini``, ``o1``, etc.).
     Defaults endpoint to ``https://api.openai.com/v1``.

* **JSON output** uses ``response_format={"type": "json_schema", ...}``
  driven by vLLM's xgrammar guided-decoding backend or each provider's
  native structured-output. Three-tier fallback: json_schema →
  json_object → unconstrained, with regex salvage as a final resort.
* User messages get ``/no_think`` appended automatically when the
  model id contains ``qwen`` (Qwen3 thinking-mode suppression). Other
  model families ignore the suffix.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
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

# A transport is an async callable that takes the OpenAI-shaped request
# args and returns the raw response string. The three transports in this
# module (OpenAI-compat, LiteLLM, OpenAI direct) all implement this.
TransportFn = Callable[
    [list[dict], int, float, dict[str, Any] | None],
    Awaitable[str],
]

# LiteLLM provider prefixes — when a model id starts with one of these and
# no explicit endpoint is configured, we route via litellm.acompletion().
# See https://docs.litellm.ai/docs/providers for the full list.
LITELLM_PROVIDER_PREFIXES: tuple[str, ...] = (
    "anthropic/",
    "azure/",
    "azure_ai/",
    "bedrock/",
    "bedrock_converse/",
    "cerebras/",
    "cloudflare/",
    "codestral/",
    "cohere/",
    "cohere_chat/",
    "databricks/",
    "deepinfra/",
    "deepseek/",
    "fireworks_ai/",
    "gemini/",
    "groq/",
    "huggingface/",
    "mistral/",
    "ollama/",
    "openrouter/",
    "perplexity/",
    "replicate/",
    "sagemaker/",
    "together_ai/",
    "vertex_ai/",
    "watsonx/",
    "xai/",
)

# Bare model ids that map to OpenAI's public API.
KNOWN_OPENAI_MODEL_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "chatgpt-",
)

logger = logging.getLogger(__name__)

CRITIC_NAMES = ["visual", "kinematic", "task", "strategy", "safety"]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Universal instruction appended to every user prompt. Reinforces the system
# prompt's "Output strict JSON only" directive at the user-message boundary,
# where many models pay closer attention.
JSON_ONLY_INSTRUCTION = "\n\nRespond with valid JSON ONLY, no commentary, no markdown fences."

# Qwen3 family (Qwen3-VL-* and Qwen3-*) defaults to a "thinking" mode that
# emits a chain-of-thought before the JSON. The `/no_think` magic token
# disables it. Other model families (GPT-4o, Claude, Gemini, Llama-Vision,
# InternVL, etc.) ignore the token, so we only append it when the model
# string suggests Qwen.
QWEN_NO_THINK_TOKEN = " /no_think"

# Backward-compatible export for callers that imported the old constant.
NO_THINK_SUFFIX = JSON_ONLY_INSTRUCTION + QWEN_NO_THINK_TOKEN


def _is_qwen_model(model_id: str | None) -> bool:
    return "qwen" in (model_id or "").lower()


def _user_message_suffix(cfg: CrucibleConfig) -> str:
    suffix = JSON_ONLY_INSTRUCTION
    if _is_qwen_model(cfg.vlm_model):
        suffix += QWEN_NO_THINK_TOKEN
    return suffix

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
        f"{_user_message_suffix(cfg)}"
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


# ---------------------------------------------------------------------------
# Transport selector — picks OpenAI-compat / LiteLLM / OpenAI-direct based on
# the config. Each transport returns an async callable with a unified shape.
# ---------------------------------------------------------------------------


def _has_litellm_prefix(model_id: str | None) -> bool:
    if not model_id:
        return False
    lower = model_id.lower()
    return any(lower.startswith(p) for p in LITELLM_PROVIDER_PREFIXES)


def _is_known_openai_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    lower = model_id.lower()
    if any(lower.startswith(p) for p in KNOWN_OPENAI_MODEL_PREFIXES):
        # Carve out: an explicit `openai/` prefix should still go through LiteLLM.
        return not lower.startswith("openai/")
    return False


def _select_transport_kind(cfg: CrucibleConfig) -> str:
    """Return one of 'openai_compat', 'litellm', 'openai_direct' for this config."""
    if cfg.vlm_endpoint and cfg.vlm_endpoint.strip() and cfg.vlm_endpoint.strip().lower() not in ("", "litellm"):
        return "openai_compat"
    if _has_litellm_prefix(cfg.vlm_model):
        return "litellm"
    if _is_known_openai_model(cfg.vlm_model):
        return "openai_direct"
    return "openai_compat"  # last-ditch — assume self-hosted vLLM at the default localhost URL


def _make_openai_compat_transport(cfg: CrucibleConfig) -> TransportFn:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package missing; reinstall crucible-curation")
    base_url = (cfg.vlm_endpoint or "http://localhost:8001/v1").strip()
    api_key = cfg.vlm_api_key or "EMPTY"
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    logger.info("crucible.transport=openai_compat endpoint=%s model=%s", base_url, cfg.vlm_model)

    async def _call(messages: list[dict], max_tokens: int, temperature: float, response_format: dict[str, Any] | None) -> str:
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

    return _call


def _make_openai_direct_transport(cfg: CrucibleConfig) -> TransportFn:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package missing; reinstall crucible-curation")
    api_key = cfg.vlm_api_key
    if not api_key or api_key.upper() == "EMPTY":
        api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    client = AsyncOpenAI(base_url="https://api.openai.com/v1", api_key=api_key)
    logger.info("crucible.transport=openai_direct model=%s", cfg.vlm_model)

    async def _call(messages: list[dict], max_tokens: int, temperature: float, response_format: dict[str, Any] | None) -> str:
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

    return _call


def _make_litellm_transport(cfg: CrucibleConfig) -> TransportFn:
    try:
        import litellm  # noqa: PLC0415  - lazy import; only required when the universal extra is in play
    except ImportError as exc:
        raise RuntimeError(
            f"Model '{cfg.vlm_model}' uses a LiteLLM provider prefix, but the "
            f"litellm package is not installed.\n\n"
            f"Install with:\n"
            f"    pip install 'crucible-curation[universal]'\n\n"
            f"Or remove the provider prefix and set CRUCIBLE_VLM_ENDPOINT to "
            f"point at an OpenAI-compatible host."
        ) from exc

    api_key = cfg.vlm_api_key if cfg.vlm_api_key and cfg.vlm_api_key.upper() != "EMPTY" else None
    logger.info("crucible.transport=litellm model=%s api_key_set=%s", cfg.vlm_model, bool(api_key))

    async def _call(messages: list[dict], max_tokens: int, temperature: float, response_format: dict[str, Any] | None) -> str:
        kwargs: dict[str, Any] = {
            "model": cfg.vlm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = await litellm.acompletion(**kwargs)
        return resp.choices[0].message.content or ""

    return _call


def _get_transport(cfg: CrucibleConfig) -> TransportFn:
    """Return the async transport callable selected for this config.

    Three transports:
    - ``openai_compat`` — when ``cfg.vlm_endpoint`` is non-empty.
    - ``litellm`` — when no endpoint and ``cfg.vlm_model`` has a LiteLLM
      provider prefix (e.g. ``anthropic/``, ``gemini/``, ``bedrock/``).
    - ``openai_direct`` — when no endpoint and the model id is a known
      OpenAI name (``gpt-4o``, ``gpt-4o-mini``, ``o1``, ...).
    """
    kind = _select_transport_kind(cfg)
    if kind == "openai_compat":
        return _make_openai_compat_transport(cfg)
    if kind == "litellm":
        return _make_litellm_transport(cfg)
    if kind == "openai_direct":
        return _make_openai_direct_transport(cfg)
    raise RuntimeError(f"unknown transport kind: {kind}")


async def _chat_once(
    transport: TransportFn,
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
    return await transport(messages, max_tokens, temperature, response_format)


async def _chat_with_retries(
    transport: TransportFn,
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
                    transport,
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
    transport: TransportFn,
) -> dict:
    system = load_prompt(name)
    user_content = build_user_message(name, bundle, cfg)
    try:
        raw = await _chat_with_retries(
            transport,
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
    transport: TransportFn,
) -> dict[str, dict]:
    if cfg.parallel_critics:
        results = await asyncio.gather(*[run_critic(n, bundle, cfg, transport) for n in CRITIC_NAMES])
    else:
        results = []
        for n in CRITIC_NAMES:
            results.append(await run_critic(n, bundle, cfg, transport))
    storage_keys = ["visual_quality", "kinematic_quality", "task_success", "strategy", "safety"]
    return dict(zip(storage_keys, results, strict=False))
