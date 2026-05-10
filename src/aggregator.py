"""Aggregator: turns five critic outputs into a final episode verdict.

Includes a deterministic Python fallback that mirrors the spec's rules so we
can always return a verdict even if the aggregator LLM call fails or returns
broken JSON.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .config import CrucibleConfig
from .critics import _chat_with_retries, _extract_json_loose, _user_message_suffix, load_aggregator_prompt

logger = logging.getLogger(__name__)

CRITIC_WEIGHTS: dict[str, float] = {
    "visual_quality": 1.0,
    "kinematic_quality": 1.0,
    "task_success": 1.5,
    "strategy": 1.5,
    "safety": 1.0,
}

REJECT_VERDICTS: set[str] = {"REJECT", "UNSAFE", "FAILED"}
MINOR_CONCERN_VERDICTS: set[str] = {"MINOR_CONCERN"}

AGGREGATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["final_score", "verdict", "summary", "top_concern"],
    "properties": {
        "final_score": {"type": "number", "minimum": 0, "maximum": 10},
        "verdict": {"type": "string", "enum": ["KEEP", "POLISH", "REJECT"]},
        "summary": {"type": "string", "minLength": 1, "maxLength": 1500},
        "top_concern": {"type": ["string", "null"], "maxLength": 400},
    },
}


def _coerce_score(payload: Any) -> float:
    v = payload.get("score", 5.0) if isinstance(payload, dict) else 5.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 5.0


def _verdict_label(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("verdict", "")).upper()
    return ""


def fallback_aggregate(critic_results: dict[str, dict]) -> dict:
    weighted_sum = 0.0
    weight_total = 0.0
    has_reject = False
    has_minor = False
    worst_concern: tuple[float, str] | None = None
    for key, weight in CRITIC_WEIGHTS.items():
        payload = critic_results.get(key, {})
        score = _coerce_score(payload)
        verdict = _verdict_label(payload)
        weighted_sum += score * weight
        weight_total += weight
        if verdict in REJECT_VERDICTS:
            has_reject = True
        if verdict in MINOR_CONCERN_VERDICTS:
            has_minor = True
        rationale = (payload or {}).get("rationale") if isinstance(payload, dict) else None
        if rationale and (worst_concern is None or score < worst_concern[0]):
            worst_concern = (score, f"{key}: {rationale}")

    final_score = weighted_sum / weight_total if weight_total else 0.0
    if has_reject or final_score < 5.0:
        verdict = "REJECT"
    elif final_score >= 7.5 and not has_minor:
        verdict = "KEEP"
    else:
        verdict = "POLISH"

    summary = (
        f"Weighted score {final_score:.2f}. "
        + ("Reject due to a critic flagging a hard failure or unsafe behavior. " if has_reject else "")
        + ("At least one minor concern was raised, recommending POLISH. " if has_minor and verdict == "POLISH" else "")
        + "Computed by deterministic fallback aggregator."
    )
    top_concern = worst_concern[1] if worst_concern and worst_concern[0] < 6.0 else None
    return {
        "final_score": round(final_score, 2),
        "verdict": verdict,
        "summary": summary,
        "top_concern": top_concern,
        "fallback": True,
    }


async def aggregate(
    critic_results: dict[str, dict],
    cfg: CrucibleConfig,
    transport,  # TransportFn from src.critics
) -> dict:
    system = load_aggregator_prompt()
    user_text = (
        "Critic outputs:\n"
        f"{json.dumps(critic_results, indent=2, default=str)}\n\n"
        "Produce the final verdict JSON."
        f"{_user_message_suffix(cfg)}"
    )
    try:
        raw = await _chat_with_retries(
            transport,
            cfg,
            system=system,
            user_content=user_text,
            max_tokens=cfg.aggregator_max_tokens,
            temperature=cfg.aggregator_temperature,
            schema=AGGREGATOR_SCHEMA,
            schema_name="aggregator_verdict",
        )
    except Exception as exc:
        logger.warning("Aggregator LLM call failed, using fallback: %s", exc)
        return fallback_aggregate(critic_results)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = _extract_json_loose(raw)

    if not isinstance(parsed, dict) or "final_score" not in parsed or "verdict" not in parsed:
        logger.info("Aggregator output malformed, using fallback")
        return fallback_aggregate(critic_results)
    return parsed
