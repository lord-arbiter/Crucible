"""Deterministic-fallback aggregator tests.

These cover the verdict rules from spec §4.4: weighted mean with
task_success and strategy weighted 1.5x, plus hard-fail rules on
REJECT / UNSAFE / FAILED critic verdicts.
"""
from __future__ import annotations

import pytest

from src.aggregator import (
    CRITIC_WEIGHTS,
    REJECT_VERDICTS,
    fallback_aggregate,
)


def _critic(score: float, verdict: str = "ACCEPTABLE", rationale: str = "ok") -> dict:
    return {"score": score, "verdict": verdict, "rationale": rationale, "evidence": []}


def test_all_high_yields_keep():
    critics = {
        "visual_quality": _critic(8.5, "EXCELLENT"),
        "kinematic_quality": _critic(8.0, "SMOOTH"),
        "task_success": _critic(9.0, "COMPLETE"),
        "strategy": _critic(9.0, "EXEMPLARY"),
        "safety": _critic(8.5, "SAFE"),
    }
    result = fallback_aggregate(critics)
    assert result["verdict"] == "KEEP"
    assert result["final_score"] >= 8.5
    assert result["fallback"] is True


def test_unsafe_critic_forces_reject_even_with_high_score():
    critics = {
        "visual_quality": _critic(9.0),
        "kinematic_quality": _critic(9.0),
        "task_success": _critic(9.0, "COMPLETE"),
        "strategy": _critic(9.0, "EXEMPLARY"),
        "safety": _critic(9.0, "UNSAFE"),  # hard-fail trigger
    }
    result = fallback_aggregate(critics)
    assert result["verdict"] == "REJECT"


def test_failed_task_forces_reject():
    critics = {
        "visual_quality": _critic(8.0),
        "kinematic_quality": _critic(8.0),
        "task_success": _critic(2.0, "FAILED"),
        "strategy": _critic(8.0, "GOOD"),
        "safety": _critic(8.0, "SAFE"),
    }
    result = fallback_aggregate(critics)
    assert result["verdict"] == "REJECT"


def test_minor_concern_demotes_keep_to_polish():
    critics = {
        "visual_quality": _critic(8.0),
        "kinematic_quality": _critic(8.0),
        "task_success": _critic(8.0, "COMPLETE"),
        "strategy": _critic(8.0, "GOOD"),
        "safety": _critic(8.0, "MINOR_CONCERN"),
    }
    result = fallback_aggregate(critics)
    assert result["verdict"] == "POLISH"


def test_low_score_yields_reject():
    critics = {k: _critic(3.0) for k in CRITIC_WEIGHTS}
    result = fallback_aggregate(critics)
    assert result["verdict"] == "REJECT"
    assert result["final_score"] < 5.0


def test_polish_band_when_score_between_5_and_75():
    critics = {k: _critic(6.0) for k in CRITIC_WEIGHTS}
    result = fallback_aggregate(critics)
    assert result["verdict"] == "POLISH"
    assert 5.0 <= result["final_score"] < 7.5


def test_weighted_mean_emphasises_task_and_strategy():
    """Strategy and task scores are weighted 1.5x; final score should reflect that."""
    critics = {
        "visual_quality": _critic(2.0),
        "kinematic_quality": _critic(2.0),
        "task_success": _critic(10.0, "COMPLETE"),
        "strategy": _critic(10.0, "EXEMPLARY"),
        "safety": _critic(2.0),
    }
    result = fallback_aggregate(critics)
    # Weighted mean: (2 + 2 + 1.5*10 + 1.5*10 + 2) / 6 = 36 / 6 = 6.0
    assert result["final_score"] == pytest.approx(6.0, abs=0.05)


def test_string_score_is_coerced():
    critics = {k: _critic(5.0) for k in CRITIC_WEIGHTS}
    critics["visual_quality"]["score"] = "8.5"  # type: ignore[assignment]
    result = fallback_aggregate(critics)
    # Should not raise; score is coerced to float.
    assert isinstance(result["final_score"], float)


def test_missing_critic_does_not_crash():
    critics = {"visual_quality": _critic(7.0)}
    result = fallback_aggregate(critics)
    assert "verdict" in result and "final_score" in result


def test_reject_verdict_set_includes_documented_terms():
    assert {"REJECT", "UNSAFE", "FAILED"} <= REJECT_VERDICTS
