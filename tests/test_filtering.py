"""Filtering selection tests.

`select_episodes` should keep only episodes meeting both score >= threshold
and verdict != REJECT.
"""
from __future__ import annotations

from src.filtering import select_episodes


def _result(idx: int, score: float, verdict: str) -> dict:
    return {
        "episode_index": idx,
        "verdict": {"final_score": score, "verdict": verdict},
    }


def test_keep_above_threshold():
    results = [
        _result(0, 8.0, "KEEP"),
        _result(1, 6.0, "POLISH"),
        _result(2, 3.0, "REJECT"),
    ]
    kept, filtered = select_episodes(results, threshold=7.0)
    assert [r["episode_index"] for r in kept] == [0]
    assert [r["episode_index"] for r in filtered] == [1, 2]


def test_reject_verdict_drops_even_if_score_high():
    """Defensive: a REJECT verdict should never sneak into kept set."""
    results = [
        _result(0, 9.5, "REJECT"),
        _result(1, 7.5, "KEEP"),
    ]
    kept, _ = select_episodes(results, threshold=7.0)
    assert [r["episode_index"] for r in kept] == [1]


def test_threshold_inclusive():
    results = [_result(0, 7.0, "POLISH")]
    kept, _ = select_episodes(results, threshold=7.0)
    assert len(kept) == 1


def test_missing_verdict_block_is_filtered():
    results = [{"episode_index": 0}]
    kept, filtered = select_episodes(results, threshold=5.0)
    assert kept == []
    assert filtered == results


def test_lowercase_verdict_still_works():
    results = [_result(0, 8.0, "keep")]
    kept, _ = select_episodes(results, threshold=7.0)
    assert len(kept) == 1
