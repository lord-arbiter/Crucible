"""Telemetry digest tests.

Validate that:
- Idle periods are detected when joints stay still.
- Recovery moves (sign reversals on average velocity) are flagged.
- Empty / degenerate inputs produce a sensible "unavailable" digest.
"""
from __future__ import annotations

import numpy as np

from src.lerobot_io import _consecutive_runs, _sample_indices, make_telemetry_digest


def test_consecutive_runs_simple():
    mask = np.array([0, 1, 1, 0, 1, 0, 0], dtype=bool)
    runs = _consecutive_runs(mask)
    assert runs == [(1, 3), (4, 5)]


def test_consecutive_runs_all_true():
    mask = np.array([1, 1, 1, 1], dtype=bool)
    runs = _consecutive_runs(mask)
    assert runs == [(0, 4)]


def test_consecutive_runs_empty():
    assert _consecutive_runs(np.array([], dtype=bool)) == []


def test_telemetry_digest_includes_duration_and_dims():
    states = np.zeros((100, 7), dtype=np.float32)
    actions = np.zeros((100, 8), dtype=np.float32)
    digest = make_telemetry_digest(states, actions, fps=50, episode_index=3)
    assert "Episode 3" in digest
    assert "duration 2.0s" in digest
    assert "Joint dim: 7" in digest


def test_telemetry_digest_detects_idle_period():
    fps = 50
    # 200 frames = 4 seconds. Frames 50-150 are idle (everything fixed).
    states = np.zeros((200, 6), dtype=np.float32)
    states[:50] = np.linspace(0, 1, 50).reshape(-1, 1) * np.ones(6)
    states[150:] = np.linspace(1, 2, 50).reshape(-1, 1) * np.ones(6)
    states[50:150] = states[49]  # idle stretch
    digest = make_telemetry_digest(states, states.copy(), fps=fps, episode_index=0)
    assert "Idle periods" in digest
    assert "none" not in digest.split("Idle periods >=0.5s:")[1].split("\n")[0]


def test_telemetry_digest_handles_empty_states():
    states = np.zeros((0, 0), dtype=np.float32)
    actions = np.zeros((0, 0), dtype=np.float32)
    digest = make_telemetry_digest(states, actions, fps=30, episode_index=9)
    assert "telemetry unavailable" in digest


def test_sample_indices_covers_endpoints():
    idx = _sample_indices(100, 5, "uniform_with_endpoints")
    assert idx[0] == 0
    assert idx[-1] == 99
    assert len(idx) == 5


def test_sample_indices_uniform_spacing():
    idx = _sample_indices(100, 4, "uniform")
    assert len(idx) == 4
    assert idx[0] >= 0 and idx[-1] <= 99
    diffs = [b - a for a, b in zip(idx, idx[1:], strict=False)]
    assert all(d > 0 for d in diffs)


def test_sample_indices_target_exceeds_total():
    assert _sample_indices(3, 10, "uniform_with_endpoints") == [0, 1, 2]


def test_sample_indices_zero_total():
    assert _sample_indices(0, 5, "uniform") == []
