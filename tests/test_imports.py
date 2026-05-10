"""Smoke test: every src/* module imports without optional GPU/network deps."""
from __future__ import annotations

import importlib

import pytest

MODULES = [
    "src.config",
    "src.aggregator",
    "src.critics",
    "src.filtering",
    "src.pipeline",
    "src.lerobot_io",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_config_round_trip() -> None:
    from src.config import CrucibleConfig

    cfg = CrucibleConfig()
    assert cfg.frames_per_episode > 0
    assert cfg.keep_threshold > cfg.polish_threshold
    assert cfg.vlm_endpoint.startswith("http")


def test_critic_prompts_load() -> None:
    from src.critics import CRITIC_NAMES, load_aggregator_prompt, load_prompt

    assert set(CRITIC_NAMES) == {"visual", "kinematic", "task", "strategy", "safety"}
    for name in CRITIC_NAMES:
        text = load_prompt(name)
        assert "Output strict JSON" in text
        assert len(text) > 200
    assert "final_score" in load_aggregator_prompt()
