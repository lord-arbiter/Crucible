"""Transport-selection tests.

Pure-Python unit tests with mocks — no network calls, no API keys required,
no LiteLLM install required. Verify that ``_get_transport(cfg)`` and
``_select_transport_kind(cfg)`` route to the right backend based on the
config shape.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest

from src.config import CrucibleConfig
from src.critics import (
    KNOWN_OPENAI_MODEL_PREFIXES,
    LITELLM_PROVIDER_PREFIXES,
    _has_litellm_prefix,
    _is_known_openai_model,
    _select_transport_kind,
)


def _cfg(*, endpoint: str = "", model: str = "", api_key: str = "EMPTY") -> CrucibleConfig:
    cfg = CrucibleConfig()
    cfg.vlm_endpoint = endpoint
    cfg.vlm_model = model
    cfg.vlm_api_key = api_key
    return cfg


# --- prefix detection helpers -----------------------------------------------


def test_litellm_prefix_detected_for_anthropic():
    assert _has_litellm_prefix("anthropic/claude-sonnet-4-5")
    assert _has_litellm_prefix("ANTHROPIC/Claude-3-5-haiku")  # case-insensitive


def test_litellm_prefix_detected_for_bedrock_and_gemini():
    assert _has_litellm_prefix("bedrock/anthropic.claude-3-5-sonnet")
    assert _has_litellm_prefix("gemini/gemini-2.5-flash")
    assert _has_litellm_prefix("vertex_ai/gemini-pro")


def test_litellm_prefix_rejects_bare_models():
    assert not _has_litellm_prefix("gpt-4o")
    assert not _has_litellm_prefix("Qwen/Qwen3-VL-32B-Instruct")  # HF-style namespace, not a litellm prefix
    assert not _has_litellm_prefix("")
    assert not _has_litellm_prefix(None)


def test_known_openai_model_detected():
    for name in ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini", "o3-mini", "chatgpt-4o-latest"):
        assert _is_known_openai_model(name), name


def test_known_openai_model_rejects_litellm_prefixed():
    # Even though "openai/gpt-4o" starts with "gpt-" after the prefix, the
    # explicit "openai/" prefix means the user wants LiteLLM routing.
    assert not _is_known_openai_model("openai/gpt-4o")


def test_known_openai_model_rejects_other_models():
    assert not _is_known_openai_model("Qwen/Qwen3-VL-32B-Instruct")
    assert not _is_known_openai_model("anthropic/claude-sonnet-4-5")
    assert not _is_known_openai_model("")
    assert not _is_known_openai_model(None)


# --- transport-kind selector -------------------------------------------------


def test_endpoint_set_yields_openai_compat():
    cfg = _cfg(endpoint="https://api.hyperbolic.xyz/v1", model="Qwen/Qwen3-VL-72B-Instruct")
    assert _select_transport_kind(cfg) == "openai_compat"


def test_endpoint_set_with_localhost_still_openai_compat():
    cfg = _cfg(endpoint="http://localhost:8001/v1", model="crucible-vlm")
    assert _select_transport_kind(cfg) == "openai_compat"


def test_no_endpoint_with_litellm_prefix_yields_litellm():
    cfg = _cfg(endpoint="", model="anthropic/claude-sonnet-4-5")
    assert _select_transport_kind(cfg) == "litellm"


def test_no_endpoint_with_known_openai_model_yields_openai_direct():
    cfg = _cfg(endpoint="", model="gpt-4o-mini")
    assert _select_transport_kind(cfg) == "openai_direct"


def test_no_endpoint_no_known_model_falls_back_to_openai_compat():
    """Last-ditch path — assume self-hosted vLLM at the default localhost URL."""
    cfg = _cfg(endpoint="", model="some-random-model")
    assert _select_transport_kind(cfg) == "openai_compat"


# --- _get_transport returns a callable that respects the kind ---------------


def test_get_transport_openai_compat():
    from src.critics import _get_transport
    cfg = _cfg(endpoint="https://api.example.com/v1", model="some-model")
    transport = _get_transport(cfg)
    assert callable(transport)


def test_get_transport_litellm_missing_dep_raises_helpful_error():
    """When litellm prefix is used but litellm isn't installed, point at the extra."""
    from src.critics import _get_transport
    cfg = _cfg(endpoint="", model="anthropic/claude-sonnet-4-5")

    # Remove litellm from sys.modules and prevent re-import.
    with mock.patch.dict(sys.modules, {"litellm": None}), pytest.raises(RuntimeError) as excinfo:
        _get_transport(cfg)
    msg = str(excinfo.value)
    assert "anthropic/claude-sonnet-4-5" in msg
    assert "crucible-curation[universal]" in msg


def test_get_transport_openai_direct_callable():
    from src.critics import _get_transport
    cfg = _cfg(endpoint="", model="gpt-4o-mini", api_key="sk-fake")
    transport = _get_transport(cfg)
    assert callable(transport)


# --- constant sanity --------------------------------------------------------


def test_litellm_prefixes_all_end_with_slash():
    for p in LITELLM_PROVIDER_PREFIXES:
        assert p.endswith("/"), f"prefix {p!r} should end with /"


def test_known_openai_prefixes_no_slash():
    for p in KNOWN_OPENAI_MODEL_PREFIXES:
        assert "/" not in p, f"OpenAI bare-model prefix {p!r} should not contain /"
