"""Runtime configuration for Crucible.

Every knob is overridable via a ``CRUCIBLE_*`` environment variable.
Crucible is **model-agnostic** — any vision-language model exposed through
an OpenAI-compatible chat-completions endpoint works. Examples below;
see ``docs/recipes/`` for end-to-end setup per backend.

| Backend / model | CRUCIBLE_VLM_ENDPOINT | CRUCIBLE_VLM_MODEL |
|---|---|---|
| OpenAI GPT-4o | ``https://api.openai.com/v1`` | ``gpt-4o`` |
| Anthropic Claude (via LiteLLM proxy) | ``http://localhost:4000/v1`` | ``claude-sonnet-4-5`` |
| Google Gemini (OpenAI compat) | ``https://generativelanguage.googleapis.com/v1beta/openai`` | ``gemini-2.5-flash`` |
| Self-hosted vLLM (local Docker) | ``http://localhost:8001/v1`` | ``crucible-vlm`` |
| Hyperbolic Qwen3-VL | ``https://api.hyperbolic.xyz/v1`` | ``Qwen/Qwen3-VL-72B-Instruct`` |
| Together AI Qwen3-VL | ``https://api.together.xyz/v1`` | ``Qwen/Qwen3-VL-32B-Instruct`` |
| DashScope (Alibaba intl) | ``https://dashscope-intl.aliyuncs.com/compatible-mode/v1`` | ``qwen3-vl-plus`` |
| Local Mac MLX | ``http://localhost:8001/v1`` | ``mlx-community/Qwen3-VL-2B-Instruct-4bit`` |

For Qwen3-family models the critic auto-appends a ``/no_think`` suffix to
suppress the chain-of-thought preamble; other model families are unaffected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class CrucibleConfig:
    vlm_endpoint: str = field(default_factory=lambda: _env_str("CRUCIBLE_VLM_ENDPOINT", "http://localhost:8001/v1"))
    vlm_model: str = field(default_factory=lambda: _env_str("CRUCIBLE_VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct"))
    vlm_api_key: str = field(default_factory=lambda: _env_str("CRUCIBLE_VLM_API_KEY", "EMPTY"))
    max_model_len: int = field(default_factory=lambda: _env_int("CRUCIBLE_MAX_MODEL_LEN", 65536))

    frames_per_episode: int = field(default_factory=lambda: _env_int("CRUCIBLE_FRAMES_PER_EPISODE", 16))
    frame_sample_strategy: str = field(default_factory=lambda: _env_str("CRUCIBLE_FRAME_SAMPLE_STRATEGY", "uniform_with_endpoints"))
    image_max_dim: int = field(default_factory=lambda: _env_int("CRUCIBLE_IMAGE_MAX_DIM", 768))

    critic_max_tokens: int = field(default_factory=lambda: _env_int("CRUCIBLE_CRITIC_MAX_TOKENS", 600))
    aggregator_max_tokens: int = field(default_factory=lambda: _env_int("CRUCIBLE_AGGREGATOR_MAX_TOKENS", 400))
    critic_temperature: float = field(default_factory=lambda: _env_float("CRUCIBLE_CRITIC_TEMPERATURE", 0.2))
    aggregator_temperature: float = field(default_factory=lambda: _env_float("CRUCIBLE_AGGREGATOR_TEMPERATURE", 0.1))
    parallel_critics: bool = True

    max_episodes_per_run: int = field(default_factory=lambda: _env_int("CRUCIBLE_MAX_EPISODES", 25))
    timeout_per_episode_s: int = field(default_factory=lambda: _env_int("CRUCIBLE_TIMEOUT_PER_EPISODE_S", 90))
    request_retries: int = field(default_factory=lambda: _env_int("CRUCIBLE_REQUEST_RETRIES", 2))

    keep_threshold: float = field(default_factory=lambda: _env_float("CRUCIBLE_KEEP_THRESHOLD", 7.5))
    polish_threshold: float = field(default_factory=lambda: _env_float("CRUCIBLE_POLISH_THRESHOLD", 5.0))

    cache_dir: str = field(default_factory=lambda: _env_str("CRUCIBLE_CACHE_DIR", "data/precached"))
    hf_download_cache: str = field(default_factory=lambda: _env_str("CRUCIBLE_HF_CACHE", os.path.expanduser("~/.cache/huggingface")))

    api_base: str = field(default_factory=lambda: _env_str("CRUCIBLE_API_BASE", "http://localhost:8000"))


DEFAULT_CONFIG = CrucibleConfig()
