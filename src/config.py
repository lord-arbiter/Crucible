"""Runtime configuration for Crucible.

Every knob is overridable via a ``CRUCIBLE_*`` environment variable. Common
backend configurations:

| Backend | CRUCIBLE_VLM_ENDPOINT | CRUCIBLE_VLM_MODEL | CRUCIBLE_VLM_API_KEY |
|---|---|---|---|
| Self-hosted vLLM (local Docker) | ``http://localhost:8001/v1`` | ``crucible-vlm`` | ``EMPTY`` |
| Hyperbolic | ``https://api.hyperbolic.xyz/v1`` | ``Qwen/Qwen3-VL-72B-Instruct`` | ``<key>`` |
| Together AI | ``https://api.together.xyz/v1`` | ``Qwen/Qwen3-VL-32B-Instruct`` | ``<key>`` |
| DashScope (Alibaba intl) | ``https://dashscope-intl.aliyuncs.com/compatible-mode/v1`` | ``qwen3-vl-plus`` | ``<key>`` |
| Local Mac MLX | ``http://localhost:8001/v1`` | ``mlx-community/Qwen3-VL-2B-Instruct-4bit`` | ``EMPTY`` |

See ``docs/recipes/`` for end-to-end setup per backend.
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
