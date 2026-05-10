"""End-to-end episode → verdict pipeline."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

from .aggregator import aggregate, fallback_aggregate
from .config import CrucibleConfig
from .critics import run_all_critics
from .lerobot_io import EpisodeBundle, stream_episodes

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, dict], None] | Callable[[int, int, dict], Awaitable[None]]


def _bundle_to_meta(bundle: EpisodeBundle) -> dict:
    return {
        "episode_index": bundle.episode_index,
        "task_description": bundle.task_description,
        "duration_s": round(bundle.duration_s, 2),
        "fps": bundle.fps,
        "n_frames_total": bundle.n_frames_total,
        "primary_camera": bundle.primary_camera,
        "raw_video_url": bundle.raw_video_url,
    }


def precache_path(cache_dir: str | Path, repo_id: str) -> Path:
    safe = repo_id.replace("/", "__")
    return Path(cache_dir) / f"{safe}.json"


def load_precached(cache_dir: str | Path, repo_id: str) -> list[dict] | None:
    p = precache_path(cache_dir, repo_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        logger.warning("Failed to load precache %s: %s", p, exc)
        return None


def save_precache(cache_dir: str | Path, repo_id: str, results: list[dict]) -> Path:
    p = precache_path(cache_dir, repo_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, default=str))
    return p


async def _maybe_await(cb: ProgressCallback | None, *args) -> None:
    if cb is None:
        return
    out = cb(*args)
    if asyncio.iscoroutine(out):
        await out


async def score_dataset(
    repo_id: str,
    cfg: CrucibleConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """Stream a LeRobot dataset and score every episode end-to-end."""
    if use_cache:
        cached = load_precached(cfg.cache_dir, repo_id)
        if cached:
            logger.info("Using precached results for %s (%d episodes)", repo_id, len(cached))
            for i, ep in enumerate(cached):
                await _maybe_await(progress_callback, i + 1, len(cached), ep)
            return cached

    if AsyncOpenAI is None:
        raise RuntimeError("openai package missing; install dependencies before scoring")

    # Resolve which backend transport this run uses (OpenAI-compat,
    # LiteLLM, or OpenAI direct) — picked from cfg.vlm_endpoint + cfg.vlm_model.
    from .critics import _get_transport
    transport = _get_transport(cfg)
    results: list[dict] = []
    bundles = stream_episodes(
        repo_id,
        cfg.max_episodes_per_run,
        frames_per_episode=cfg.frames_per_episode,
        sample_strategy=cfg.frame_sample_strategy,
    )

    started = time.time()
    for i, bundle in enumerate(bundles):
        ep_started = time.time()
        try:
            critic_results = await asyncio.wait_for(
                run_all_critics(bundle, cfg, transport),
                timeout=cfg.timeout_per_episode_s,
            )
            verdict = await asyncio.wait_for(
                aggregate(critic_results, cfg, transport),
                timeout=cfg.timeout_per_episode_s,
            )
        except TimeoutError:
            logger.warning("Episode %d timed out", bundle.episode_index)
            critic_results = {}
            verdict = fallback_aggregate(critic_results)
            verdict["timed_out"] = True
        except Exception as exc:
            logger.exception("Episode %d failed: %s", bundle.episode_index, exc)
            critic_results = {}
            verdict = fallback_aggregate(critic_results)
            verdict["error"] = str(exc)

        record = {
            **_bundle_to_meta(bundle),
            "critics": critic_results,
            "verdict": verdict,
            "elapsed_s": round(time.time() - ep_started, 2),
        }
        results.append(record)
        await _maybe_await(progress_callback, i + 1, cfg.max_episodes_per_run, record)

    logger.info("Scored %d episodes from %s in %.1fs", len(results), repo_id, time.time() - started)
    save_precache(cfg.cache_dir, repo_id, results)
    return results
