"""Threshold filtering + push filtered LeRobot subset to the HuggingFace Hub.

The v3 LeRobot layout packs many episodes per parquet/video shard. Re-sharding
to a filtered subset requires non-trivial work (recompute episode pointer
columns, regenerate stats, re-chunk videos with ffmpeg). For a hackathon demo
we take the pragmatic path:

- Mirror the source dataset's metadata (info.json, tasks.parquet, episodes
  parquet shards) and the chunk files that contain at least one kept episode.
- Add a ``crucible_curation`` block to ``info.json`` listing the kept episode
  indices and instructing downstream consumers to load via
  ``LeRobotDataset(repo, episodes=[...])`` rather than the default
  range-based loader.
- Write a dataset card that explains exactly what was filtered and why.

This produces a Hub repo that judges and humans can browse meaningfully and
that LeRobotDataset can load when given the explicit episode list. It does
NOT pretend to be a fully re-shared dataset.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_download

from .config import CrucibleConfig
from .lerobot_io import (
    DEFAULT_TASKS_PATH,
    LEGACY_EPISODES_JSONL,
    LEGACY_TASKS_JSON,
    LEGACY_TASKS_JSONL,
    _data_parquet_candidates,
    _list_repo_files,
    _load_dataset_meta,
    _video_path_candidates,
)

logger = logging.getLogger(__name__)


def select_episodes(results: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    filtered: list[dict] = []
    for r in results:
        score = float(((r.get("verdict") or {}).get("final_score")) or 0.0)
        verdict = ((r.get("verdict") or {}).get("verdict") or "").upper()
        if score >= threshold and verdict != "REJECT":
            kept.append(r)
        else:
            filtered.append(r)
    return kept, filtered


def _safe_download(repo_id: str, path: str) -> str | None:
    try:
        return hf_hub_download(repo_id, path, repo_type="dataset")
    except Exception as exc:
        logger.debug("download failed for %s/%s: %s", repo_id, path, exc)
        return None


def _resolve_remote_path(candidates: Iterable[str], remote_files: set[str]) -> str | None:
    for c in candidates:
        if c in remote_files:
            return c
    return None


def _copy_into(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _build_dataset_card(
    source_repo: str,
    target_repo: str,
    threshold: float,
    kept: list[dict],
    filtered: list[dict],
    layout_version: int,
) -> str:
    avg_kept = sum(((r.get("verdict") or {}).get("final_score") or 0) for r in kept) / max(len(kept), 1)
    avg_filtered = sum(((r.get("verdict") or {}).get("final_score") or 0) for r in filtered) / max(len(filtered), 1)
    rejected_lines = []
    for r in filtered[:25]:
        v = r.get("verdict") or {}
        score = v.get("final_score") or 0.0
        rejected_lines.append(
            f"- Episode {r.get('episode_index')}: score {score:.2f} "
            f"({v.get('verdict')}) — {(v.get('top_concern') or v.get('summary') or '').strip()[:200]}"
        )
    rejected_block = "\n".join(rejected_lines) if rejected_lines else "_No episodes were filtered out._"
    kept_indices = [r.get("episode_index") for r in kept]

    load_block = (
        "from lerobot import LeRobotDataset\n"
        f'ds = LeRobotDataset("{target_repo}", episodes={kept_indices[:50]}{"  # ...truncated" if len(kept_indices) > 50 else ""})'
    ) if layout_version == 3 else (
        f'from huggingface_hub import snapshot_download\n'
        f'snapshot_download("{target_repo}", repo_type="dataset")'
    )

    return f"""---
license: mit
tags:
- robotics
- lerobot
- curated
- crucible
- amd-hackathon-2026
---

# {target_repo}

A curated subset of [`{source_repo}`](https://huggingface.co/datasets/{source_repo}) selected by
[Crucible](https://github.com/lord-arbiter/Crucible) — a VLM-judged data
curation studio for robot demonstrations, built for the AMD Developer
Hackathon 2026.

## Filter

- Keep threshold: **score ≥ {threshold:.2f}** (KEEP / POLISH only; REJECTs always dropped)
- Episodes kept: **{len(kept)} of {len(kept) + len(filtered)}**
- Average score of kept episodes: **{avg_kept:.2f}**
- Average score of filtered episodes: **{avg_filtered:.2f}**

## How Crucible scored episodes

Each episode was reviewed by five Qwen3-VL specialist critics running on AMD MI300X:

| Critic | Axis |
|---|---|
| Visual Quality | Lighting, blur, occlusion, camera issues |
| Kinematic Quality | Jerk, recovery moves, idle drift, saturation |
| Task Success | Did the task actually complete? |
| Strategy | Was the operator's approach efficient and teachable? |
| Safety | Near-collisions, drops, unsafe contact |

A weighted aggregator (task + strategy weighted 1.5×) produced the final score
and a KEEP / POLISH / REJECT verdict.

## Loading this curated subset

This repo mirrors the source dataset's chunked layout. The kept episode
indices are recorded in `meta/info.json` under `crucible_curation`. To load
only the kept episodes:

```python
{load_block}
```

If you want raw access without the LeRobot library, use `huggingface_hub.snapshot_download`
and read the parquet shards directly.

## Filtered episodes (top 25 with rationale)

{rejected_block}

## Provenance

- Source: [`{source_repo}`](https://huggingface.co/datasets/{source_repo})
- Curated by: Crucible (https://github.com/lord-arbiter/Crucible)
- Dataset layout: LeRobotDataset {"v3.0" if layout_version == 3 else "v2.x"}
"""


def _write_curation_metadata(
    target_dir: Path,
    kept_records: list[dict],
    source_repo: str,
    threshold: float,
    layout_version: int,
) -> None:
    """Write info.json (with crucible_curation block) and a per-episode index summary."""
    info_path = target_dir / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info: dict = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            info = {}
    kept_indices = [int(r.get("episode_index")) for r in kept_records if r.get("episode_index") is not None]
    info["crucible_curation"] = {
        "source_repo": source_repo,
        "layout_version": layout_version,
        "threshold": threshold,
        "n_kept": len(kept_records),
        "kept_episode_indices": kept_indices,
        "note": (
            "This dataset preserves the source layout. Episode indices are NOT "
            "re-keyed; non-kept episodes' rows may still be present in chunked "
            "data/video files. Load with `LeRobotDataset(repo, episodes=kept_episode_indices)` "
            "to consume only the curated subset."
        ),
    }
    # Don't lie about total_episodes — keep the source value but make filtered count explicit.
    info_path.write_text(json.dumps(info, indent=2))

    summary_path = target_dir / "meta" / "crucible_episode_index.json"
    summary_path.write_text(json.dumps([
        {
            "episode_index": r.get("episode_index"),
            "task_description": r.get("task_description"),
            "verdict": (r.get("verdict") or {}).get("verdict"),
            "final_score": (r.get("verdict") or {}).get("final_score"),
            "top_concern": (r.get("verdict") or {}).get("top_concern"),
        }
        for r in kept_records
    ], indent=2))


async def push_filtered_to_hub(
    source_repo: str,
    results: list[dict],
    threshold: float,
    target_repo: str,
    hf_token: str,
    *,
    cfg: CrucibleConfig | None = None,
) -> dict:
    cfg = cfg or CrucibleConfig()
    if not hf_token:
        raise ValueError("HF token is required to push to the Hub")
    if not target_repo or "/" not in target_repo:
        raise ValueError("target_repo must be in the form 'username/dataset_name'")

    kept, filtered = select_episodes(results, threshold)
    if not kept:
        return {"ok": False, "error": "No episodes passed the threshold; refusing to push an empty dataset."}

    return await asyncio.to_thread(
        _push_filtered_sync,
        source_repo,
        target_repo,
        threshold,
        kept,
        filtered,
        hf_token,
    )


def _push_filtered_sync(
    source_repo: str,
    target_repo: str,
    threshold: float,
    kept: list[dict],
    filtered: list[dict],
    hf_token: str,
) -> dict:
    api = HfApi(token=hf_token)
    create_repo(target_repo, repo_type="dataset", exist_ok=True, token=hf_token)

    source_meta = _load_dataset_meta(source_repo, cache_dir=None)
    remote_files = set(_list_repo_files(source_repo))

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)

        # 1. Mirror metadata files. v3 prefers tasks.parquet + meta/episodes/*.parquet.
        for meta_file in (
            "meta/info.json",
            DEFAULT_TASKS_PATH,          # v3 tasks.parquet
            LEGACY_TASKS_JSONL,          # v2 fallback
            LEGACY_TASKS_JSON,
            "meta/stats.json",
            LEGACY_EPISODES_JSONL,       # v2 fallback
        ):
            if meta_file in remote_files:
                local = _safe_download(source_repo, meta_file)
                if local:
                    _copy_into(local, staging / meta_file)

        # v3: copy every meta/episodes/chunk-*/file-*.parquet shard wholesale —
        # they're small and judges may want to inspect the per-episode pointer rows.
        for path in remote_files:
            if path.startswith("meta/episodes/") and path.endswith(".parquet"):
                local = _safe_download(source_repo, path)
                if local:
                    _copy_into(local, staging / path)

        # 2. For v3, resolve each kept episode to its (chunk, file) shard and copy
        # only those (deduped). For v2, fall back to per-episode candidate paths.
        copied_data_paths: set[str] = set()
        copied_video_paths: set[str] = set()

        # Build episode -> raw row lookup so we can read pointer columns for v3.
        raw_by_idx = {int(e.get("episode_index", -1)): e for e in source_meta.episodes}

        for record in kept:
            idx = record.get("episode_index")
            if idx is None:
                continue
            idx = int(idx)

            if source_meta.layout_version == 3 and idx in raw_by_idx:
                row = raw_by_idx[idx]
                ci = row.get("data/chunk_index")
                fi = row.get("data/file_index")
                if ci is not None and fi is not None:
                    data_path = source_meta.data_path_template.format(
                        chunk_index=int(ci), file_index=int(fi)
                    )
                    if data_path not in copied_data_paths and data_path in remote_files:
                        local = _safe_download(source_repo, data_path)
                        if local:
                            _copy_into(local, staging / data_path)
                            copied_data_paths.add(data_path)
                for cam in source_meta.camera_keys:
                    vci = row.get(f"videos/{cam}/chunk_index")
                    vfi = row.get(f"videos/{cam}/file_index")
                    if vci is None or vfi is None:
                        continue
                    vid_path = source_meta.video_path_template.format(
                        video_key=cam, chunk_index=int(vci), file_index=int(vfi)
                    )
                    if vid_path not in copied_video_paths and vid_path in remote_files:
                        local = _safe_download(source_repo, vid_path)
                        if local:
                            _copy_into(local, staging / vid_path)
                            copied_video_paths.add(vid_path)
            else:
                # v2 fallback: per-episode candidate list.
                data_path = _resolve_remote_path(_data_parquet_candidates(idx), remote_files)
                if data_path and data_path not in copied_data_paths:
                    local = _safe_download(source_repo, data_path)
                    if local:
                        _copy_into(local, staging / data_path)
                        copied_data_paths.add(data_path)
                for cam in source_meta.camera_keys:
                    vid_path = _resolve_remote_path(_video_path_candidates(idx, cam), remote_files)
                    if vid_path and vid_path not in copied_video_paths:
                        local = _safe_download(source_repo, vid_path)
                        if local:
                            _copy_into(local, staging / vid_path)
                            copied_video_paths.add(vid_path)

        # 3. Write the curation metadata + dataset card.
        _write_curation_metadata(staging, kept, source_repo, threshold, source_meta.layout_version)
        card = _build_dataset_card(source_repo, target_repo, threshold, kept, filtered, source_meta.layout_version)
        (staging / "README.md").write_text(card)

        # 4. Upload everything in one go.
        api.upload_folder(
            folder_path=str(staging),
            repo_id=target_repo,
            repo_type="dataset",
            commit_message=(
                f"Crucible curated subset of {source_repo} "
                f"(kept {len(kept)} of {len(kept) + len(filtered)}, threshold {threshold:.2f})"
            ),
        )

    return {
        "ok": True,
        "target_repo_url": f"https://huggingface.co/datasets/{target_repo}",
        "n_kept": len(kept),
        "n_filtered": len(filtered),
        "n_data_files_copied": len(copied_data_paths),
        "n_video_files_copied": len(copied_video_paths),
        "layout_version": source_meta.layout_version,
    }
