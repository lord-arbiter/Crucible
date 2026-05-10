"""Threshold filtering + push filtered LeRobot subset to the HuggingFace Hub.

The filtered upload is intentionally minimal — we mirror the original
dataset metadata, keep only the selected episodes' parquet shards and
videos, regenerate the per-episode index, and write a dataset card that
explains exactly what Crucible filtered and why. We do **not** rewrite
data files; we copy the originals so the schemas stay byte-identical.
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
from .lerobot_io import _data_parquet_candidates, _list_repo_files, _load_dataset_meta, _video_path_candidates

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
) -> str:
    avg_kept = sum(((r.get("verdict") or {}).get("final_score") or 0) for r in kept) / max(len(kept), 1)
    avg_filtered = sum(((r.get("verdict") or {}).get("final_score") or 0) for r in filtered) / max(len(filtered), 1)
    rejected_lines = []
    for r in filtered[:25]:
        v = r.get("verdict") or {}
        rejected_lines.append(
            f"- Episode {r.get('episode_index')}: score {v.get('final_score'):.2f} "
            f"({v.get('verdict')}) — {(v.get('top_concern') or v.get('summary') or '').strip()[:200]}"
        )
    rejected_block = "\n".join(rejected_lines) if rejected_lines else "_No episodes were filtered out._"

    return f"""---
license: mit
tags:
- robotics
- lerobot
- curated
- crucible
---

# {target_repo}

A subset of [`{source_repo}`](https://huggingface.co/datasets/{source_repo}) curated by
[Crucible](https://huggingface.co/spaces/) — a VLM-judged data curation studio for
robot demonstrations, built for the AMD Developer Hackathon 2026.

## Filter

- Keep threshold: **score ≥ {threshold:.2f}** (KEEP / POLISH only; REJECTs always dropped)
- Episodes kept: **{len(kept)}**
- Episodes filtered out: **{len(filtered)}**
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

## Filtered episodes (top 25 with rationale)

{rejected_block}

## Reproduce

```python
from src.pipeline import score_dataset
from src.config import CrucibleConfig
results = await score_dataset("{source_repo}", CrucibleConfig())
```
"""


def _write_episode_index(target_dir: Path, kept_records: list[dict], source_meta) -> None:
    """Rewrite ``meta/info.json`` and a synthetic episode index JSON for the curated subset."""
    info_path = target_dir / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info: dict = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            info = {}
    info["total_episodes"] = len(kept_records)
    info["crucible_curation"] = {
        "source_repo": source_meta.repo_id,
        "n_kept": len(kept_records),
        "kept_episode_indices": [r.get("episode_index") for r in kept_records],
    }
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
        # Mirror metadata: info.json, tasks.jsonl, stats files if present.
        for meta_file in ("meta/info.json", "meta/tasks.jsonl", "meta/tasks.json", "meta/stats.json"):
            if meta_file in remote_files:
                local = _safe_download(source_repo, meta_file)
                if local:
                    _copy_into(local, staging / meta_file)

        # Copy parquet + video for each kept episode.
        for record in kept:
            idx = record.get("episode_index")
            if idx is None:
                continue
            data_path = _resolve_remote_path(_data_parquet_candidates(idx), remote_files)
            if data_path:
                local = _safe_download(source_repo, data_path)
                if local:
                    _copy_into(local, staging / data_path)
            for cam in source_meta.camera_keys:
                vid_path = _resolve_remote_path(_video_path_candidates(idx, cam), remote_files)
                if vid_path:
                    local = _safe_download(source_repo, vid_path)
                    if local:
                        _copy_into(local, staging / vid_path)

        _write_episode_index(staging, kept, source_meta)

        card = _build_dataset_card(source_repo, target_repo, threshold, kept, filtered)
        (staging / "README.md").write_text(card)

        # Upload everything.
        api.upload_folder(
            folder_path=str(staging),
            repo_id=target_repo,
            repo_type="dataset",
            commit_message=f"Crucible curated subset of {source_repo} (kept {len(kept)} of {len(kept) + len(filtered)})",
        )

    return {
        "ok": True,
        "target_repo_url": f"https://huggingface.co/datasets/{target_repo}",
        "n_kept": len(kept),
        "n_filtered": len(filtered),
    }
