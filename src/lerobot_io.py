"""
Streaming LeRobot dataset reader.

Pulls metadata + per-episode parquet + per-episode video from a HuggingFace
LeRobot repo and yields :class:`EpisodeBundle` objects ready for the critics.

Designed to work without a runtime dependency on the ``lerobot`` library —
we read v3 layout (preferred) and fall back to v2 if the v3 metadata files
are absent. We handle the realistic edge cases the spec calls out:
- missing task descriptions (use a placeholder)
- fewer episodes than requested (yield what we have)
- unexpected camera key names (auto-detect "observation.images.*")
- v2 vs v3 metadata layout
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

try:
    import av  # PyAV for video decoding
except ImportError:  # pragma: no cover - PyAV is required at runtime
    av = None

from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)


@dataclass
class EpisodeBundle:
    episode_index: int
    task_description: str
    sampled_frames: list[Image.Image]
    sample_timestamps: list[float]
    telemetry_digest: str
    duration_s: float
    fps: int
    raw_video_url: str | None = None
    primary_camera: str | None = None
    n_frames_total: int = 0
    raw_joint_states: np.ndarray | None = field(default=None, repr=False)
    raw_actions: np.ndarray | None = field(default=None, repr=False)


@dataclass
class _DatasetMeta:
    repo_id: str
    fps: int
    total_episodes: int
    layout_version: int  # 2 or 3
    tasks: dict[int, str]  # task_index -> description
    episodes: list[dict]  # entries: {episode_index, length, task_index, data_path, video_paths{cam: path}}
    camera_keys: list[str]
    state_key: str | None
    action_key: str | None


def _list_repo_files(repo_id: str) -> list[str]:
    api = HfApi()
    return api.list_repo_files(repo_id, repo_type="dataset")


def _safe_download(repo_id: str, path: str, cache_dir: str | None = None) -> str | None:
    try:
        return hf_hub_download(repo_id, path, repo_type="dataset", cache_dir=cache_dir)
    except Exception as exc:
        logger.debug("hf_hub_download failed for %s/%s: %s", repo_id, path, exc)
        return None


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _parse_tasks(repo_id: str, files: list[str], cache_dir: str | None) -> dict[int, str]:
    tasks: dict[int, str] = {}
    candidates = [
        "meta/tasks.jsonl",
        "meta/tasks.json",
    ]
    for candidate in candidates:
        if candidate not in files:
            continue
        local = _safe_download(repo_id, candidate, cache_dir)
        if not local:
            continue
        try:
            if candidate.endswith(".jsonl"):
                rows = _read_jsonl(local)
                for row in rows:
                    idx = int(row.get("task_index", row.get("task_id", -1)))
                    desc = row.get("task", row.get("task_description", row.get("description", "")))
                    if idx >= 0 and desc:
                        tasks[idx] = desc
            else:
                payload = _read_json(local)
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        try:
                            tasks[int(k)] = v if isinstance(v, str) else v.get("task", "")
                        except ValueError:
                            continue
                elif isinstance(payload, list):
                    for i, v in enumerate(payload):
                        tasks[i] = v if isinstance(v, str) else v.get("task", "")
            if tasks:
                break
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", candidate, exc)
    return tasks


def _parse_episodes_v3(repo_id: str, files: list[str], cache_dir: str | None) -> list[dict]:
    """Read v3 ``meta/episodes/*.parquet``. Returns episode rows."""
    import pyarrow.parquet as pq

    episode_files = sorted(p for p in files if p.startswith("meta/episodes/") and p.endswith(".parquet"))
    if not episode_files:
        return []
    rows: list[dict] = []
    for ep_file in episode_files:
        local = _safe_download(repo_id, ep_file, cache_dir)
        if not local:
            continue
        table = pq.read_table(local)
        df = table.to_pandas()
        for _, r in df.iterrows():
            rows.append(r.to_dict())
    return rows


def _parse_episodes_v2(repo_id: str, files: list[str], cache_dir: str | None) -> list[dict]:
    """Read v2 ``meta/episodes.jsonl``."""
    if "meta/episodes.jsonl" not in files:
        return []
    local = _safe_download(repo_id, "meta/episodes.jsonl", cache_dir)
    if not local:
        return []
    return _read_jsonl(local)


def _detect_layout_version(info: dict, files: list[str]) -> int:
    if any(p.startswith("meta/episodes/") and p.endswith(".parquet") for p in files):
        return 3
    if "meta/episodes.jsonl" in files:
        return 2
    if isinstance(info.get("codebase_version"), str) and info["codebase_version"].startswith("v3"):
        return 3
    return 2


def _detect_camera_keys(info: dict) -> list[str]:
    features = info.get("features", {})
    keys: list[str] = []
    for k, spec in features.items():
        dtype = (spec or {}).get("dtype", "")
        if k.startswith("observation.images.") and dtype in {"video", "image"}:
            keys.append(k)
    keys.sort()
    return keys


def _detect_state_action_keys(info: dict) -> tuple[str | None, str | None]:
    features = info.get("features", {})
    state_candidates = ["observation.state", "observation.qpos", "observation.joints"]
    action_candidates = ["action", "actions"]
    state_key = next((k for k in state_candidates if k in features), None)
    action_key = next((k for k in action_candidates if k in features), None)
    return state_key, action_key


def _build_episode_index(
    repo_id: str,
    files: list[str],
    layout_version: int,
    cache_dir: str | None,
) -> list[dict]:
    rows = _parse_episodes_v3(repo_id, files, cache_dir) if layout_version == 3 else _parse_episodes_v2(repo_id, files, cache_dir)
    episodes: list[dict] = []
    for r in rows:
        idx = int(r.get("episode_index", r.get("episode_id", len(episodes))))
        length = int(r.get("length", r.get("num_frames", 0)) or 0)
        # task_index may be missing, may be a list
        task_index = r.get("task_index", r.get("tasks", None))
        if isinstance(task_index, (list, tuple, np.ndarray)):
            task_index = int(task_index[0]) if len(task_index) else None
        elif isinstance(task_index, (int, np.integer)):
            task_index = int(task_index)
        else:
            task_index = None
        # Some datasets store the task string directly per episode
        task_str = r.get("task", None)
        if isinstance(task_str, (list, tuple, np.ndarray)) and len(task_str):
            task_str = str(task_str[0])
        episodes.append({
            "episode_index": idx,
            "length": length,
            "task_index": task_index,
            "task_str": task_str if isinstance(task_str, str) else None,
            "raw": r,
        })
    episodes.sort(key=lambda e: e["episode_index"])
    return episodes


def _data_parquet_candidates(episode_index: int) -> list[str]:
    """Possible paths for the per-episode parquet file across LeRobot layouts."""
    chunk = episode_index // 1000
    return [
        f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet",
        f"data/chunk-{chunk:03d}/file_{episode_index:06d}.parquet",
        f"data/episode_{episode_index:06d}.parquet",
    ]


def _video_path_candidates(episode_index: int, cam_key: str) -> list[str]:
    chunk = episode_index // 1000
    return [
        f"videos/chunk-{chunk:03d}/{cam_key}/episode_{episode_index:06d}.mp4",
        f"videos/{cam_key}/episode_{episode_index:06d}.mp4",
        f"videos/chunk-{chunk:03d}/{cam_key}/file_{episode_index:06d}.mp4",
    ]


def _resolve_first_existing(repo_id: str, candidates: Sequence[str], files: set[str], cache_dir: str | None) -> str | None:
    for c in candidates:
        if c in files:
            local = _safe_download(repo_id, c, cache_dir)
            if local:
                return local
    return None


def _load_dataset_meta(repo_id: str, cache_dir: str | None) -> _DatasetMeta:
    files = _list_repo_files(repo_id)
    info_path = _safe_download(repo_id, "meta/info.json", cache_dir)
    if not info_path:
        raise RuntimeError(f"meta/info.json missing on dataset {repo_id}; cannot proceed")
    info = _read_json(info_path)
    layout_version = _detect_layout_version(info, files)
    fps = int(info.get("fps", 30))
    total_episodes = int(info.get("total_episodes", info.get("num_episodes", 0)) or 0)

    tasks = _parse_tasks(repo_id, files, cache_dir)
    episodes = _build_episode_index(repo_id, files, layout_version, cache_dir)
    if not total_episodes and episodes:
        total_episodes = len(episodes)

    camera_keys = _detect_camera_keys(info)
    state_key, action_key = _detect_state_action_keys(info)

    return _DatasetMeta(
        repo_id=repo_id,
        fps=fps,
        total_episodes=total_episodes,
        layout_version=layout_version,
        tasks=tasks,
        episodes=episodes,
        camera_keys=camera_keys,
        state_key=state_key,
        action_key=action_key,
    )


def _resolve_task_description(meta: _DatasetMeta, episode_row: dict) -> str:
    if episode_row.get("task_str"):
        return str(episode_row["task_str"])
    ti = episode_row.get("task_index")
    if ti is not None and ti in meta.tasks:
        return meta.tasks[ti]
    if meta.tasks:
        return next(iter(meta.tasks.values()))
    return "Unknown task (no task description present in dataset metadata)."


def _sample_indices(n_total: int, n_target: int, strategy: str) -> list[int]:
    if n_total <= 0:
        return []
    if n_target >= n_total:
        return list(range(n_total))
    if strategy == "uniform_with_endpoints" and n_target >= 2:
        return [int(round(i * (n_total - 1) / (n_target - 1))) for i in range(n_target)]
    step = n_total / n_target
    return [min(n_total - 1, int(round((i + 0.5) * step))) for i in range(n_target)]


def _decode_frames_from_video(video_path: str, frame_indices: Sequence[int], fps: int) -> tuple[list[Image.Image], list[float]]:
    """
    Decode a small set of frames at given indices via PyAV.

    PyAV's seek is keyframe-based, so for sparse uniform sampling we just
    iterate sequentially and keep the frames whose indices we need. For
    typical episode lengths (~1k–4k frames) this is fast enough.
    """
    if av is None:
        raise RuntimeError("PyAV is required for video decoding; install `av`")
    needed = set(int(i) for i in frame_indices)
    if not needed:
        return [], []
    frames: dict[int, Image.Image] = {}
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        decoded_idx = 0
        max_needed = max(needed)
        for packet in container.demux(stream):
            for frame in packet.decode():
                if decoded_idx in needed:
                    img = frame.to_ndarray(format="rgb24")
                    frames[decoded_idx] = Image.fromarray(img)
                decoded_idx += 1
                if decoded_idx > max_needed:
                    break
            if decoded_idx > max_needed:
                break
    finally:
        container.close()
    ordered = sorted(needed)
    images = [frames[i] for i in ordered if i in frames]
    timestamps = [i / max(fps, 1) for i in ordered if i in frames]
    return images, timestamps


def _read_episode_arrays(parquet_path: str, state_key: str | None, action_key: str | None) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    def _col_to_array(col: str | None) -> np.ndarray:
        if not col or col not in df.columns:
            return np.zeros((len(df), 0), dtype=np.float32)
        arr = df[col].tolist()
        try:
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            return np.zeros((len(df), 0), dtype=np.float32)

    return _col_to_array(state_key), _col_to_array(action_key)


def make_telemetry_digest(
    joint_states: np.ndarray,
    actions: np.ndarray,
    fps: int,
    episode_index: int,
) -> str:
    duration = len(joint_states) / max(fps, 1) if len(joint_states) else 0.0
    if joint_states.ndim != 2 or joint_states.shape[0] < 2 or joint_states.shape[1] == 0:
        return f"Episode {episode_index}: telemetry unavailable (no joint state column)."

    velocities = np.diff(joint_states, axis=0) * fps  # rad/s
    peak_vel = np.abs(velocities).max(axis=0)
    mean_abs_vel = np.abs(velocities).mean(axis=0)

    speed = np.linalg.norm(velocities, axis=1)
    idle_mask = speed < 0.01
    idle_runs = _consecutive_runs(idle_mask)
    long_idles = [(round(s / fps, 2), round(e / fps, 2)) for s, e in idle_runs if (e - s) / fps >= 0.5]

    avg_vel = velocities.mean(axis=1)
    sign_changes = np.where(np.diff(np.sign(avg_vel)) != 0)[0]
    recoveries = [round(t / fps, 2) for t in sign_changes if abs(avg_vel[t]) > 0.5 and t + 1 < len(avg_vel) and abs(avg_vel[t + 1]) > 0.5]

    gripper_events: list[str] = []
    if actions is not None and actions.ndim == 2 and actions.shape[1] >= 1:
        # Heuristic: last action dim is often the gripper.
        gripper_dim = actions.shape[1] - 1
        g = actions[:, gripper_dim]
        if g.size:
            gnorm = (g - g.min()) / max(g.max() - g.min(), 1e-6)
            high = gnorm > 0.7
            transitions = np.where(np.diff(high.astype(int)) != 0)[0]
            for t in transitions[:6]:
                state_after = "open" if high[t + 1] else "close"
                gripper_events.append(f"{round(int(t) / fps, 2)}s:{state_after}")

    return "\n".join([
        f"Episode {episode_index}: duration {duration:.1f}s @ {fps}fps",
        f"Joint dim: {joint_states.shape[1]}, frames: {joint_states.shape[0]}",
        f"Peak |joint velocity| (rad/s): {peak_vel.round(2).tolist()}",
        f"Mean |joint velocity| (rad/s): {mean_abs_vel.round(2).tolist()}",
        f"Idle periods >=0.5s: {long_idles[:5] if long_idles else 'none'}",
        f"Possible recovery moves at: {recoveries[:5] if recoveries else 'none'}",
        f"Gripper transitions (heuristic): {gripper_events if gripper_events else 'none detected'}",
    ])


def _consecutive_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = (np.where(diff == 1)[0] + 1).tolist()
    ends = (np.where(diff == -1)[0] + 1).tolist()
    if mask[0]:
        starts = [0, *starts]
    if mask[-1]:
        ends = [*ends, int(mask.size)]
    return list(zip(starts, ends, strict=False))


def _hub_video_url(repo_id: str, path: str | None) -> str | None:
    if not path:
        return None
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"


def _pick_primary_camera(meta: _DatasetMeta) -> str | None:
    if not meta.camera_keys:
        return None
    for preferred in ("observation.images.cam_high", "observation.images.top", "observation.images.image"):
        if preferred in meta.camera_keys:
            return preferred
    return meta.camera_keys[0]


def stream_episodes(
    repo_id: str,
    n: int,
    *,
    frames_per_episode: int = 16,
    sample_strategy: str = "uniform_with_endpoints",
    cache_dir: str | None = None,
) -> Iterator[EpisodeBundle]:
    meta = _load_dataset_meta(repo_id, cache_dir)
    files_set = set(_list_repo_files(repo_id))
    primary_cam = _pick_primary_camera(meta)
    target_episodes = meta.episodes[: max(0, int(n))] if meta.episodes else []
    if not target_episodes:
        logger.warning("No episodes resolved from %s metadata; nothing to score", repo_id)
        return

    for ep in target_episodes:
        idx = ep["episode_index"]
        try:
            parquet_local = _resolve_first_existing(repo_id, _data_parquet_candidates(idx), files_set, cache_dir)
            if not parquet_local:
                logger.warning("No data parquet found for episode %d in %s", idx, repo_id)
                continue
            joint_states, actions = _read_episode_arrays(parquet_local, meta.state_key, meta.action_key)
            n_frames = ep["length"] or len(joint_states)
            duration_s = n_frames / max(meta.fps, 1)

            sampled_indices = _sample_indices(n_frames, frames_per_episode, sample_strategy)
            sampled_frames: list[Image.Image] = []
            sample_timestamps: list[float] = []
            video_relpath: str | None = None
            if primary_cam:
                video_local = _resolve_first_existing(
                    repo_id, _video_path_candidates(idx, primary_cam), files_set, cache_dir
                )
                if video_local:
                    video_relpath = next(
                        (p for p in _video_path_candidates(idx, primary_cam) if p in files_set), None
                    )
                    try:
                        sampled_frames, sample_timestamps = _decode_frames_from_video(
                            video_local, sampled_indices, meta.fps
                        )
                    except Exception as exc:
                        logger.warning("Frame decode failed for episode %d: %s", idx, exc)

            # Image-only datasets — fall back to reading PNGs from the parquet rows.
            if not sampled_frames and primary_cam and primary_cam in _read_parquet_columns(parquet_local):
                sampled_frames, sample_timestamps = _decode_image_column(parquet_local, primary_cam, sampled_indices, meta.fps)

            telemetry_digest = make_telemetry_digest(joint_states, actions, meta.fps, idx)

            yield EpisodeBundle(
                episode_index=idx,
                task_description=_resolve_task_description(meta, ep),
                sampled_frames=sampled_frames,
                sample_timestamps=sample_timestamps,
                telemetry_digest=telemetry_digest,
                duration_s=duration_s,
                fps=meta.fps,
                raw_video_url=_hub_video_url(repo_id, video_relpath),
                primary_camera=primary_cam,
                n_frames_total=n_frames,
                raw_joint_states=joint_states,
                raw_actions=actions,
            )
        except Exception as exc:
            logger.exception("Failed to bundle episode %d from %s: %s", idx, repo_id, exc)
            continue


def _read_parquet_columns(parquet_path: str) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_schema(parquet_path).names


def _decode_image_column(
    parquet_path: str,
    column: str,
    indices: Sequence[int],
    fps: int,
) -> tuple[list[Image.Image], list[float]]:
    """Image-only LeRobot datasets store frames as PNG bytes (or relative paths) per row."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=[column])
    raws = table.column(column).to_pylist()
    images: list[Image.Image] = []
    timestamps: list[float] = []
    for i in indices:
        if i >= len(raws):
            continue
        raw = raws[i]
        try:
            if isinstance(raw, dict) and "bytes" in raw:
                from io import BytesIO
                images.append(Image.open(BytesIO(raw["bytes"])).convert("RGB"))
            elif isinstance(raw, (bytes, bytearray)):
                from io import BytesIO
                images.append(Image.open(BytesIO(raw)).convert("RGB"))
            else:
                continue
            timestamps.append(i / max(fps, 1))
        except Exception:
            continue
    return images, timestamps
