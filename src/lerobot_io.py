"""
Streaming LeRobot dataset reader.

Pulls metadata + per-chunk parquet + per-chunk video from a HuggingFace
LeRobot repo and yields :class:`EpisodeBundle` objects ready for the critics.

Designed to work without a runtime dependency on the ``lerobot`` library —
we read the v3 chunked layout natively (and fall back to v2 per-episode
file layout when the source predates v3).

Canonical layout reference (verified against the lerobot repo):
- info.json["data_path"]   = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
- info.json["video_path"]  = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
- meta/tasks.parquet       (task_index column, task string as pandas index)
- meta/episodes/chunk-XXX/file-YYY.parquet
    columns: episode_index, length, tasks (list[str]),
             data/chunk_index, data/file_index,
             dataset_from_index, dataset_to_index,
             videos/<cam>/chunk_index, videos/<cam>/file_index,
             videos/<cam>/from_timestamp, videos/<cam>/to_timestamp

A single chunk file packs many episodes' rows/frames; per-episode bounds
are read out of the episodes-index parquet pointer columns. Video frames
must be decoded by seeking to the episode's `from_timestamp`, not by
sequential demux from the start of the chunk.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

try:
    import av  # PyAV for video decoding
except ImportError:  # pragma: no cover
    av = None

from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)

# Canonical default templates (matches huggingface/lerobot src/lerobot/datasets/utils.py).
DEFAULT_DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_EPISODES_PATH_PREFIX = "meta/episodes/"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
LEGACY_TASKS_JSONL = "meta/tasks.jsonl"
LEGACY_TASKS_JSON = "meta/tasks.json"
LEGACY_EPISODES_JSONL = "meta/episodes.jsonl"


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
    tasks: dict[int, str]  # task_index -> description (best-effort)
    episodes: list[dict]  # raw rows from episodes.parquet (v3) or .jsonl (v2)
    camera_keys: list[str]
    video_camera_keys: list[str]  # cameras with dtype == "video"
    image_camera_keys: list[str]  # cameras with dtype == "image"
    state_key: str | None
    action_key: str | None
    data_path_template: str
    video_path_template: str


def _list_repo_files(repo_id: str) -> list[str]:
    api = HfApi()
    return api.list_repo_files(repo_id, repo_type="dataset")


def _safe_download(repo_id: str, path: str, cache_dir: str | None = None) -> str | None:
    try:
        return hf_hub_download(repo_id, path, repo_type="dataset", cache_dir=cache_dir)
    except Exception as exc:
        logger.debug("hf_hub_download failed for %s/%s: %s", repo_id, path, exc)
        return None


def _read_json(path: str) -> dict:
    import json
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: str) -> list[dict]:
    import json
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _detect_layout_version(info: dict, files: list[str]) -> int:
    cv = str(info.get("codebase_version", "")).lower()
    if cv.startswith("v3") or any(p.startswith(DEFAULT_EPISODES_PATH_PREFIX) and p.endswith(".parquet") for p in files):
        return 3
    if LEGACY_EPISODES_JSONL in files:
        return 2
    return 3  # default to v3 — the modern layout is overwhelmingly common


def _parse_tasks_v3(repo_id: str, files: list[str], cache_dir: str | None) -> dict[int, str]:
    """Read v3 ``meta/tasks.parquet``: task_index column, task string in the pandas index."""
    if DEFAULT_TASKS_PATH not in files:
        return {}
    local = _safe_download(repo_id, DEFAULT_TASKS_PATH, cache_dir)
    if not local:
        return {}
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(local)
        df = table.to_pandas()
        # Schema: index = task description, column = task_index.
        if "task_index" in df.columns:
            return {int(row["task_index"]): str(idx) for idx, row in df.iterrows()}
        return {}
    except Exception as exc:
        logger.warning("Failed to read tasks.parquet: %s", exc)
        return {}


def _parse_tasks_v2(repo_id: str, files: list[str], cache_dir: str | None) -> dict[int, str]:
    for candidate in (LEGACY_TASKS_JSONL, LEGACY_TASKS_JSON):
        if candidate not in files:
            continue
        local = _safe_download(repo_id, candidate, cache_dir)
        if not local:
            continue
        try:
            if candidate.endswith(".jsonl"):
                rows = _read_jsonl(local)
                tasks = {}
                for r in rows:
                    idx = r.get("task_index", r.get("task_id"))
                    desc = r.get("task", r.get("task_description", r.get("description")))
                    if idx is not None and desc:
                        tasks[int(idx)] = str(desc)
                return tasks
            else:
                payload = _read_json(local)
                if isinstance(payload, dict):
                    return {int(k): (v if isinstance(v, str) else v.get("task", "")) for k, v in payload.items()}
                if isinstance(payload, list):
                    return {i: (v if isinstance(v, str) else v.get("task", "")) for i, v in enumerate(payload)}
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", candidate, exc)
    return {}


def _parse_episodes_v3(repo_id: str, files: list[str], cache_dir: str | None) -> list[dict]:
    """Walk ``meta/episodes/chunk-XXX/file-YYY.parquet`` and concatenate rows."""
    import pyarrow.parquet as pq

    episode_files = sorted(
        p for p in files
        if p.startswith(DEFAULT_EPISODES_PATH_PREFIX) and p.endswith(".parquet")
    )
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
    rows.sort(key=lambda r: int(r.get("episode_index", 0)))
    return rows


def _parse_episodes_v2(repo_id: str, files: list[str], cache_dir: str | None) -> list[dict]:
    if LEGACY_EPISODES_JSONL not in files:
        return []
    local = _safe_download(repo_id, LEGACY_EPISODES_JSONL, cache_dir)
    if not local:
        return []
    return _read_jsonl(local)


def _detect_camera_keys(info: dict) -> tuple[list[str], list[str], list[str]]:
    """Returns (all_camera_keys, video_keys, image_keys)."""
    features = info.get("features", {})
    all_keys: list[str] = []
    video_keys: list[str] = []
    image_keys: list[str] = []
    for k, spec in features.items():
        dtype = (spec or {}).get("dtype", "")
        if not k.startswith("observation.images."):
            continue
        if dtype == "video":
            video_keys.append(k)
            all_keys.append(k)
        elif dtype == "image":
            image_keys.append(k)
            all_keys.append(k)
    all_keys.sort()
    video_keys.sort()
    image_keys.sort()
    return all_keys, video_keys, image_keys


def _detect_state_action_keys(info: dict) -> tuple[str | None, str | None]:
    features = info.get("features", {})
    state_candidates = ["observation.state", "observation.qpos", "observation.joints"]
    action_candidates = ["action", "actions"]
    state_key = next((k for k in state_candidates if k in features), None)
    action_key = next((k for k in action_candidates if k in features), None)
    return state_key, action_key


def _load_dataset_meta(repo_id: str, cache_dir: str | None) -> _DatasetMeta:
    files = _list_repo_files(repo_id)
    info_path = _safe_download(repo_id, "meta/info.json", cache_dir)
    if not info_path:
        raise RuntimeError(f"meta/info.json missing on dataset {repo_id}; cannot proceed")
    info = _read_json(info_path)
    layout_version = _detect_layout_version(info, files)
    fps = int(info.get("fps", 30))
    total_episodes = int(info.get("total_episodes", info.get("num_episodes", 0)) or 0)

    if layout_version == 3:
        tasks = _parse_tasks_v3(repo_id, files, cache_dir) or _parse_tasks_v2(repo_id, files, cache_dir)
        episodes = _parse_episodes_v3(repo_id, files, cache_dir)
    else:
        tasks = _parse_tasks_v2(repo_id, files, cache_dir)
        episodes = _parse_episodes_v2(repo_id, files, cache_dir)

    if not total_episodes and episodes:
        total_episodes = len(episodes)

    cam_all, cam_video, cam_image = _detect_camera_keys(info)
    state_key, action_key = _detect_state_action_keys(info)

    data_template = info.get("data_path") or DEFAULT_DATA_PATH_TEMPLATE
    video_template = info.get("video_path") or DEFAULT_VIDEO_PATH_TEMPLATE

    return _DatasetMeta(
        repo_id=repo_id,
        fps=fps,
        total_episodes=total_episodes,
        layout_version=layout_version,
        tasks=tasks,
        episodes=episodes,
        camera_keys=cam_all,
        video_camera_keys=cam_video,
        image_camera_keys=cam_image,
        state_key=state_key,
        action_key=action_key,
        data_path_template=data_template,
        video_path_template=video_template,
    )


def _resolve_task_description(meta: _DatasetMeta, episode_row: dict) -> str:
    """v3: ``tasks`` column is list[str]. v2: ``task_index`` -> tasks.jsonl lookup."""
    raw_tasks = episode_row.get("tasks")
    if isinstance(raw_tasks, (list, tuple, np.ndarray)) and len(raw_tasks):
        first = raw_tasks[0]
        if isinstance(first, (bytes, bytearray)):
            first = first.decode("utf-8", errors="replace")
        if first:
            return str(first)
    if isinstance(raw_tasks, str) and raw_tasks:
        return raw_tasks
    ti = episode_row.get("task_index")
    if isinstance(ti, (list, tuple, np.ndarray)) and len(ti):
        ti = int(ti[0])
    elif isinstance(ti, (int, np.integer)):
        ti = int(ti)
    else:
        ti = None
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


def _pick_primary_camera(meta: _DatasetMeta) -> tuple[str | None, bool]:
    """Returns (camera_key, is_video). Prefer video cameras over image-only."""
    pool = meta.video_camera_keys or meta.image_camera_keys
    if not pool:
        return None, False
    for preferred in (
        "observation.images.cam_high",
        "observation.images.top",
        "observation.images.image",
        "observation.images.front",
        "observation.images.image_top",
    ):
        if preferred in pool:
            return preferred, preferred in meta.video_camera_keys
    return pool[0], pool[0] in meta.video_camera_keys


def _format_data_path(template: str, episode_row: dict) -> str | None:
    chunk_index = episode_row.get("data/chunk_index")
    file_index = episode_row.get("data/file_index")
    if chunk_index is None or file_index is None:
        return None
    return template.format(chunk_index=int(chunk_index), file_index=int(file_index))


def _format_video_path(template: str, cam: str, episode_row: dict) -> tuple[str | None, float, float]:
    chunk_index = episode_row.get(f"videos/{cam}/chunk_index")
    file_index = episode_row.get(f"videos/{cam}/file_index")
    from_ts = episode_row.get(f"videos/{cam}/from_timestamp", 0.0)
    to_ts = episode_row.get(f"videos/{cam}/to_timestamp", 0.0)
    if chunk_index is None or file_index is None:
        return None, 0.0, 0.0
    path = template.format(video_key=cam, chunk_index=int(chunk_index), file_index=int(file_index))
    return path, float(from_ts), float(to_ts)


def _slice_episode_rows(parquet_path: str, episode_row: dict, episode_index: int) -> Any:
    """Return a pyarrow Table containing only this episode's rows from a chunk file.

    Prefers the (from_index, to_index) pointer columns; falls back to filtering by
    episode_index if the columns are absent.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from_idx = episode_row.get("dataset_from_index")
    to_idx = episode_row.get("dataset_to_index")
    if from_idx is not None and to_idx is not None:
        # Compute chunk-local row offsets. dataset_from_index is global across the dataset
        # but the parquet file holds a contiguous global slice, so we need to know its
        # offset. The simplest correct path is to read the whole file and filter by
        # episode_index — fast on these chunked files (~20–40 MB typical).
        table = pq.read_table(parquet_path)
        if "episode_index" in table.column_names:
            mask = pa.compute.equal(table["episode_index"], pa.scalar(int(episode_index)))
            return table.filter(mask)
        # Otherwise treat from_idx/to_idx as offsets from the start of the file.
        n_rows = table.num_rows
        local_from = max(0, int(from_idx))
        local_to = min(n_rows, int(to_idx))
        return table.slice(local_from, max(0, local_to - local_from))
    table = pq.read_table(parquet_path)
    if "episode_index" in table.column_names:
        import pyarrow.compute as pc
        mask = pc.equal(table["episode_index"], pa.scalar(int(episode_index)))
        return table.filter(mask)
    return table


def _column_to_array(table, col: str | None) -> np.ndarray:
    """Convert a list-valued parquet column to a 2D numpy array (n_frames, dim).

    Real LeRobot v3 stores observation.state and action as list<float> columns.
    """
    if not col or col not in table.column_names:
        return np.zeros((table.num_rows, 0), dtype=np.float32)
    pylist = table.column(col).to_pylist()
    if not pylist:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        arr = np.stack([np.asarray(row, dtype=np.float32) for row in pylist])
        return arr
    except Exception:
        try:
            return np.asarray(pylist, dtype=np.float32)
        except Exception as exc:
            logger.warning("Failed to coerce column %s to array: %s", col, exc)
            return np.zeros((len(pylist), 0), dtype=np.float32)


def _decode_frames_seek(
    video_path: str,
    from_timestamp: float,
    to_timestamp: float,
    fps: int,
    episode_length: int,
    target_in_episode_indices: Sequence[int],
) -> tuple[list[Image.Image], list[float]]:
    """Seek to ``from_timestamp`` then decode forward, picking frames closest to each target.

    Returns frames in their original order plus their timestamps RELATIVE to the
    start of the episode (so 0.0 = first frame of the episode).
    """
    if av is None:
        raise RuntimeError("PyAV is required for video decoding; install `av`")
    if not target_in_episode_indices:
        return [], []

    # Absolute target times in the chunk video (seconds).
    targets: list[tuple[int, float]] = [
        (int(idx), from_timestamp + idx / max(fps, 1))
        for idx in target_in_episode_indices
    ]
    targets.sort(key=lambda x: x[1])

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        seek_target_seconds = max(0.0, from_timestamp - 0.2)
        try:
            seek_pts = int(seek_target_seconds / float(stream.time_base))
            container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
        except Exception:
            container.seek(int(seek_target_seconds * av.time_base), backward=True)

        frames: dict[int, tuple[Image.Image, float]] = {}
        target_idx = 0
        cutoff = to_timestamp + 0.5  # generous tail to avoid missing the last target
        for frame in container.decode(stream):
            if target_idx >= len(targets):
                break
            t = float(frame.time) if frame.time is not None else float(frame.pts * stream.time_base)
            if t > cutoff:
                break
            ep_idx, target_t = targets[target_idx]
            if t < target_t - 1.0 / max(fps, 1):
                continue
            img = Image.fromarray(frame.to_ndarray(format="rgb24"))
            frames[ep_idx] = (img, t - from_timestamp)
            target_idx += 1
    finally:
        container.close()

    ordered = sorted(frames.items())
    return [img for _, (img, _) in ordered], [round(max(0.0, rel_t), 3) for _, (_, rel_t) in ordered]


def _decode_image_column_v3(
    table,
    cam_key: str,
    target_in_episode_indices: Sequence[int],
    fps: int,
) -> tuple[list[Image.Image], list[float]]:
    """For image-only datasets, frames are stored per-row in the parquet."""
    from io import BytesIO
    if cam_key not in table.column_names:
        return [], []
    pylist = table.column(cam_key).to_pylist()
    images: list[Image.Image] = []
    timestamps: list[float] = []
    for ep_idx in target_in_episode_indices:
        ep_idx = int(ep_idx)
        if ep_idx >= len(pylist):
            continue
        raw = pylist[ep_idx]
        try:
            if isinstance(raw, dict) and raw.get("bytes") is not None:
                images.append(Image.open(BytesIO(raw["bytes"])).convert("RGB"))
            elif isinstance(raw, (bytes, bytearray)):
                images.append(Image.open(BytesIO(raw)).convert("RGB"))
            else:
                continue
            timestamps.append(round(ep_idx / max(fps, 1), 3))
        except Exception as exc:
            logger.debug("image decode failed at idx %d: %s", ep_idx, exc)
            continue
    return images, timestamps


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
    recoveries = [
        round(t / fps, 2)
        for t in sign_changes
        if abs(avg_vel[t]) > 0.5 and t + 1 < len(avg_vel) and abs(avg_vel[t + 1]) > 0.5
    ]

    gripper_events: list[str] = []
    if actions is not None and actions.ndim == 2 and actions.shape[1] >= 1:
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


def stream_episodes(
    repo_id: str,
    n: int,
    *,
    frames_per_episode: int = 16,
    sample_strategy: str = "uniform_with_endpoints",
    cache_dir: str | None = None,
) -> Iterator[EpisodeBundle]:
    meta = _load_dataset_meta(repo_id, cache_dir)
    if not meta.episodes:
        logger.warning("No episodes resolved from %s metadata; nothing to score", repo_id)
        return

    primary_cam, primary_is_video = _pick_primary_camera(meta)
    target_episodes = meta.episodes[: max(0, int(n))]

    # Per-(chunk, file) caches keyed locally so the same shard isn't downloaded twice.
    parquet_cache: dict[tuple[int, int], str] = {}
    video_cache: dict[tuple[str, int, int], str] = {}

    for ep_row in target_episodes:
        idx = int(ep_row.get("episode_index", -1))
        if idx < 0:
            continue
        try:
            data_path = _format_data_path(meta.data_path_template, ep_row)
            if data_path is None:
                logger.warning("Episode %d missing data/chunk_index; skipping", idx)
                continue
            cache_key = (int(ep_row["data/chunk_index"]), int(ep_row["data/file_index"]))
            local_parquet = parquet_cache.get(cache_key)
            if local_parquet is None:
                local_parquet = _safe_download(repo_id, data_path, cache_dir)
                if not local_parquet:
                    logger.warning("Could not download data parquet %s", data_path)
                    continue
                parquet_cache[cache_key] = local_parquet

            episode_table = _slice_episode_rows(local_parquet, ep_row, idx)
            if episode_table.num_rows == 0:
                logger.warning("Episode %d resolved to 0 rows in %s", idx, data_path)
                continue
            joint_states = _column_to_array(episode_table, meta.state_key)
            actions = _column_to_array(episode_table, meta.action_key)
            n_frames = int(ep_row.get("length") or episode_table.num_rows)
            duration_s = n_frames / max(meta.fps, 1)
            sampled_in_episode_idx = _sample_indices(n_frames, frames_per_episode, sample_strategy)

            sampled_frames: list[Image.Image] = []
            sample_timestamps: list[float] = []
            video_url: str | None = None
            if primary_cam:
                if primary_is_video:
                    vp_path, from_ts, to_ts = _format_video_path(meta.video_path_template, primary_cam, ep_row)
                    if vp_path:
                        vid_key = (primary_cam, int(ep_row[f"videos/{primary_cam}/chunk_index"]),
                                   int(ep_row[f"videos/{primary_cam}/file_index"]))
                        local_video = video_cache.get(vid_key)
                        if local_video is None:
                            local_video = _safe_download(repo_id, vp_path, cache_dir)
                            if local_video:
                                video_cache[vid_key] = local_video
                        if local_video:
                            try:
                                sampled_frames, sample_timestamps = _decode_frames_seek(
                                    local_video,
                                    from_ts,
                                    to_ts,
                                    meta.fps,
                                    n_frames,
                                    sampled_in_episode_idx,
                                )
                            except Exception as exc:
                                logger.warning("Frame decode failed for ep %d: %s", idx, exc)
                            video_url = _hub_video_url(repo_id, vp_path)
                else:
                    sampled_frames, sample_timestamps = _decode_image_column_v3(
                        episode_table, primary_cam, sampled_in_episode_idx, meta.fps
                    )

            telemetry_digest = make_telemetry_digest(joint_states, actions, meta.fps, idx)

            yield EpisodeBundle(
                episode_index=idx,
                task_description=_resolve_task_description(meta, ep_row),
                sampled_frames=sampled_frames,
                sample_timestamps=sample_timestamps,
                telemetry_digest=telemetry_digest,
                duration_s=duration_s,
                fps=meta.fps,
                raw_video_url=video_url,
                primary_camera=primary_cam,
                n_frames_total=n_frames,
                raw_joint_states=joint_states,
                raw_actions=actions,
            )
        except Exception as exc:
            logger.exception("Failed to bundle episode %d from %s: %s", idx, repo_id, exc)
            continue


# Legacy aliases retained for filtering.py and tests that reference internal helpers.
def _data_parquet_candidates(episode_index: int) -> list[str]:
    """Legacy v2 candidates only — v3 uses episode pointer columns instead."""
    chunk = episode_index // 1000
    return [
        f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet",
        f"data/episode_{episode_index:06d}.parquet",
    ]


def _video_path_candidates(episode_index: int, cam_key: str) -> list[str]:
    """Legacy v2 candidates only — v3 uses episode pointer columns instead."""
    chunk = episode_index // 1000
    return [
        f"videos/chunk-{chunk:03d}/{cam_key}/episode_{episode_index:06d}.mp4",
        f"videos/{cam_key}/episode_{episode_index:06d}.mp4",
    ]
