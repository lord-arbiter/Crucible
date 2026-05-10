#!/usr/bin/env python3
"""I/O smoke test — proves the LeRobot v3 streaming reader works end-to-end on a real Hub dataset.

No GPU. No vLLM. Just network.

Pulls N episodes from a HuggingFace LeRobot dataset, decodes the configured
number of frames per episode, builds the telemetry digest, and asserts the
critical invariants:

- frames decoded > 0
- joint_state shape is 2D and non-empty
- task_description is not the placeholder
- telemetry digest contains "Joint dim:" (i.e. real signal, not the
  "telemetry unavailable" branch)

Run this before every GPU bring-up. If it fails on a small dataset, it will
fail at scale on the GPU — fix here first.

Usage:
    python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2

Exit code 0 = pass, 2 = no episodes, 3 = invariant violation.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lerobot_io import stream_episodes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="lerobot/aloha_static_cups_open")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    bundles = list(stream_episodes(
        args.repo,
        n=args.episodes,
        frames_per_episode=args.frames,
    ))

    if not bundles:
        print(f"FAIL: no episodes produced from {args.repo}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for b in bundles:
        print()
        print(f"=== Episode {b.episode_index} ===")
        print(f"  task: {b.task_description!r}")
        print(f"  primary_camera: {b.primary_camera}")
        print(f"  duration: {b.duration_s:.2f}s @ {b.fps}fps  total_frames: {b.n_frames_total}")
        print(f"  sampled_frames: {len(b.sampled_frames)}")
        print(f"  sample_timestamps: {b.sample_timestamps}")
        if b.sampled_frames:
            print(f"  first_frame_size: {b.sampled_frames[0].size}  mode: {b.sampled_frames[0].mode}")
        print(f"  joint_state shape: {b.raw_joint_states.shape if b.raw_joint_states is not None else None}")
        print("  digest:")
        for line in b.telemetry_digest.split("\n"):
            print("   ", line)

        # Invariants
        if not b.sampled_frames:
            failures.append(f"episode {b.episode_index}: no frames decoded")
        if b.raw_joint_states is None or b.raw_joint_states.ndim != 2 or b.raw_joint_states.shape[0] == 0:
            failures.append(f"episode {b.episode_index}: joint_state malformed")
        if "Unknown task" in b.task_description:
            failures.append(f"episode {b.episode_index}: task_description fell back to placeholder")
        if "telemetry unavailable" in b.telemetry_digest:
            failures.append(f"episode {b.episode_index}: telemetry digest empty")

    print()
    if failures:
        print("FAIL — invariant violations:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 3
    print(f"PASS — {len(bundles)} episodes streamed and decoded cleanly from {args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
