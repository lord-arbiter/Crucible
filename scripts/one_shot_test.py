#!/usr/bin/env python3
"""Smoke test: pull one episode, run one critic against the configured vLLM endpoint.

Usage:
    python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual

Prints the resulting JSON. Use this to validate the full path
HF Hub → frame decode → telemetry digest → vLLM → JSON in less than 30 seconds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CrucibleConfig  # noqa: E402
from src.critics import CRITIC_NAMES, run_critic  # noqa: E402
from src.lerobot_io import stream_episodes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="lerobot/aloha_static_cups_open")
    parser.add_argument("--critic", default="visual", choices=CRITIC_NAMES)
    parser.add_argument("--episode", type=int, default=0, help="Skip to the Nth yielded episode.")
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()

    cfg = CrucibleConfig()
    cfg.frames_per_episode = args.frames

    bundles = stream_episodes(
        args.repo,
        n=args.episode + 1,
        frames_per_episode=cfg.frames_per_episode,
        sample_strategy=cfg.frame_sample_strategy,
    )
    bundle = None
    for i, b in enumerate(bundles):
        if i == args.episode:
            bundle = b
            break
    if bundle is None:
        print("No episode produced — check repo_id and metadata.", file=sys.stderr)
        sys.exit(2)

    print("=== EpisodeBundle ===")
    print(f"index={bundle.episode_index} task={bundle.task_description!r}")
    print(f"frames={len(bundle.sampled_frames)} duration={bundle.duration_s:.2f}s fps={bundle.fps}")
    print("Telemetry digest:\n" + bundle.telemetry_digest)

    try:
        from src.critics import _get_transport, _select_transport_kind
    except ImportError as exc:
        print(f"Failed to import critic transport: {exc}", file=sys.stderr)
        sys.exit(1)

    kind = _select_transport_kind(cfg)
    print(f"\n=== Running {args.critic} critic via transport={kind} model={cfg.vlm_model} ===")
    transport = _get_transport(cfg)
    out = await run_critic(args.critic, bundle, cfg, transport)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
