#!/usr/bin/env python3
"""Run a full LeRobot dataset through Crucible and save the JSON to data/precached/.

Run on the GPU box (or wherever the vLLM endpoint is reachable). The output
file is committed to the repo so the HF Space can serve the demo dataset
even when the GPU droplet is offline.

Usage:
    python scripts/precache_demo.py --repo lerobot/aloha_mobile_cabinet --episodes 25
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CrucibleConfig  # noqa: E402
from src.pipeline import save_precache, score_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--no-existing-cache", action="store_true",
                        help="Do not reuse a precache file even if one exists.")
    args = parser.parse_args()

    cfg = CrucibleConfig()
    cfg.max_episodes_per_run = args.episodes
    cfg.frames_per_episode = args.frames

    def cb(i: int, total: int, ep: dict) -> None:
        v = ep.get("verdict") or {}
        print(f"[{i}/{total}] ep {ep.get('episode_index')} score={v.get('final_score')} verdict={v.get('verdict')}")

    results = await score_dataset(
        args.repo,
        cfg,
        progress_callback=cb,
        use_cache=not args.no_existing_cache,
    )
    saved = save_precache(cfg.cache_dir, args.repo, results)
    print(f"Saved {len(results)} episodes to {saved}")


if __name__ == "__main__":
    asyncio.run(main())
