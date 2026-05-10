# Crucible

**A multi-axis VLM-judged data curation studio for robot demonstrations.**

[![Tests](https://github.com/lord-arbiter/Crucible/actions/workflows/test.yml/badge.svg)](https://github.com/lord-arbiter/Crucible/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Powered by Qwen3-VL](https://img.shields.io/badge/powered%20by-Qwen3--VL-9cf)](https://huggingface.co/Qwen)
[![LeRobot v3](https://img.shields.io/badge/LeRobot-v3.0-orange)](https://huggingface.co/lerobot)

[Live Demo](https://huggingface.co/spaces/) · [Docs](docs/) · [Recipes](docs/recipes/) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md)

---

Crucible reads a HuggingFace LeRobot dataset and runs five Qwen3-VL specialist critic agents on every episode — visual quality, kinematic quality, task success, **strategy**, **safety** — then fuses their outputs into a single `KEEP / POLISH / REJECT` verdict with rationale and timestamp evidence. Drag a threshold and push the curated subset back to the Hub in one click.

It works against any OpenAI-compatible Qwen3-VL endpoint (self-hosted vLLM, Hyperbolic, Together AI, DashScope, AWS-hosted) so the barrier to "try it" is a free API key, not a $32k GPU.

## Why this exists

Robotics teams spend ~$118 / hour on teleoperation data, and 20–40 % of it silently teaches policies bad habits. Heuristic curation (jerk, path-efficiency, actuator saturation) catches motion artefacts. A VLM that only judges binary task success (à la `score_lerobot_episodes` calling Gemini) catches outright failures. **Neither catches the silent killers**: operator hesitation, multi-attempt regrasps, inefficient strategies, near-miss safety incidents.

A bad-strategy episode that *succeeds* is worse than a good-strategy episode that fails — the former teaches the policy unreliable recovery as if it were normal behavior.

Crucible's contribution is the **multi-axis behavioral rubric** with structured per-frame timestamp evidence, plus an inference economics story that lets a robotics lab curate millions of frames on their own hardware.

## How it compares

| Tool | Approach | Output | What it misses |
|---|---|---|---|
| `score_lerobot_episodes` (Oct 2025) | 7 CV heuristics + 1 binary Gemini call | scalar score per axis | Strategy, hesitation, behavioral nuance |
| AgiBot Genie Centurion / Task Sentinel (May 2025) | fine-tuned MiniGPT-4 success classifier in teleop loop | online intervention during collection | Offline curation of completed datasets |
| LeRobot dataset visualizer | low-movement / jerk / outlier-length filters | filter UI suggestions | Anything semantic |
| Stanford Quality-over-Quantity | influence-function math on policy gradients | continuous demo importance | Requires a trained policy; per-axis explainability |
| **Crucible** | **5-axis behavioral rubric, 1 aggregator, 1 served VLM, structured rationale + timestamp evidence** | **per-episode JSON, KEEP / POLISH / REJECT** | **n/a — this is the gap** |

## Quickstart (5 minutes, no GPU required)

The fastest way to try Crucible is against any hosted OpenAI-compatible Qwen3-VL endpoint. See [docs/recipes/hosted-api-quickstart.md](docs/recipes/hosted-api-quickstart.md) for the generic recipe; provider-specific recipes for [Hyperbolic](docs/recipes/hyperbolic.md), [Together AI](docs/recipes/together-ai.md), and [DashScope](docs/recipes/dashscope.md) are also in the same folder.

```bash
# 1. Clone & install
git clone https://github.com/lord-arbiter/Crucible
cd Crucible
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Point at your endpoint
export CRUCIBLE_VLM_ENDPOINT=https://api.your-provider.com/v1
export CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
export CRUCIBLE_VLM_API_KEY=sk-your-key

# 3. Score one episode end-to-end
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual

# 4. Score a small dataset and inspect results
python scripts/precache_demo.py \
    --repo lerobot/aloha_static_cups_open \
    --episodes 5
```

You can also run Crucible **without any VLM** to validate the I/O layer:

```bash
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2
# PASS — 2 episodes streamed and decoded cleanly
```

## Self-host on a GPU box

Crucible ships Docker images for both NVIDIA (CUDA) and AMD (ROCm) GPUs. Concrete recipes for the most common deployment targets:

- [AWS EC2 g6e.xlarge — L40S 48 GB, $1.86/hr](docs/recipes/aws-ec2.md) — recommended NVIDIA path, runs Qwen3-VL-8B comfortably.
- [AWS SageMaker JumpStart](docs/recipes/aws-sagemaker.md) — managed alternative.
- [AMD MI300X / Developer Cloud](docs/recipes/amd-mi300x.md) — full-precision Qwen3-VL-32B at 65k context.
- [Local Mac (MLX, Qwen3-VL-2B)](docs/recipes/local-mac.md) — laptop development path.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full deployment guide and a 9-symptom troubleshooting table.

## How it works

```
LeRobot v3 dataset (chunked parquet + AV1 video)
       │
       ▼  src/lerobot_io.py
  EpisodeBundle (frames + telemetry digest + task description)
       │
       ▼  src/critics.py   → any OpenAI-compatible VLM endpoint
  5 critic verdicts (visual / kinematic / task / strategy / safety)
       │
       ▼  src/aggregator.py
  Final verdict (KEEP / POLISH / REJECT + summary + top_concern)
       │
       ▼  src/pipeline.py   → cache to data/precached/<repo>.json
  List[record]
       │
       ├──▶  src/filtering.py   → curated subset on HuggingFace Hub
       │
       └──▶  src/api.py         → SSE stream to the Gradio frontend
```

Reliability features:

- **Three-tier JSON output** — `json_schema` (xgrammar) → `json_object` → unconstrained, with regex JSON salvage as a final fallback. Every critic returns a structured dict even on a misbehaving model.
- **Deterministic Python aggregator fallback** — if the LLM aggregator call fails or returns malformed output, a pure-Python implementation of the verdict rules guarantees a verdict.
- **LeRobot v3 native** — reads chunked parquet/video shards via the `meta/episodes/*.parquet` pointer columns and seeks PyAV to per-episode timestamps. v2 layouts also supported.
- **Curation manifest** — pushed datasets carry a `crucible_curation` block in `info.json` listing kept episode indices and a documented `LeRobotDataset(repo, episodes=[...])` load instruction.

Architecture deep-dive: [docs/architecture.md](docs/architecture.md). Build history: [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md).

## Configuration

All knobs live in `src/config.py` and are overridable via `CRUCIBLE_*` environment variables:

```bash
CRUCIBLE_VLM_ENDPOINT=http://localhost:8001/v1   # any OpenAI-compatible
CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
CRUCIBLE_VLM_API_KEY=EMPTY                       # or your provider key
CRUCIBLE_FRAMES_PER_EPISODE=16                   # frames per critic
CRUCIBLE_IMAGE_MAX_DIM=768                       # client-side resize
CRUCIBLE_KEEP_THRESHOLD=7.5
CRUCIBLE_POLISH_THRESHOLD=5.0
```

See [`.env.example`](.env.example) for the full list with provider-specific examples.

## Limitations

- Qwen3-VL is not specifically trained on robotics teleoperation video, so it can miss embodiment-specific failure modes a domain expert would catch. We mitigate by feeding the kinematic critic a pre-computed telemetry digest (joint velocities, idle periods, recovery moves, gripper events) where pure VLMs are weakest.
- Critic outputs are stochastic at temperature 0.2. For production deployment, run each episode 3× and aggregate, or fine-tune on human-labeled episode quality data (planned for v0.3).
- AV1 video decoding requires PyAV with libdav1d. Most modern wheels include it.
- The push-to-Hub flow preserves the source LeRobot v3 chunked layout and adds a `crucible_curation` block to `info.json`. It does not regenerate stats or re-shard videos — load via `LeRobotDataset(repo, episodes=kept_indices)` to consume only the curated subset.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full milestone breakdown. The short version:

- **v0.1** (current): multi-axis rubric, 5 critics + aggregator, LeRobot v3 reader, multi-backend deployment recipes.
- **v0.2**: PyPI package, examples notebooks, multi-GPU recipe for Qwen3-VL-32B, polished HF Space.
- **v0.3**: human-VLM agreement dataset (the real moat), Cohen's κ benchmark per rubric axis, fine-tuned judge model.
- **v1.0**: production-grade fine-tuned judge model on AMD ROCm; multi-model support (Claude / GPT-4o / Gemini drop-in).

## Contributing

PRs welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup, design principles, and PR workflow. New critics, new dataset format readers, new deployment recipes are all in scope.

For security issues see [SECURITY.md](SECURITY.md).

## Acknowledgements

- The five critic prompts and the strategy / safety rubric are original to Crucible.
- The LeRobot v3 path templates and chunked-layout I/O patterns were verified against the official `huggingface/lerobot` library.
- Qwen3-VL is the model that makes this practical at fp16 with native video understanding.
- AMD/LMSYS published the MI300X-specific vLLM optimizations (rocJPEG, AITER + prefill-decode attention) we use in the AMD recipe.
- `score_lerobot_episodes` (RoboticsData, Oct 2025) and AgiBot Genie Centurion are the closest prior art and the baselines we explicitly out-position with the multi-axis behavioral rubric.

## License

MIT. See [LICENSE](LICENSE).
