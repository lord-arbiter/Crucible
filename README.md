# Crucible

*The first multi-axis behavioral rubric for robot demonstration data — VLM-judged on AMD MI300X.*

[Live Space](https://huggingface.co/spaces/) ·
[Demo Video](#) ·
[Architecture](docs/architecture.md) ·
[Runbook](docs/RUNBOOK.md) ·
[GPU access](docs/GPU_ACCESS.md) ·
[Test plan](TESTING.md)

> Built solo for the **AMD Developer Hackathon 2026** (lablab.ai).
> Track: AI Agents & Agentic Workflows.
> Stack: Qwen3-VL-32B on AMD MI300X via vLLM, FastAPI orchestrator, Gradio frontend on HuggingFace Spaces.

## The problem

Robotics teams in 2026 spend **$118 / hour** collecting teleoperation data, and 20–40 % of it shouldn't be used to train policies. The damage is silent: bad episodes don't fail loudly — they teach the policy bad habits.

Heuristic curation (jerk, path-efficiency ratio, actuator saturation) catches motion artefacts. A VLM that only judges binary task success (à la `score_lerobot_episodes` calling Gemini, Oct 2025) catches outright failures. Neither catches the silent killers: **operator hesitation, multi-attempt regrasps, inefficient strategies, near-miss safety incidents, ambiguous task execution.**

A bad-strategy episode that *succeeds* is worse than a good-strategy episode that fails — the former teaches the policy unreliable recovery as if it were normal behavior.

## What Crucible does

Five Qwen3-VL specialist critic agents review every episode along orthogonal axes, and one aggregator agent fuses their outputs into a final verdict with rationale and timestamp evidence:

| Critic | Axis (none of these are scalar success/fail) |
|---|---|
| **Visual Quality** | Lighting consistency, motion blur, lens fog, occlusion, frame integrity |
| **Kinematic Quality** | Jerk, joint saturation, recovery moves, idle drift, gripper instability |
| **Task Success** | Did the task described actually complete? Confidence, not just yes/no |
| **Strategy** | Was the operator's approach efficient and teachable? Hesitation, regrasps, inefficient paths |
| **Safety** | Near-collisions, items knocked off the workspace, unsafe contact, dangerous arcs |

The aggregator weights `task_success` and `strategy` 1.5× and applies hard-fail rules (any `REJECT / UNSAFE / FAILED` critic verdict ⇒ overall `REJECT`).

You drag a threshold slider, hit "Push filtered dataset," and Crucible writes a curated subset back to the HuggingFace Hub — preserving the original LeRobot v3 layout, with a `crucible_curation` block in `info.json` listing the kept episode indices and a generated dataset card explaining what was filtered and why.

## How Crucible compares to existing tools

| Tool | Approach | Output | What it misses |
|---|---|---|---|
| `score_lerobot_episodes` (RoboticsData, Oct 2025) | 7 CV heuristics + 1 binary Gemini call | scalar score per axis | Strategy, hesitation, behavioral nuance |
| AgiBot Genie Centurion / Task Sentinel (May 2025) | fine-tuned MiniGPT-4 success classifier inside teleop loop | online intervention during collection | Offline curation of completed datasets |
| LeRobot dataset visualizer | low-movement / jerk / outlier-length flags | filter UI suggestions | Anything semantic |
| Stanford Quality-over-Quantity | influence-function math on policy gradients | continuous demo importance | Requires a trained policy; per-axis explainability |
| **Crucible** | **5-axis behavioral rubric, 1 aggregator, 1 served VLM, structured rationale + evidence** | **per-episode JSON with timestamp citations + KEEP/POLISH/REJECT** | **n/a — this is the gap** |

Crucible's contribution is not "we thought of using a VLM as a judge" — that ground was broken in 2025. It is the **multi-axis behavioral rubric**, the **structured per-episode rationale with timestamp evidence**, and an inference economics story (on-prem AMD MI300X) that lets a robotics lab curate millions of frames without metering every call to a third-party API.

## Why AMD MI300X

Crucible runs Qwen3-VL-32B at bf16 with a 65k-token context window — enough to ingest a full episode's frame stack (16 frames at 768px) plus the telemetry digest plus the task instruction in a single shot, with no aggressive trimming. On an H100 80 GB this requires either model quantization or context shrinkage. On MI300X 192 GB it runs at full precision, with room to batch five concurrent critics through the same model.

That headroom is the load-bearing capability for this product, not a nice-to-have. Most submissions in this hackathon will be H100-bound. Crucible is designed for the GPU that exists in this hackathon.

## Why Qwen3-VL

Qwen3-VL is the strongest open-weight multimodal model with native video understanding and 256K context as of late 2025. AMD/LMSYS published MI300X-specific optimizations (rocJPEG, batch-level data parallelism for the vision encoder, AITER + prefill-decode attention) that make it the right model for this hardware. Crucible is end-to-end Qwen3-VL: critics + aggregator share a single served model, and outputs are JSON-schema-constrained via vLLM's xgrammar guided-decoding backend.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  HuggingFace Space (Gradio frontend, free CPU tier)          │
│  - Dataset URL input                                         │
│  - Live SSE-driven progress streaming                        │
│  - Episode dashboard with embedded video + critic cards      │
│  - Filter slider + push-to-Hub button                        │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTPS / SSE
┌──────────────────────┴───────────────────────────────────────┐
│  AMD Developer Cloud — MI300X GPU Droplet                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  FastAPI orchestrator (port 8000)                      │  │
│  └─────────────────────┬──────────────────────────────────┘  │
│                        │                                     │
│  ┌─────────────────────┴──────────────────────────────────┐  │
│  │  v3-native LeRobot streaming reader + telemetry digest │  │
│  └─────────────────────┬──────────────────────────────────┘  │
│                        │ json_schema-constrained requests    │
│  ┌─────────────────────┴──────────────────────────────────┐  │
│  │  vLLM serving Qwen3-VL-32B (port 8001)                 │  │
│  │  - 5 specialist critics (parallel via asyncio.gather)  │  │
│  │  - 1 aggregator                                        │  │
│  │  - xgrammar guided decoding (json_schema)              │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

Full diagram + writeup: [docs/architecture.md](docs/architecture.md).

## Reproduce locally

```bash
# 1. Clone & install
git clone https://github.com/lord-arbiter/Crucible
cd Crucible
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. (No GPU needed) Smoke-test the v3 streaming reader on a real dataset
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2

# 3. Serve Qwen3-VL via vLLM on the GPU box (MI300X recipe in docker/entrypoint.sh)
docker build -f docker/Dockerfile.gpu -t crucible:gpu .
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --group-add video \
    -p 8000:8000 -p 8001:8001 \
    crucible:gpu

# 4. End-to-end smoke test against the live vLLM endpoint
python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual

# 5. Pre-cache the demo dataset
python scripts/precache_demo.py --repo lerobot/aloha_mobile_cabinet --episodes 25

# 6. Run the Gradio frontend (or deploy to HF Spaces)
CRUCIBLE_API_BASE=http://<gpu-droplet>:8000 python frontend/app.py
```

Full sequence with provisioning steps: [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Configuration

All knobs live in `src/config.py` and are overridable via environment variables (`CRUCIBLE_*`). See `.env.example`.

## Limitations

- Qwen3-VL is not specifically trained on robotics teleoperation video, so it can miss embodiment-specific failure modes a domain expert would catch. We mitigate by feeding the kinematic critic a pre-computed telemetry digest (joint velocities, idle periods, recovery moves, gripper events) where pure VLMs are weakest.
- Critic outputs are stochastic at temperature 0.2. For production deployment, run each episode 3× and aggregate, or fine-tune on human-labeled episode quality data.
- AV1 video decoding requires `pyav` with codec support (libdav1d). The `Dockerfile.gpu` installs both; pure-pip wheels on macOS arm64 also include dav1d.
- The push-to-Hub flow preserves the source LeRobot v3 chunked layout and adds a `crucible_curation` block in `info.json`. It does not regenerate stats or re-shard videos — load via `LeRobotDataset(repo, episodes=kept_indices)` to consume only the curated subset.
- The LeRobot dataset reader handles v3 chunked layout (the modern default) and falls back to v2 per-episode layout. Custom forks with non-standard path templates may not parse cleanly.

## Bonus tracks

- **Qwen-powered project bonus** — Qwen3-VL is the entire model story (5 critics + 1 aggregator share one served model).
- **Build in Public bonus** — thread documenting the build at <link to be filled> · template at [docs/build_in_public.md](docs/build_in_public.md).

## Defensible moat (post-hackathon)

Beyond the hackathon, Crucible's defensibility is not in the model (Qwen3-VL is a commodity). It's in three follow-up artefacts:

1. **A human-VLM agreement dataset** — co-score N episodes with both expert humans and the VLM judge, publish Cohen's κ per rubric axis. The only artefact competitors can't quickly reproduce.
2. **Critic prompt library, validated against that κ dataset, tuned per task family** (pick-and-place, contact-rich, bimanual, mobile manipulation).
3. **A small fine-tuned judge model** running on-prem on AMD ROCm — turns Crucible into a one-binary deployment story versus Gemini-API dependency.

## References

- *State of Robotics 2026 Report* — Silicon Valley Robotics Center (teleop cost benchmark).
- Stanford "Quality over Quantity: Demonstration Curation via Influence Functions" (arXiv 2603.09056).
- AgiBot Genie Centurion / Task Sentinel (arXiv 2505.18793, May 2025).
- `score_lerobot_episodes` (RoboticsData, Oct 2025) — heuristic + Gemini-task-success baseline we explicitly out-position.
- Qwen3-VL Technical Report (arXiv 2511.21631).
- LMSYS *Ultimate Latency Optimization of Qwen3-VL on AMD MI300X Series* (Feb 2026).
- LeRobotDataset v3.0 release blog — the data format Crucible reads from and writes to.

## License

MIT.
