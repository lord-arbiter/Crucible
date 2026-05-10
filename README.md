# Crucible

*A VLM-judged data curation studio for robot demonstrations.*

[Live Space](https://huggingface.co/spaces/) ·
[Demo Video](#) ·
[Architecture](docs/architecture.md)

> Built solo for the **AMD Developer Hackathon 2026** (lablab.ai).
> Track: AI Agents & Agentic Workflows.
> Stack: Qwen3-VL on AMD MI300X via vLLM, FastAPI orchestrator, Gradio frontend on HuggingFace Spaces.

## The problem

Robotics teams in 2026 spend $118/hour collecting teleoperation data, and 20-40% of it shouldn't be used to train policies. The damage is silent: bad episodes don't fail loudly — they teach the policy bad habits. Existing curation tools score classical heuristics (blur via Laplacian, jitter via joint derivatives) and miss the things humans catch on visual inspection: operator hesitation, inefficient strategy, ambiguous task success, near-miss safety incidents.

Stanford research shows filtering the bottom 20% of demonstrations improves downstream policy success rates by 15–25%. SVRC reports teleop costs at $118/hour. The problem exists today (UR + Scale AI launched UR AI Trainer March 2026; AgiBot Genie Centurion targets the same bottleneck).

## What Crucible does

1. Paste a HuggingFace LeRobot dataset URL (e.g. `lerobot/aloha_mobile_cabinet`).
2. Crucible streams the episodes, samples frames, builds a telemetry digest, and runs each episode through five Qwen3-VL specialist critic agents in parallel on a single AMD MI300X.
3. An aggregator agent fuses the five critic outputs into a final `KEEP` / `POLISH` / `REJECT` verdict with rationale and timestamp citations.
4. The dashboard shows a score distribution, per-episode rationale cards, and an embedded video clip for inspection.
5. Drag a threshold slider, hit "Push filtered dataset" — Crucible writes a curated subset (preserving the original LeRobot v3 layout) to a new HuggingFace dataset, generates a dataset card explaining what was filtered and why.

## The five critics

| Critic | What it scores |
|---|---|
| **Visual Quality** | Lighting, motion blur, occlusion, lens fog, exposure, frame integrity |
| **Kinematic Quality** | Jerk, joint saturation, recovery moves, idle drift, gripper instability |
| **Task Success** | Did the task described actually complete by the end of the episode? |
| **Strategy** | Was the operator's approach efficient and teachable, or full of regrasps and hesitation? |
| **Safety** | Near-collisions, items knocked off the table, gripper crushing, self-collision risk |

Each critic returns `{score, verdict, rationale, evidence[]}`. The aggregator uses a weighted mean (`task_success` and `strategy` weighted 1.5×) and applies hard-fail rules (any `REJECT` / `UNSAFE` / `FAILED` critic verdict ⇒ overall `REJECT`).

The strategy critic is the one no other tool does. *A bad strategy episode that succeeds is worse than a good strategy episode that fails — the former teaches the policy bad habits silently.*

## Why AMD MI300X

Crucible runs Qwen3-VL-32B at fp16 with a 65k-token context window — long enough to ingest a full episode's frame stack plus the telemetry digest plus the task instruction in a single shot, with no aggressive trimming. On an H100 80GB this requires either model quantization or context shrinkage. On MI300X 192GB it runs at full precision with room to batch multiple episodes through the same model in parallel.

That headroom is the load-bearing capability for this product, not a nice-to-have. Most submissions in this hackathon will be H100-bound. Crucible is designed for the GPU that exists in this hackathon.

## Why Qwen3-VL

Qwen3-VL is the strongest open-weight multimodal model with native video understanding and a 256K context as of late 2025. AMD/LMSYS published MI300X-specific optimizations (rocJPEG, batch-level data parallelism for the vision encoder, CUDA IPC for multimodal data transfer) that make it the right model for this hardware. Crucible is end-to-end Qwen3-VL: critics + aggregator share a single served model.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  HuggingFace Space (Gradio frontend, free CPU tier)          │
│  - Dataset URL input                                         │
│  - Live progress streaming via SSE                           │
│  - Episode dashboard with embedded video + critic cards      │
│  - Filter slider + push-to-Hub button                        │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTPS / SSE
┌──────────────────────┴───────────────────────────────────────┐
│  AMD Developer Cloud — MI300X GPU Droplet                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  FastAPI orchestrator (port 8000)                      │  │
│  │  - /score_dataset, /progress (SSE), /push_filtered     │  │
│  └─────────────────────┬──────────────────────────────────┘  │
│                        │                                     │
│  ┌─────────────────────┴──────────────────────────────────┐  │
│  │  Episode processor: LeRobotDataset streaming I/O,      │  │
│  │  PyAV frame sampling, telemetry digest builder         │  │
│  └─────────────────────┬──────────────────────────────────┘  │
│                        │ batched VLM requests                │
│  ┌─────────────────────┴──────────────────────────────────┐  │
│  │  vLLM serving Qwen3-VL-32B (port 8001)                 │  │
│  │  - 5 specialist critics (parallel)                     │  │
│  │  - 1 aggregator                                        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

Full diagram + writeup: [docs/architecture.md](docs/architecture.md).

## Reproduce locally

```bash
# 1. Clone & install
git clone https://github.com/<you>/crucible
cd crucible
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Serve Qwen3-VL via vLLM (on a GPU box; example for MI300X)
HSA_FORCE_FINE_GRAIN_PCIE=1 HIP_VISIBLE_DEVICES=0 \
    vllm serve Qwen/Qwen3-VL-32B-Instruct \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.85 \
        --port 8001

# 3. Smoke-test one episode through one critic
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual

# 4. Pre-cache the demo dataset
python scripts/precache_demo.py \
    --repo lerobot/aloha_mobile_cabinet \
    --episodes 25

# 5. Run the FastAPI orchestrator
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 6. Run the Gradio frontend (or deploy to HF Spaces)
CRUCIBLE_API_BASE=http://localhost:8000 \
    python frontend/app.py
```

## Configuration

All knobs live in `src/config.py` and are also overridable via environment variables (`CRUCIBLE_*`). See `.env.example`.

## Limitations

- Qwen3-VL is not specifically trained on robotics teleoperation video, so it can miss embodiment-specific failure modes a domain expert would catch. We mitigate by feeding the kinematic critic a pre-computed telemetry digest (joint velocities, idle periods, recovery moves, gripper events) where pure VLMs are weakest.
- Critic outputs are stochastic at temperature 0.2. For production deployment, run each episode 3× and aggregate, or fine-tune on human-labeled episode quality data.
- AV1 video decoding requires `pyav` with codec support. We ship fallback paths to image-only datasets (`lerobot/aloha_sim_insertion_human_image`).
- The LeRobot dataset reader is hand-rolled (not lerobot library at runtime) — handles v2 + v3 layouts, but unusual custom forks may not parse cleanly.

## Bonus tracks

- **Qwen-powered project bonus** — Qwen3-VL is the entire model story (5 critics + 1 aggregator).
- **Build in Public bonus** — thread documenting the build at <link to be filled>.

## References

- *State of Robotics 2026 Report* — Silicon Valley Robotics Center (teleop cost benchmark).
- Stanford "Quality over Quantity: Demonstration Curation via Influence Functions" — formal grounding for the curation thesis.
- AgiBot **Genie Centurion** — same problem area, different approach (rewind-and-refine vs. post-hoc scoring).
- HuggingFace `score_lerobot_episodes` — heuristic baseline we improve on.
- Qwen3-VL Technical Report (arXiv 2511.21631).
- LMSYS *Ultimate Latency Optimization of Qwen3-VL on AMD MI300X Series*.
- LeRobotDataset v3.0 release blog — the data format Crucible reads from and writes to.

## License

MIT.
