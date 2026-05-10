# Architecture

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
│  AMD Developer Cloud — MI300X GPU Droplet (Ubuntu 22.04)     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  FastAPI orchestrator (src/api.py, port 8000)          │  │
│  │  - POST /score_dataset            kick off a job       │  │
│  │  - GET  /progress/{job_id}        SSE stream           │  │
│  │  - GET  /results/{job_id}         final results        │  │
│  │  - POST /push_filtered            curated subset → Hub │  │
│  └─────────────────────┬──────────────────────────────────┘  │
│                        │                                     │
│  ┌─────────────────────┴──────────────────────────────────┐  │
│  │  Episode processor (src/lerobot_io.py, src/pipeline.py)│  │
│  │  - LeRobotDataset streaming I/O (v2 and v3 layouts)    │  │
│  │  - Frame sampling via PyAV (AV1-compatible)            │  │
│  │  - Telemetry digest builder                            │  │
│  │     - Per-joint peak/mean velocities                   │  │
│  │     - Idle period detection (>0.5s)                    │  │
│  │     - Recovery-move detection (sign reversals)         │  │
│  │     - Heuristic gripper transitions                    │  │
│  └─────────────────────┬──────────────────────────────────┘  │
│                        │ batched VLM requests                │
│  ┌─────────────────────┴──────────────────────────────────┐  │
│  │  vLLM serving Qwen3-VL-32B (port 8001)                 │  │
│  │  - 5 specialist critics (parallel via asyncio.gather)  │  │
│  │  - 1 aggregator                                        │  │
│  │  - JSON-schema constrained outputs                     │  │
│  │  - 65k-token max context (room for ~16 frames @ 768px) │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Why this split

The Gradio frontend on HuggingFace Spaces is free, gives us a public URL, and is what the lablab judges expect to see. The actual GPU compute lives on the AMD MI300X droplet.

This split is also operationally robust — if the GPU droplet shuts down (or if AMD credits run out mid-judging), the Space still loads. We bundle precached results for the demo dataset (`lerobot/aloha_mobile_cabinet`, 25 episodes) so the dashboard always has data to render. A "live score" still works the moment the droplet comes back up.

## Concurrency model

Per dataset:

1. The FastAPI request handler enqueues a `score_dataset` coroutine on the event loop and returns a `job_id`.
2. The pipeline iterates the dataset's episodes one at a time. For each episode it issues five concurrent VLM calls (the five critics) via `asyncio.gather`. The aggregator runs sequentially after.
3. Each critic emits a JSON object validated against a small Pydantic-flavored schema; malformed outputs fall through a regex-based JSON salvage path.
4. As each episode finishes, the pipeline calls `progress_callback`. The API turns each callback into an SSE event so the frontend updates live.

## Caching strategy

`data/precached/<repo_id_with_underscores>.json` holds the most recent full run for any dataset. The pipeline checks the cache before issuing any VLM calls. The Space ships with this directory mounted, so a cold-loaded Space serves the demo run instantly even if the GPU is offline.

## Failure modes & fallbacks

| Failure | Fallback |
|---|---|
| GPU droplet unreachable | Space loads precached results; a banner explains the state. |
| Single critic returns malformed JSON | `_extract_json_loose` salvages the JSON block; otherwise records a `PARSE_ERROR` verdict. |
| Aggregator LLM call fails | Deterministic Python fallback applies the same weighting + verdict rules. |
| AV1 decode fails (missing codec) | Pipeline can use image-only datasets (`*_image` variants) — image bytes read directly from parquet. |
| LeRobot v2 vs v3 layout | I/O layer detects layout from `meta/info.json` + file presence and reads the right metadata. |
| Episode missing task description | Falls back to first known task description, or a placeholder string. |
