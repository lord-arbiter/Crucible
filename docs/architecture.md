# Architecture

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
│  Compute box (any GPU host or hosted API frontend)           │
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
│  │  - LeRobotDataset v2/v3 streaming I/O                  │  │
│  │  - Frame sampling via PyAV (AV1-compatible)            │  │
│  │  - Telemetry digest builder                            │  │
│  └─────────────────────┬──────────────────────────────────┘  │
│                        │ json_schema-constrained requests    │
│  ┌─────────────────────┴──────────────────────────────────┐  │
│  │  Transport selector (src/critics.py:_get_transport):   │  │
│  │   1. OpenAI-compat   when CRUCIBLE_VLM_ENDPOINT is set │  │
│  │      (self-hosted vLLM, Hyperbolic, Together,          │  │
│  │       DashScope, OpenAI, Gemini OpenAI-compat, ...)    │  │
│  │   2. OpenAI direct   when model id is gpt-*/o1/o3/o4   │  │
│  │   3. LiteLLM         when model id has provider prefix │  │
│  │      (anthropic/, bedrock/, vertex_ai/, cohere/,       │  │
│  │       groq/, replicate/, xai/, ... ~100 providers)     │  │
│  │  Five specialist critics (parallel via asyncio.gather) │  │
│  │  + one aggregator share the selected transport.        │  │
│  │  json_schema → json_object → unconstrained fallback.   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Why this split

The Gradio frontend on HuggingFace Spaces is free, gives a public URL, and is what most evaluators expect to see. The actual compute lives wherever you point `CRUCIBLE_VLM_ENDPOINT` — your laptop, a self-hosted EC2 instance, an MI300X droplet, or a hosted Qwen3-VL provider.

This split is also operationally robust — if the compute box shuts down, the Space still loads. Crucible bundles a precache for the demo dataset (`lerobot/aloha_mobile_cabinet`, 25 episodes) so the dashboard always has data to render. A "live score" still works the moment the endpoint comes back up.

## Concurrency model

Per dataset:

1. The FastAPI request handler enqueues a `score_dataset` coroutine on the event loop and returns a `job_id`.
2. The pipeline iterates the dataset's episodes one at a time. For each episode it issues five concurrent VLM calls (the five critics) via `asyncio.gather`. The aggregator runs sequentially after.
3. Each critic emits a JSON object validated against a per-critic JSON schema; malformed outputs fall through json_object mode and then a regex-based JSON salvage path.
4. As each episode finishes, the pipeline calls `progress_callback`. The API turns each callback into an SSE event so the frontend updates live.

For hosted APIs with strict rate limits, set `CRUCIBLE_PARALLEL_CRITICS=false` to serialize the five critic calls per episode. The trade-off is ~5× per-episode latency.

## Caching strategy

`data/precached/<repo_id_with_underscores>.json` holds the most recent full run for any dataset. The pipeline checks the cache before issuing any VLM calls. The Space ships with this directory mounted, so a cold-loaded Space serves the demo run instantly even if the compute is offline.

## Failure modes & fallbacks

| Failure | Fallback |
|---|---|
| Compute box / hosted API unreachable | Space loads precached results; a banner explains the state. |
| Single critic returns malformed JSON | `_extract_json_loose` salvages the JSON block; otherwise records a `PARSE_ERROR` verdict. |
| Aggregator LLM call fails | Deterministic Python fallback applies the same weighting + verdict rules. |
| AV1 decode fails (missing codec) | Pipeline can use image-only datasets — image bytes read directly from parquet. |
| LeRobot v2 vs v3 layout | I/O layer detects layout from `meta/info.json` + file presence and reads the right metadata. |
| Episode missing task description | Falls back to first known task description, or a placeholder string. |
| `response_format=json_schema` not supported by provider | Three-tier fallback chain: json_schema → json_object → unconstrained. |
