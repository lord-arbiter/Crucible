# Recipe: hosted-API quickstart (any OpenAI-compatible Qwen3-VL provider)

## When to use

You don't have a GPU and don't want to set one up. You want to try
Crucible end-to-end in 5 minutes against a hosted Qwen3-VL endpoint.

This is the easiest path. Provider-specific recipes for
[Hyperbolic](./hyperbolic.md), [Together AI](./together-ai.md), and
[DashScope](./dashscope.md) follow the same shape.

## What you need

- A provider that hosts Qwen3-VL (32B-Instruct, 72B-Instruct, or any
  Qwen3-VL variant) behind an OpenAI-compatible chat completions API.
- An API key from that provider.
- Python 3.11+ on your laptop. No GPU.

Verify the provider supports:

- `POST /v1/chat/completions` with `messages: [{role, content: [...]}]`
- `image_url` content type with `data:image/jpeg;base64,...` URLs
- A Qwen3-VL model id (e.g. `Qwen/Qwen3-VL-32B-Instruct`)

`response_format={"type":"json_schema",...}` is preferred but **not
required** — Crucible's three-tier fallback handles providers that only
support `json_object` or unconstrained output.

## Setup

### 1. Install Crucible

```bash
git clone https://github.com/lord-arbiter/Crucible
cd Crucible
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Set environment variables

```bash
export CRUCIBLE_VLM_ENDPOINT=https://api.your-provider.com/v1
export CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
export CRUCIBLE_VLM_API_KEY=sk-your-key
```

The exact `CRUCIBLE_VLM_MODEL` value depends on the provider. Common
patterns:

| Provider | Endpoint | Model id |
|---|---|---|
| Hyperbolic | `https://api.hyperbolic.xyz/v1` | `Qwen/Qwen3-VL-72B-Instruct` |
| Together AI | `https://api.together.xyz/v1` | `Qwen/Qwen3-VL-32B-Instruct` |
| DashScope (Alibaba intl) | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `qwen3-vl-plus` |
| Self-hosted vLLM | `http://localhost:8001/v1` | whatever you set `--served-model-name` to |

See each provider's recipe in this folder for the exact values they
support today.

### 3. Smoke test

```bash
# I/O smoke (no VLM call — proves dataset reading works)
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2

# Single-critic call against the hosted endpoint
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual \
    --frames 8
```

Expect a JSON object with:

```json
{
  "score": <float 0-10>,
  "verdict": "EXCELLENT" | "ACCEPTABLE" | "MARGINAL" | "REJECT",
  "rationale": "...",
  "evidence": [{"timestamp": "...", "observation": "..."}, ...]
}
```

If the verdict is `PARSE_ERROR` or `ERROR`, see the troubleshooting
section below.

### 4. Full pipeline on a small dataset

```bash
python scripts/precache_demo.py \
    --repo lerobot/aloha_static_cups_open \
    --episodes 5 \
    --frames 12
```

Expected: 5 episodes scored in ~60–120 seconds (latency dominated by
the hosted-API round-trip), all records have non-fallback verdicts,
no `PARSE_ERROR`. Output saved to
`data/precached/lerobot__aloha_static_cups_open.json`.

### 5. (Optional) Run the Gradio frontend locally

```bash
# Point the frontend at the same endpoint as the CLI
export CRUCIBLE_API_BASE=http://localhost:8000

# Start the FastAPI orchestrator
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Start the Gradio app
python frontend/app.py
```

Open <http://localhost:7860>. Score the cached dataset, click an
episode to see the critic cards, drag the threshold slider.

## Cost

Per episode: 6 LLM calls (5 critics + 1 aggregator). Each critic call
sends ~1k tokens of text + 4–16 images at 768px (~1.5k tokens of vision
context per image). Total: ~10–30k input tokens + ~3k output tokens
per episode.

At Qwen3-VL hosted pricing of ~$0.20–$0.50 per million input tokens and
~$0.40–$0.80 per million output tokens (May 2026 ranges), expect
**~$0.005–$0.02 per episode**. A 25-episode dataset costs **~$0.13–$0.50**.

Cheaper than your morning coffee. Also cheaper than the EC2 path
($1.86/hr) if you're scoring fewer than ~100 episodes.

## Provider-specific quirks

Each provider has its own gotchas (model id aliases, rate limits, JSON
mode support, image format constraints). See the provider-specific
recipe files:

- [Hyperbolic](./hyperbolic.md)
- [Together AI](./together-ai.md)
- [DashScope](./dashscope.md)

If you bring up Crucible against a provider not yet covered, please
send a PR with a new recipe file using the template in
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Critics return `PARSE_ERROR` | Provider rejects `json_schema` | Crucible falls through to `json_object` automatically; check the verdict on retry. If still PARSE_ERROR, model is producing prose-wrapped JSON — try lowering `CRUCIBLE_CRITIC_TEMPERATURE` to 0.1 |
| HTTP 429 (rate limit) | Concurrent critic dispatch exceeds your provider's rate limit | Set `CRUCIBLE_PARALLEL_CRITICS=false` to serialize the 5 critic calls per episode (5× per-episode latency, but no rate-limit hits) |
| HTTP 400 "model not found" | Wrong `CRUCIBLE_VLM_MODEL` for this provider | Check the provider's model catalog; some use `qwen3-vl-plus`, others `Qwen/Qwen3-VL-32B-Instruct` |
| HTTP 401 / 403 | API key wrong or scope missing | Verify the key in your provider's dashboard |
| Image too large errors | Provider has tighter limits than 768px max | Lower `CRUCIBLE_IMAGE_MAX_DIM` to 512 or 384 |
| Latency very high (>30 s per critic) | Cold start on the provider, or you're hitting a small / slow model variant | Pick a higher-tier model id; or warmup with a small request |
