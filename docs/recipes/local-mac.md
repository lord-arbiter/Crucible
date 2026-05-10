# Recipe: Local Mac (Apple Silicon, MLX, Qwen3-VL-2B)

## When to use

You want to develop Crucible end-to-end on your laptop without paying
for cloud GPU. Apple Silicon's unified memory architecture lets you run
Qwen3-VL-2B-Instruct comfortably on M1/M2/M3/M4/M5 Macs with 16+ GB RAM.

This is the **slowest** path (~30 seconds per critic call vs. ~1 second
on a GPU) but the most accessible — no cloud account, no Docker, no
network dependency on a hosted API.

## Cost

Free, after the cost of the Mac.

Quality trade-off: Qwen3-VL-2B is meaningfully weaker than 8B/32B at the
strategy and safety axes (smaller models compress behavioral nuance
worse). Use this for development; validate against a hosted API or
self-hosted larger model before drawing conclusions.

## Setup

### 1. Install MLX

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework
optimized for Apple Silicon. The `mlx-vlm` package serves vision-language
models with an OpenAI-compatible API.

```bash
pip install mlx-vlm
```

### 2. Serve Qwen3-VL-2B

```bash
python -m mlx_vlm.server \
    --model mlx-community/Qwen3-VL-2B-Instruct-4bit \
    --port 8001 \
    --host 127.0.0.1
```

First boot downloads ~1.2 GB of quantized weights. Allow 1–2 minutes.

The 4-bit quantization is the practical choice — 8-bit is also
available (`-8bit` suffix) at ~2× the memory cost.

### 3. Configure Crucible

```bash
export CRUCIBLE_VLM_ENDPOINT=http://localhost:8001/v1
export CRUCIBLE_VLM_MODEL=mlx-community/Qwen3-VL-2B-Instruct-4bit
export CRUCIBLE_VLM_API_KEY=EMPTY
```

### 4. Smoke test

```bash
# I/O smoke (no VLM)
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2

# Single critic — expect ~20–40 seconds wall-clock per call
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual \
    --frames 6
```

If the JSON parses with a sensible score and verdict, you're set.

### 5. Full pipeline (be patient)

```bash
python scripts/precache_demo.py \
    --repo lerobot/aloha_static_cups_open \
    --episodes 3 \
    --frames 8
```

3 episodes × 6 calls × ~30 s each ≈ **9 minutes** wall-clock. This is
the "develop on your laptop" path, not the "live demo" path.

## Known quirks

- **Concurrency.** mlx-vlm's server processes one request at a time.
  Set `CRUCIBLE_PARALLEL_CRITICS=false` so critics dispatch serially —
  otherwise queued requests time out.
- **Image size.** Lower `CRUCIBLE_IMAGE_MAX_DIM` to 512 or 384 to
  speed things up; the 2B model isn't sharper at 768px in practice.
- **`response_format=json_schema`** is not supported by mlx-vlm.
  Crucible falls through to `json_object` mode automatically; the 2B
  model is sometimes wobbly on JSON, so the regex JSON salvage path
  occasionally kicks in.
- **Model size.** The 2B model has noticeably weaker reasoning than
  8B/32B. For the strategy and safety critics especially, expect more
  generic rationale strings. Ground-truth your behavioral findings
  against a larger model before drawing conclusions.

## Why MLX (not vLLM or llama.cpp)

- **vLLM** doesn't have first-class Apple Silicon support.
- **llama.cpp** runs Qwen3-VL-2B but image preprocessing is rough.
- **MLX** + `mlx-vlm` is the cleanest local-Mac story today, with a
  proper OpenAI-compatible server and tight Apple Silicon performance.

## When to upgrade

The moment your laptop run is bottlenecking development, point
`CRUCIBLE_VLM_ENDPOINT` at any hosted API
([hosted-api-quickstart.md](./hosted-api-quickstart.md)) — Crucible
itself doesn't change.
