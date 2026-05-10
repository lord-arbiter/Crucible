# Recipe: AWS EC2 (g6e.xlarge — L40S 48 GB)

## When to use

You want full control of the GPU box, you have AWS credits, and you don't
need the 192 GB headroom of MI300X. This is **the recommended NVIDIA
self-hosted recipe** for Crucible.

A single L40S 48 GB comfortably fits Qwen3-VL-8B-Instruct at fp16 with
32k context and 16-image multimodal requests. Qwen3-VL-32B at fp16
(~64 GB) does *not* fit on a single L40S — see the
[multi-GPU recipe](./aws-ec2-multi-gpu.md) (planned for v0.2) or use
[AMD MI300X](./amd-mi300x.md) if you need the larger model.

## Cost

- `g6e.xlarge` on-demand: **$1.861 / hour** in `us-east-1` (May 2026).
- Storage: $0.08 / GB-month for gp3; we use 200 GB → ~$16/month if left
  attached. Detach or delete the volume when not running.
- Validation run estimate: **~$2 of credits** for a 1-hour bring-up,
  smoke tests, and a 25-episode precache.

Verify current pricing at <https://aws.amazon.com/ec2/instance-types/g6e/>.

## Setup

### 1. Provision the instance

AWS Console → EC2 → **Launch instance**:

- **Name:** `crucible-gpu`
- **AMI:** Deep Learning Base GPU AMI (Ubuntu 22.04). Pre-installs
  NVIDIA drivers, Docker, and `nvidia-container-toolkit`. Pick the
  newest version available.
- **Instance type:** `g6e.xlarge` (1× NVIDIA L40S 48 GB, 4 vCPU, 32 GB
  RAM).
- **Key pair:** existing or generate new.
- **Network settings:** Default VPC. Create or select a security group
  with these inbound rules (limit source to your IP):
  - SSH (22) → My IP
  - Custom TCP (8000) → My IP — FastAPI orchestrator
  - Custom TCP (8001) → My IP — vLLM endpoint (only needed if you want
    to call vLLM directly from your laptop; otherwise leave 8001 closed
    and access it from inside the box)
- **Storage:** 200 GB gp3 root volume.
- **Launch.**

Provisioning takes ~2 minutes. Note the **public IPv4** when the
instance reaches Running.

### 2. SSH in and verify the GPU

```bash
ssh -i ~/.ssh/<your-key>.pem ubuntu@<public-ipv4>

# Verify the GPU
nvidia-smi
# Expected: 1× NVIDIA L40S, ~46068 MiB total, driver 550+, CUDA 12.x
```

If `nvidia-smi` errors, the AMI didn't pre-install the driver — pick a
different Deep Learning AMI variant.

### 3. Pull the repo and build the CUDA image

```bash
git clone https://github.com/lord-arbiter/Crucible
cd Crucible

# Optional: confirm git status is clean
git log --oneline -3

# Build the CUDA Dockerfile
docker build -f docker/Dockerfile.cuda -t crucible:cuda .
```

First build pulls the upstream `vllm/vllm-openai:latest` (~12 GB) and
adds Crucible deps on top. Allow ~5–8 minutes.

### 4. Run the container

```bash
# Set your HF token if you have one (faster downloads, higher rate limits)
export HF_TOKEN=hf_...

docker run --rm --gpus all \
    -p 8000:8000 -p 8001:8001 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    -e CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct \
    -e VLLM_MAX_LEN=32768 \
    -e VLLM_GPU_UTIL=0.90 \
    -e VLLM_MAX_IMAGES_PER_PROMPT=16 \
    crucible:cuda
```

The container starts vLLM in the background and waits up to 10 minutes
for `/v1/models` to come up before launching the FastAPI orchestrator.

**First boot** downloads ~16 GB of weights for Qwen3-VL-8B from the
HuggingFace Hub. With `HF_HUB_ENABLE_HF_TRANSFER=1` (set in the
Dockerfile) expect 2–4 minutes. Subsequent boots use the cached weights
and finish in <90 seconds.

### 5. Verify endpoints

In a second terminal on the box:

```bash
curl -s http://localhost:8001/v1/models | python3 -m json.tool
# Expect: {"object": "list", "data": [{"id": "crucible-vlm", ...}]}

curl -s http://localhost:8000/healthz
# Expect: {"ok": true, "default_model": "Qwen/Qwen3-VL-8B-Instruct"}
```

### 6. End-to-end smoke

```bash
# Set client-side env vars
export CRUCIBLE_VLM_ENDPOINT=http://localhost:8001/v1
export CRUCIBLE_VLM_MODEL=crucible-vlm
export CRUCIBLE_VLM_API_KEY=EMPTY

# Single-critic test
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual \
    --frames 8

# Expected: JSON with score 0-10, verdict in {EXCELLENT, ACCEPTABLE,
# MARGINAL, REJECT}, non-empty rationale, ≥1 evidence item.
```

If this passes, run the full pipeline on a small dataset:

```bash
python scripts/precache_demo.py \
    --repo lerobot/aloha_static_cups_open \
    --episodes 5 \
    --frames 12

# Expected: ~60–90 second wall-clock on L40S, all 5 records have
# non-fallback verdicts, no PARSE_ERROR.
```

### 7. Stop the instance when done

```bash
# Inside the container terminal: Ctrl-C to shut down

# AWS Console → EC2 → instances → select crucible-gpu → Instance state
# → Stop. Storage stays attached at $16/month; if you don't need to
# resume, also Terminate to delete the volume.
```

**Do not leave g6e.xlarge running idle**. At $1.86/hour, an idle weekend
costs ~$90.

## Known quirks

- **First request is slow** (3–5 seconds) because vLLM CUDA graphs
  capture on the first inference. Subsequent requests are 0.5–1.5 s.
  If this matters, send a warmup request after `/v1/models` returns.
- **`response_format=json_schema` works** with `--guided-decoding-backend
  xgrammar` (set in the Dockerfile entrypoint). Crucible's three-tier
  fallback handles providers that don't support it.
- **L40S doesn't support FP8.** Qwen3-VL ships BF16 native; that's what
  the Dockerfile pins via `--dtype bfloat16`.
- **Spot instances** are available for ~30% off ($1.30/hour vs $1.86)
  but get interrupted ~daily on average. Useful for offline batch
  scoring; not recommended for live demos.
- **HF Hub rate limits** without a token are tight. Export `HF_TOKEN`
  before `docker run` for any non-trivial dataset.

## Verification (paste output in the issue if filing one)

```bash
# 1. tests pass on the box (no GPU needed)
python -m pytest tests -q

# 2. I/O smoke against real data
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2

# 3. vLLM healthy
curl -s http://localhost:8001/v1/models | python3 -m json.tool

# 4. Single critic against live VLM
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open --critic visual

# 5. Full pipeline
python scripts/precache_demo.py \
    --repo lerobot/aloha_static_cups_open --episodes 5

# 6. nvidia-smi snapshot for the README
nvidia-smi
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker build` fails: "could not select device driver" | nvidia-container-toolkit not installed | `sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker` |
| vLLM exits with `CUDA out of memory` | KV cache too big for the GPU | Lower `VLLM_MAX_LEN` to 16384 or `VLLM_GPU_UTIL` to 0.80 |
| Critics return `PARSE_ERROR` | xgrammar didn't load | Check container logs for `Successfully imported xgrammar`; if missing, rebuild image |
| `precache_demo.py` slow (>5 min for 5 eps) | Cold cache + first-request CUDA graph capture | Run a warmup request, or expect first dataset to be slower |
| HF download fails with 429 | Rate-limited as anonymous user | Export `HF_TOKEN` and re-run |
| No frames decoded from videos | Missing libdav1d in container | Should be in the Dockerfile; rebuild if you customized |
