# Recipe: AMD MI300X / Developer Cloud

## When to use

You want **Qwen3-VL-32B-Instruct at full bf16 precision with 65k token
context**, and you have access to AMD MI300X (192 GB HBM3). This is the
recipe Crucible was originally built around — the MI300X's memory
headroom lets the model run with no quantization and room to spare for
batched multimodal requests.

Best for: production deployments, the larger Qwen3-VL-72B variant,
high-throughput batch curation jobs.

## Cost

- **Free path:** AMD AI Developer Program ships a $100 credit on the
  AMD Developer Cloud. Sign up at
  <https://www.amd.com/en/developer/ai-dev-program.html>. No approval
  gate; provisioning is 2–4 minutes once registered.
- **Paid path:** RunPod community MI300X at ~$0.50/hr (variable),
  Hot Aisle on-demand at ~$1.99/hr.

A 25-episode validation run uses well under $1 of credit.

## Setup

### 1. Provision the droplet

If using AMD Developer Cloud:

1. Sign up at the AMD AI Developer Program link above.
2. Inside the member portal → Member Perks → claim the AMD Developer
   Cloud credit.
3. Land on <https://devcloud.amd.com>.
4. Add an SSH key.
5. Create GPU Droplet → 1× MI300X / Ubuntu 22.04 + ROCm.
6. Droplet Active in 2–4 minutes.

If using RunPod:

1. <https://console.runpod.io/deploy> → filter "MI300X".
2. Pick the cheapest community pod with a recent ROCm image.
3. Deploy, wait ~2 minutes for SSH access.

### 2. Verify the GPU

```bash
ssh ubuntu@<public-ip>

rocm-smi --showproductname        # MI300X line
rocm-smi --showmeminfo vram       # Total VRAM ≈ 196,592 MB
rocm-smi --showtemp --showpower   # idle temp <60°C, power <100W
```

### 3. Pull the repo and build the ROCm image

```bash
sudo apt-get update && sudo apt-get install -y git docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER
# log out and back in for the docker group to take effect

git clone https://github.com/lord-arbiter/Crucible
cd Crucible

docker build -f docker/Dockerfile.gpu -t crucible:rocm .
```

If the default base tag has rotated, override:

```bash
docker build -f docker/Dockerfile.gpu \
    --build-arg BASE=vllm/vllm-openai-rocm:latest \
    -t crucible:rocm .
```

### 4. Run the container

```bash
export HF_TOKEN=hf_...   # optional but recommended

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HF_TOKEN \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 -p 8001:8001 \
    crucible:rocm
```

The container starts vLLM and waits up to 10 minutes for
`/v1/models` to come up before launching the FastAPI orchestrator.

First boot downloads ~64 GB of weights for Qwen3-VL-32B. With
`HF_HUB_ENABLE_HF_TRANSFER=1` (set in the Dockerfile) expect 3–5
minutes from a fast network. Subsequent boots <2 minutes.

### 5. Verify and run the test sequence

See [TESTING.md](../../TESTING.md) for the full 10-layer test plan.
Quick version:

```bash
curl -s http://localhost:8001/v1/models | python3 -m json.tool
curl -s http://localhost:8000/healthz

export CRUCIBLE_VLM_ENDPOINT=http://localhost:8001/v1
export CRUCIBLE_VLM_MODEL=crucible-vlm
export CRUCIBLE_VLM_API_KEY=EMPTY

python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open --critic visual

python scripts/precache_demo.py \
    --repo lerobot/aloha_mobile_cabinet --episodes 25 --frames 16
```

Expected: 25 episodes scored in ~3 minutes on MI300X.

### 6. Stop the droplet when done

AMD Cloud / RunPod consoles → stop or terminate the instance. The
HF cache disk usually carries through stop/start cycles; terminate
deletes it.

## Known quirks

- **Base image rotates.** AMD's `rocm/vllm-dev:nightly_main_*` tag is
  date-stamped. If the default fails to pull, use the override above.
- **`HSA_FORCE_FINE_GRAIN_PCIE=1`** is set in the Dockerfile; required
  for MI300X performance.
- **`VLLM_USE_V1=1` + `VLLM_ROCM_USE_AITER=1`** are the AMD performance
  path; both are set in the entrypoint.
- **Image preprocessing** uses rocJPEG when available — first request
  may compile kernels (3–5 s). Warmup with one request before live
  demos.
- **Concurrency.** MI300X comfortably handles 5 concurrent multimodal
  requests at 16 images each. For higher concurrency, raise
  `--max-num-seqs` past 16 in the entrypoint.

## Performance reference

LMSYS published a Feb 2026 benchmark for Qwen3-VL-235B on 8× MI300:

- TTFT: 1.08 s (5 images @ 960×1280, 8k text, 500 output tokens)
- TPOT: 12.5 ms

Scaling down to **32B on 1× MI300X** with smaller images (768px) and
tighter context, expect:

- Per-request: ~600–900 ms TTFT, ~6–9 ms / token
- 5 critics + 1 aggregator per episode: ~5–8 s
- 25 episodes: ~2–3 minutes wall-clock

## Multi-GPU scaling

For Qwen3-VL-72B or higher concurrency, use a multi-GPU MI300X box
(`MI300X-8` configurations on Hot Aisle / RunPod). Set
`--tensor-parallel-size 8` in the vLLM serve command. Scaling is
near-linear up to 8× for compute-bound stages.

## When to use this vs. AWS EC2

| | AMD MI300X | AWS EC2 g6e.xlarge |
|---|---|---|
| GPU | 1× MI300X 192 GB | 1× L40S 48 GB |
| Best model | Qwen3-VL-32B fp16 | Qwen3-VL-8B fp16 |
| Cost (free) | $100 AMD credit | n/a |
| Cost (paid) | $0.50–$2/hr | $1.86/hr |
| Bring-up time | ~10 min | ~30 min |
| Bf16 throughput | Highest | Lower |

If the `ml.g6e.12xlarge` (4× L40S) is in your budget, Qwen3-VL-32B fits
there with tensor-parallel — see the planned multi-GPU recipe.
