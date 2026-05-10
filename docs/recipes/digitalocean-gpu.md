# Recipe: DigitalOcean GPU Droplet (H100 / H100x8)

## When to use

You want the simplest possible cloud experience — DigitalOcean's UX is famously friction-free, no quota requests, no IAM ceremony, just click "Create" and SSH in 2 minutes later.

DigitalOcean added GPU droplets in 2024 with NVIDIA H100 80 GB at competitive pricing. Best for solo developers and small teams that want to avoid AWS/Azure/GCP complexity.

## Cost

Verify current pricing at <https://www.digitalocean.com/pricing/gpu-droplets>.

Available sizes (May 2026):

| Size | GPU | VRAM | Approx $/hr |
|---|---|---|---|
| `gpu-h100x1-80gb` | 1× H100 | 80 GB | ~$4.89 |
| `gpu-h100x8-640gb` | 8× H100 | 8×80 GB | ~$39.12 |

Both are SXM-form-factor (full-bandwidth NVLink). Single-H100 at $4.89/hr is roughly half AWS p5.x's price.

DO bills per hour with no minimum commitment. There is no Spot equivalent.

## Setup

### 1. Provision via the doctl CLI (or web console)

```bash
# Install doctl if you haven't:
# https://docs.digitalocean.com/reference/doctl/how-to/install/
doctl auth init   # paste your API token

# List available GPU droplet sizes
doctl compute size list | grep gpu

# Create the droplet
doctl compute droplet create crucible-gpu \
  --image gpu-h100x1-80gb-base \
  --size gpu-h100x1-80gb \
  --region nyc2 \
  --ssh-keys $(doctl compute ssh-key list --no-header --format ID | head -1) \
  --enable-monitoring \
  --wait

# Get the public IP
doctl compute droplet list crucible-gpu --no-header --format "PublicIPv4"
```

DO offers `gpu-h100x1-80gb-base` as the official GPU droplet image (Ubuntu 22.04 + NVIDIA driver + Docker + nvidia-container-toolkit pre-installed).

### 2. SSH in and run the container

```bash
ssh root@<public-ip>
nvidia-smi   # confirm H100

git clone https://github.com/lord-arbiter/Crucible
cd Crucible
docker build -f docker/Dockerfile.cuda -t crucible:cuda .

export HF_TOKEN=hf_...
docker run --rm --gpus all \
    -p 8000:8000 -p 8001:8001 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    -e CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct \
    -e VLLM_MAX_LEN=65536 \
    -e VLLM_GPU_UTIL=0.90 \
    crucible:cuda
```

H100 80 GB fits Qwen3-VL-32B comfortably with 65k context. For bigger models on the 8× variant, set `VLLM_TENSOR_PARALLEL=8`.

### 3. Open inbound ports

DO Cloud Firewalls aren't enabled by default. Either accept that port 8000/8001 are open to the internet (DO assigns the droplet a public IP), or attach a firewall:

```bash
doctl compute firewall create \
  --name crucible-firewall \
  --inbound-rules "protocol:tcp,ports:22,address:0.0.0.0/0 protocol:tcp,ports:8000,address:$(curl -s ifconfig.me)/32 protocol:tcp,ports:8001,address:$(curl -s ifconfig.me)/32" \
  --droplet-ids $(doctl compute droplet list crucible-gpu --no-header --format "ID")
```

### 4. Verify and smoke test

Same as [cloud-gpu-vm.md](./cloud-gpu-vm.md) sections 6–7.

### 5. Power off and destroy when done

```bash
# Stops billing for compute. Storage continues at ~$0.10/GB/month.
doctl compute droplet-action power-off crucible-gpu --wait

# Or destroy entirely
doctl compute droplet delete crucible-gpu --force
```

## Known quirks

- **No GPU quota requests.** Unlike AWS / Azure / GCP where new accounts have 0 GPU quota, DO lets you provision GPU droplets immediately. This is the main reason to pick DO for one-off scoring runs.
- **Limited regions.** GPU droplets are only in `nyc2`, `tor1`, `ams3`, `fra1`, `sgp1` (May 2026). If your target region isn't listed, latency may be noticeably worse.
- **No image marketplace for ML stacks.** The `-base` image is bare-metal Ubuntu plus the GPU stack — no PyTorch / CUDA toolkit pre-baked. The Crucible Docker image bundles everything we need, so this is fine.
- **H100 SXM** here means PCIe-attached SXM cards in a custom DigitalOcean chassis — not the NVLink-mesh you get on H100 SXM5. For multi-GPU scaling beyond 8 cards, AWS p5.48xlarge or Azure ND96isr_H100_v5 will outperform.
- **Block storage** is $0.10/GB/month if attached. The droplet's local SSD (200 GB on `gpu-h100x1-80gb`) is plenty for the HF cache.
- **Egress** is included up to 10 TB/month per droplet. Pulling Qwen3-VL-32B (~64 GB) once is a rounding error.

## Verification

```bash
nvidia-smi   # H100 80 GB visible
curl -s http://localhost:8001/v1/models
python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual
```

Pass: JSON parses, score ∈ [0, 10], total round-trip <5 seconds on H100.

## When to use DO vs other clouds

- **Solo dev, want zero ceremony**: DO. No quota wait, no IAM setup, droplet ready in 2 minutes.
- **Scoring 100s of episodes one-time**: DO H100x1 at $4.89/hr is competitive with AWS p4de ($4-5/hr A100) but with H100's faster compute.
- **You're on a long-running production deployment**: AWS / Azure / GCP committed-use discounts beat DO's flat hourly.
- **You need 8× H100 with NVLink mesh for >70B models**: AWS p5.48xlarge or Azure ND96isr_H100 are the proper paths; DO's 8x variant is fine but PCIe-bandwidth-limited.
