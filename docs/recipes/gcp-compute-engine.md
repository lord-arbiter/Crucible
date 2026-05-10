# Recipe: Google Cloud — Compute Engine (G2 / A3 / N1+T4)

## When to use

You're on Google Cloud (existing project, $300 free trial credits, billing already in place). GCP has strong GPU coverage with the L4 (G2 series) being one of the cheapest 24 GB GPUs anywhere, plus competitive A100 / H100 availability.

## Cost

Verify current pricing at <https://cloud.google.com/compute/gpus-pricing>.

Recommended instance types (May 2026 retail in `us-central1`):

| Instance type | GPU | VRAM | Approx $/hr |
|---|---|---|---|
| `g2-standard-4` | 1× L4 | 24 GB | ~$0.69 |
| `g2-standard-8` | 1× L4 | 24 GB | ~$0.85 |
| `a2-highgpu-1g` | 1× A100 | 40 GB | ~$3.67 |
| `a2-ultragpu-1g` | 1× A100 | 80 GB | ~$5.07 |
| `a3-highgpu-1g` | 1× H100 | 80 GB | ~$10.83 |
| `a3-megagpu-8g` | 8× H100 | 8×80 GB | ~$87 |

Recommended for Crucible v0.1: **`g2-standard-4`** (1× L4 24 GB, ~$0.69/hr) for development, or **`a2-ultragpu-1g`** (1× A100 80 GB, ~$5.07/hr) for production-quality scoring with Qwen3-VL-32B.

## Setup

### 1. Provision via gcloud CLI

```bash
PROJECT_ID=your-project-id
ZONE=us-central1-a
VM_NAME=crucible-gpu

# Authenticate if needed
gcloud auth login
gcloud config set project $PROJECT_ID

# Create a VM with a Deep Learning VM image
# Family: pytorch-latest-gpu (includes NVIDIA driver, CUDA, Docker, nvidia-container-toolkit)
gcloud compute instances create $VM_NAME \
  --zone=$ZONE \
  --machine-type=g2-standard-4 \
  --accelerator=count=1,type=nvidia-l4 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --metadata=install-nvidia-driver=True \
  --tags=crucible-gpu

# Open inbound ports 8000 and 8001 to your IP only
MY_IP=$(curl -s https://api.ipify.org)
gcloud compute firewall-rules create crucible-allow-app \
  --network=default \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:8000,tcp:8001 \
  --source-ranges=${MY_IP}/32 \
  --target-tags=crucible-gpu

# Get the public IP
gcloud compute instances describe $VM_NAME --zone=$ZONE \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

For an A100 instead of L4:
```bash
--machine-type=a2-ultragpu-1g --accelerator=count=1,type=nvidia-a100-80gb
```

For an H100:
```bash
--machine-type=a3-highgpu-1g --accelerator=count=1,type=nvidia-h100-80gb
```

### 2. SSH in and run the container

```bash
gcloud compute ssh $VM_NAME --zone=$ZONE
nvidia-smi   # confirm GPU visible

git clone https://github.com/lord-arbiter/Crucible
cd Crucible
docker build -f docker/Dockerfile.cuda -t crucible:cuda .

export HF_TOKEN=hf_...
docker run --rm --gpus all \
    -p 8000:8000 -p 8001:8001 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    -e CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct \
    -e VLLM_MAX_LEN=32768 \
    crucible:cuda
```

On L4 (24 GB), stick with the 8B model. On A100 80 GB or H100, you can run Qwen3-VL-32B with `VLLM_MAX_LEN=65536`.

### 3. Verify and smoke test

Same as [cloud-gpu-vm.md](./cloud-gpu-vm.md) sections 6–7.

### 4. Stop / delete the instance

```bash
# Stop (keeps disk; bills storage only)
gcloud compute instances stop $VM_NAME --zone=$ZONE

# Or delete entirely
gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet
gcloud compute firewall-rules delete crucible-allow-app --quiet
```

## Known quirks

- **GPU quota.** New projects start with 0 GPU quota in most regions. Request a quota increase via Console → IAM & Admin → Quotas before provisioning, or you'll get `QUOTA_EXCEEDED`. Approval is usually 1–24 hours.
- **Spot / preemptible**: 60–91% discount, can be evicted any time. Useful for offline batch curation jobs only.
- **Image family `pytorch-latest-gpu`** comes with NVIDIA driver pre-installed. If you opt for a vanilla Ubuntu image, set `--metadata=install-nvidia-driver=True` and the GCP startup script will install drivers on first boot (adds ~5 minutes to provisioning).
- **L4 (G2)** is excellent value at $0.69/hr but has only 24 GB VRAM — fine for Qwen3-VL-8B fp16, tight for 32B even with int8.
- **H100 (A3)** stock is constrained in `us-central1`; try `us-east4`, `us-east5`, `europe-west4`, or `asia-southeast1` if `us-central1` is out.
- **TPU.** GCP also offers TPUs but vLLM TPU support is experimental — stick with GPUs unless you have specific TPU expertise.

## Verification

```bash
nvidia-smi
curl -s http://localhost:8001/v1/models
python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual
```

Pass: JSON parses, score ∈ [0, 10], total round-trip <10 seconds (faster on H100, ~3 seconds).

## When to use GCP vs other clouds

- **Cheapest 24 GB GPU**: GCP L4 at $0.69/hr beats every cloud's equivalent (AWS g5.xlarge at ~$1.00, Azure NV6 at similar).
- **Cheapest A100 80 GB**: GCP a2-ultragpu-1g at ~$5/hr is roughly tied with Azure NCads_A100_v4 and slightly cheaper than AWS p4de.
- **You're already on GCP**: stay on GCP. Cross-cloud data egress is expensive.
- **Other**: see the [universal cloud-gpu-vm recipe](./cloud-gpu-vm.md).
