# Recipe: RunPod (NVIDIA — A100 / H100 / L40S)

## When to use

You want the cheapest NVIDIA GPU per hour on the market, and you're OK with marketplace-style availability where prices and stock fluctuate. RunPod has two tiers — Community Cloud (peer-hosted, lowest prices, occasional reboots) and Secure Cloud (data-center, ~50% more expensive, enterprise SLA). Both are available via the same web console and CLI.

For Crucible scoring runs, Community Cloud is a perfect fit — interruptions don't lose progress (the precache JSON resumes from disk).

## Cost

Verify current pricing at <https://www.runpod.io/pricing>.

Community Cloud (May 2026):

| GPU | VRAM | Approx $/hr |
|---|---|---|
| 1× L40 | 48 GB | ~$0.79 |
| 1× A40 | 48 GB | ~$0.39 |
| 1× A100 PCIe | 80 GB | ~$1.89 |
| 1× H100 PCIe | 80 GB | ~$1.99 |
| 1× H100 SXM | 80 GB | ~$2.99 |

Secure Cloud is roughly 1.3–1.5× these prices.

L40 at $0.79/hr is the cheapest 48 GB GPU anywhere — solid value for Qwen3-VL-8B development.

## Setup

### 1. Provision via the web console (or runpodctl CLI)

Easiest path — web console at <https://console.runpod.io/deploy>:

1. Sign up / log in.
2. Click "Deploy" → pick GPU (filter by type / price).
3. Pick a template — `RunPod PyTorch 2.4` is the standard ML image (NVIDIA driver, CUDA 12.4, Docker, nvidia-container-toolkit).
4. Configure storage (50 GB included; add 200 GB persistent volume for the HF cache).
5. Optional: set environment variables (HF_TOKEN, etc.) so you don't have to ssh-export them later.
6. Deploy. Pod is ready in <2 minutes.

For the CLI:

```bash
# Install runpodctl: https://docs.runpod.io/runpodctl/install-runpodctl
runpodctl create pod \
  --name crucible-gpu \
  --gpuType "NVIDIA H100 PCIe" \
  --gpuCount 1 \
  --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
  --containerDiskSize 50 \
  --volumeSize 200 \
  --volumePath /workspace
```

### 2. SSH in and run the container

RunPod gives you both an SSH command and a web terminal in the dashboard. Either works.

```bash
ssh root@<pod-ip> -p <pod-port>
nvidia-smi

git clone https://github.com/lord-arbiter/Crucible
cd Crucible

# Build inside the pod (uses the host's docker daemon)
docker build -f docker/Dockerfile.cuda -t crucible:cuda .

export HF_TOKEN=hf_...
docker run --rm --gpus all \
    -p 8000:8000 -p 8001:8001 \
    -v /workspace/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    -e CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct \
    crucible:cuda
```

Note: RunPod's persistent volume is mounted at `/workspace`, not `/root` — point the HF cache mount there to survive pod restarts.

### 3. Expose ports

RunPod proxies HTTP traffic for ports you declare at deploy time. To use the FastAPI orchestrator from your laptop:

1. In the deploy form, add 8000 and 8001 to "Expose HTTP Ports".
2. RunPod gives you a `https://<pod-id>-8000.proxy.runpod.net` URL.
3. Set `CRUCIBLE_API_BASE` to that URL in your local Gradio frontend or curl commands.

### 4. Verify and smoke test

Same as [cloud-gpu-vm.md](./cloud-gpu-vm.md) sections 6–7.

### 5. Stop / terminate the pod

```bash
# Stop (keeps the volume; bills storage only)
runpodctl stop pod <pod-id>

# Or terminate entirely
runpodctl remove pod <pod-id>
```

## Known quirks

- **Community Cloud reliability.** Pods occasionally reboot when the underlying host needs maintenance. Crucible's precache resumes from disk, so this is fine for scoring jobs. For live demos, use Secure Cloud.
- **Storage tiers.** Container disk (free, ephemeral) vs. Network Volume ($0.10/GB/month, persistent). The HF cache should go on Network Volume so model weights survive pod restarts.
- **GPU selection.** "L40S" and "L40" are different — L40S has FP8 support, L40 doesn't. Both work for our workload.
- **Templates.** The official `runpod/pytorch:*-devel-ubuntu22.04` image has the GPU stack. The `-runtime` variants don't include the build toolchain — fine for inference, but you can't `docker build` from them. We need devel for our Dockerfile.cuda build.
- **API key.** Set in account settings; export `RUNPOD_API_KEY=...` for runpodctl.

## Verification

```bash
nvidia-smi
curl -s http://localhost:8001/v1/models
python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual
```

Pass: JSON parses, score ∈ [0, 10], total round-trip <5 seconds on H100.

## When to use RunPod NVIDIA vs other clouds

- **Cheapest H100**: RunPod Community at $1.99/hr (PCIe) or $2.99/hr (SXM). Lambda comes close when stock is available.
- **Cheapest 48 GB GPU**: RunPod Community L40 at $0.79/hr.
- **You also want AMD MI300X**: RunPod has both — see [amd-mi300x.md](./amd-mi300x.md) for the AMD path.
- **Production with reserved capacity**: RunPod has Secure Cloud monthly contracts; AWS / Azure / GCP committed-use is more mature though.
