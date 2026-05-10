# Recipe: Any cloud GPU VM (universal NVIDIA path)

## When to use

Your cloud isn't covered by a dedicated recipe, OR you want the lowest-common-denominator path that works on AWS, Azure, GCP, DigitalOcean, Lambda, RunPod, Vast.ai, Paperspace, OVH, Hetzner, Oracle Cloud, IBM Cloud, and any bare-metal NVIDIA host.

This recipe is the universal pattern. Cloud-specific recipes ([AWS](./aws-ec2.md), [Azure](./azure-vm.md), [GCP](./gcp-compute-engine.md), [DigitalOcean](./digitalocean-gpu.md), [Lambda](./lambda-labs.md), [RunPod](./runpod-nvidia.md), [Vast](./vast-ai.md)) cover the same flow with a more specific provisioning section.

## What you need

- A VM with at least one NVIDIA GPU (≥ 24 GB VRAM recommended; ≥ 48 GB for Qwen3-VL-32B fp16)
- NVIDIA driver installed (`nvidia-smi` works)
- Docker installed
- `nvidia-container-toolkit` installed (so `docker run --gpus all` works)
- 200 GB of disk for the HuggingFace model cache
- Inbound TCP 22 (SSH), 8000 (FastAPI orchestrator), 8001 (vLLM)

Most clouds offer a pre-baked "Deep Learning" or "GPU-optimized" image that comes with all four prereqs pre-installed. Use that — saves 30 minutes of setup.

## GPU sizing

Match the GPU VRAM to the model you want to run:

| GPU | VRAM | Recommended model | Why |
|---|---|---|---|
| T4, L4, A10G | 16-24 GB | Qwen3-VL-8B (4-bit quantized) | Cheapest path; OK for development |
| L40, A40, RTX 6000 Ada, L40S | 40-48 GB | Qwen3-VL-8B (fp16) | Sweet spot for cost vs. quality |
| A100 80 GB, H100 80 GB | 80 GB | Qwen3-VL-32B (fp16) | Premium quality |
| MI300X | 192 GB | Qwen3-VL-32B / 72B at 65k context | See [amd-mi300x.md](./amd-mi300x.md) |

Crucible itself doesn't care about GPU brand or memory beyond fitting your chosen model. If you can run `vllm serve <model>` on the box, Crucible runs.

## Setup

### 1. Provision a GPU VM

Cloud-specific. See the dedicated recipes. The universal flow:

1. Pick a region (closer to you = lower latency; usually `us-east-*` has best stock).
2. Pick an instance type with the GPU you want.
3. Pick a Deep Learning AMI / image (Ubuntu 22.04+ recommended).
4. Configure 200 GB storage.
5. Configure inbound rules: 22 (SSH from your IP), 8000 + 8001 (from your IP only).
6. Add your SSH public key.
7. Launch.

### 2. SSH in and verify

```bash
ssh -i ~/.ssh/<key>.pem <user>@<public-ip>

# Confirm GPU + CUDA are visible
nvidia-smi
# Should show your GPU, driver version, CUDA version

# Confirm Docker + GPU toolkit are working
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# Should print the same nvidia-smi output from inside a container
```

If `nvidia-smi` errors or `docker run --gpus all` fails, the image you picked didn't have the GPU stack pre-installed. See [Bare Ubuntu fallback](#bare-ubuntu-fallback) below.

### 3. Pull the repo

```bash
git clone https://github.com/lord-arbiter/Crucible
cd Crucible
```

### 4. Build the CUDA Docker image

```bash
docker build -f docker/Dockerfile.cuda -t crucible:cuda .
```

First build pulls `vllm/vllm-openai:latest` (~12 GB) and adds Crucible's deps. Allow 5–8 minutes.

If your cloud blocks Docker Hub pulls (rare), use a mirror via `--build-arg BASE=...`. See the troubleshooting table.

### 5. Run the container

```bash
# Optional: HF token for faster downloads + higher rate limits
export HF_TOKEN=hf_...

docker run --rm --gpus all \
    -p 8000:8000 -p 8001:8001 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    -e CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct \
    -e VLLM_MAX_LEN=32768 \
    -e VLLM_GPU_UTIL=0.90 \
    crucible:cuda
```

The container starts vLLM in the background (waits up to 10 minutes for `/v1/models`) and then launches the FastAPI orchestrator on `:8000`.

First boot downloads ~16 GB for Qwen3-VL-8B. Allow 3–5 minutes. The container caches weights to the mounted `~/.cache/huggingface`, so subsequent boots use cached weights and finish in <90 seconds.

To run a different model, change `CRUCIBLE_VLM_MODEL`:

```bash
# Bigger model, needs 80 GB VRAM
-e CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
-e VLLM_MAX_LEN=65536

# Different family
-e CRUCIBLE_VLM_MODEL=meta-llama/Llama-3.2-11B-Vision-Instruct
-e CRUCIBLE_VLM_MODEL=OpenGVLab/InternVL3-8B
```

### 6. Verify endpoints

In a second terminal on the box:

```bash
curl -s http://localhost:8001/v1/models | python3 -m json.tool
curl -s http://localhost:8000/healthz
```

### 7. Run the smoke tests

Still on the VM:

```bash
# Set client-side env vars (point at the local container)
export CRUCIBLE_VLM_ENDPOINT=http://localhost:8001/v1
export CRUCIBLE_VLM_MODEL=crucible-vlm
export CRUCIBLE_VLM_API_KEY=EMPTY

# Pull dependencies into a venv (the Crucible Python package, not the
# container — we use the host venv for the CLI scripts)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# I/O smoke (no LLM)
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2

# End-to-end smoke (single critic against the live container)
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual

# Full pipeline on a small dataset
python scripts/precache_demo.py \
    --repo lerobot/aloha_static_cups_open \
    --episodes 5
```

### 8. Stop the VM when done

GPU VMs bill while running. Stop or terminate the moment you're done:

- **AWS:** EC2 console → Instance state → Stop (or Terminate)
- **Azure:** Portal → Virtual machines → Stop / Deallocate
- **GCP:** Console → Compute Engine → Stop
- **DigitalOcean:** Droplets → ⋯ → Power Off → Destroy
- **Lambda Labs:** On-Demand → Terminate
- **RunPod:** Pods → Stop / Terminate
- **Vast.ai:** Instances → Destroy

Pricing varies $0.50–$10/hr depending on GPU; an idle weekend on H100 ≈ $400. Don't forget.

## Bare Ubuntu fallback

If your cloud's image doesn't pre-install the GPU stack, run this once on a fresh Ubuntu 22.04 VM:

```bash
# 1. NVIDIA driver
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot

# After reboot, confirm:
nvidia-smi   # should print GPU details

# 2. Docker
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
# log out and back in for the docker group

# 3. NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

This is the same setup AWS Deep Learning AMI / Lambda's stock images do for you. Allow 10 minutes plus reboot.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi: command not found` | Image lacks NVIDIA driver | Use a Deep Learning image, or run the bare Ubuntu fallback above |
| `docker run --gpus all` fails: "could not select device driver" | nvidia-container-toolkit missing | Run section 3 of the bare Ubuntu fallback |
| `docker pull` is slow or fails | Cloud blocks Docker Hub | Use a mirror: `--build-arg BASE=public.ecr.aws/vllm/vllm-openai:latest` (AWS) or `--build-arg BASE=ghcr.io/vllm-project/vllm/vllm-openai:latest` |
| vLLM exits with `CUDA out of memory` | Model too big for GPU | Lower `VLLM_MAX_LEN` to 16384, or pick a smaller model (`Qwen/Qwen3-VL-8B-Instruct` instead of 32B) |
| `curl /v1/models` hangs | Weights still downloading | `du -sh ~/.cache/huggingface/hub` to watch progress |
| Critics return `PARSE_ERROR` | xgrammar didn't load in container | Rebuild with `--no-cache`; ensure `transformers >= 4.57` and `xgrammar >= 0.1.18` are pinned in the Dockerfile |
| Cannot reach `:8000` from your laptop | Inbound rules block it | Open TCP 8000 to your IP only in the cloud's security group / NSG / firewall |
| First request takes 5+ seconds | Cold CUDA graph capture | Send a warmup request after `/v1/models` returns; subsequent requests are faster |

## Verification

```bash
# 1. tests pass on the box
python -m pytest tests -q

# 2. I/O smoke against real data
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2

# 3. vLLM healthy
curl -s http://localhost:8001/v1/models | python3 -m json.tool

# 4. Single critic against live VLM
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open --critic visual

# 5. Full pipeline (5 episodes < 90 seconds)
python scripts/precache_demo.py \
    --repo lerobot/aloha_static_cups_open --episodes 5

# 6. nvidia-smi snapshot for the README
nvidia-smi
```

Pass: all 6 succeed end-to-end. If any fail, see the troubleshooting table above.
