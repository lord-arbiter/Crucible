# Recipe: Vast.ai

## When to use

You want the absolute cheapest GPU rental on the market and you're OK with peer-to-peer hosting where reliability varies by host. Vast.ai is a marketplace where independent data centers and individuals rent out spare GPU capacity — prices float by supply and demand, and you pick which host to rent from based on their reliability score.

Best for: cost-sensitive batch curation, exploratory development, or running into stock issues on the major clouds.

## Cost

Highly variable — that's the whole point. Browse <https://cloud.vast.ai/create/> and you'll see live pricing. As of May 2026, typical ranges:

| GPU | VRAM | Approx $/hr (interruptible) | Approx $/hr (on-demand) |
|---|---|---|---|
| RTX 4090 | 24 GB | $0.18–0.30 | $0.30–0.50 |
| L40 | 48 GB | $0.50–0.80 | $0.80–1.20 |
| A100 80 GB | 80 GB | $0.80–1.20 | $1.30–1.80 |
| H100 80 GB | 80 GB | $1.50–2.00 | $2.00–2.80 |

Vast frequently has the cheapest H100 anywhere. Trade-off is host quality varies; pick hosts with ≥ 99% reliability score.

## Setup

### 1. Browse and rent

Web console at <https://cloud.vast.ai/create/>:

1. Sign up / log in. Add credit ($10 minimum to deploy).
2. Filter by GPU type, price, and reliability (≥ 99% recommended).
3. Pick a host. Look at the host's history (`Reliability` column) and DLPerf score.
4. Choose a Docker image — pick `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` or any official ML image.
5. Set "On-Demand" pricing for guaranteed availability, or "Interruptible" for cheaper but evictable.
6. Click "Rent".

Or via CLI (`vastai`):

```bash
pip install vastai
vastai set api-key <your-key>

# Search for cheap H100 instances
vastai search offers 'gpu_name=H100 num_gpus=1 reliability>0.99' \
  --order 'dphtotal' --limit 5

# Pick one and rent
vastai create instance <offer-id> \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \
  --disk 200 \
  --label crucible-gpu
```

### 2. SSH in and run the container

Vast assigns an SSH command in the instance dashboard. Format is typically:

```bash
ssh -p <port> root@<vast-ip>
nvidia-smi   # confirm GPU

git clone https://github.com/lord-arbiter/Crucible
cd Crucible
docker build -f docker/Dockerfile.cuda -t crucible:cuda .

export HF_TOKEN=hf_...
docker run --rm --gpus all \
    -p 8000:8000 -p 8001:8001 \
    -v /workspace/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    -e CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct \
    crucible:cuda
```

Vast persistent storage is at `/workspace`, similar to RunPod. Mount the HF cache there to survive instance restarts.

### 3. Port forwarding

Vast doesn't proxy HTTP by default. Two options:

**SSH tunneling (recommended for solo dev):**

```bash
# From your laptop:
ssh -p <vast-port> -L 8000:localhost:8000 -L 8001:localhost:8001 root@<vast-ip>
# Then localhost:8000 on your laptop hits the container on the rented box.
```

**Open the ports publicly (for the Gradio Space backend):**

In the instance dashboard, "Forward Port" → assign 8000 and 8001 to public ports. Vast gives you a public IP + port mapping.

### 4. Verify and smoke test

Same as [cloud-gpu-vm.md](./cloud-gpu-vm.md) sections 6–7.

### 5. Destroy when done

```bash
# Web: Instance → ⋯ → Destroy
# CLI:
vastai destroy instance <instance-id>
```

Vast bills per second, so destroying immediately after a job stops the meter.

## Known quirks

- **Reliability score** is the most important filter. Hosts below 99% will lose your job to a reboot at random. The cheapest hosts often have low scores; pay 20% more for one with 99%+.
- **Bandwidth varies by host.** Some hosts have residential gigabit, others have data-center 10 Gbps. Pulling Qwen3-VL-32B (~64 GB) on a slow host is ~30 minutes; on a fast host, ~5 minutes. The instance card shows download speed.
- **No SLA.** Hosts can disappear. Always set a budget alert and don't run anything you can't restart.
- **Interruptible pricing** is 30–40% cheaper but the instance can be evicted any time. Crucible's precache resumes from disk — this is fine for batch curation.
- **`pytorch/pytorch:*-devel`** is the right image; `-runtime` variants lack the build toolchain.
- **GPU passthrough** uses the host's NVIDIA driver — Vast publishes the driver version on the instance card. Drivers older than 535 lack support for some recent vLLM features; filter for ≥ 550.

## Verification

```bash
nvidia-smi
curl -s http://localhost:8001/v1/models
python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual
```

Pass: JSON parses, score ∈ [0, 10], total round-trip < 8 seconds.

## When to use Vast vs other clouds

- **Absolute cheapest GPU**: Vast wins on raw $/hr almost always.
- **Reliability matters**: Lambda > RunPod Secure > AWS/Azure/GCP > RunPod Community > Vast.
- **You're scoring 100s of episodes one-time**: Vast interruptible is fine; precache resumes from disk on eviction.
- **You're running a production live demo**: pick a major cloud or Lambda; Vast is too volatile.
