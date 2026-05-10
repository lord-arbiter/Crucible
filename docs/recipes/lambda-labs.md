# Recipe: Lambda Labs On-Demand

## When to use

You want a no-ceremony GPU instance where pricing is the lowest among major US providers, and you're scoring batch jobs that can interrupt without losing work. Lambda Labs is a developer-friendly GPU cloud with the simplest possible pricing model.

Lambda has been the cheapest H100 SXM on-demand provider since 2023. The trade-off: stock is intermittent — sometimes you'll see "out of capacity" for hours.

## Cost

Verify current pricing at <https://lambdalabs.com/service/gpu-cloud>.

On-demand pricing (May 2026):

| Instance | GPU | VRAM | Approx $/hr |
|---|---|---|---|
| 1× A100 PCIe | 1× A100 | 40 GB | ~$1.29 |
| 1× A100 SXM | 1× A100 | 80 GB | ~$1.79 |
| 1× H100 PCIe | 1× H100 | 80 GB | ~$2.99 |
| 1× H100 SXM | 1× H100 | 80 GB | ~$3.49 |
| 8× H100 SXM | 8× H100 | 8×80 GB | ~$27.92 |

Lambda's H100 SXM at $3.49/hr is roughly half AWS p5.x and a third of Azure NCads_H100. Lowest H100 pricing among the major US providers — when stock is available.

## Setup

### 1. Provision via the web console

Lambda doesn't have a public CLI for instance creation; use the web console at <https://cloud.lambdalabs.com/instances>.

1. Sign up / log in.
2. Click "Launch instance".
3. Pick a region and instance type. Stock is shown live in the UI; if a configuration is greyed out, it's out of capacity — try another region.
4. Pick a filesystem (Lambda Filesystem, persistent across instances) or skip for ephemeral storage.
5. Select your SSH key.
6. Launch.

The instance boots with `Lambda Stack` pre-installed: NVIDIA drivers, CUDA, cuDNN, PyTorch, Docker, nvidia-container-toolkit. Provisioning is 1–3 minutes.

### 2. SSH in and run the container

```bash
ssh ubuntu@<lambda-instance-ip>
nvidia-smi   # confirm GPU

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
    crucible:cuda
```

On A100 80 GB or H100, run Qwen3-VL-32B at full context. On A100 40 GB, drop to Qwen3-VL-8B.

### 3. Verify and smoke test

Same as [cloud-gpu-vm.md](./cloud-gpu-vm.md) sections 6–7.

### 4. Terminate when done

Lambda's web console → Instances → ⋯ → Terminate. **There is no "stop, keep storage" mode** unless you used a persistent filesystem; otherwise terminating destroys local data. The HF model cache is gone after termination — next run re-downloads.

If you want to preserve the cache between runs, attach a Lambda Filesystem at instance launch and mount it at `~/.cache/huggingface`.

## Known quirks

- **Stock volatility.** Lambda's prices are low because their utilization is high — popular configurations are often out of capacity, especially `8× H100 SXM` and `1× H100 SXM`. Have a fallback (RunPod, Vast) ready.
- **No CLI** for provisioning instances at the time of writing. There's a [public API](https://docs.lambdalabs.com/cloud/api/) but it's read-only for instance management; use the web console for launches.
- **No spot tier.** All on-demand. If a job is interruptible-tolerant, Vast.ai or AWS Spot have cheaper options.
- **Lambda Filesystem.** Optional persistent network filesystem ($0.20/GB/month). Useful if you want the HF cache preserved across instance terminations; not needed for one-off runs.
- **Public IPs are dynamic** unless you reserve a static one ($5/month). For short-lived debugging, just use the IP that comes with the instance.
- **Inbound ports** — Lambda's default firewall opens 22, 8888 (Jupyter), 6006 (TensorBoard) to the world. To open 8000/8001 to your IP, use `iptables` or `ufw` on the instance:

```bash
sudo ufw allow from $(curl -s ifconfig.me) to any port 8000
sudo ufw allow from $(curl -s ifconfig.me) to any port 8001
sudo ufw enable
```

## Verification

```bash
nvidia-smi
curl -s http://localhost:8001/v1/models
python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual
```

Pass: JSON parses, score ∈ [0, 10], total round-trip <5 seconds on H100.

## When to use Lambda vs other clouds

- **Lowest H100 / A100 hourly**: Lambda, when stock is available.
- **Stock unavailable on Lambda**: RunPod community, then Vast.ai, then DigitalOcean H100x1.
- **Production with reserved capacity**: AWS / Azure committed-use beats Lambda's flat hourly.
- **Long-running, persistent state**: Lambda Filesystem works but at this point you might as well rent reserved capacity.
