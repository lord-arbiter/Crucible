# Runbook — bring Crucible up on a fresh GPU box

This is the path you walk the moment your GPU credits land. Each step is
copy-pasteable. If a step fails, the troubleshooting section at the bottom
maps the symptom to the fix.

Total expected wall-clock: **~25 minutes** from `ssh` to first scored episode
on AMD MI300X. Add ~10 minutes if you have to fall back to RunPod H100.

## 0. Prereqs

- A running MI300X droplet on AMD Developer Cloud (or H100 80 GB on RunPod) with Ubuntu 22.04+, Docker, and HF_TOKEN exported.
- This repo cloned at `~/Crucible` on the box.
- Public HuggingFace dataset access (no auth needed for read).

## 1. Local sanity check before touching the GPU

Run this on your laptop. **If it fails here, the GPU run will also fail.** No GPU needed for this step.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2 --frames 6
# Expect: "PASS — 2 episodes streamed and decoded cleanly"
```

If you see `FAIL`, fix the dataset reader before bringing up the GPU.

## 2. AMD MI300X bring-up via Docker

The `docker/Dockerfile.gpu` builds on AMD's official ROCm vLLM image. Build
on the GPU box (the base image is multi-GB; don't push it from your laptop).

```bash
cd ~/Crucible

# Verify the GPU is visible.
rocm-smi --showproductname

# Build (single-arg override if AMD's nightly tag has rotated).
docker build -f docker/Dockerfile.gpu -t crucible:gpu .
# If the default base tag is unavailable:
# docker build -f docker/Dockerfile.gpu \
#     --build-arg BASE=vllm/vllm-openai-rocm:latest \
#     -t crucible:gpu .

# Run with both GPUs and HF cache mounted (so re-runs don't re-download Qwen3-VL).
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HF_TOKEN=${HF_TOKEN} \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 -p 8001:8001 \
    crucible:gpu
```

The container starts vLLM in the background and waits up to 10 minutes for
`/v1/models` to come up before launching the FastAPI orchestrator. First run
downloads ~64 GB of weights for Qwen3-VL-32B (set `HF_HUB_ENABLE_HF_TRANSFER=1`
already in the image — expect ~500 MB/s if peering is good). Subsequent runs
boot in <2 minutes.

## 3. Verify the endpoints

In a second terminal on the box:

```bash
curl -s http://localhost:8001/v1/models | python3 -m json.tool
# Expect: {"data": [{"id": "crucible-vlm", ...}]}

curl -s http://localhost:8000/healthz
# Expect: {"ok": true, "default_model": "Qwen/Qwen3-VL-32B-Instruct"}
```

## 4. End-to-end smoke against the live VLM

Still on the GPU box:

```bash
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual \
    --frames 8
```

Expected output: a JSON object with numeric `score` and a verdict from
`{EXCELLENT, ACCEPTABLE, MARGINAL, REJECT}`. If it returns `PARSE_ERROR`,
check `var/log/crucible/vllm.log` for guided-decoding warnings — see
troubleshooting #4.

## 5. Pre-cache the demo dataset

This produces `data/precached/lerobot__aloha_mobile_cabinet.json`. Commit it
back to the repo so the HF Space serves the demo even when this droplet is
offline.

```bash
python scripts/precache_demo.py \
    --repo lerobot/aloha_mobile_cabinet \
    --episodes 25 \
    --frames 16

# Expect ~3 minutes wall-clock at 5 concurrent critics on MI300X.
```

Then on your laptop:

```bash
git pull origin main
git add data/precached/lerobot__aloha_mobile_cabinet.json
git commit -m "chore(data): precached lerobot/aloha_mobile_cabinet (25 episodes)"
git push origin main
```

## 6. Deploy the Gradio Space

Still on your laptop (or a CI runner, not the GPU box):

```bash
huggingface-cli login  # paste a write-scoped token
HF_USER=<your-hf-username> SPACE_NAME=crucible ./scripts/deploy_space.sh
```

Open the Space URL and confirm:
- The dropdown loads with `lerobot/aloha_mobile_cabinet`.
- "Score Dataset" with `use_cache=true` returns instantly from the precache.
- Episode 47 (or wherever your most damning example lands) renders the five critic cards.

## 7. Set the Space's API base to point at the GPU droplet

In the Space settings → Variables and secrets, add:

```
CRUCIBLE_API_BASE = http://<droplet-public-ip>:8000
```

Restart the Space. Now "Score Dataset" with `use_cache=false` will trigger a
live run on the MI300X via SSE.

## 8. Demo recording

Follow `docs/demo_script.md` exactly. Don't ad-lib the killer-detail moment —
pre-pick episode index from the precache during pre-flight (step 5).

## 9. Submit + shut down

1. Fill in `docs/SUBMISSION.md` checklist on lablab.ai.
2. **Stop the GPU droplet** the moment submission is confirmed:
   ```bash
   # On AMD Developer Cloud:
   gcloud compute instances stop crucible-gpu --zone=...   # or click Stop in the UI
   ```
3. Tweet the final build-in-public message linking the submission.

---

## Troubleshooting

### 1. `docker build` fails: `manifest unknown` for `rocm/vllm-dev:nightly_main_*`

The nightly tag rotates. Run `docker pull vllm/vllm-openai-rocm:latest` and rebuild with `--build-arg BASE=vllm/vllm-openai-rocm:latest`. Both base images bundle ROCm 7.x + vLLM 0.16+ + a pre-built ROCm torch wheel.

### 2. `vllm serve` exits with `OutOfMemoryError` during profiling

Lower `VLLM_MAX_IMAGES_PER_PROMPT` to 8 (env var on `docker run`). Known interaction between `--limit-mm-per-prompt.image` and Qwen3-VL profiling (vLLM #38459) over-allocates KV. If still OOM, drop `VLLM_MAX_LEN` to 32768 — you'll lose context budget but five 4-image critics still fit comfortably.

### 3. `curl /v1/models` hangs

The model is still downloading. Check `docker logs <container>` and `du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct`. With `HF_HUB_ENABLE_HF_TRANSFER=1` expect ~64 GB in 3-5 minutes; without it 15+ minutes.

### 4. Critics return `PARSE_ERROR` with empty rationale

`json_object` mode is misbehaving on Qwen3-VL. Verify the entrypoint passed `--guided-decoding-backend xgrammar` (check `/var/log/crucible/vllm.log`). The Crucible client falls through json_schema → json_object → unconstrained automatically; if all three fail the issue is xgrammar isn't installed in the image. Re-pull / rebuild.

### 5. Frame decode silently produces 0 frames

The `aloha_static_cups_open` dataset uses AV1 video. Confirm `libdav1d` is installed in the container: `docker exec <container> dpkg -l | grep dav1d`. The Dockerfile installs it; if you swapped the base image, you may need to add it manually.

### 6. SSE stream from the Space hangs

The free-tier HF Space CPU has a 30-second request timeout for non-SSE responses. Confirm the Space code uses `requests.get(url, stream=True)` + `sseclient.SSEClient(resp)` (it does in `frontend/app.py`). If the droplet IP isn't reachable from the Space, you'll see this — set `CRUCIBLE_API_BASE` to a public IP, not a private subnet IP.

### 7. Push-to-Hub fails with `403`

The HF token in the Gradio form needs **write scope** (read scope creates the repo but rejects upload_folder). Generate a new token at https://huggingface.co/settings/tokens with "write" selected.

### 8. `precache_demo.py` runs out of disk

The HF cache plus 25 episodes' worth of frames at 65k context can hit ~80 GB on the droplet's local disk. Mount `/root/.cache/huggingface` from a larger volume (`-v /mnt/data/hf-cache:/root/.cache/huggingface`).

### 9. Live demo starts but stalls after 5 episodes

Concurrency cap. Lower `cfg.max_episodes_per_run` from 25 to 15 in the live UI; the precache covers 25 already. Or raise `--max-num-seqs` to 32 in the entrypoint (memory permitting).

---

## Time budget reference

| Task | Estimate |
|---|---|
| Local I/O smoke test | 2 min |
| GPU droplet provisioning | 5 min |
| Docker build | 8 min (first time, then cached) |
| Model download to HF cache | 4 min (with `hf_transfer`) |
| vLLM cold start | 90 s |
| `one_shot_test.py` | 30 s |
| `precache_demo.py` 25 episodes | 3 min |
| Space deploy + verify | 5 min |
| Demo recording | 10 min |
| Submission form | 10 min |
| **Total** | **~50 min** end-to-end |
