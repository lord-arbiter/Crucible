# Deploying Crucible

Crucible is an OSS Python project that runs against any OpenAI-compatible
Qwen3-VL endpoint. There's no single "right" way to deploy it — pick the
recipe that matches your infrastructure and budget.

## Pick a path

### Hosted multimodal APIs

| You want | Pick |
|---|---|
| Most reliable JSON output, easy setup | [OpenAI GPT-4o](recipes/openai-gpt4o.md) |
| Cheapest hosted, largest context | [Google Gemini](recipes/google-gemini.md) |
| Strongest strategy / safety rationale | [Anthropic Claude (via LiteLLM)](recipes/anthropic-claude.md) |
| Cheapest open-weight hosted | [Hyperbolic Qwen3-VL](recipes/hyperbolic.md) |
| Other hosted Qwen3-VL providers | [Together AI](recipes/together-ai.md) / [DashScope](recipes/dashscope.md) |
| Generic guide for any OpenAI-compat host | [Hosted API quickstart](recipes/hosted-api-quickstart.md) |

### Self-hosted on cloud GPU

| You want | Pick |
|---|---|
| **Universal NVIDIA cloud-VM recipe** | [Any cloud GPU VM](recipes/cloud-gpu-vm.md) |
| AWS EC2, full control | [AWS EC2 g6e.xlarge](recipes/aws-ec2.md) |
| AWS, managed | [AWS SageMaker JumpStart](recipes/aws-sagemaker.md) |
| Azure | [Azure VM (NCads_H100 / A100)](recipes/azure-vm.md) |
| Google Cloud | [GCP Compute Engine (G2 / A3)](recipes/gcp-compute-engine.md) |
| DigitalOcean (zero ceremony) | [DO H100 GPU droplet](recipes/digitalocean-gpu.md) |
| Cheapest US H100 / A100 | [Lambda Labs On-Demand](recipes/lambda-labs.md) |
| RunPod NVIDIA | [RunPod (A100 / H100 / L40S)](recipes/runpod-nvidia.md) |
| Marketplace, lowest cost | [Vast.ai](recipes/vast-ai.md) |
| Maximum performance, AMD hardware | [AMD MI300X](recipes/amd-mi300x.md) |
| Develop on a Mac without cloud | [Local Mac MLX](recipes/local-mac.md) |
| Other backend not listed here | Adapt the [universal recipe](recipes/cloud-gpu-vm.md); PRs welcome |

The [hosted-API quickstart](recipes/hosted-api-quickstart.md) is the
recommended starting point if you've never run Crucible before. It
takes 5 minutes and validates the full pipeline end-to-end.

## What every deployment needs

Three environment variables, regardless of backend:

```bash
CRUCIBLE_VLM_ENDPOINT   # https://api.provider.com/v1 — OpenAI-compatible
CRUCIBLE_VLM_MODEL      # the model id the provider expects
CRUCIBLE_VLM_API_KEY    # provider key, or "EMPTY" for self-hosted
```

That's it. Crucible's code is backend-agnostic — `src/critics.py`
sends standard `chat/completions` requests with multimodal content
and a JSON schema response format. Provider-specific quirks
(rate limits, model id naming, JSON-mode support) are documented in
each recipe.

For pre-deployment validation that doesn't touch a VLM, run:

```bash
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2
```

This proves the LeRobot v3 reader works on real Hub data. It's fast,
free, and catches the most fragile layer of the system before you sink
GPU time on a misconfigured endpoint.

## Validation walk

After your endpoint is up, follow [TESTING.md](../TESTING.md) layer by
layer. The full walk is ~25 minutes and covers everything from
`rocm-smi` / `nvidia-smi` health checks to the live demo dry-run.

## Operational concerns

### Cost control

- **Stop GPU instances when idle.** EC2 / RunPod / Hot Aisle bill while
  running, regardless of whether you're sending requests. Set a daily
  reminder.
- **Use spot instances** where the workload tolerates interruption
  (offline batch curation jobs).
- **Hosted APIs** are cheaper than self-hosting if you're scoring
  fewer than ~100 episodes.

### Reliability

- The Gradio Space falls back to `data/precached/*.json` when the
  compute endpoint is unreachable. Pre-cache your demo dataset before
  going live.
- Crucible's three-tier JSON output handles providers with limited
  `response_format=json_schema` support — you don't need to pick
  json-mode-supporting providers exclusively.
- The deterministic Python aggregator fallback (`fallback_aggregate`
  in `src/aggregator.py`) guarantees a verdict even when the LLM
  aggregator call fails.

### Security

- Don't commit API keys. Use Space secrets, AWS Secrets Manager, or
  `.env` (gitignored). See [SECURITY.md](../SECURITY.md) for the
  audit guide.
- Open ports 8000 / 8001 only to trusted IPs in your security group.
- The push-to-Hub flow accepts a write-scoped HF token via the UI;
  it's used in-process and not persisted.

### Observability

- vLLM logs to `/var/log/crucible/vllm.log` inside the GPU container.
- The FastAPI orchestrator logs to stdout — capture via
  `docker logs <container>`.
- Per-job state is kept in memory by `src/api.py` — restart loses
  jobs. Acceptable for hackathon scope; for production, swap in
  Redis-backed job state.

## Common failure modes

(Same table as [TESTING.md](../TESTING.md), repeated here for
convenience.)

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker build` fails: `manifest unknown` | Base image tag rotated | Use `--build-arg BASE=...` to specify a current tag (see recipes) |
| vLLM `OutOfMemoryError` during profiling | KV cache too big | Lower `VLLM_MAX_LEN`; or `VLLM_GPU_UTIL` |
| `curl /v1/models` hangs | Weights still downloading | Check `du -sh ~/.cache/huggingface/hub/...` |
| Critics return `PARSE_ERROR` | `--guided-decoding-backend xgrammar` not set, or provider doesn't support json_schema | Confirm flag is set; or rely on Crucible's fallback chain |
| Frame decode produces 0 frames | Missing libdav1d (AV1 codec) | Install `libdav1d-dev` in the container |
| SSE stream hangs | Free-tier Space has 30 s request timeout for non-SSE | Confirm streaming code in `frontend/app.py`; check `CRUCIBLE_API_BASE` is a public IP |
| Push-to-Hub fails with `403` | HF token is read-only | Regenerate with **write** scope |
| `precache_demo.py` runs out of disk | HF cache + 25 episodes' frames at 65k context can hit ~80 GB | Mount cache from larger volume |
| Live demo stalls after 5 episodes | Concurrency cap | Lower `cfg.max_episodes_per_run`; or raise `--max-num-seqs` |
