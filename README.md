# Crucible

**A multi-axis VLM-as-judge data curation studio for robot demonstrations.**

[![Tests](https://github.com/lord-arbiter/Crucible/actions/workflows/test.yml/badge.svg)](https://github.com/lord-arbiter/Crucible/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![VLM-agnostic](https://img.shields.io/badge/VLM-agnostic-9cf)](#supported-models)
[![LeRobot v3](https://img.shields.io/badge/LeRobot-v3.0-orange)](https://huggingface.co/lerobot)

[Live Demo](https://huggingface.co/spaces/) · [Docs](docs/) · [Recipes](docs/recipes/) · [Test plan](docs/TESTING.md) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md)

---

Crucible reads a HuggingFace LeRobot dataset and runs five specialist VLM critic agents on every episode — visual quality, kinematic quality, task success, **strategy**, **safety** — then fuses their outputs into a single `KEEP / POLISH / REJECT` verdict with rationale and timestamp evidence. Drag a threshold and push the curated subset back to the Hub in one click.

The critics are model-agnostic. Drop in **any API key from any provider** — Crucible supports the OpenAI chat-completions API surface natively for self-hosted vLLM / Hyperbolic / Together AI / DashScope / OpenAI / Gemini's OpenAI-compat endpoint, and ships a [LiteLLM](https://github.com/BerriAI/litellm)-powered universal gateway as an opt-in extra (`pip install 'crucible-curation[universal]'`) for everything else: Anthropic native, AWS Bedrock, Google Vertex AI, Cohere, Replicate, Groq, Mistral, OpenRouter, xAI, and ~100 more. See [docs/recipes/any-provider-litellm.md](docs/recipes/any-provider-litellm.md).

## Why this exists

Robotics teams spend ~$118 / hour on teleoperation data, and 20–40 % of it silently teaches policies bad habits. Heuristic curation (jerk, path-efficiency, actuator saturation) catches motion artefacts. A VLM that only judges binary task success (à la `score_lerobot_episodes` calling Gemini) catches outright failures. **Neither catches the silent killers**: operator hesitation, multi-attempt regrasps, inefficient strategies, near-miss safety incidents.

A bad-strategy episode that *succeeds* is worse than a good-strategy episode that fails — the former teaches the policy unreliable recovery as if it were normal behavior.

Crucible's contribution is the **multi-axis behavioral rubric** with structured per-frame timestamp evidence — a curation contract that's independent of which VLM you plug in. Drop in the cheapest model that hits the kappa target on your task family; swap when a better one ships.

## How it compares

| Tool | Approach | Output | What it misses |
|---|---|---|---|
| `score_lerobot_episodes` (Oct 2025) | 7 CV heuristics + 1 binary Gemini call | scalar score per axis | Strategy, hesitation, behavioral nuance |
| AgiBot Genie Centurion / Task Sentinel (May 2025) | fine-tuned MiniGPT-4 success classifier in teleop loop | online intervention during collection | Offline curation of completed datasets |
| LeRobot dataset visualizer | low-movement / jerk / outlier-length filters | filter UI suggestions | Anything semantic |
| Stanford Quality-over-Quantity | influence-function math on policy gradients | continuous demo importance | Requires a trained policy; per-axis explainability |
| **Crucible** | **5-axis behavioral rubric, 1 aggregator, any VLM behind an OpenAI-compatible endpoint, structured rationale + timestamp evidence** | **per-episode JSON, KEEP / POLISH / REJECT** | **n/a — this is the gap** |

## Quickstart (5 minutes, no GPU required)

The fastest way to try Crucible is against any hosted OpenAI-compatible vision endpoint. Pick the one you have an account with — see [docs/recipes/](docs/recipes/) for setup notes per provider.

```bash
# 1. Clone & install
git clone https://github.com/lord-arbiter/Crucible
cd Crucible
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Point at any provider you have a key for
# Pick ONE block below — Crucible auto-detects which transport to use.

# (a) OpenAI direct (no endpoint URL needed — known model)
export CRUCIBLE_VLM_MODEL=gpt-4o-mini
export CRUCIBLE_VLM_API_KEY=sk-...

# (b) LiteLLM universal gateway (Anthropic, Bedrock, Vertex, ...)
# Requires `pip install 'crucible-curation[universal]'`. No endpoint URL.
# export CRUCIBLE_VLM_MODEL=anthropic/claude-sonnet-4-5
# export CRUCIBLE_VLM_API_KEY=sk-ant-...

# (c) OpenAI-compatible endpoint (Hyperbolic, Together, DashScope, self-hosted vLLM)
# export CRUCIBLE_VLM_ENDPOINT=https://api.hyperbolic.xyz/v1
# export CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-72B-Instruct
# export CRUCIBLE_VLM_API_KEY=<your-key>

# 3. Score one episode end-to-end
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual

# 4. Score a small dataset and inspect results
python scripts/precache_demo.py \
    --repo lerobot/aloha_static_cups_open \
    --episodes 5
```

You can also run Crucible **without any VLM** to validate the I/O layer:

```bash
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2
# PASS — 2 episodes streamed and decoded cleanly
```

## Supported models

Any vision-language model reachable through a chat-completions API works. Crucible has three transports and picks the right one automatically:

| Transport | When it's used | Setup |
|---|---|---|
| **OpenAI-compat** | `CRUCIBLE_VLM_ENDPOINT` is set | endpoint URL + model id + API key |
| **OpenAI direct** | `CRUCIBLE_VLM_MODEL` is `gpt-*` / `o1` / `o3` / `o4` and no endpoint set | model id + `OPENAI_API_KEY` |
| **LiteLLM universal** | model id has a provider prefix like `anthropic/`, `bedrock/`, `vertex_ai/` | `pip install 'crucible-curation[universal]'` + provider creds |

| Model family | Provider | Best transport | Notes |
|---|---|---|---|
| **Qwen3-VL** (2B / 8B / 32B / 72B / 235B) | self-hosted vLLM, Hyperbolic, Together AI, DashScope | OpenAI-compat | Auto-appends `/no_think` to suppress thinking-mode preamble |
| **GPT-4o / GPT-4o-mini / o1** | OpenAI | OpenAI direct | Best JSON-mode reliability; recommended starting point |
| **Claude Sonnet 4.5 / Opus 4.5** | Anthropic | LiteLLM (`anthropic/...`) | Strongest reasoning on strategy + safety axes |
| **Gemini 2.5 Pro / Flash** | Google AI Studio | OpenAI-compat OR LiteLLM (`gemini/...`) | Cheapest hosted multimodal |
| **Claude / Llama / Nova on Bedrock** | AWS Bedrock | LiteLLM (`bedrock/...`) | Uses standard AWS credential chain |
| **Gemini / PaLM on Vertex** | Google Vertex AI | LiteLLM (`vertex_ai/...`) | Uses gcloud ADC |
| **Llama-3.2-Vision** (11B / 90B) | self-hosted vLLM, Groq, Fireworks | OpenAI-compat or LiteLLM | Open-weight alternative |
| **InternVL** (2B / 8B / 26B / 78B) | self-hosted vLLM | OpenAI-compat | Strong open-weight |
| **Cohere / Mistral / Replicate / Groq / xAI** | their native APIs | LiteLLM (`<provider>/...`) | One env var change |

Crucible's three-tier output mode (`json_schema` → `json_object` → unconstrained, with regex JSON salvage) covers the variation in structured-output support across these models. You don't need to pick a model that supports `json_schema` specifically.

Want a provider not listed? Open an [issue](https://github.com/lord-arbiter/Crucible/issues/new?template=feature_request.md) — LiteLLM probably already supports it; usually a one-line addition to the prefix list.

## Self-host on a GPU box

Crucible ships Docker images for both NVIDIA (CUDA) and AMD (ROCm) GPUs. The same `docker run` works on any cloud — concrete recipes for the most common providers:

| Provider | Recipe |
|---|---|
| AWS EC2 | [aws-ec2.md](docs/recipes/aws-ec2.md) |
| AWS SageMaker JumpStart | [aws-sagemaker.md](docs/recipes/aws-sagemaker.md) |
| Azure VM | [azure-vm.md](docs/recipes/azure-vm.md) |
| Google Cloud Compute Engine | [gcp-compute-engine.md](docs/recipes/gcp-compute-engine.md) |
| DigitalOcean | [digitalocean-gpu.md](docs/recipes/digitalocean-gpu.md) |
| Lambda Labs | [lambda-labs.md](docs/recipes/lambda-labs.md) |
| RunPod (NVIDIA) | [runpod-nvidia.md](docs/recipes/runpod-nvidia.md) |
| Vast.ai | [vast-ai.md](docs/recipes/vast-ai.md) |
| AMD MI300X (Hot Aisle / RunPod / AMD Cloud) | [amd-mi300x.md](docs/recipes/amd-mi300x.md) |
| Local Mac (MLX) | [local-mac.md](docs/recipes/local-mac.md) |
| **Other cloud / bare metal NVIDIA** | [Universal cloud-gpu-vm recipe](docs/recipes/cloud-gpu-vm.md) |

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full deployment guide and a troubleshooting table.

## How it works

```
LeRobot v3 dataset (chunked parquet + AV1 video)
       │
       ▼  src/lerobot_io.py
  EpisodeBundle (frames + telemetry digest + task description)
       │
       ▼  src/critics.py   → any OpenAI-compatible vision-language endpoint
  5 critic verdicts (visual / kinematic / task / strategy / safety)
       │
       ▼  src/aggregator.py
  Final verdict (KEEP / POLISH / REJECT + summary + top_concern)
       │
       ▼  src/pipeline.py   → cache to data/precached/<repo>.json
  List[record]
       │
       ├──▶  src/filtering.py   → curated subset on HuggingFace Hub
       │
       └──▶  src/api.py         → SSE stream to the Gradio frontend
```

Reliability features:

- **Three-tier JSON output** — `json_schema` (xgrammar) → `json_object` → unconstrained, with regex JSON salvage as a final fallback. Every critic returns a structured dict even on a misbehaving model.
- **Deterministic Python aggregator fallback** — if the LLM aggregator call fails or returns malformed output, a pure-Python implementation of the verdict rules guarantees a verdict.
- **LeRobot v3 native** — reads chunked parquet/video shards via the `meta/episodes/*.parquet` pointer columns and seeks PyAV to per-episode timestamps. v2 layouts also supported.
- **Curation manifest** — pushed datasets carry a `crucible_curation` block in `info.json` listing kept episode indices and a documented `LeRobotDataset(repo, episodes=[...])` load instruction.

Architecture deep-dive: [docs/architecture.md](docs/architecture.md).

## Configuration

All knobs live in `src/config.py` and are overridable via `CRUCIBLE_*` environment variables:

```bash
CRUCIBLE_VLM_ENDPOINT=http://localhost:8001/v1   # any OpenAI-compatible
CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
CRUCIBLE_VLM_API_KEY=EMPTY                       # or your provider key
CRUCIBLE_FRAMES_PER_EPISODE=16                   # frames per critic
CRUCIBLE_IMAGE_MAX_DIM=768                       # client-side resize
CRUCIBLE_KEEP_THRESHOLD=7.5
CRUCIBLE_POLISH_THRESHOLD=5.0
```

See [`.env.example`](.env.example) for the full list with provider-specific examples.

## Limitations

- No general-purpose VLM (Qwen3-VL, GPT-4o, Claude, Gemini, Llama-Vision) is specifically trained on robotics teleoperation video, so any of them can miss embodiment-specific failure modes a domain expert would catch. We mitigate by feeding the kinematic critic a pre-computed telemetry digest (joint velocities, idle periods, recovery moves, gripper events) where VLMs are weakest.
- Critic outputs are stochastic at temperature 0.2. For production deployment, run each episode 3× and aggregate, or fine-tune on human-labeled episode quality data (planned for v0.3).
- AV1 video decoding requires PyAV with libdav1d. Most modern wheels include it.
- The push-to-Hub flow preserves the source LeRobot v3 chunked layout and adds a `crucible_curation` block to `info.json`. It does not regenerate stats or re-shard videos — load via `LeRobotDataset(repo, episodes=kept_indices)` to consume only the curated subset.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full milestone breakdown. The short version:

- **v0.1** (current): multi-axis rubric, 5 critics + aggregator, LeRobot v3 reader, multi-backend deployment recipes.
- **v0.2**: PyPI package, examples notebooks, multi-GPU recipe for Qwen3-VL-32B, polished HF Space.
- **v0.3**: human-VLM agreement dataset (the real moat), Cohen's κ benchmark per rubric axis, fine-tuned judge model.
- **v1.0**: production-grade fine-tuned judge model on AMD ROCm; multi-model support (Claude / GPT-4o / Gemini drop-in).

## Contributing

PRs welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup, design principles, and PR workflow. New critics, new dataset format readers, new deployment recipes are all in scope.

For security issues see [SECURITY.md](SECURITY.md).

## Acknowledgements

- The five critic prompts and the strategy / safety rubric are original to Crucible.
- The LeRobot v3 path templates and chunked-layout I/O patterns were verified against the official `huggingface/lerobot` library.
- The OpenAI chat-completions API surface is the lingua franca that makes the multi-VLM story practical — we'd rewrite this to a different abstraction the day a better one wins.
- AMD/LMSYS published the MI300X-specific vLLM optimizations (rocJPEG, AITER + prefill-decode attention) we use in the AMD recipe.
- `score_lerobot_episodes` (RoboticsData, Oct 2025) and AgiBot Genie Centurion are the closest prior art and the baselines we explicitly out-position with the multi-axis behavioral rubric.

## Citation

If you use Crucible in research or production, please cite it:

```bibtex
@software{crucible_2026,
  author  = {Chakradhari},
  title   = {Crucible: Multi-axis VLM-judged data curation for robot demonstrations},
  year    = {2026},
  url     = {https://github.com/lord-arbiter/Crucible},
  version = {0.1.0}
}
```

GitHub also reads [CITATION.cff](CITATION.cff) — click "Cite this repository" on the repo sidebar for an auto-generated APA / BibTeX block.

## Contributors

[![Contributors](https://contrib.rocks/image?repo=lord-arbiter/Crucible)](https://github.com/lord-arbiter/Crucible/graphs/contributors)

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup, design principles, and PR workflow.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lord-arbiter/Crucible&type=Date)](https://star-history.com/#lord-arbiter/Crucible&Date)

## License

MIT. See [LICENSE](LICENSE).
