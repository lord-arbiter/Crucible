# Roadmap

Crucible's milestones, in priority order. Anything not listed here is out
of scope for the foreseeable future — prefer to file an issue for
discussion before sending a PR for it.

## v0.1.x — current line

Stabilize the OSS pivot. Bug-fix releases only after v0.1.0. No new
features.

- Live HF Space pointed at a stable hosted-API backend.
- README hero with screenshots from a real validation run.
- A handful of community recipes contributed via PR (any Qwen3-VL host
  not currently covered).

## v0.2 — discoverability + polish

Target: 4–6 weeks after v0.1.0.

- Publish to PyPI as `crucible-curation` (or `crucible-vlm` if the
  bare name is taken). `pip install crucible-curation` works.
- Three Jupyter notebooks under `examples/`:
  - `01_quickstart.ipynb` — score 3 episodes against any backend.
  - `02_full_pipeline.ipynb` — full curation walk: score 25 → histogram
    → threshold → push to Hub.
  - `03_custom_critic.ipynb` — add a 6th critic with prompt + schema.
- Multi-GPU recipe for Qwen3-VL-32B on `g6e.12xlarge` (4× L40S
  tensor-parallel).
- Local-Mac MLX recipe for Qwen3-VL-2B (laptop development path).
- Documented CLI: `crucible score`, `crucible filter`, `crucible push`
  as wrappers over the existing scripts.

## v0.3 — the agreement dataset (the moat)

Target: 2–3 months after v0.1.0.

- Co-score 200 episodes from `lerobot/aloha_mobile_cabinet`,
  `lerobot/aloha_static_cups_open`, and one bimanual contact-rich
  dataset with both Crucible and 3+ human expert raters.
- Publish the agreement dataset to HuggingFace Hub.
- Compute Cohen's κ per rubric axis. Target ≥ 0.7 for task_success and
  visual_quality; ≥ 0.5 for the harder axes (strategy, safety) given
  inherent rater disagreement.
- Publish a leaderboard / benchmarks page.

This is the artefact competitors can't quickly reproduce. It's also the
input we need for v1.0.

## v1.0 — production-grade fine-tuned judge

Target: 4–6 months after v0.1.0.

- Fine-tune Qwen2-VL-7B (or successor) on the agreement dataset to
  match expert raters at lower per-call cost.
- Single-binary deployment: one Docker image, one model, no
  hosted-API dependency. Runs on any AMD ROCm or NVIDIA CUDA box.
- Multi-model judge support: drop-in evaluation of Claude, GPT-4o,
  Gemini against our agreement dataset, so users can pick the
  cost/quality trade-off that fits.
- Production observability: per-critic latency histograms,
  per-axis κ tracking against a held-out expert set, drift alarms.

## v1.x — broader integration

Speculative; subject to community pull.

- Integration with LeRobot's training loops: filter episodes inline
  before each fine-tune run.
- Critic prompts as first-class artefacts: a registry where teams
  publish task-family-tuned prompts (pick-and-place, contact-rich,
  bimanual, mobile manipulation).
- Multi-modal support beyond Qwen3-VL: native handlers for Claude
  Sonnet 4.5, Gemini 3, GPT-5.

## Explicitly out of scope

- A web UI more elaborate than the current Gradio Space. If you want
  rich annotation tooling, that's a separate product.
- A custom file format. We read and write LeRobot v3 because that's
  what the ecosystem uses.
- A managed cloud service. Crucible is an OSS library you self-host.
  If hosted offerings emerge, they'll be third-party.
