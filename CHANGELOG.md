# Changelog

All notable changes to Crucible are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Model-agnostic story.** Crucible's positioning is no longer
  Qwen3-VL-first; the project is built around the OpenAI chat-completions
  API surface and works with any vision-language model exposed through
  it (Qwen3-VL, GPT-4o, Claude, Gemini, Llama-3.2-Vision, InternVL).
  README, CITATION.cff, pyproject.toml keywords, and config docstring
  updated to reflect this. The Qwen-specific `/no_think` token is now
  auto-applied only when the configured model id contains `qwen`.
- New recipes for non-Qwen backends: `docs/recipes/openai-gpt4o.md`,
  `docs/recipes/anthropic-claude.md` (via LiteLLM proxy),
  `docs/recipes/google-gemini.md`.
- Multi-backend deployment recipes under `docs/recipes/` (AWS EC2, AWS
  SageMaker, AMD MI300X, Hyperbolic, Together AI, DashScope, local Mac).
- `docker/Dockerfile.cuda` for NVIDIA GPUs (A10G, L40S, H100, A100).
- Examples directory with three Jupyter notebooks
  (quickstart, full pipeline, custom critic).
- PyPI publishing workflow at `.github/workflows/publish.yml`.
- Issue + PR templates at `.github/`.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `ROADMAP.md`, `SECURITY.md`.
- `CITATION.cff` for academic citations via GitHub's "Cite this
  repository" button.
- Backend-agnostic `CRUCIBLE_VLM_API_KEY` and `extra_body` plumbing in
  `src/critics.py` for providers that need provider-specific request
  parameters.
- Frontend now exposes VLM endpoint / model name / API key fields so the
  Space works against any user-supplied OpenAI-compatible endpoint.
- README sections: Star History (api.star-history.com badge),
  Contributors (contrib.rocks avatar grid), Citation (BibTeX block).

### Changed
- Repositioned from a one-off hackathon submission to an OSS project.
  README rewritten to lead with product value rather than event framing.
- `docs/RUNBOOK.md` renamed to `docs/DEPLOYMENT.md` and generalized to
  point at per-backend recipes.
- `docs/GPU_ACCESS.md` renamed to `docs/COMPUTE_OPTIONS.md` covering
  hosted APIs, AWS, GCP, Azure, AMD Cloud, and local options.
- `TESTING.md` moved to `docs/TESTING.md` (standard convention) and
  made backend-agnostic.
- `docs/architecture.md` generalized to "any OpenAI-compatible Qwen3-VL
  endpoint" rather than MI300X-specific.

### Changed
- Repositioned from a one-off hackathon submission to an OSS project.
  README rewritten to lead with product value rather than event framing.
- `docs/RUNBOOK.md` renamed to `docs/DEPLOYMENT.md` and generalized to
  point at per-backend recipes.
- `docs/GPU_ACCESS.md` renamed to `docs/COMPUTE_OPTIONS.md` covering
  hosted APIs, AWS, GCP, Azure, AMD Cloud, and local options.
- `TESTING.md` made backend-agnostic.
- `docs/architecture.md` generalized to "any OpenAI-compatible Qwen3-VL
  endpoint" rather than MI300X-specific.

### Removed
- Hackathon-specific docs: `docs/SUBMISSION.md`, `docs/build_in_public.md`,
  `docs/demo_script.md`, `docs/pitch.md`, `eval/manual_check.md`.
  Provenance preserved in git history.
- `docs/CODEBASE_MAP.md` — internal build-history walkthrough not useful
  to library users; the same information lives in `git log`.
- `assets/README.md` — placeholder doc removed; image files will land
  here directly when captured.

## [0.1.0] — Initial release

The initial public release of Crucible.

### Added

#### Core capabilities
- Five Qwen3-VL specialist critic agents scoring orthogonal axes:
  visual quality, kinematic quality, task success, strategy, safety.
- One aggregator agent fusing critic outputs into a final
  KEEP / POLISH / REJECT verdict with rationale and timestamp evidence.
- Weighted-mean scoring (`task_success` and `strategy` weighted 1.5×)
  with hard-fail rules on `REJECT / UNSAFE / FAILED` critic verdicts.
- Deterministic Python aggregator fallback when the LLM aggregator call
  fails or returns malformed output.

#### LeRobot v3 native
- Streaming dataset reader for the chunked v3 layout
  (`data/chunk-XXX/file-YYY.parquet`,
  `videos/{cam}/chunk-XXX/file-YYY.mp4`,
  `meta/episodes/chunk-*/file-*.parquet`,
  `meta/tasks.parquet`).
- v2 fallback path for legacy datasets.
- PyAV seek-to-`from_timestamp` frame decoding so episode-aligned frames
  come out of multi-episode chunk videos correctly.
- Telemetry digest builder (peak/mean joint velocities, idle periods,
  recovery moves, gripper transitions) used as primary input for the
  kinematic critic.

#### Robust JSON handling
- Three-tier output mode: `json_schema` with xgrammar guided decoding →
  `json_object` → unconstrained, with regex JSON salvage as a final
  fallback. Per-critic schemas pin score range and verdict vocabulary.
- `/no_think` suffix on user prompts to suppress Qwen3 thinking-mode
  preambles.

#### Threshold filtering + push-to-Hub
- `select_episodes` enforces both `score >= threshold` and
  `verdict != REJECT` (REJECT-with-high-score never sneaks through).
- `push_filtered_to_hub` mirrors source metadata (`info.json`,
  `tasks.parquet`, `episodes/*.parquet`), dedupes chunked shard copies,
  and writes a `crucible_curation` block in `info.json` with kept
  episode indices and a documented load instruction.

#### Service surface
- FastAPI orchestrator with POST `/score_dataset`, SSE
  `/progress/{job_id}`, GET `/results/{job_id}`, POST `/push_filtered`,
  GET `/healthz`.
- Gradio frontend deployable to HuggingFace Spaces with score histogram,
  per-episode detail cards, embedded video, threshold slider, push
  button, and JSON export.

#### Deployment
- `docker/Dockerfile.gpu` for AMD MI300X with ROCm + vLLM, env vars set
  for the AMD performance path (`HSA_FORCE_FINE_GRAIN_PCIE`,
  `VLLM_USE_V1`, `VLLM_ROCM_USE_AITER`).
- Production vLLM serve flags including
  `--guided-decoding-backend xgrammar`,
  `--limit-mm-per-prompt.image 16`,
  `--dtype bfloat16`, `--max-model-len 65536`.

#### Quality
- 40 unit tests covering aggregator math, JSON salvage, telemetry
  digest, episode selection, and import smoke.
- I/O smoke script `scripts/io_smoke.py` proven against
  `lerobot/aloha_static_cups_open` (no GPU required).
- GitHub Actions CI running pytest + ruff on every push.

### Provenance
- Originally built in October 2025 — May 2026 as a submission to the
  AMD Developer Hackathon 2026 on lablab.ai. Spun off as a serious OSS
  project after the hackathon's value as a one-off was eclipsed by the
  multi-axis behavioral rubric's broader applicability to any robotics
  team curating teleoperation data.
- Initial build history and 27-phase commit-by-commit walkthrough is
  preserved at `docs/CODEBASE_MAP.md`.
