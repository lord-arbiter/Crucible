# Contributing to Crucible

Thanks for your interest in Crucible. PRs that fix bugs, add deployment recipes, add new dataset format readers, or add new critics are all in scope. This doc tells you the dev workflow, the design principles to respect, and the things we'd rather you not change.

## Quick dev setup

```bash
git clone https://github.com/lord-arbiter/Crucible
cd Crucible
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # installs runtime + dev deps (pytest, ruff)

# Sanity check
python -m pytest tests -q     # 40/40 should pass
ruff check src tests frontend scripts
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2
```

You don't need a GPU to develop most of Crucible — only the live VLM critic
calls require one. For everything else (I/O, telemetry, filtering, push-to-Hub,
Gradio UI), a laptop is enough.

## Design principles

Read these before sending a PR. Crucible is opinionated about a few things:

1. **Multi-backend by default.** Any OpenAI-compatible Qwen3-VL endpoint
   should work. Don't add code paths that assume a specific provider's API
   surface, model id, or auth scheme. If a provider needs special handling,
   route it through `extra_body` rather than hard-coding.
2. **Deterministic fallback everywhere.** When the LLM misbehaves, a
   structured-data path takes over: `_extract_json_loose` for malformed
   critic JSON, `fallback_aggregate` for malformed aggregator output. Don't
   remove these even if the model "should" never produce broken output.
3. **No runtime dependency on `lerobot`.** Crucible reads the LeRobot
   format directly via `huggingface_hub` + `pyarrow` + `pyav`. This keeps
   the install lightweight and prevents API-drift regressions when LeRobot
   refactors.
4. **Production-ready, not toy.** Real type hints, no docstrings on every
   helper (only where the WHY is non-obvious), defensive at boundaries, no
   over-abstraction. If you find yourself adding a class hierarchy for one
   subclass, it's wrong.
5. **No emojis in code.** README and docs can have them sparingly. Code,
   commits, log lines: no.

## Tests

We have 40 unit tests covering the deterministic logic (aggregator math,
JSON salvage, telemetry digest, episode selection). They run without a GPU
or network:

```bash
python -m pytest tests -q
```

If you add a new feature with deterministic behavior (a new critic verdict
rule, a new filter mode, a new layout-detection branch), add a test for it.
If your change is integration-only and can't be tested without a real VLM
endpoint, document the manual test in the PR description.

We also have an I/O smoke test that hits the real HuggingFace Hub:

```bash
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2
```

Run this before sending a PR that touches `src/lerobot_io.py`.

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting:

```bash
ruff check src tests frontend scripts
ruff format src tests frontend scripts   # if you want auto-format
```

The CI workflow (`.github/workflows/test.yml`) runs both pytest and ruff
on every push and PR. Both must be green to merge.

## Adding a new critic

The five critics live in `src/prompts/critic_*.txt` plus a row in
`CRITIC_VERDICT_VOCAB` and `CRITIC_NAMES` in `src/critics.py`. To add a
sixth (e.g. "Reachability" for novel arm geometries):

1. Write the prompt as `src/prompts/critic_reachability.txt`. Match the
   structure of the existing prompts (task description, frames + telemetry,
   strict JSON schema with a verdict vocabulary).
2. Add `"reachability"` to `CRITIC_NAMES` in `src/critics.py`.
3. Add `"reachability": ["GOOD", "ACCEPTABLE", "MARGINAL", "POOR"]` (or
   whatever vocab fits) to `CRITIC_VERDICT_VOCAB`.
4. Update the storage-key list at the bottom of `run_all_critics` if you
   want it visible in pipeline output.
5. Add a row to `CRITIC_WEIGHTS` in `src/aggregator.py`.
6. Add tests in `tests/test_aggregator.py` covering the new critic in the
   verdict-rule scenarios.

A working example is `examples/03_custom_critic.ipynb`.

## Adding a deployment recipe

If you've successfully run Crucible on a backend not in `docs/recipes/`,
please send a PR. Recipe template:

```markdown
# Recipe: <provider>

## When to use
1-2 paragraphs. Best for ___, less good for ___.

## Cost
Concrete pricing as of <date>.

## Setup
1. Sign up / provision steps.
2. Get an API key / compute box.
3. Set environment variables (full block).
4. Run `scripts/one_shot_test.py` to verify.

## Known quirks
Any provider-specific things (rate limits, model name aliases, JSON mode
support, image format constraints).

## Verification commands
3-5 lines that prove the recipe works end-to-end.
```

## Pull request workflow

1. Fork → branch → make your change.
2. `python -m pytest tests -q` and `ruff check ...` both green.
3. Update `CHANGELOG.md` under "Unreleased" with a one-line entry.
4. Open a PR using the template. Link to the issue if one exists.
5. CI will run; address any failures.
6. Maintainer reviews and merges (or asks for changes).

For larger changes (architectural refactors, new dataset format readers,
breaking API changes), open an issue first to discuss the design before
writing code.

## Security disclosures

See [SECURITY.md](SECURITY.md). Don't open public issues for security
problems.

## Code of conduct

We follow the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). Be kind.

## Ground truth

If anything in this doc disagrees with the actual code, the code wins.
Send a PR to fix the doc.
