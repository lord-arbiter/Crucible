# Examples

Three Jupyter notebooks demonstrating Crucible end-to-end. They work
against any OpenAI-compatible Qwen3-VL endpoint — set the three
`CRUCIBLE_VLM_*` env vars before launching.

| Notebook | What it shows | Time |
|---|---|---|
| [01_quickstart.ipynb](01_quickstart.ipynb) | Score 3 episodes from a tiny dataset; inspect per-critic verdicts | ~2 min |
| [02_full_pipeline.ipynb](02_full_pipeline.ipynb) | Score 25 episodes, render a verdict histogram, drag a threshold, push the curated subset to a personal HF dataset repo | ~10 min |
| [03_custom_critic.ipynb](03_custom_critic.ipynb) | Add a 6th critic ("Reachability") and score the same dataset against the extended rubric | ~5 min |

## Running

```bash
# From the repo root, with the dev extras installed
pip install -e ".[notebooks]"

# Set your VLM backend (any OpenAI-compatible Qwen3-VL endpoint)
export CRUCIBLE_VLM_ENDPOINT=https://api.your-provider.com/v1
export CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
export CRUCIBLE_VLM_API_KEY=sk-your-key

jupyter lab examples/
```

Each notebook starts with an environment check cell that verifies your
endpoint is reachable before running the longer cells.

## Cost per notebook

- 01_quickstart: ~$0.02 against a hosted Qwen3-VL API
- 02_full_pipeline: ~$0.30
- 03_custom_critic: ~$0.05

If you're running against self-hosted vLLM, GPU-hour cost dominates
($1.86/hr on AWS g6e.xlarge, $0.50/hr on RunPod community MI300X).

## Modifying the notebooks

PRs welcome. If you add a new notebook, please:

1. Number it sequentially (`04_foo.ipynb`).
2. Open with an env-check cell.
3. Make sure it runs end-to-end against any backend (don't hard-code a
   provider's quirks).
4. Update this README.
