---
name: Bug report
about: Crucible behaved in a way you didn't expect
title: "[bug] "
labels: bug
assignees: ''
---

## What happened

A clear, concise description of the bug.

## What you expected

A clear, concise description of the expected behavior.

## Repro

```bash
# the exact commands you ran, including env vars
export CRUCIBLE_VLM_ENDPOINT=...
python scripts/...
```

## Output / traceback

```
paste the full traceback or relevant log lines here
```

## Environment

- Crucible version (or commit hash):
- Python version: `python --version`
- OS:
- Backend (which VLM endpoint): self-hosted vLLM / Hyperbolic / Together AI / DashScope / SageMaker / other
- Model id: e.g. `Qwen/Qwen3-VL-8B-Instruct`
- Dataset (if applicable): e.g. `lerobot/aloha_static_cups_open`

## Anything else

Any context that might help — what changed since it last worked, related
recent changes, screenshots if it's a frontend bug.
