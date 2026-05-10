## What this PR does

A clear, concise description of the change.

## Linked issue

Fixes #___ / Closes #___ / Refs #___

## How I tested it

- [ ] `python -m pytest tests -q` — green
- [ ] `ruff check src tests frontend scripts` — clean
- [ ] `python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2` — passes
  (only required if you touched `src/lerobot_io.py`)
- [ ] Manual end-to-end check against a real VLM endpoint
  (only if your change affects the critic or aggregator path; describe what you ran below)

```
# manual test commands and output (if any)
```

## Checklist

- [ ] CHANGELOG.md updated under "Unreleased"
- [ ] If you added a new public API, docs updated
- [ ] If you added a new dependency, justified in the PR description
- [ ] No emojis in code; only sparingly in prose if at all
- [ ] No docstrings on every helper — only where the WHY is non-obvious

## Anything reviewers should focus on

Highlight non-obvious decisions, performance trade-offs, or places where
you're unsure of the right approach.
