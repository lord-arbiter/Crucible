---
title: Crucible
emoji: 🛠
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Multi-axis VLM-judged data curation for robot demonstrations
---

# Crucible — Space frontend

Paste a HuggingFace LeRobot dataset URL → Crucible runs five Qwen3-VL critics
on every episode (visual quality, kinematic quality, task success, strategy,
safety) and lets you filter + push the curated subset back to the Hub.

The Space is the UI. VLM inference runs on whatever endpoint you point
`CRUCIBLE_API_BASE` at — a self-hosted vLLM box, a hosted Qwen3-VL API, or
the user-supplied endpoint exposed as fields in the UI itself. If the
backend is unreachable, the Space loads precached results from disk so the
demo still works end-to-end.

GitHub: <https://github.com/lord-arbiter/Crucible>
