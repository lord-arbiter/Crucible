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
short_description: VLM-judged data curation for robot demos (Qwen3-VL on AMD MI300X)
---

# Crucible — Space frontend

Paste a HuggingFace LeRobot dataset URL → Crucible runs five Qwen3-VL critics
on every episode (visual quality, kinematic quality, task success, strategy,
safety) and lets you filter + push the curated subset back to the Hub.

The Space is the UI. The actual VLM inference runs on an **AMD MI300X**
droplet behind the `CRUCIBLE_API_BASE` URL configured in the Space settings.
If the droplet is offline, the Space loads precached results from disk so the
demo still works end-to-end.

Built for the AMD Developer Hackathon 2026 (lablab.ai).
