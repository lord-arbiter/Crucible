# Build-in-Public thread template

For the **Build in Public bonus** track. Post on X (Twitter) over the build
window. Tag `@AMDDevCloud` `@lablabai` `@huggingface` and the
`#AMDDevHackathon` `#Qwen3VL` hashtags.

Adapt freely — these are starting points, not scripts.

---

## Tweet 1 — kickoff

> Day 0 of the AMD Developer Hackathon 2026. Solo build.
>
> I'm building **Crucible**: a VLM-judged data curation studio for robot demonstrations. Five Qwen3-VL critics on AMD MI300X scoring every episode of a HuggingFace LeRobot dataset.
>
> The thesis: 20–40 % of teleop data silently teaches policies bad habits. Existing tools score jitter and blur. They miss strategy.
>
> #AMDDevHackathon #Qwen3VL

---

## Tweet 2 — architecture sketch

Attach the architecture diagram (or screenshot of `docs/architecture.md`).

> Architecture for Crucible.
>
> Gradio on HF Spaces (free CPU) ──► FastAPI orchestrator on MI300X ──► vLLM serving Qwen3-VL-32B.
>
> Five specialist critics + one aggregator share the same model. Single-GPU, single-model, parallel via asyncio.
>
> The MI300X 192 GB HBM3 is the load-bearing piece — fits 32B at fp16 with 65k context. Wouldn't fit on H100.

---

## Tweet 3 — first end-to-end run

Attach a screenshot of the histogram + episode detail view.

> First end-to-end Crucible run on `lerobot/aloha_mobile_cabinet`.
>
> 25 episodes scored in <X> minutes on MI300X. Five critics per episode (visual / kinematic / task / strategy / safety) → weighted aggregator → KEEP / POLISH / REJECT.
>
> Watching the histogram fill in live is satisfying.

---

## Tweet 4 — the moment a critique lands

Pick the most damning episode in the precache. Quote the strategy critic's verdict.

> Crucible just flagged a "passed" episode as REJECT.
>
> Strategy critic: *"Operator hesitated 3.1s after initial reach, attempted regrasp suggesting object slipped. Policy fine-tuned on this episode would learn unreliable grasp recovery rather than confident first-attempt grasping."*
>
> Score 4.6. Existing heuristic tools score this episode "fine."

---

## Tweet 5 — push to Hub

Show the auto-generated dataset card.

> One click: **Push filtered dataset to Hub**.
>
> Crucible writes a curated subset back as a new HuggingFace dataset, with a generated dataset card listing exactly which episodes were dropped and why.
>
> Dollar value: $118/hr teleop × bad-episodes-removed = real saved spend.

---

## Tweet 6 — submission

> Crucible is submitted to @lablabai for the AMD Developer Hackathon 2026.
>
> Live: <Space URL>
> Code: <GitHub URL>
> Demo: <video URL>
>
> Six Qwen3-VL agents on a single MI300X turning frontier multimodal models from interesting research into shipping infrastructure for physical AI.
>
> #AMDDevHackathon

---

## Notes

- Don't pre-write all six and post on a timer. The judges follow build-in-public threads partly to see real, human-paced progress.
- Responses ("how did you decide on Qwen3-VL?") are worth more than the original tweets. Stay engaged.
- Pin tweet 6 (the submission) once posted.
