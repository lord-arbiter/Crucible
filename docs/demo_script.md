# Demo video script (2 minutes)

## [0:00 – 0:15] The hook

Title card "Crucible" over a still of a robot teleop episode that's clearly going wrong (operator fumbling, regrasp visible).

> **Voiceover:** "Robotics teams spent $118 an hour to collect this data. Half of it is silently teaching the wrong lessons. Crucible catches that, automatically."

## [0:15 – 0:35] The product, fast

Cut to the Gradio UI. Paste `lerobot/aloha_mobile_cabinet`. Hit **Score dataset**. The score-distribution histogram fills in as episodes complete in real time.

> **Voiceover:** "Five Qwen3-VL critics on a single AMD MI300X — visual quality, kinematic quality, task success, strategy, safety. They score every episode in parallel."

## [0:35 – 1:15] The killer detail

Click on the most damning episode from the precache (pick beforehand — typically a multi-regrasp episode with hesitation). Show:

- **Embedded video clip** — point out the operator hesitation around 4.2s.
- **Strategy critic verdict** "MEDIOCRE" with rationale: *"Operator hesitated for 3.1s after initial reach, then attempted regrasp suggesting object slipped. Policy fine-tuned on this episode would learn unreliable grasp recovery rather than confident first-attempt grasping."*
- **Final score 4.6 — REJECT.**

> **Voiceover:** "Existing curation tools score this episode 'fine' on jitter and lighting. They miss the strategy. Crucible doesn't."

## [1:15 – 1:45] The output

Drag the threshold slider to **7.0**. Show the filtered count: "65 of 85 episodes kept." Click **Push filtered dataset**. Cut to the new HuggingFace dataset page, dataset card auto-generated, listing what was filtered and why.

> **Voiceover:** "One click, curated dataset back to the Hub, with a generated card so anyone training on it knows exactly what was kept."

## [1:45 – 2:00] The why-MI300X close

Cut to a terminal showing `rocm-smi` with the GPU running.

> **Voiceover:** "Qwen3-VL-32B at full precision, 65k context, processing video at 768p in batched parallel. 192 gigabytes of HBM3 makes this practical. Crucible scales — every episode you don't waste training on saves 40 dollars in teleop labor."

Hard cut to logo. Done.

---

## Pre-flight checklist

- [ ] Demo dataset precached (`data/precached/lerobot__aloha_mobile_cabinet.json`).
- [ ] At least one obviously bad episode identified by index, with rationale verified.
- [ ] Gradio Space loads in <5s from a fresh browser session.
- [ ] HF token to push filtered dataset prepared (use a throwaway repo).
- [ ] `rocm-smi` snapshot saved as a still in case the live shot fails.
- [ ] Audio levels checked.
- [ ] Caption titles ready as overlay (model name, GPU, score values).
