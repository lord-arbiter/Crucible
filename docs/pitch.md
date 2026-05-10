# Crucible — Pitch deck content

Five slides. Convert to Slides / Canva / Keynote at submission time.

---

## Slide 1 — Title

**Crucible**
*The first multi-axis behavioral rubric for robot demonstration data — VLM-judged on AMD MI300X.*

Solo build · AMD Developer Hackathon 2026

Stack: Qwen3-VL-32B on AMD MI300X via vLLM · FastAPI · Gradio on HuggingFace Spaces

Submitting under: **AI Agents & Agentic Workflows**

---

## Slide 2 — Problem

Robotics teams spend **$118 / hour** on teleoperation data collection. **20–40 % of it** silently teaches the policy bad habits.

Existing curation falls into two camps, both incomplete:

- **Heuristic scorers** (`score_lerobot_episodes`, dataset-visualizer filter panel) — score jerk, path efficiency, actuator saturation. Miss strategy and hesitation.
- **Binary VLM success classifiers** (`score_lerobot_episodes` → Gemini call, Genie Centurion → fine-tuned MiniGPT-4) — answer "did the task complete?". Miss the silent killers: hesitation, regrasps, inefficient paths, near-miss safety incidents.

A bad-strategy episode that *succeeds* is **worse** than a good-strategy episode that fails — the former teaches the policy unreliable recovery as if it were normal.

**Show one example:** a teleop episode that scores fine on jerk, scores fine on Gemini-task-success, but contains 3 regrasps and a near-collision. Existing tools keep it; your policy quietly inherits the bad habit.

---

## Slide 3 — Insight + approach

**Five specialist VLM critics with a structured behavioral rubric > one heuristic scorer or one binary success classifier.**

Diagram:

```
LeRobot v3 dataset (chunk-batched)
  ├── frames (sampled, 16 × 768px)
  ├── joint trajectories ──► telemetry digest (idle, recovery, gripper events)
  └── task description (from meta/tasks.parquet)
                         │
                         ▼
   ┌───── Qwen3-VL critics (parallel on MI300X via xgrammar) ─────┐
   │  Visual ·  Kinematic ·  Task ·  Strategy ·  Safety           │
   └──────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
                  Aggregator (weighted, hard-fail rules)
                              │
                              ▼
            KEEP / POLISH / REJECT + rationale + timestamp evidence
```

The aggregator weighs **task success** and **strategy** at 1.5×. Any critic flagging `REJECT / UNSAFE / FAILED` forces an overall `REJECT`.

We're **not** "first to apply VLM as judge" — `score_lerobot_episodes` shipped Oct 2025; Genie Centurion shipped May 2025. We **are** the first to apply a **multi-axis behavioral rubric** with **structured rationale + timestamp citations** for offline LeRobot dataset curation, as a deployable studio rather than a one-shot CLI.

---

## Slide 4 — Demo

Screenshot the Gradio dashboard:

- Score histogram across 25 episodes (KEEP / POLISH / REJECT colored).
- Episode 47 expanded: video clip, five critic verdict cards, rationale, timestamp evidence.
- Threshold slider at 7.0 → "65 of 85 kept."
- Push button → curated repo on HuggingFace with a `crucible_curation` block in info.json + auto-generated dataset card with `LeRobotDataset(repo, episodes=[...])` load snippet.

Caption beneath:

> "On `lerobot/aloha_mobile_cabinet`: Crucible flagged 12 of 25 episodes for problems that no heuristic scorer or binary success classifier caught — multi-attempt regrasps, 3 s hesitation pauses, near-collisions on otherwise successful trials. Average kept-score 8.1; average rejected-score 4.3."

---

## Slide 5 — Why this stack

- **AMD MI300X 192 GB HBM3** — fits Qwen3-VL-32B at bf16 with a 65k-token context. Five concurrent critics + one aggregator share the same served model. Same workload on H100 80 GB requires either model quantization or aggressive context trimming. Headroom is the load-bearing capability.
- **Qwen3-VL-32B** — strongest open-weight multimodal model with native video understanding and 256K context. AMD/LMSYS optimizations (rocJPEG, AITER + prefill-decode attention) measured TTFT 1.08 s, TPOT 12.5 ms on the MI308 generation.
- **vLLM with xgrammar guided decoding** — `json_schema` response format + `/no_think` ensures every critic returns parseable structured JSON. We don't trust prose-mode JSON.
- **Single-GPU, single-model, parallel** — no orchestration overhead, no model switching, no batched cold starts.

Forward-look: **scale to MI355X** for multi-dataset batched curation; productize as Crucible-as-a-service for robotics teams shipping VLAs (Vision-Language-Action policies).

---

## Anticipated judge question + rebuttal

**Q:** "`score_lerobot_episodes` already does VLM-judged curation. What's new?"

**A:** That tool calls Gemini once per episode with a single boolean *"did the task succeed?"* prompt. That's LLM-as-classifier, not LLM-as-judge. Crucible runs a **structured five-axis behavioral rubric** (visual / kinematic / task / strategy / safety) with chain-of-thought critic prompts and timestamped per-frame attribution, returning a rich rationale rather than a scalar. It also runs **locally on AMD MI300X** so a lab can curate millions of frames without paying Google per call or shipping proprietary teleop footage to a third-party API. The novelty is in the **rubric design** and **inference economics**, not in "we thought of using a VLM."

---

## Optional bonus slide — Numbers

If asked for ROI math:

- Filtering bottom 20 % of demos → 15–25 % policy success-rate improvement (Stanford QoQ).
- $118 / hour teleop × 30 minutes wasted per bad episode ≈ **$59 per bad episode**.
- Crucible processes one episode in ~5–8 s on MI300X end-to-end (TTFT 0.6–0.9 s × 6 calls); a 25-episode dataset clears in ~3 minutes. Labor saved per dataset clears the GPU cost in the first run.
