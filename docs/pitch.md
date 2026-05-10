# Crucible — Pitch deck content

Five slides. Convert to Slides / Canva / Keynote at submission time.

---

## Slide 1 — Title

**Crucible**
*A VLM-judged data curation studio for robot demonstrations.*

Solo build · AMD Developer Hackathon 2026

Stack: Qwen3-VL-32B on AMD MI300X via vLLM · FastAPI · Gradio on HuggingFace Spaces

Submitting under: **AI Agents & Agentic Workflows**

---

## Slide 2 — Problem

Robotics teams spend **$118 / hour** on teleoperation data collection. **20–40 % of it** silently teaches the policy bad habits.

Existing curation tools score classical heuristics:

- Joint jitter (derivative thresholds)
- Motion blur (Laplacian variance)
- Episode length

They cannot answer:

- "Did the operator hesitate before grasping?"
- "Did the task actually complete?"
- "Was the strategy efficient or just lucky?"
- "Was there a near-miss safety incident?"

Result: silent miscalibration of every fine-tune downstream.

**Show one example:** a teleop episode that scores fine on jitter and lighting but contains 3 regrasps and a near-collision. Existing tools keep it; your policy quietly inherits the bad habit.

---

## Slide 3 — Insight + approach

**Five specialist VLM critics > one heuristic scorer.**

Diagram:

```
LeRobot dataset
  ├── frames (sampled)
  ├── joint trajectories ──► telemetry digest (idle, recovery, gripper events)
  └── task description
                         │
                         ▼
   ┌───── Qwen3-VL critics (parallel on MI300X) ─────┐
   │  Visual ·  Kinematic ·  Task ·  Strategy ·  Safety │
   └────────────────────────┬────────────────────────┘
                            │
                            ▼
                Aggregator (weighted, hard-fail rules)
                            │
                            ▼
              KEEP / POLISH / REJECT + rationale + evidence
```

The aggregator weighs **task success** and **strategy** at 1.5×. Any critic flagging `REJECT / UNSAFE / FAILED` forces an overall `REJECT`.

We're the first to apply LLM-as-judge to physical-AI demonstration data at this scale.

---

## Slide 4 — Demo

Screenshot the Gradio dashboard:

- Score histogram across 25 episodes (KEEP / POLISH / REJECT colored).
- Episode 47 expanded: video clip, five critic verdict cards, rationale, timestamp evidence.
- Threshold slider at 7.0 → "65 of 85 kept."
- Push button → curated repo on HuggingFace with auto-generated dataset card.

Caption beneath:

> "On `lerobot/aloha_mobile_cabinet`: Crucible flagged 12 of 25 episodes for problems that no heuristic scorer caught. Average kept-score 8.1; average rejected-score 4.3."

---

## Slide 5 — Why this stack

- **AMD MI300X 192 GB HBM3** — fits Qwen3-VL-32B at fp16 with a 65k-token context. Same workload on H100 80 GB requires either model quantization or aggressive context trimming. Headroom is the load-bearing capability.
- **Qwen3-VL-32B** — strongest open-weight multimodal model with native video understanding and 256K context. AMD/LMSYS published MI300X-specific optimizations (rocJPEG, batch-level data parallelism for the vision encoder).
- **vLLM** — single served model handles all six agents (5 critics + aggregator) via batched OpenAI-compatible API. Single-GPU, single-model, parallel.
- **HuggingFace Spaces** — free CPU tier hosts the Gradio frontend; GPU work lives on the MI300X droplet.

Forward-look: **scale to MI355X** for multi-dataset batched curation; productize as Crucible-as-a-service for robotics teams shipping VLAs.

---

## Optional bonus slide — Numbers

If asked for ROI math:

- Filtering bottom 20 % of demos → 15–25 % policy success-rate improvement (Stanford).
- $118 / hour teleop × 30 minutes wasted per bad episode ≈ **$59 per bad episode**.
- Crucible processes one episode in ~30 s on MI300X end-to-end; the labor saved per dataset clears the GPU cost in the first run.
