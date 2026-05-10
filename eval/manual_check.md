# Manual sanity check — 10 episodes

Use this template to spot-check Crucible's verdicts against your own judgment on 10 episodes from the demo dataset. The hackathon judges may ask "do these scores reflect reality" — having even a coarse human-vs-Crucible agreement number is a strong defense.

## Procedure

1. Run the precache script on the demo dataset.
2. Open `data/precached/<dataset>.json` and pick 10 episodes spanning the score range.
3. Watch each episode in the Gradio UI (or `lerobot-dataset-visualizer`).
4. Fill in the table below in 30–60 seconds per episode.
5. Compute simple agreement (% of human verdicts that match Crucible's verdict).

## Worksheet

| Episode | Crucible score | Crucible verdict | Human verdict | Notes |
|---|---|---|---|---|
| 0 | | | KEEP / POLISH / REJECT | |
| 5 | | | | |
| 12 | | | | |
| 20 | | | | |
| 33 | | | | |
| 41 | | | | |
| 47 | | | | |
| 58 | | | | |
| 71 | | | | |
| 84 | | | | |

## Agreement summary

- Verdict-level match: __ / 10
- Disagreements: list the episode index + which direction Crucible was off (over- or under-rating) + your one-line take on why.

## Known systematic biases (fill after the run)

- (e.g. "Crucible over-rates strategy when operator pauses are very short — frame rate downsampling washes them out.")

The point of this sheet is not perfection; it's evidence that the system was looked at by a human, not just shipped blind.
