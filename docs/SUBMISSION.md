# Submission checklist — lablab.ai AMD Developer Hackathon 2026

Walk this list top-to-bottom on submission day. Each item must be green
before you click "Submit".

## 1. Working prototype is live

- [ ] HuggingFace Space loads from a fresh browser in <5s: `https://huggingface.co/spaces/<user>/crucible`
- [ ] Pasting `lerobot/aloha_mobile_cabinet` and clicking "Score" returns results from the precache instantly.
- [ ] Episode detail view renders the embedded video + five critic cards.
- [ ] Threshold slider changes the kept-count display.
- [ ] (If GPU is up) live scoring on `lerobot/aloha_static_cups_open` finishes in under 2 minutes.

## 2. Demo video

- [ ] Length: 1:30 – 2:00 minutes (lablab tolerates up to 3:00).
- [ ] Script followed: `docs/demo_script.md`.
- [ ] Hosted on YouTube **unlisted** (lablab accepts unlisted) or Loom.
- [ ] Captions / on-screen titles for: model name, GPU, score values.
- [ ] Audio levels checked.
- [ ] One `rocm-smi` shot (live or still) showing MI300X utilization.

## 3. Pitch deck

- [ ] 5 slides max, content from `docs/pitch.md`.
- [ ] Title slide includes: project name, builder, track, hackathon name.
- [ ] Stack call-outs: AMD MI300X, Qwen3-VL.

## 4. GitHub repo

- [ ] Repo public at `https://github.com/lord-arbiter/crucible`.
- [ ] README renders cleanly with architecture diagram (`docs/architecture.md`).
- [ ] LICENSE present (MIT).
- [ ] `data/precached/<demo_dataset>.json` committed for offline demo safety.
- [ ] `.env.example` present; no real secrets in the repo.
- [ ] CI badge or pytest results visible (optional).

## 5. lablab submission form

Fill in:

- **Project name:** Crucible
- **Tagline:** A VLM-judged data curation studio for robot demonstrations.
- **Track:** AI Agents & Agentic Workflows
- **Technologies:** AMD Developer Cloud, AMD ROCm, AMD MI300X, vLLM, Qwen3-VL, HuggingFace Spaces, FastAPI, Gradio, LeRobot
- **Participation format:** Online
- **Demo video URL:** <fill>
- **Live demo URL:** <Space URL>
- **GitHub URL:** <repo URL>
- **Pitch deck URL:** <Slides / Canva URL>
- **Builder:** Chakradhari (solo)
- **Bonus tracks claimed:** Qwen-powered project, Build in Public

## 6. Bonus track artifacts

- [ ] **Qwen-powered project** — README has a prominent "Powered by Qwen3-VL" section. The model name appears in the demo video.
- [ ] **Build in Public** — Twitter/X thread posted with at least 5 progress tweets and a link to the Space; thread URL pasted into the lablab submission notes.
- [ ] (Optional) **X402 Payments** — only if `/score_paid` endpoint is wired up and tested.

## 7. Post-submission

- [ ] Shut down the GPU droplet immediately to stop credit burn.
- [ ] Save the lablab confirmation email.
- [ ] Tweet final thread post linking the submission.
- [ ] Note any judge feedback received during the office hours window.

## 8. Last-30-minutes checklist

- [ ] Open the submitted Space in an incognito window — does it actually work for someone who isn't you?
- [ ] Click each button in the UI once.
- [ ] Verify the demo video link plays for someone not signed into your YouTube.
- [ ] Verify the GitHub repo is public, not private.
- [ ] Submit, then re-open the lablab submission to confirm everything saved.
