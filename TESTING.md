# Crucible — Layered Test Plan

Walk top-to-bottom the moment GPU credits land. Green at every layer = demo safe. Total ~25 min. All commands assume `cd ~/Crucible && source .venv/bin/activate`.

> **Need a GPU first?** See `docs/GPU_ACCESS.md` for the AMD Dev Cloud sign-up flow + paid fallbacks (RunPod MI300X $0.50/hr, RunPod H100 $1.99/hr).

---

## Layer -1 — GPU box health (run once, on first SSH)

**Time: 30 sec.** Confirm the box is real before sinking model-download time.

```bash
rocm-smi --showproductname           # one line per GPU; expect "MI300X" or similar
rocm-smi --showmeminfo vram          # Total VRAM ≈ 196,592 MB on MI300X
rocm-smi --showtemp --showpower      # idle temp <60°C, idle power <100W
rocm-smi --showbus                   # PCIe Gen5 x16 at 32 GT/s
rocminfo | grep -E "gfx|MI300|Marketing Name" | head -5
```

**Pass:** product name says MI300X (or H100 if on the fallback path), VRAM reports the full HBM, no `[!]` warnings.

Live monitor (run in a tmux pane while testing):
```bash
watch -n 1 'rocm-smi --showuse --showmemuse --showtemp --showpower | head -20'
```

**Most-likely failure:** `rocm-smi: command not found`. **Fix:** the base image isn't AMD's vLLM image — `apt list --installed | grep rocm` and re-pull `rocm/vllm-dev:nightly_main_*` or `vllm/vllm-openai-rocm:latest`.

---

## Layer 0 — Local logic (no GPU)

**Time: 2 min**

```bash
ruff check src/ scripts/ frontend/ tests/
pytest -q
python scripts/io_smoke.py --repo lerobot/aloha_static_cups_open --episodes 2 --frames 6
```

**Pass:** ruff clean; pytest **35 passed** (aggregator 10 + critics_json 7 + telemetry 10 + filtering 5 + imports 3); io_smoke prints `PASS — 2 episodes streamed and decoded cleanly`.

**Most-likely failure:** io_smoke exits 3 with `telemetry digest empty`. **Fix:** `pip install --upgrade --force-reinstall av` (libdav1d codec missing).

---

## Layer 1 — vLLM endpoint health

**Time: 1 min**

```bash
curl -s http://localhost:8001/v1/models | python3 -m json.tool          # data[0].id == Qwen/Qwen3-VL-32B-Instruct
curl -s http://localhost:8000/healthz                                   # {"ok": true, "default_model": "..."}
curl -s http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-32B-Instruct","messages":[{"role":"user","content":"reply OK"}],"max_tokens":4,"temperature":0}'
# choices[0].message.content == "OK"
curl -s http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-32B-Instruct","messages":[{"role":"user","content":"emit {\"x\":1}"}],"max_tokens":16,"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"t","schema":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]},"strict":true}}}'
# parses to {"x": 1} → confirms xgrammar
```

**Pass:** all four return correct shapes; total <5s after warm-up.

**Most-likely failure:** `/v1/models` hangs. **Fix:** weights still downloading — tail `docker logs` and watch HF cache grow.

---

## Layer 2 — Single critic against live VLM

**Time: 3 min**

```bash
for c in visual kinematic task strategy safety; do
  python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic $c --frames 8
done
```

**Pass (each critic):** JSON parses (no `PARSE_ERROR`/`ERROR`); `score` ∈ [0,10]; `verdict` in vocab — visual `{EXCELLENT,ACCEPTABLE,MARGINAL,REJECT}`, kinematic `{SMOOTH,ACCEPTABLE,JERKY,REJECT}`, task `{COMPLETE,PARTIAL,FAILED,UNCLEAR}`, strategy `{EXEMPLARY,GOOD,MEDIOCRE,POOR}`, safety `{SAFE,MINOR_CONCERN,MODERATE_CONCERN,UNSAFE}`; non-empty `rationale`; ≥1 `evidence` item.

**Most-likely failure:** `verdict: PARSE_ERROR`. **Fix:** vLLM started without `--guided-decoding-backend xgrammar`; restart container with the flag.

---

## Layer 3 — Full pipeline on 1 episode

**Time: 1 min**

```bash
python -c "
import asyncio, json
from src.config import CrucibleConfig
from src.pipeline import score_dataset
cfg = CrucibleConfig(); cfg.max_episodes_per_run = 1
res = asyncio.run(score_dataset('lerobot/aloha_static_cups_open', cfg, use_cache=False))
print(json.dumps(res[0], indent=2, default=str))"
```

**Pass:** 1 record; `critics` has all 5 keys (`visual_quality, kinematic_quality, task_success, strategy, safety`), none `PARSE_ERROR`/`ERROR`; `verdict` has `final_score` ∈ [0,10], `verdict` ∈ {KEEP,POLISH,REJECT}, non-empty `summary`; `verdict.fallback` is **not** `True`; `elapsed_s` < 60.

**Most-likely failure:** `verdict.fallback == True` while critics succeeded. **Fix:** aggregator schema rejection or truncation; lower `aggregator_max_tokens` or check vLLM log.

---

## Layer 4 — Small dataset with cache

**Time: ~3 min**

```bash
rm -f data/precached/lerobot__aloha_static_cups_open.json
python scripts/precache_demo.py --repo lerobot/aloha_static_cups_open --episodes 5 --frames 12
python -c "
import json
recs = json.load(open('data/precached/lerobot__aloha_static_cups_open.json'))
assert len(recs) == 5
for r in recs:
    v = r['verdict']
    assert v['verdict'] in {'KEEP','POLISH','REJECT'}
    assert not v.get('fallback'), f'ep {r[\"episode_index\"]} hit fallback'
    for n,c in r['critics'].items():
        assert c.get('verdict') not in {'PARSE_ERROR','ERROR'}, (n, c.get('verdict'))
print('OK — 5 episodes, no fallbacks')"
```

**Pass:** file written; all 5 records non-fallback, no parse errors; wall-clock < 60s on MI300X.

**Most-likely failure:** sporadic `PARSE_ERROR` on 1 critic. **Fix:** `CRUCIBLE_REQUEST_RETRIES=3`; if persistent, drop `image_max_dim` to 640.

---

## Layer 5 — FastAPI + SSE

**Time: 2 min**

```bash
JOB=$(curl -s -X POST http://localhost:8000/score_dataset -H 'Content-Type: application/json' \
  -d '{"repo_id":"lerobot/aloha_static_cups_open","n_episodes":3,"use_cache":true}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')

curl -N -s "http://localhost:8000/progress/$JOB"
# Expect: 3× `data: {"type":"episode",...}` then `data: {"type":"done"}`

curl -s "http://localhost:8000/results/$JOB" | python3 -c "
import sys,json
p=json.load(sys.stdin); assert p['status']=='complete'; assert len(p['results'])==3
print('OK')"

# Optional async/httpx framing check:
python -c "
import httpx,json
with httpx.Client(timeout=None) as c, c.stream('GET','http://localhost:8000/progress/$JOB') as r:
    for line in r.iter_lines():
        if line.startswith('data:'):
            m=json.loads(line[5:].strip()); print(m.get('type'),m.get('i'))
            if m.get('type')=='done': break"
```

**Pass:** POST returns `job_id` <1s; SSE emits 3 episode events in order then `done`; `/results` `status=complete` with 3 records matching SSE.

**Most-likely failure:** `/progress/$JOB` 404s. **Fix:** runner crashed pre-queue (commonly `openai` import missing); check uvicorn stderr.

---

## Layer 6 — Gradio UI manual QA (60s judge walkthrough)

**Time: 2 min.** Open the public Space URL.

1. Cold load < 5s; no stuck spinner.
2. Dropdown shows 4 options (`aloha_mobile_cabinet` default, `aloha_static_cups_open`, `aloha_sim_insertion_human_image`, `aloha_mobile_shrimp`); custom value typeable.
3. Score Dataset on `aloha_mobile_cabinet` with `use_cache=true` — histogram + dataframe fill within 3s.
4. Histogram has all 3 verdict colors (green KEEP / yellow POLISH / red REJECT) where present.
5. Dataframe sorts by `score` on header click.
6. Type `47` (or pre-picked killer index), Load — five critic cards render with rationale + `MM:SS:` evidence; video thumbnail loads.
7. Each card: score, verdict, ≥1-sentence rationale, ≥1 evidence line.
8. Threshold slider drags 0→10 smoothly.
9. Push guard rails: empty target_repo → `Set target_repo as 'username/dataset_name'.`; empty token → `Provide your HF token (write scope) ...`.
10. Live: switch to `aloha_static_cups_open`, uncheck cache, Score — SSE rows appear within 15s, histogram fills incrementally.
11. Download results JSON → file appears, valid JSON.
12. No browser-console errors beyond CORS preflight.

**Pass:** 12/12. Items 10–11 may fall back to precache-with-banner if GPU offline.

**Most-likely failure:** item 10 hangs. **Fix:** Space's `CRUCIBLE_API_BASE` not set or pointing at private IP — set public IP, restart Space.

---

## Layer 7 — Push-to-Hub

**Time: 3 min**

```bash
TARGET="lord-arbiter/crucible-test-curated-$(date +%s)"
JOB=$(curl -s -X POST http://localhost:8000/score_dataset -H 'Content-Type: application/json' \
  -d '{"repo_id":"lerobot/aloha_static_cups_open","n_episodes":5,"use_cache":true}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
until [ "$(curl -s http://localhost:8000/results/$JOB | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')" = "complete" ]; do sleep 2; done

curl -s -X POST http://localhost:8000/push_filtered -H 'Content-Type: application/json' \
  -d "{\"job_id\":\"$JOB\",\"threshold\":7.0,\"target_repo\":\"$TARGET\",\"hf_token\":\"$HF_TOKEN\",\"source_repo\":\"lerobot/aloha_static_cups_open\"}"
echo "https://huggingface.co/datasets/$TARGET"

python -c "
from huggingface_hub import hf_hub_download
import json
from lerobot import LeRobotDataset
p = hf_hub_download('$TARGET','meta/info.json',repo_type='dataset')
cur = json.load(open(p))['crucible_curation']
assert cur['threshold']==7.0 and cur['n_kept']==len(cur['kept_episode_indices'])
ds = LeRobotDataset('$TARGET', episodes=cur['kept_episode_indices'])
assert len(ds) > 0
print('frames in curated subset:', len(ds), 'kept:', cur['kept_episode_indices'])"
```

**Pass:** (1) browser shows rendered README, `meta/info.json`, `meta/episodes/*.parquet`, ≥1 `data/.../*.parquet`, ≥1 `videos/.../*.mp4`; (2) `crucible_curation` block has `threshold=7.0` + non-empty `kept_episode_indices`; (3) `LeRobotDataset(repo, episodes=kept)` loads with `len(ds) > 0`.

**Most-likely failure:** `403`. **Fix:** HF token is read-only — regenerate with **write** scope.

---

## Layer 8 — Concurrency stress (optional)

**Time: 4 min**

```bash
for i in 1 2 3 4 5; do
  curl -s -X POST http://localhost:8000/score_dataset -H 'Content-Type: application/json' \
    -d '{"repo_id":"lerobot/aloha_static_cups_open","n_episodes":5,"use_cache":false}' \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])' &
done; wait
watch -n 1 'rocm-smi --showuse --showmemuse | head -20; curl -s http://localhost:8000/jobs | python3 -m json.tool | tail -40'
```

**Pass:** all 5 jobs reach `status: complete`; vLLM no `OutOfMemoryError`/`CUDA error`; HBM peak <85%; each job's `n_done == n_target`.

**Most-likely failure:** jobs stuck at `n_done < n_target`. **Fix:** raise vLLM `--max-num-seqs` to 32 in `docker/entrypoint.sh`, or cap to 3 concurrent for the demo.

---

## Layer 9 — Live demo dry-run

**Time: 2 min.** With stopwatch + OBS, walk `docs/demo_script.md`:

1. Open Space, narrate problem (15s).
2. `aloha_mobile_cabinet` + Score Dataset (cache hit, instant) — narrate rubric (20s).
3. Click pre-picked killer-detail episode — read hesitation rationale verbatim (25s).
4. Drag threshold to 7.0 — note kept-count change (10s).
5. Push to fresh `lord-arbiter/crucible-demo-curated-<ts>`, show rendered README (40s).

**Pass:** total < 2:00; no spinner > 3s except push; killer episode renders with timestamp evidence; pushed URL opens cleanly in fresh tab.

**Most-likely failure:** killer-detail index changed after re-precache. **Fix:** re-pick during pre-flight, update demo script before recording.

---

## Time budget

| Layer | Time | Blocking |
|---|---|---|
| 0 local logic | 2m | yes |
| 1 vLLM health | 1m | yes |
| 2 five critics | 3m | yes |
| 3 one episode | 1m | yes |
| 4 small + cache | 3m | yes |
| 5 SSE | 2m | yes |
| 6 Gradio QA | 2m | yes |
| 7 push-to-Hub | 3m | yes |
| 8 concurrency | 4m | optional |
| 9 demo dry-run | 2m | yes |
| **Total (skip 8)** | **~19m** | |
| **Full** | **~23m** | |

Red at any layer = fix before continuing. Green at Layer 9 = ship.
