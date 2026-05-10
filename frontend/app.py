"""Crucible Gradio frontend.

Deployed to a HuggingFace Space (CPU tier). Talks to the FastAPI orchestrator
over HTTPS + SSE. When the user pastes a repo_id that we have precached
locally, we serve the cached results instantly so the demo is robust to a
GPU droplet outage.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

API_BASE = os.environ.get("CRUCIBLE_API_BASE", "http://localhost:8000")
PRECACHE_DIR = Path(os.environ.get("CRUCIBLE_CACHE_DIR", "data/precached"))

DEFAULT_REPO = "lerobot/aloha_mobile_cabinet"
SAMPLE_REPOS = [
    "lerobot/aloha_mobile_cabinet",
    "lerobot/aloha_static_cups_open",
    "lerobot/aloha_sim_insertion_human_image",
    "lerobot/aloha_mobile_shrimp",
]

VERDICT_COLORS = {
    "KEEP": "#16a34a",
    "POLISH": "#eab308",
    "REJECT": "#dc2626",
}


def _precache_path(repo_id: str) -> Path:
    return PRECACHE_DIR / f"{repo_id.replace('/', '__')}.json"


def _load_precached(repo_id: str) -> list[dict] | None:
    p = _precache_path(repo_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _results_to_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        v = r.get("verdict") or {}
        rows.append({
            "episode": r.get("episode_index"),
            "task": (r.get("task_description") or "")[:80],
            "score": float(v.get("final_score") or 0.0),
            "verdict": v.get("verdict") or "?",
            "duration_s": r.get("duration_s") or 0,
            "top_concern": v.get("top_concern") or "",
        })
    return pd.DataFrame(rows)


def _score_histogram(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig = px.histogram(
        df,
        x="score",
        color="verdict",
        nbins=20,
        color_discrete_map=VERDICT_COLORS,
        title=f"Score distribution across {len(df)} episodes",
    )
    fig.update_layout(bargap=0.1, height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _kick_off(
    repo_id: str,
    n_episodes: int,
    use_cache: bool,
    vlm_endpoint: str,
    vlm_model: str,
    vlm_api_key: str,
) -> tuple[str, str]:
    if use_cache:
        cached = _load_precached(repo_id)
        if cached:
            return f"precache::{repo_id}", f"Loaded {len(cached)} precached episodes from disk."
    body: dict = {"repo_id": repo_id, "n_episodes": int(n_episodes), "use_cache": use_cache}
    # Forward any UI-specified VLM endpoint overrides so the same orchestrator
    # can serve different users without a restart.
    if vlm_endpoint and vlm_endpoint.strip():
        body["vlm_endpoint"] = vlm_endpoint.strip()
    if vlm_model and vlm_model.strip():
        body["vlm_model"] = vlm_model.strip()
    if vlm_api_key and vlm_api_key.strip():
        body["vlm_api_key"] = vlm_api_key.strip()
    r = requests.post(f"{API_BASE}/score_dataset", json=body, timeout=15)
    r.raise_for_status()
    payload = r.json()
    return payload["job_id"], f"Started {payload['job_id']} on {repo_id} ({n_episodes} episodes)"


def _stream_results(job_id: str) -> Iterator[tuple[pd.DataFrame, go.Figure, str]]:
    if job_id.startswith("precache::"):
        repo_id = job_id.split("::", 1)[1]
        cached = _load_precached(repo_id) or []
        df = _results_to_df(cached)
        yield df, _score_histogram(df), f"Loaded {len(cached)} precached episodes."
        return

    try:
        import sseclient  # type: ignore
    except ImportError:
        sseclient = None

    url = f"{API_BASE}/progress/{job_id}"
    results: list[dict] = []
    if sseclient:
        with requests.get(url, stream=True, headers={"Accept": "text/event-stream"}, timeout=None) as resp:
            client = sseclient.SSEClient(resp)
            for event in client.events():
                if not event.data:
                    continue
                try:
                    msg = json.loads(event.data)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "done":
                    break
                if msg.get("type") == "error":
                    yield pd.DataFrame(), go.Figure(), f"Error: {msg.get('error')}"
                    return
                ep = msg.get("episode")
                if ep:
                    results.append(ep)
                    df = _results_to_df(results)
                    yield df, _score_histogram(df), f"Scored {len(results)} / {msg.get('total')} episodes"
    else:
        # Polling fallback
        while True:
            r = requests.get(f"{API_BASE}/results/{job_id}", timeout=15)
            r.raise_for_status()
            payload = r.json()
            results = payload["results"]
            df = _results_to_df(results)
            yield df, _score_histogram(df), f"{payload['status']} — {len(results)} episodes"
            if payload["status"] in {"complete", "error"}:
                break
            time.sleep(2)


def _episode_detail(repo_id: str, job_id: str | None, episode_index_str: str) -> tuple[str, str, gr.Video]:
    """Render the per-critic cards + telemetry digest + embedded video for one episode."""
    if not episode_index_str:
        return "", "", gr.Video(value=None)
    try:
        target = int(episode_index_str)
    except ValueError:
        return f"Invalid episode index: {episode_index_str}", "", gr.Video(value=None)

    results: list[dict] = []
    if job_id and job_id.startswith("precache::"):
        results = _load_precached(job_id.split("::", 1)[1]) or []
    elif job_id:
        try:
            r = requests.get(f"{API_BASE}/results/{job_id}", timeout=10)
            r.raise_for_status()
            results = r.json().get("results") or []
        except Exception:
            results = []
    if not results:
        results = _load_precached(repo_id) or []

    record = next((r for r in results if r.get("episode_index") == target), None)
    if not record:
        return f"No episode {target} in current results.", "", gr.Video(value=None)

    verdict = record.get("verdict") or {}
    cards: list[str] = []
    for key in ("visual_quality", "kinematic_quality", "task_success", "strategy", "safety"):
        c = (record.get("critics") or {}).get(key) or {}
        score = c.get("score", "—")
        ev = "<br>".join(
            f"<code>{e.get('timestamp', '')}</code>: {e.get('observation', '')}"
            for e in (c.get("evidence") or [])[:4]
        ) or "<i>no evidence cited</i>"
        cards.append(
            f"""<div style='border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin:6px 0;background:#fafafa'>
<b>{key.replace('_', ' ').title()}</b> &nbsp; <span style='font-size:1.1em'>{score}</span>
&nbsp; <span style='color:#6b7280'>({c.get('verdict', '?')})</span>
<div style='color:#374151;margin-top:4px'>{c.get('rationale', '')}</div>
<div style='font-size:0.85em;color:#6b7280;margin-top:4px'>{ev}</div>
</div>"""
        )
    final_score = verdict.get("final_score", "—")
    color = VERDICT_COLORS.get(str(verdict.get("verdict") or "").upper(), "#6b7280")
    summary_html = f"""<div style='padding:12px;border-radius:8px;background:{color}20;border-left:4px solid {color}'>
<h3 style='margin:0'>Episode {record.get('episode_index')} — {verdict.get('verdict')} ({final_score})</h3>
<p><b>Task:</b> {record.get('task_description', '')}</p>
<p>{verdict.get('summary', '')}</p>
<p><b>Top concern:</b> {verdict.get('top_concern') or '—'}</p>
</div>
{''.join(cards)}"""
    video_url = record.get("raw_video_url") or ""
    return summary_html, record.get("task_description", ""), gr.Video(value=video_url or None)


def _export_results(job_id: str | None, repo_id: str) -> str | None:
    """Write the current results bundle to a tempfile and return its path for download."""
    results: list[dict] = []
    if job_id and job_id.startswith("precache::"):
        results = _load_precached(job_id.split("::", 1)[1]) or []
    elif job_id:
        try:
            r = requests.get(f"{API_BASE}/results/{job_id}", timeout=10)
            r.raise_for_status()
            results = r.json().get("results") or []
        except Exception:
            results = []
    if not results:
        results = _load_precached(repo_id) or []
    if not results:
        return None
    safe = (repo_id or "results").replace("/", "__")
    tmp = Path(tempfile.gettempdir()) / f"crucible_{safe}.json"
    tmp.write_text(json.dumps(results, indent=2, default=str))
    return str(tmp)


def _push_handler(
    job_id: str,
    repo_id: str,
    threshold: float,
    target_repo: str,
    hf_token: str,
) -> str:
    if not target_repo or "/" not in target_repo:
        return "Set target_repo as 'username/dataset_name'."
    if not hf_token:
        return "Provide your HF token (write scope) — never commit it to a repo."

    payload = {
        "job_id": job_id or "",
        "threshold": float(threshold),
        "target_repo": target_repo,
        "hf_token": hf_token,
        "source_repo": repo_id,
    }
    try:
        r = requests.post(f"{API_BASE}/push_filtered", json=payload, timeout=600)
        r.raise_for_status()
        out = r.json()
    except Exception as exc:
        return f"Push failed: {exc}"
    if not out.get("ok"):
        return f"Push refused: {out.get('error')}"
    return f"Pushed {out['n_kept']} episodes (filtered {out['n_filtered']}). View: {out['target_repo_url']}"


with gr.Blocks(title="Crucible — Robot Demo Curator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# Crucible
*A multi-axis VLM-judged data curation studio for robot demonstrations.*

Paste a HuggingFace LeRobot dataset URL. Crucible runs five Qwen3-VL critics
(visual, kinematic, task success, strategy, safety) on every episode, fuses
their outputs into a final KEEP / POLISH / REJECT verdict with rationale and
timestamp evidence, and lets you push the curated subset back to the Hub.

GitHub: <https://github.com/lord-arbiter/Crucible> · Recipes: <https://github.com/lord-arbiter/Crucible/tree/main/docs/recipes>
"""
    )

    with gr.Row():
        repo_id = gr.Dropdown(
            label="HuggingFace LeRobot dataset",
            choices=SAMPLE_REPOS,
            value=DEFAULT_REPO,
            allow_custom_value=True,
        )
        n_eps = gr.Slider(label="Episodes to score", minimum=1, maximum=100, value=25, step=1)
        use_cache = gr.Checkbox(label="Use precache when available", value=True)

    with gr.Accordion("Custom VLM endpoint (optional)", open=False):
        gr.Markdown(
            """Override the orchestrator's default model per request. Leave blank to
use the server-side defaults.

- **OpenAI-compatible**: set the endpoint URL + model id + API key
  (Hyperbolic, Together, DashScope, self-hosted vLLM, ...).
- **OpenAI direct**: leave endpoint blank, set model id like `gpt-4o-mini` and the API key.
- **LiteLLM universal** (orchestrator must have `[universal]` extra installed):
  leave endpoint blank, set model id with a provider prefix —
  `anthropic/claude-sonnet-4-5`, `bedrock/anthropic.claude-3-5-sonnet`,
  `gemini/gemini-2.5-flash`, `vertex_ai/gemini-2.5-pro`, `cohere/command-a`,
  `groq/llama-3.2-90b-vision-preview`, etc."""
        )
        with gr.Row():
            vlm_endpoint = gr.Textbox(
                label="VLM endpoint (OpenAI-compatible)",
                placeholder="https://api.your-provider.com/v1",
                value="",
            )
            vlm_model = gr.Textbox(
                label="Model id",
                placeholder="Qwen/Qwen3-VL-32B-Instruct",
                value="",
            )
            vlm_api_key = gr.Textbox(
                label="API key",
                type="password",
                value="",
            )

    with gr.Row():
        go_btn = gr.Button("Score dataset", variant="primary")
        status = gr.Markdown()

    job_state = gr.State()

    with gr.Row():
        with gr.Column(scale=2):
            score_hist = gr.Plot(label="Score distribution")
        with gr.Column(scale=3):
            results_df = gr.Dataframe(
                label="Per-episode scores", headers=["episode", "task", "score", "verdict", "duration_s", "top_concern"],
                interactive=False,
            )

    gr.Markdown("## Episode detail")
    with gr.Row():
        episode_idx = gr.Textbox(label="Episode index", value="0")
        load_btn = gr.Button("Load episode", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            episode_html = gr.HTML()
        with gr.Column(scale=1):
            episode_video = gr.Video(label="Primary camera", interactive=False)
            episode_task = gr.Textbox(label="Task instruction", interactive=False)

    gr.Markdown("## Filter & push")
    with gr.Row():
        threshold = gr.Slider(label="Keep threshold", minimum=0, maximum=10, value=7.0, step=0.1)
        target_repo = gr.Textbox(label="Target dataset repo (username/name)")
        hf_token = gr.Textbox(label="HF token (write scope)", type="password")
    push_btn = gr.Button("Push filtered dataset to Hub", variant="primary")
    push_status = gr.Markdown()

    gr.Markdown("## Export raw results (judge-friendly)")
    with gr.Row():
        export_btn = gr.Button("Download results JSON", variant="secondary")
        export_file = gr.File(label="Results JSON", interactive=False)

    go_btn.click(
        _kick_off,
        inputs=[repo_id, n_eps, use_cache, vlm_endpoint, vlm_model, vlm_api_key],
        outputs=[job_state, status],
    ).then(
        _stream_results,
        inputs=job_state,
        outputs=[results_df, score_hist, status],
    )

    load_btn.click(
        _episode_detail,
        inputs=[repo_id, job_state, episode_idx],
        outputs=[episode_html, episode_task, episode_video],
    )

    push_btn.click(
        _push_handler,
        inputs=[job_state, repo_id, threshold, target_repo, hf_token],
        outputs=push_status,
    )

    export_btn.click(_export_results, inputs=[job_state, repo_id], outputs=export_file)

    gr.Markdown(
        """---
*Architecture:* Gradio (HF Spaces, CPU) → FastAPI orchestrator → any
OpenAI-compatible Qwen3-VL endpoint. Five specialist critics + one aggregator
share the same backend. See [docs/architecture.md](https://github.com/lord-arbiter/Crucible/blob/main/docs/architecture.md)
for the full diagram.
"""
    )


if __name__ == "__main__":
    demo.queue().launch(server_name=os.environ.get("GRADIO_HOST", "0.0.0.0"))
