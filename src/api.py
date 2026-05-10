"""FastAPI orchestrator. Lives on the GPU droplet and exposes:

- POST /score_dataset      kick off a job, returns job_id
- GET  /progress/{job_id}  SSE stream of per-episode verdicts
- GET  /results/{job_id}   final results once done
- GET  /jobs               list jobs
- POST /push_filtered      push curated subset to the Hub
- GET  /healthz            simple health check (used by the Space)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .config import DEFAULT_CONFIG, CrucibleConfig
from .filtering import push_filtered_to_hub
from .pipeline import load_precached, score_dataset

logger = logging.getLogger("crucible.api")
logging.basicConfig(level=os.environ.get("CRUCIBLE_LOG_LEVEL", "INFO"))

app = FastAPI(title="Crucible API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict[str, Any]] = {}


class ScoreRequest(BaseModel):
    repo_id: str
    n_episodes: int = Field(default=DEFAULT_CONFIG.max_episodes_per_run, ge=1, le=200)
    use_cache: bool = True
    frames_per_episode: int | None = None


class PushRequest(BaseModel):
    job_id: str
    threshold: float = Field(ge=0, le=10)
    target_repo: str
    hf_token: str
    source_repo: str | None = None  # optional override; otherwise pulled from job state


def _new_job_id() -> str:
    return f"job_{secrets.token_hex(4)}"


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "default_model": DEFAULT_CONFIG.vlm_model}


@app.get("/jobs")
async def list_jobs() -> dict:
    return {
        job_id: {
            "repo_id": j["repo_id"],
            "status": j["status"],
            "n_done": len(j["results"]),
            "n_target": j["n_target"],
        }
        for job_id, j in JOBS.items()
    }


@app.post("/score_dataset")
async def score_endpoint(req: ScoreRequest) -> dict:
    job_id = _new_job_id()
    queue: asyncio.Queue = asyncio.Queue()
    cfg = CrucibleConfig()
    cfg.max_episodes_per_run = req.n_episodes
    if req.frames_per_episode:
        cfg.frames_per_episode = int(req.frames_per_episode)

    JOBS[job_id] = {
        "queue": queue,
        "results": [],
        "repo_id": req.repo_id,
        "n_target": req.n_episodes,
        "status": "running",
        "error": None,
    }

    async def progress_cb(i: int, total: int, ep: dict) -> None:
        JOBS[job_id]["results"].append(ep)
        await queue.put({"type": "episode", "i": i, "total": total, "episode": ep})

    async def runner() -> None:
        try:
            await score_dataset(req.repo_id, cfg, progress_callback=progress_cb, use_cache=req.use_cache)
            JOBS[job_id]["status"] = "complete"
        except Exception as exc:
            logger.exception("Job %s crashed: %s", job_id, exc)
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(exc)
            await queue.put({"type": "error", "error": str(exc)})
        await queue.put({"type": "done"})

    asyncio.create_task(runner())
    return {"job_id": job_id, "repo_id": req.repo_id, "n_target": req.n_episodes}


@app.get("/progress/{job_id}")
async def progress_stream(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, f"unknown job_id {job_id}")
    queue: asyncio.Queue = JOBS[job_id]["queue"]

    async def event_gen():
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=300)
            except TimeoutError:
                yield {"event": "ping", "data": "{}"}
                continue
            yield {"data": json.dumps(msg, default=str)}
            if msg.get("type") == "done":
                break

    return EventSourceResponse(event_gen())


@app.get("/results/{job_id}")
async def results_endpoint(job_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(404, f"unknown job_id {job_id}")
    j = JOBS[job_id]
    return {"status": j["status"], "results": j["results"], "error": j["error"]}


@app.post("/push_filtered")
async def push_endpoint(req: PushRequest) -> dict:
    job = JOBS.get(req.job_id)
    if not job:
        # Allow pushing from a precache when there is no live job.
        if not req.source_repo:
            raise HTTPException(404, f"unknown job_id {req.job_id} and no source_repo provided")
        results = load_precached(DEFAULT_CONFIG.cache_dir, req.source_repo) or []
        source_repo = req.source_repo
    else:
        results = job["results"]
        source_repo = req.source_repo or job["repo_id"]

    if not results:
        raise HTTPException(400, "no results to filter")

    return await push_filtered_to_hub(
        source_repo=source_repo,
        results=results,
        threshold=req.threshold,
        target_repo=req.target_repo,
        hf_token=req.hf_token,
    )


def main() -> None:
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=os.environ.get("CRUCIBLE_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("CRUCIBLE_API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
