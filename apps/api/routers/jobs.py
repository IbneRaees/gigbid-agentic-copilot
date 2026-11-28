"""Jobs, Threads & Freelancers Router for GigBid API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from agent.marketplace_port import build_marketplace_adapter
from agent.nodes.scout import scout_jobs
from apps.api.auth import verify_optional_api_token

router = APIRouter(tags=["Jobs & Marketplace"])

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT_DIR / "data"


@router.get("/jobs")
async def list_jobs(
    q: str | None = Query(default=None),
    skill: str | None = Query(default=None),
    min_budget: float | None = Query(default=None, ge=0.0),
    domain: str | None = Query(default=None),
    limit: int = Query(default=48, ge=1, le=100),
    freelancer_id: str = Query(default="persona_aiml_01"),
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """Browse marketplace jobs via Marketplace MCP / GigSim / local seed fallback."""
    jobs = await scout_jobs(
        q=q,
        skill=skill,
        min_budget=min_budget,
        domain=domain,
        limit=limit,
        freelancer_id=freelancer_id,
    )
    return {
        "total": len(jobs),
        "jobs": jobs,
        "marketplace_mode": os.getenv("MARKETPLACE_MODE", "gigsim"),
    }


@router.get("/jobs/{job_id}")
async def get_job_detail(
    job_id: str,
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """Retrieve a single job and its associated client history."""
    jobs_path = DATA_DIR / "jobs.json"
    clients_path = DATA_DIR / "client_history.json"
    job_data: dict[str, Any] | None = None

    try:
        adapter = build_marketplace_adapter()
        j = await adapter.get_job(job_id)
        c = await adapter.get_client_history(j.client_id)
        return {"job": j.model_dump(mode="json"), "client": c.model_dump(mode="json")}
    except Exception:
        pass

    if jobs_path.exists():
        for item in json.loads(jobs_path.read_text(encoding="utf-8")):
            if item.get("job_id") == job_id:
                job_data = item
                break
    if job_data is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    client_data: dict[str, Any] = {}
    if clients_path.exists():
        for c_item in json.loads(clients_path.read_text(encoding="utf-8")):
            if c_item.get("client_id") == job_data.get("client_id"):
                client_data = c_item
                break
    return {"job": job_data, "client": client_data}


@router.get("/threads")
async def list_threads(
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """List active client conversation threads."""
    try:
        adapter = build_marketplace_adapter()
        threads = await adapter.list_messages()
        return {
            "total": len(threads),
            "threads": [t.model_dump(mode="json") for t in threads],
        }
    except Exception:
        msg_path = DATA_DIR / "messages.json"
        threads_raw = json.loads(msg_path.read_text(encoding="utf-8")) if msg_path.exists() else []
        return {"total": len(threads_raw), "threads": threads_raw}


@router.get("/freelancers")
async def list_freelancers(
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """List freelancer personas and live Connects balances."""
    fl_path = DATA_DIR / "freelancers.json"
    freelancers = json.loads(fl_path.read_text(encoding="utf-8")) if fl_path.exists() else []

    try:
        adapter = build_marketplace_adapter()
        for fl in freelancers:
            bal = await adapter.get_connects_balance(fl["freelancer_id"])
            fl["connects_balance"] = bal.balance
    except Exception:
        pass

    return {"total": len(freelancers), "freelancers": freelancers}
