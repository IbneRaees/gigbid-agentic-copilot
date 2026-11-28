"""Job Scout Node: queries Marketplace MCP read-only tools & wraps untrusted job data."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent.llm import ModelRouter
from agent.marketplace_port import build_marketplace_adapter
from agent.memory.store import GigBidMemoryStore
from agent.nodes._runtime import get_mcp_client_manager
from agent.state import BidGraphState, make_budget_delta, make_trace_event

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"


def wrap_untrusted_job_post(job: dict[str, Any]) -> str:
    """Wrap raw marketplace job text in explicit untrusted delimiters (`POL-009`)."""
    title = str(job.get("title", ""))
    desc = str(job.get("description", ""))
    skills = ", ".join(str(s) for s in job.get("required_skills", []))
    questions = "\n".join(f"- {q}" for q in job.get("screening_questions", []))
    return (
        "<UNTRUSTED_JOB_POST>\n"
        f"TITLE: {title}\n"
        f"SKILLS: {skills}\n"
        f"DESCRIPTION:\n{desc}\n"
        f"SCREENING_QUESTIONS:\n{questions or 'None'}\n"
        "</UNTRUSTED_JOB_POST>"
    )


def load_freelancer_profile(freelancer_id: str, data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Load freelancer persona metadata from `data/freelancers.json`."""
    path = Path(data_dir) / "freelancers.json"
    if path.exists():
        profiles = json.loads(path.read_text(encoding="utf-8"))
        for prof in profiles:
            if prof.get("freelancer_id") == freelancer_id:
                return dict(prof)
        if profiles:
            return dict(profiles[0])
    return {
        "freelancer_id": freelancer_id,
        "name": "GigBid Freelancer",
        "headline": "Senior AI & Full-Stack Engineer",
        "hourly_rate_usd": 80.0,
        "min_fixed_budget_usd": 750.0,
        "connects_balance": 80,
        "skills": ["Python", "FastAPI", "LangGraph", "RAG", "PostgreSQL"],
        "style_preferences": ["Cite verified evidence_ids and measurable outcomes."],
    }


def _load_local_job_and_client(
    job_id: str, client_id: str | None = None, data_dir: Path = DATA_DIR
) -> tuple[dict[str, Any], dict[str, Any]]:
    jobs_path = Path(data_dir) / "jobs.json"
    clients_path = Path(data_dir) / "client_history.json"

    raw_job: dict[str, Any] = {}
    if jobs_path.exists():
        for j in json.loads(jobs_path.read_text(encoding="utf-8")):
            if j.get("job_id") == job_id:
                raw_job = dict(j)
                break
    if not raw_job:
        raise KeyError(f"Job '{job_id}' not found in marketplace or local seed data.")

    resolved_client_id = client_id or str(raw_job.get("client_id", ""))
    client_hist: dict[str, Any] = {
        "client_id": resolved_client_id or "unknown_client",
        "name": "Marketplace Client",
        "payment_verified": False,
        "rating": 0.0,
        "hire_rate": 0.0,
        "total_spent_usd": 0.0,
        "past_reviews": [],
    }
    if clients_path.exists() and resolved_client_id:
        for c in json.loads(clients_path.read_text(encoding="utf-8")):
            if c.get("client_id") == resolved_client_id:
                client_hist = dict(c)
                break
    return raw_job, client_hist


async def _fetch_via_mcp_or_adapter(
    *,
    job_id: str,
    freelancer_id: str,
    marketplace_mode: str,
    trace_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Attempt Marketplace MCP read-only calls first, falling back to adapter/seed data."""
    mcp_mgr = get_mcp_client_manager(trace_id=trace_id, user_id=freelancer_id)
    if mcp_mgr is not None:
        try:
            async with mcp_mgr as mcp:
                job_data = await mcp.call_tool("marketplace", "get_job", {"job_id": job_id})
                client_id = str(job_data.get("client_id", ""))
                client_data = await mcp.call_tool(
                    "marketplace", "get_client_history", {"client_id": client_id}
                )
                connects_data = await mcp.call_tool(
                    "marketplace", "get_connects_balance", {"freelancer_id": freelancer_id}
                )
                return job_data, client_data, connects_data, "marketplace_mcp"
        except Exception:
            pass

    # Fallback 1: Direct MarketplacePort adapter (GigSim HTTP if running)
    try:
        adapter = build_marketplace_adapter(marketplace_mode)
        job_obj = await adapter.get_job(job_id)
        client_obj = await adapter.get_client_history(job_obj.client_id)
        connects_obj = await adapter.get_connects_balance(freelancer_id)
        return (
            job_obj.model_dump(mode="json"),
            client_obj.model_dump(mode="json"),
            connects_obj.model_dump(mode="json"),
            f"adapter:{marketplace_mode}",
        )
    except Exception:
        pass

    # Fallback 2: Deterministic local seed store (offline-safe)
    raw_job, client_hist = _load_local_job_and_client(job_id)
    profile = load_freelancer_profile(freelancer_id)
    connects_data = {
        "freelancer_id": freelancer_id,
        "balance": int(profile.get("connects_balance", 80)),
        "weekly_budget": int(profile.get("weekly_connects_budget", 60)),
        "recent_ledger": [],
        "source_mode": marketplace_mode,
    }
    return raw_job, client_hist, connects_data, "local_seed"


async def scout_jobs(
    *,
    q: str | None = None,
    skill: str | None = None,
    min_budget: float | None = None,
    domain: str | None = None,
    limit: int = 25,
    freelancer_id: str = "persona_aiml_01",
    trace_id: str = "trace-scout",
) -> list[dict[str, Any]]:
    """Query Marketplace MCP `search_jobs` (read-only) with local seed fallback."""
    mcp_mgr = get_mcp_client_manager(trace_id=trace_id, user_id=freelancer_id)
    if mcp_mgr is not None:
        try:
            async with mcp_mgr as mcp:
                res = await mcp.call_tool(
                    "marketplace",
                    "search_jobs",
                    {
                        "q": q,
                        "skill": skill,
                        "min_budget": min_budget,
                        "domain": domain,
                        "limit": limit,
                    },
                )
                return list(res.get("jobs", []))
        except Exception:
            pass

    try:
        adapter = build_marketplace_adapter()
        res = await adapter.search_jobs(
            q=q, skill=skill, min_budget=min_budget, domain=domain, limit=limit
        )
        return [j.model_dump(mode="json") for j in res.jobs]
    except Exception:
        jobs_path = DATA_DIR / "jobs.json"
        if not jobs_path.exists():
            return []
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
        filtered: list[dict[str, Any]] = []
        for j in jobs:
            if q and q.lower() not in f"{j.get('title', '')} {j.get('description', '')}".lower():
                continue
            if skill and not any(skill.lower() in s.lower() for s in j.get("required_skills", [])):
                continue
            if min_budget is not None and float(j.get("budget", {}).get("max", 0)) < min_budget:
                continue
            if domain and j.get("domain_category") != domain:
                continue
            filtered.append(j)
        return filtered[:limit]


async def scout_node(state: BidGraphState) -> dict[str, Any]:
    """Fetch job details, client history, Connects balance, and freelancer profile."""
    t0 = time.perf_counter()
    job_id = state["job_id"]
    freelancer_id = state.get("freelancer_id", "persona_aiml_01")
    marketplace_mode = state.get("marketplace_mode", "gigsim")
    trace_id = state.get("trace_id", "trace-local")

    router = ModelRouter(profile=state.get("model_profile"))
    router.check_budget(state)

    raw_job = dict(state.get("raw_job") or {})
    client_history = dict(state.get("client_history") or {})
    connects_balance = dict(state.get("connects_balance") or {})
    source = "state_preloaded"

    if not raw_job or not client_history or not connects_balance:
        fetched_job, fetched_client, fetched_connects, source = await _fetch_via_mcp_or_adapter(
            job_id=job_id,
            freelancer_id=freelancer_id,
            marketplace_mode=marketplace_mode,
            trace_id=trace_id,
        )
        if not raw_job:
            raw_job = fetched_job
        if not client_history:
            client_history = fetched_client
        if not connects_balance:
            connects_balance = fetched_connects

    freelancer_profile = dict(state.get("freelancer_profile") or {})
    if not freelancer_profile:
        freelancer_profile = load_freelancer_profile(freelancer_id)

    wrapped_job_text = wrap_untrusted_job_post(raw_job)

    # Retrieve relevant promoted semantic memories for this freelancer & job domain
    memory_store = GigBidMemoryStore()
    semantic_hits = memory_store.query_semantic(
        freelancer_id=freelancer_id,
        query=f"{raw_job.get('title', '')} {' '.join(raw_job.get('required_skills', []))}",
        limit=4,
    )
    semantic_memories = [m.model_dump(mode="json") for m in semantic_hits]

    duration_ms = (time.perf_counter() - t0) * 1000.0
    event = make_trace_event(
        node="scout",
        phase="completed",
        duration_ms=duration_ms,
        summary=(
            f"Scouted job {job_id} via {source} | client={client_history.get('client_id')} "
            f"| connects_balance={connects_balance.get('balance', 0)}"
        ),
        rag_evidence_ids=[],
    )
    return {
        "raw_job": raw_job,
        "wrapped_job_text": wrapped_job_text,
        "client_history": client_history,
        "freelancer_profile": freelancer_profile,
        "connects_balance": connects_balance,
        "semantic_memories": semantic_memories,
        "budgets": make_budget_delta(steps=1),
        "node_timings": {"scout": round(duration_ms, 2)},
        "trace_events": [event],
        "status": "scouted",
    }


__all__ = ["scout_node", "scout_jobs", "wrap_untrusted_job_post", "load_freelancer_profile"]
