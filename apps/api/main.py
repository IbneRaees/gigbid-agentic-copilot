"""FastAPI Application Entrypoint for GigBid (Port 8000)."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.llm import resolve_model_profile
from apps.api.routers import jobs_router, observability_router, runs_router
from apps.api.schemas import HealthResponse
from apps.api.streaming import get_run_registry

load_dotenv()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(
    title="GigBid Agentic Freelancer Copilot API",
    description="Multi-agent LangGraph orchestration, SSE streaming, HITL approvals, Memory & Eval API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)
app.include_router(runs_router)
app.include_router(observability_router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    registry = get_run_registry()
    all_runs = registry.list_runs()
    pending = sum(1 for r in all_runs if r.get("awaiting_approval"))
    active = sum(1 for r in all_runs if r.get("status") == "running")
    return HealthResponse(
        status="ok",
        marketplace_mode=os.getenv("MARKETPLACE_MODE", "gigsim"),
        model_profile=resolve_model_profile(),
        active_runs=active,
        pending_approvals=pending,
    )
