"""FastAPI REST Server for GigSim Marketplace Simulator (Port 8081).

Implements:
  - Bearer token authentication (`GIGSIM_API_TOKEN`)
  - Sliding-window per-token rate limiting
  - Deterministic fault-injection header (`X-GigSim-Simulate-Error`) for resilience testing
  - Full read/write marketplace API backed by SQLite (`GigSimStore`)
"""

from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gigsim import __version__
from gigsim.models import (
    ClientRecord,
    ConnectsBalanceResponse,
    HealthResponse,
    JobListResponse,
    JobRecord,
    ProposalDraftRequest,
    ProposalResponse,
    ProposalSubmitRequest,
    SendMessageRequest,
    SendMessageResponse,
    ThreadResponse,
)
from gigsim.store import GigSimStore

EXPECTED_TOKEN = os.getenv("GIGSIM_API_TOKEN", "gigsim-dev-token-2026")
RATE_LIMIT_RPM = int(os.getenv("GIGSIM_RATE_LIMIT_RPM", "120"))

bearer_scheme = HTTPBearer(auto_error=False)
_request_timestamps: dict[str, deque[float]] = defaultdict(deque)
_store: GigSimStore | None = None


def get_store() -> GigSimStore:
    global _store
    if _store is None:
        _store = GigSimStore()
    return _store


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    get_store()
    yield


app = FastAPI(
    title="GigSim Marketplace Simulator",
    description="Deterministic freelance marketplace API simulator for GigBid (AI-4016).",
    version=__version__,
    lifespan=lifespan,
)


def verify_auth_and_rate_limit(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_gigsim_simulate_error: str | None = Header(default=None),
) -> str:
    """Enforce Bearer token authentication, sliding-window rate limits, and optional fault injection."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    if not secrets.compare_digest(token, EXPECTED_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid GIGSIM_API_TOKEN.",
        )

    # Support deterministic fault injection in integration/viva tests
    if x_gigsim_simulate_error == "429":
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Simulated GigSim rate limit exceeded.",
            headers={"Retry-After": "2"},
        )
    if x_gigsim_simulate_error == "503":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Simulated GigSim upstream service unavailable.",
        )

    # Sliding window rate limiter (per token, 60-second window)
    now = time.monotonic()
    window = _request_timestamps[token]
    while window and now - window[0] > 60.0:
        window.popleft()
    if len(window) >= RATE_LIMIT_RPM:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({RATE_LIMIT_RPM} requests/minute).",
            headers={"Retry-After": "5"},
        )
    window.append(now)
    return token


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(store: GigSimStore = Depends(get_store)) -> HealthResponse:
    jobs_cnt, clients_cnt, threads_cnt = store.get_counts()
    return HealthResponse(
        status="ok",
        version=__version__,
        jobs_count=jobs_cnt,
        clients_count=clients_cnt,
        threads_count=threads_cnt,
    )


@app.get("/jobs", response_model=JobListResponse, tags=["Marketplace Read"])
def list_jobs(
    q: str | None = Query(default=None, description="Keyword search across title and description"),
    skill: str | None = Query(default=None, description="Filter by required or nice-to-have skill"),
    min_budget: float | None = Query(default=None, ge=0.0),
    max_budget: float | None = Query(default=None, ge=0.0),
    domain: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _: str = Depends(verify_auth_and_rate_limit),
    store: GigSimStore = Depends(get_store),
) -> JobListResponse:
    return store.list_jobs(
        q=q,
        skill=skill,
        min_budget=min_budget,
        max_budget=max_budget,
        domain=domain,
        limit=limit,
        offset=offset,
    )


@app.get("/jobs/{job_id}", response_model=JobRecord, tags=["Marketplace Read"])
def get_job(
    job_id: str,
    _: str = Depends(verify_auth_and_rate_limit),
    store: GigSimStore = Depends(get_store),
) -> JobRecord:
    return store.get_job(job_id)


@app.get("/clients/{client_id}", response_model=ClientRecord, tags=["Marketplace Read"])
def get_client(
    client_id: str,
    _: str = Depends(verify_auth_and_rate_limit),
    store: GigSimStore = Depends(get_store),
) -> ClientRecord:
    return store.get_client(client_id)


@app.get("/account/connects", response_model=ConnectsBalanceResponse, tags=["Account"])
def get_account_connects(
    freelancer_id: str = Query(default="persona_aiml_01"),
    _: str = Depends(verify_auth_and_rate_limit),
    store: GigSimStore = Depends(get_store),
) -> ConnectsBalanceResponse:
    return store.get_connects_balance(freelancer_id=freelancer_id)


@app.post("/proposals/draft", response_model=ProposalResponse, tags=["Proposals"])
def draft_proposal(
    body: ProposalDraftRequest,
    _: str = Depends(verify_auth_and_rate_limit),
    store: GigSimStore = Depends(get_store),
) -> ProposalResponse:
    return store.create_proposal_draft(body)


@app.post("/proposals/submit", response_model=ProposalResponse, tags=["Proposals"])
def submit_proposal(
    body: ProposalSubmitRequest,
    _: str = Depends(verify_auth_and_rate_limit),
    store: GigSimStore = Depends(get_store),
) -> ProposalResponse:
    return store.submit_proposal(body)


@app.get("/threads", response_model=list[ThreadResponse], tags=["Messaging"])
def list_threads(
    _: str = Depends(verify_auth_and_rate_limit),
    store: GigSimStore = Depends(get_store),
) -> list[ThreadResponse]:
    return store.list_threads()


@app.get("/threads/{thread_id}/messages", response_model=ThreadResponse, tags=["Messaging"])
def get_thread_messages(
    thread_id: str,
    _: str = Depends(verify_auth_and_rate_limit),
    store: GigSimStore = Depends(get_store),
) -> ThreadResponse:
    return store.get_thread_messages(thread_id)


@app.post("/threads/{thread_id}/messages", response_model=SendMessageResponse, tags=["Messaging"])
def send_thread_message(
    thread_id: str,
    body: SendMessageRequest,
    _: str = Depends(verify_auth_and_rate_limit),
    store: GigSimStore = Depends(get_store),
) -> SendMessageResponse:
    return store.post_thread_message(thread_id, body)