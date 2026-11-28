"""Integration tests for GigBid FastAPI REST & SSE endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app


@pytest.fixture(autouse=True)
def _isolate_api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROFILE", "offline_demo")
    monkeypatch.setenv("GIGBID_USE_MCP", "false")
    monkeypatch.setenv("GIGSIM_DB_PATH", str(tmp_path / "gigsim_api.db"))
    monkeypatch.setenv("WORKSPACE_DB_PATH", str(tmp_path / "workspace_api.db"))
    monkeypatch.setenv("GIGBID_MEMORY_PATH", str(tmp_path / "memory_api.json"))
    monkeypatch.setenv("MCP_AUDIT_LOG_PATH", str(tmp_path / "mcp_audit.jsonl"))


def test_api_health_jobs_and_observability() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    jobs = client.get("/jobs")
    assert jobs.status_code == 200
    assert jobs.json()["total"] >= 12

    mem = client.get("/memory")
    assert mem.status_code == 200
    assert len(mem.json()["portfolio_items"]) >= 10

    evals = client.get("/evals/results")
    assert evals.status_code == 200
    assert "rag_comparison" in evals.json()


def test_api_bid_run_interrupt_and_approve_flow() -> None:
    client = TestClient(app)

    start_resp = client.post(
        "/runs/bid?wait=true",
        json={
            "job_id": "job_001",
            "freelancer_id": "persona_aiml_01",
            "model_profile": "offline_demo",
        },
    )
    assert start_resp.status_code == 200
    run = start_resp.json()
    assert run["awaiting_approval"] is True
    assert run["status"] == "awaiting_human_approval"
    thread_id = run["thread_id"]

    approve_resp = client.post(
        f"/runs/{thread_id}/approve",
        json={"approved": True},
    )
    assert approve_resp.status_code == 200
    resumed = approve_resp.json()
    assert resumed["awaiting_approval"] is False
    assert resumed["state_snapshot"]["submission_result"]["connects_charged"] == 8

    logs_resp = client.get("/audit-logs")
    assert logs_resp.status_code == 200
    assert logs_resp.json()["total"] >= 1
