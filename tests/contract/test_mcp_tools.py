"""Contract & Security Test Suite for GigBid Phase 2B MCP Servers.

Verifies:
  1. Exact MCP tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`)
     across Portfolio, Marketplace, and Workspace MCP servers.
  2. Portfolio MCP read-only tools (`search_evidence`, `get_case_study`, `list_skills`),
     resource templates (`portfolio://case/{id}`, `portfolio://proposal/{id}`), and
     prompt templates (`answer_screening_question`).
  3. Dry-run non-mutation across Marketplace (`submit_proposal`, `send_message`) and
     Workspace (`create_tentative_interview_hold`, `create_email_draft`) write tools.
  4. Idempotency-key deduplication (no double Connects deduction or duplicate records)
     and payload-mismatch conflict detection.
  5. HTTP transport security middleware: Bearer-token auth (`401`/`403`), sliding-window
     rate limiting (`429`), and PII-scrubbed JSONL audit logging (`logs/mcp_audit.jsonl`).
  6. Upwork Live draft-only guardrail (`MARKETPLACE_MODE=upwork_live` never executes live
     submissions or spends Connects even when `dry_run=False`).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx
from mcp import ClientSession
from mcp.types import Implementation
from starlette.testclient import TestClient

from agent.mcp_client import GigBidMCPClientManager
from gigsim.app import app as gigsim_fastapi_app
from gigsim.app import get_store
from gigsim.store import GigSimStore
from mcp_servers.marketplace.adapters.gigsim import GigSimMarketplaceAdapter
from mcp_servers.marketplace.adapters.upwork_live import UpworkLiveMarketplaceAdapter
from mcp_servers.marketplace.server import (
    build_marketplace_mcp_server,
)
from mcp_servers.marketplace.server import (
    create_app as create_marketplace_http_app,
)
from mcp_servers.portfolio.server import build_portfolio_mcp_server
from mcp_servers.security import (
    MCPAuditLogger,
    MCPIdempotencyStore,
    MCPServer,
    SlidingWindowRateLimiter,
)
from mcp_servers.workspace.server import build_workspace_mcp_server

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"


@asynccontextmanager
async def open_in_memory_mcp_session(
    server: MCPServer,
) -> AsyncGenerator[ClientSession, None]:
    """Connect an official MCP `ClientSession` to an in-process `MCPServer` (mcp 1.x & 2.x)."""
    try:
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(server) as session:
            yield session
            return
    except ImportError:
        pass

    # Fallback for mcp >= 2.0.0 using anyio memory object streams
    c2s_send, c2s_recv = anyio.create_memory_object_stream(100)
    s2c_send, s2c_recv = anyio.create_memory_object_stream(100)

    low_level = server._mcp_server
    init_opts = low_level.create_initialization_options()

    async with anyio.create_task_group() as tg:
        tg.start_soon(low_level.run, c2s_recv, s2c_send, init_opts)
        async with ClientSession(
            s2c_recv,
            c2s_send,
            client_info=Implementation(name="gigbid-contract-test", version="0.2.0"),
        ) as session:
            await session.initialize()
            yield session
        tg.cancel_scope.cancel()


def _extract_tool_json(call_result: Any) -> dict[str, Any]:
    """Extract structured JSON dict from an MCP `CallToolResult` across SDK versions."""
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for block in getattr(call_result, "content", []):
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            return json.loads(text)
    raise AssertionError(f"Unable to extract JSON payload from CallToolResult: {call_result}")


def _get_annotation_flags(tool_obj: Any) -> tuple[bool | None, bool | None, bool | None]:
    ann = getattr(tool_obj, "annotations", None)
    if ann is None:
        return None, None, None
    ro = getattr(ann, "readOnlyHint", None)
    if ro is None:
        ro = getattr(ann, "read_only_hint", None)
    dest = getattr(ann, "destructiveHint", None)
    if dest is None:
        dest = getattr(ann, "destructive_hint", None)
    idem = getattr(ann, "idempotentHint", None)
    if idem is None:
        idem = getattr(ann, "idempotent_hint", None)
    return ro, dest, idem


def _build_isolated_gigsim_adapter(tmp_path: Path) -> tuple[GigSimMarketplaceAdapter, GigSimStore]:
    """Create an isolated SQLite-backed GigSim instance wired via in-memory `httpx.ASGITransport`."""
    store = GigSimStore(db_path=tmp_path / "gigsim_test.db", data_dir=DATA_DIR)
    gigsim_fastapi_app.dependency_overrides[get_store] = lambda: store
    transport = httpx.ASGITransport(app=gigsim_fastapi_app)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://gigsim.test")
    adapter = GigSimMarketplaceAdapter(
        base_url="http://gigsim.test",
        api_token="gigsim-dev-token-2026",
        client=async_client,
    )
    return adapter, store


# ==============================================================================
# 1. MCP TOOL ANNOTATIONS CONTRACT TESTS
# ==============================================================================


def test_marketplace_tool_annotations(tmp_path: Path) -> None:
    async def _run() -> None:
        adapter, _ = _build_isolated_gigsim_adapter(tmp_path)
        server = build_marketplace_mcp_server(
            adapter=adapter,
            audit_logger=MCPAuditLogger(tmp_path / "audit.jsonl"),
            idempotency_store=MCPIdempotencyStore(tmp_path / "idem.db"),
        )
        async with open_in_memory_mcp_session(server) as session:
            tools_resp = await session.list_tools()
            by_name = {t.name: t for t in tools_resp.tools}

            expected = {
                "search_jobs": (True, False, True),
                "get_job": (True, False, True),
                "get_client_history": (True, False, True),
                "get_connects_balance": (True, False, True),
                "list_messages": (True, False, True),
                "create_proposal_draft": (False, False, True),
                "submit_proposal": (False, True, True),
                "send_message": (False, True, True),
            }
            assert set(expected.keys()) <= set(by_name.keys())
            for name, expected_flags in expected.items():
                assert _get_annotation_flags(by_name[name]) == expected_flags, (
                    f"Annotation mismatch on marketplace tool '{name}'"
                )

    asyncio.run(_run())


def test_portfolio_and_workspace_tool_annotations(tmp_path: Path) -> None:
    async def _run() -> None:
        port_server = build_portfolio_mcp_server(
            data_dir=DATA_DIR,
            audit_logger=MCPAuditLogger(tmp_path / "port_audit.jsonl"),
        )
        async with open_in_memory_mcp_session(port_server) as session:
            tools_resp = await session.list_tools()
            by_name = {t.name: t for t in tools_resp.tools}
            for name in ("search_evidence", "get_case_study", "list_skills"):
                assert name in by_name
                assert _get_annotation_flags(by_name[name]) == (True, False, True)

        ws_server = build_workspace_mcp_server(
            db_path=tmp_path / "ws.db",
            audit_logger=MCPAuditLogger(tmp_path / "ws_audit.jsonl"),
            idempotency_store=MCPIdempotencyStore(tmp_path / "ws_idem.db"),
        )
        async with open_in_memory_mcp_session(ws_server) as session:
            tools_resp = await session.list_tools()
            by_name = {t.name: t for t in tools_resp.tools}
            assert _get_annotation_flags(by_name["get_free_calendar_slots"]) == (
                True,
                False,
                True,
            )
            assert _get_annotation_flags(by_name["create_tentative_interview_hold"]) == (
                False,
                False,
                True,
            )
            assert _get_annotation_flags(by_name["create_email_draft"]) == (
                False,
                False,
                True,
            )

    asyncio.run(_run())


# ==============================================================================
# 2. PORTFOLIO MCP TOOLS, RESOURCES & PROMPTS CONTRACT TESTS
# ==============================================================================


def test_portfolio_mcp_tools_resources_and_prompts(tmp_path: Path) -> None:
    async def _run() -> None:
        server = build_portfolio_mcp_server(
            data_dir=DATA_DIR,
            audit_logger=MCPAuditLogger(tmp_path / "portfolio_audit.jsonl"),
        )
        async with open_in_memory_mcp_session(server) as session:
            # 1. search_evidence
            search_res = await session.call_tool(
                "search_evidence",
                {
                    "requirement": "LangGraph Postgres human-in-the-loop contract review",
                    "k": 3,
                    "filters": {"freelancer_id": "persona_aiml_01"},
                },
            )
            search_data = _extract_tool_json(search_res)
            assert search_data["total_returned"] == 3
            assert search_data["matches"][0]["evidence_id"] == "ev_aiml_001"

            # 2. get_case_study
            case_res = await session.call_tool("get_case_study", {"case_id": "ev_aiml_001"})
            case_data = _extract_tool_json(case_res)
            assert case_data["evidence_id"] == "ev_aiml_001"
            assert "62%" in case_data["full_text"]

            # 3. list_skills
            skills_res = await session.call_tool(
                "list_skills", {"freelancer_id": "persona_aiml_01"}
            )
            skills_data = _extract_tool_json(skills_res)
            assert skills_data["total_skills"] >= 15

            # 4. Resource templates: portfolio://case/{id} and portfolio://proposal/{id}
            case_resource = await session.read_resource("portfolio://case/ev_aiml_002")
            case_text = case_resource.contents[0].text
            assert "ev_aiml_002" in case_text
            assert "0.89" in case_text

            prop_resource = await session.read_resource("portfolio://proposal/ev_aiml_004")
            prop_text = prop_resource.contents[0].text
            assert "ev_aiml_004" in prop_text

            # 5. Prompt template: answer_screening_question
            prompt_res = await session.get_prompt(
                "answer_screening_question",
                arguments={
                    "question": "How do you prevent hallucinations in RAG outputs?",
                    "job_context": "Clinical RAG assistant",
                    "freelancer_id": "persona_aiml_01",
                },
            )
            prompt_text = prompt_res.messages[0].content.text
            assert "ev_aiml_004" in prompt_text or "ev_aiml_001" in prompt_text

    asyncio.run(_run())


# ==============================================================================
# 3. DRY-RUN NON-MUTATION & IDEMPOTENCY DEDUPLICATION TESTS
# ==============================================================================


def test_marketplace_dry_run_and_idempotency(tmp_path: Path) -> None:
    async def _run() -> None:
        audit_path = tmp_path / "market_audit.jsonl"
        adapter, _ = _build_isolated_gigsim_adapter(tmp_path)
        server = build_marketplace_mcp_server(
            adapter=adapter,
            audit_logger=MCPAuditLogger(audit_path),
            idempotency_store=MCPIdempotencyStore(tmp_path / "market_idem.db"),
        )
        cover = (
            "In our LangGraph legal contract copilot (ev_aiml_001), we reduced analyst "
            "review time by 62% with PostgresSaver checkpointing and a 1.8% unsupported claim rate."
        )
        async with open_in_memory_mcp_session(server) as session:
            # Check initial Connects balance (84)
            bal0 = _extract_tool_json(
                await session.call_tool(
                    "get_connects_balance", {"freelancer_id": "persona_aiml_01"}
                )
            )
            assert bal0["balance"] == 84

            # 1. Dry-run preview must NOT deduct Connects or consume idempotency_key
            dry_res = _extract_tool_json(
                await session.call_tool(
                    "submit_proposal",
                    {
                        "idempotency_key": "idem-job001-live",
                        "job_id": "job_001",
                        "freelancer_id": "persona_aiml_01",
                        "cover_letter": cover,
                        "proposed_rate": 3800.0,
                        "cited_evidence_ids": ["ev_aiml_001"],
                        "dry_run": True,
                    },
                )
            )
            assert dry_res["status"] == "dry_run_ok"
            assert dry_res["connects_charged"] == 0
            assert dry_res["dry_run_preview"]["connects_cost"] == 8
            assert dry_res["dry_run_preview"]["remaining_connects_after"] == 76

            bal_after_dry = _extract_tool_json(
                await session.call_tool(
                    "get_connects_balance", {"freelancer_id": "persona_aiml_01"}
                )
            )
            assert bal_after_dry["balance"] == 84

            # 2. Live submission (dry_run=False) deducts 8 Connects (84 -> 76)
            live_1 = _extract_tool_json(
                await session.call_tool(
                    "submit_proposal",
                    {
                        "idempotency_key": "idem-job001-live",
                        "job_id": "job_001",
                        "freelancer_id": "persona_aiml_01",
                        "cover_letter": cover,
                        "proposed_rate": 3800.0,
                        "cited_evidence_ids": ["ev_aiml_001"],
                        "dry_run": False,
                    },
                )
            )
            assert live_1["connects_charged"] == 8
            assert live_1["connects_remaining"] == 76
            assert live_1["status"] in {"submitted", "replied", "interview"}

            # 3. Retried live submission with identical idempotency_key deduplicates cleanly
            live_2 = _extract_tool_json(
                await session.call_tool(
                    "submit_proposal",
                    {
                        "idempotency_key": "idem-job001-live",
                        "job_id": "job_001",
                        "freelancer_id": "persona_aiml_01",
                        "cover_letter": cover,
                        "proposed_rate": 3800.0,
                        "cited_evidence_ids": ["ev_aiml_001"],
                        "dry_run": False,
                    },
                )
            )
            assert live_2["proposal_id"] == live_1["proposal_id"]
            assert live_2["connects_remaining"] == 76

            bal_final = _extract_tool_json(
                await session.call_tool(
                    "get_connects_balance", {"freelancer_id": "persona_aiml_01"}
                )
            )
            assert bal_final["balance"] == 76

            # 4. Reusing same idempotency_key with different proposed_rate must fail
            conflict_res = await session.call_tool(
                "submit_proposal",
                {
                    "idempotency_key": "idem-job001-live",
                    "job_id": "job_001",
                    "freelancer_id": "persona_aiml_01",
                    "cover_letter": cover,
                    "proposed_rate": 4200.0,
                    "cited_evidence_ids": ["ev_aiml_001"],
                    "dry_run": False,
                },
            )
            assert getattr(conflict_res, "isError", False) is True

    asyncio.run(_run())


def test_workspace_dry_run_and_idempotency(tmp_path: Path) -> None:
    async def _run() -> None:
        ws_db = tmp_path / "workspace.db"
        server = build_workspace_mcp_server(
            db_path=ws_db,
            audit_logger=MCPAuditLogger(tmp_path / "ws_audit.jsonl"),
            idempotency_store=MCPIdempotencyStore(tmp_path / "ws_idem.db"),
        )
        async with open_in_memory_mcp_session(server) as session:
            slots_data = _extract_tool_json(
                await session.call_tool("get_free_calendar_slots", {"days_ahead": 2})
            )
            assert slots_data["total_available"] == 6
            chosen_slot = slots_data["slots"][0]["slot_start_iso"]

            # Dry-run hold does not book slot
            dry_hold = _extract_tool_json(
                await session.call_tool(
                    "create_tentative_interview_hold",
                    {
                        "idempotency_key": "hold-key-1",
                        "thread_id": "thread_001",
                        "job_id": "job_001",
                        "slot_start_iso": chosen_slot,
                        "dry_run": True,
                    },
                )
            )
            assert dry_hold["status"] == "dry_run_ok"
            assert dry_hold["slot_available"] is True

            # Live hold books slot once and deduplicates retries
            hold_1 = _extract_tool_json(
                await session.call_tool(
                    "create_tentative_interview_hold",
                    {
                        "idempotency_key": "hold-key-1",
                        "thread_id": "thread_001",
                        "job_id": "job_001",
                        "slot_start_iso": chosen_slot,
                        "dry_run": False,
                    },
                )
            )
            hold_2 = _extract_tool_json(
                await session.call_tool(
                    "create_tentative_interview_hold",
                    {
                        "idempotency_key": "hold-key-1",
                        "thread_id": "thread_001",
                        "job_id": "job_001",
                        "slot_start_iso": chosen_slot,
                        "dry_run": False,
                    },
                )
            )
            assert hold_1["hold_id"] == hold_2["hold_id"]
            assert hold_1["status"] == "tentative_hold"

            # Free slots decreased from 6 to 5
            slots_after = _extract_tool_json(
                await session.call_tool("get_free_calendar_slots", {"days_ahead": 2})
            )
            assert slots_after["total_available"] == 5

    asyncio.run(_run())


# ==============================================================================
# 4. UPWORK LIVE DRAFT-ONLY GUARDRAIL TEST
# ==============================================================================


def test_upwork_live_draft_only_guardrail(tmp_path: Path) -> None:
    async def _run() -> None:
        live_adapter = UpworkLiveMarketplaceAdapter(access_token="")
        server = build_marketplace_mcp_server(
            adapter=live_adapter,
            mode="upwork_live",
            audit_logger=MCPAuditLogger(tmp_path / "upwork_audit.jsonl"),
            idempotency_store=MCPIdempotencyStore(tmp_path / "upwork_idem.db"),
        )
        async with open_in_memory_mcp_session(server) as session:
            # Even with dry_run=False, submit_proposal MUST NOT submit live or spend Connects
            res = _extract_tool_json(
                await session.call_tool(
                    "submit_proposal",
                    {
                        "idempotency_key": "upwork-live-guardrail-001",
                        "job_id": "job_upwork_999",
                        "freelancer_id": "persona_aiml_01",
                        "cover_letter": "Grounded proposal draft citing ev_aiml_001.",
                        "proposed_rate": 3200.0,
                        "dry_run": False,
                    },
                )
            )
            assert res["status"] == "manual_review_required"
            assert res["requires_manual_submission"] is True
            assert res["connects_charged"] == 0
            assert "upwork.com/ab/proposals/job/job_upwork_999/apply" in (
                res["manual_review_url"] or ""
            )

            # Even with dry_run=False, send_message MUST NOT send externally
            msg_res = _extract_tool_json(
                await session.call_tool(
                    "send_message",
                    {
                        "idempotency_key": "upwork-msg-guardrail-001",
                        "thread_id": "room_888",
                        "text": "Thank you for reaching out!",
                        "dry_run": False,
                    },
                )
            )
            assert msg_res["status"] == "draft_only_manual_review"
            assert msg_res["requires_manual_send"] is True
            assert msg_res["sent_message"] is None

    asyncio.run(_run())


# ==============================================================================
# 5. HTTP SECURITY MIDDLEWARE: BEARER AUTH, RATE LIMITING & PII-SCRUBBED AUDIT LOG
# ==============================================================================


def test_http_bearer_auth_rate_limit_and_audit_log(tmp_path: Path) -> None:
    audit_path = tmp_path / "http_audit.jsonl"
    limiter = SlidingWindowRateLimiter(max_requests_per_minute=2)
    adapter, _ = _build_isolated_gigsim_adapter(tmp_path)

    http_app = create_marketplace_http_app(
        adapter=adapter,
        expected_token="secret-mcp-token",
        rate_limiter=limiter,
        audit_logger=MCPAuditLogger(audit_path),
        idempotency_store=MCPIdempotencyStore(tmp_path / "http_idem.db"),
    )

    with TestClient(http_app) as client:
        # Unauthenticated /health succeeds
        h = client.get("/health")
        assert h.status_code == 200

        # Missing Bearer token -> 401
        unauth = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert unauth.status_code == 401

        # Wrong Bearer token -> 403
        bad = client.post(
            "/mcp",
            headers={"Authorization": "Bearer wrong-token"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        assert bad.status_code == 403

        # Valid Bearer token allowed up to rate limit (2 RPM), 3rd request -> 429
        valid_headers = {
            "Authorization": "Bearer secret-mcp-token",
            "Accept": "application/json, text/event-stream",
            "X-User-Id": "user_rate_test",
            "X-Trace-Id": "trace-abc-123",
        }
        r1 = client.post(
            "/mcp",
            headers=valid_headers,
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        )
        r2 = client.post(
            "/mcp",
            headers=valid_headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
        )
        r3 = client.post(
            "/mcp",
            headers=valid_headers,
            json={"jsonrpc": "2.0", "id": 3, "method": "ping"},
        )
        assert r1.status_code != 429
        assert r2.status_code != 429
        assert r3.status_code == 429
        assert r3.json()["error"] == "rate_limit_exceeded"

    # Verify PII redaction in JSONL audit logger
    logger = MCPAuditLogger(audit_path)
    logger.record(
        server_name="marketplace",
        tool_name="send_message",
        raw_args={
            "text": "Reach me at alice@example.com or +1-555-867-5309",
            "api_key": "super-secret-key",
        },
        result_summary="ok",
        trace_id="trace-pii-check",
        user_id="persona_aiml_01",
    )
    lines = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    last = lines[-1]
    assert last["trace_id"] == "trace-pii-check"
    assert last["sanitized_args"]["api_key"] == "[REDACTED_SECRET]"
    assert "[REDACTED_EMAIL]" in last["sanitized_args"]["text"]
    assert "[REDACTED_PHONE]" in last["sanitized_args"]["text"]
    assert "alice@example.com" not in json.dumps(last)


# ==============================================================================
# 6. UNIFIED MCP CLIENT MANAGER CONFIGURATION TEST
# ==============================================================================


def test_mcp_client_manager_header_and_server_configuration() -> None:
    mgr = GigBidMCPClientManager(
        auth_token="test-token-999",
        github_token="ghp_test_token_123",
        trace_id="trace-mgr-001",
        user_id="persona_fullstack_02",
    )
    conns = mgr.build_server_connections()
    assert set(conns.keys()) == {"portfolio", "marketplace", "workspace", "github"}
    assert conns["marketplace"]["headers"]["Authorization"] == "Bearer test-token-999"
    assert conns["marketplace"]["headers"]["X-Trace-Id"] == "trace-mgr-001"
    assert conns["marketplace"]["headers"]["X-User-Id"] == "persona_fullstack_02"
    assert conns["github"]["transport"] == "stdio"