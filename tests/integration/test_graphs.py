"""End-to-end integration tests for GigBid Phase 4 LangGraph workflows & Memory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from agent.graphs.bid_graph import build_bid_graph
from agent.graphs.checkpointer import LocalSqliteCheckpointSaver
from agent.graphs.conversation_graph import build_conversation_graph
from agent.memory.promotion_rules import evaluate_bid_promotion
from agent.memory.store import EpisodicEvent, GigBidMemoryStore
from agent.state import create_initial_bid_state, create_initial_conversation_state
from gigsim.store import GigSimStore

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"


def _load_job(job_id: str) -> dict[str, Any]:
    jobs = json.loads((DATA_DIR / "jobs.json").read_text(encoding="utf-8"))
    return next(j for j in jobs if j["job_id"] == job_id)


def _load_client(client_id: str) -> dict[str, Any]:
    clients = json.loads((DATA_DIR / "client_history.json").read_text(encoding="utf-8"))
    return next(c for c in clients if c["client_id"] == client_id)


def _load_freelancer(freelancer_id: str) -> dict[str, Any]:
    fls = json.loads((DATA_DIR / "freelancers.json").read_text(encoding="utf-8"))
    return next(f for f in fls if f["freelancer_id"] == freelancer_id)


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force offline_demo mode, disable external network/MCP, and isolate DB paths."""
    monkeypatch.setenv("MODEL_PROFILE", "offline_demo")
    monkeypatch.setenv("GIGBID_USE_MCP", "false")
    monkeypatch.setenv("GIGSIM_DB_PATH", str(tmp_path / "gigsim_test.db"))
    monkeypatch.setenv("WORKSPACE_DB_PATH", str(tmp_path / "workspace_test.db"))
    monkeypatch.setenv("GIGBID_MEMORY_PATH", str(tmp_path / "memory_test.json"))
    monkeypatch.setenv("MCP_AUDIT_LOG_PATH", str(tmp_path / "mcp_audit.jsonl"))
    monkeypatch.setenv("MCP_IDEMPOTENCY_DB_PATH", str(tmp_path / "mcp_idem.db"))


@pytest.mark.parametrize(
    ("job_id", "expected_reason_substr"),
    [
        ("job_005", "off-platform"),
        ("job_006", "free"),
        ("job_008", "mismatch"),
        ("job_012", "skill fit"),
    ],
)
def test_bid_graph_rejects_policy_traps_and_bad_fits(
    tmp_path: Path, job_id: str, expected_reason_substr: str
) -> None:
    saver = LocalSqliteCheckpointSaver(db_path=tmp_path / f"cp_{job_id}.db")
    graph = build_bid_graph(checkpointer=saver)
    state = create_initial_bid_state(
        job_id=job_id,
        freelancer_id="persona_aiml_01",
        model_profile="offline_demo",
    )
    config = {"configurable": {"thread_id": state["thread_id"]}}

    result = asyncio.run(graph.ainvoke(state, config=config))

    assert "__interrupt__" not in result
    assert result["status"] == "skipped"
    assert result["bid_decision"]["action"] == "skip"
    assert result["bid_decision"]["auto_skipped"] is True
    assert expected_reason_substr in result["bid_decision"]["rationale"].lower()


def test_bid_graph_neutralizes_prompt_injection_and_stays_in_budget(tmp_path: Path) -> None:
    saver = LocalSqliteCheckpointSaver(db_path=tmp_path / "cp_inj.db")
    graph = build_bid_graph(checkpointer=saver)
    job = _load_job("job_007")
    state = create_initial_bid_state(
        job_id="job_007",
        freelancer_id="persona_fullstack_02",
        model_profile="offline_demo",
    )
    config = {"configurable": {"thread_id": state["thread_id"]}}

    result = asyncio.run(graph.ainvoke(state, config=config))

    assert result["analyst_report"]["injection_detected"] is True
    assert result["risk_report"]["has_prompt_injection"] is True
    assert result["bid_decision"]["action"] == "bid"
    assert (
        job["budget"]["min"] <= result["price_quote"]["recommended_amount"] <= job["budget"]["max"]
    )
    assert result["price_quote"]["recommended_amount"] < 5000.0
    assert "evil-exfil" not in result["proposal_draft"]["cover_letter"].lower()
    assert result["compliance_verdict"]["passed"] is True
    assert "__interrupt__" in result


def test_bid_graph_interrupt_dry_run_then_human_approve_spends_connects_once(
    tmp_path: Path,
) -> None:
    store = GigSimStore(db_path=tmp_path / "gigsim_test.db")
    initial_balance = store.get_connects_balance("persona_aiml_01").balance

    saver = LocalSqliteCheckpointSaver(db_path=tmp_path / "cp_bid.db")
    graph = build_bid_graph(checkpointer=saver)
    state = create_initial_bid_state(
        job_id="job_001",
        freelancer_id="persona_aiml_01",
        model_profile="offline_demo",
    )
    config = {"configurable": {"thread_id": state["thread_id"]}}

    # Phase 1: Run until mandatory HITL interrupt (dry_run=True preview)
    interrupted = asyncio.run(graph.ainvoke(state, config=config))

    assert "__interrupt__" in interrupted
    interrupt_payload = interrupted["__interrupt__"][0].value
    assert interrupt_payload["action"] == "submit_proposal"
    assert interrupt_payload["dry_run_preview"]["status"] == "dry_run_ok"
    assert interrupt_payload["dry_run_preview"]["connects_charged"] == 0
    assert interrupted["verification_report"]["grounding_score"] >= 0.85
    assert interrupted["compliance_verdict"]["passed"] is True

    # Connects balance MUST NOT change before human approval
    assert store.get_connects_balance("persona_aiml_01").balance == initial_balance

    # Phase 2: Resume with human approval -> executes submit_proposal(dry_run=False)
    resumed = asyncio.run(
        graph.ainvoke(
            Command(
                resume={
                    "approved": True,
                    "cover_letter": interrupt_payload["proposal_draft"]["cover_letter"],
                }
            ),
            config=config,
        )
    )

    assert "__interrupt__" not in resumed
    assert resumed["human_approval"]["approved"] is True
    sub = resumed["submission_result"]
    assert sub["connects_charged"] == 8
    assert sub["connects_remaining"] == initial_balance - 8
    assert store.get_connects_balance("persona_aiml_01").balance == initial_balance - 8
    assert len(resumed["promoted_memories"]) >= 1


def test_bid_graph_human_rejection_leaves_connects_untouched(tmp_path: Path) -> None:
    store = GigSimStore(db_path=tmp_path / "gigsim_test.db")
    initial_balance = store.get_connects_balance("persona_aiml_01").balance

    saver = LocalSqliteCheckpointSaver(db_path=tmp_path / "cp_reject.db")
    graph = build_bid_graph(checkpointer=saver)
    state = create_initial_bid_state(
        job_id="job_002",
        freelancer_id="persona_aiml_01",
        model_profile="offline_demo",
    )
    config = {"configurable": {"thread_id": state["thread_id"]}}

    interrupted = asyncio.run(graph.ainvoke(state, config=config))
    assert "__interrupt__" in interrupted

    resumed = asyncio.run(
        graph.ainvoke(
            Command(resume={"approved": False, "reason": "Saving Connects for higher ROI."}),
            config=config,
        )
    )
    assert resumed["status"] == "rejected_by_human"
    assert store.get_connects_balance("persona_aiml_01").balance == initial_balance


def test_conversation_graph_interview_hold_and_scam_refusal(tmp_path: Path) -> None:
    saver = LocalSqliteCheckpointSaver(db_path=tmp_path / "cp_conv.db")
    graph = build_conversation_graph(checkpointer=saver)

    # 1. Interview request (thread_001) -> checks calendar slots -> interrupt -> approves hold & send
    state1 = create_initial_conversation_state(
        thread_id="thread_001",
        freelancer_id="persona_aiml_01",
        model_profile="offline_demo",
    )
    cfg1 = {"configurable": {"thread_id": f"conv_1_{tmp_path.name}"}}
    interrupted1 = asyncio.run(graph.ainvoke(state1, config=cfg1))

    assert "__interrupt__" in interrupted1
    payload1 = interrupted1["__interrupt__"][0].value
    assert payload1["reply_intent"]["intent"] == "interview_request"
    assert payload1["reply_intent"]["requires_calendar_slots"] is True
    assert len(payload1["available_slots"]) >= 1
    assert payload1["hold_preview"]["status"] == "dry_run_ok"
    assert payload1["send_preview"]["status"] == "dry_run_ok"

    resumed1 = asyncio.run(graph.ainvoke(Command(resume={"approved": True}), config=cfg1))
    assert resumed1["send_result"]["status"] == "sent"
    assert resumed1["hold_result"]["status"] == "tentative_hold"

    # 2. Off-platform scam message (thread_005) -> flags POL-001/002 & drafts safe refusal
    state2 = create_initial_conversation_state(
        thread_id="thread_005",
        freelancer_id="persona_aiml_01",
        model_profile="offline_demo",
    )
    cfg2 = {"configurable": {"thread_id": f"conv_2_{tmp_path.name}"}}
    interrupted2 = asyncio.run(graph.ainvoke(state2, config=cfg2))
    payload2 = interrupted2["__interrupt__"][0].value
    assert payload2["reply_intent"]["intent"] == "scam_or_offplatform"
    assert payload2["reply_intent"]["scam_or_policy_flag"] is True
    assert "exclusively within the platform" in payload2["reply_draft"]["reply_text"].lower()
    assert payload2["compliance_verdict"]["passed"] is True


def test_memory_promotion_rules_gate_ungrounded_events(tmp_path: Path) -> None:
    store = GigBidMemoryStore(store_path=tmp_path / "mem.json")
    bad_event = EpisodicEvent(
        event_id="ep_bad",
        freelancer_id="persona_aiml_01",
        job_id="job_001",
        event_type="proposal_outcome",
        outcome="won",
        summary="Hallucinated claim",
        grounding_score=0.40,
        compliance_passed=True,
        evidence_ids=["ev_aiml_001"],
    )
    bad_decision = evaluate_bid_promotion(bad_event)
    assert bad_decision.promoted is False

    good_event = EpisodicEvent(
        event_id="ep_good",
        freelancer_id="persona_aiml_01",
        job_id="job_001",
        event_type="proposal_outcome",
        outcome="won",
        summary="Grounded legal contract copilot proposal",
        grounding_score=0.95,
        compliance_passed=True,
        evidence_ids=["ev_aiml_001"],
        screening_answers={
            "Describe a LangGraph system you shipped.": "In ev_aiml_001, I architected a 6-node parallel LangGraph workflow with PostgresSaver."
        },
    )
    good_decision = evaluate_bid_promotion(good_event)
    assert good_decision.promoted is True
    assert len(good_decision.promoted_items) == 2
    for item in good_decision.promoted_items:
        store.upsert_semantic(item)
    assert len(store.list_semantic(freelancer_id="persona_aiml_01")) == 2


def test_critic_bounded_revision_loop_and_sanitize_fallback(tmp_path: Path) -> None:
    from agent.nodes.critic import critic_node
    from agent.nodes.fit_matcher import fit_matcher_node
    from agent.nodes.scout import scout_node

    state = create_initial_bid_state(
        job_id="job_001",
        freelancer_id="persona_aiml_01",
        model_profile="offline_demo",
        max_revisions=2,
    )
    state.update(asyncio.run(scout_node(state)))
    state.update(asyncio.run(fit_matcher_node(state)))

    # Inject a draft containing a fabricated 99.9% metric & unowned Rust/CUDA claim
    state["proposal_draft"] = {
        "job_id": "job_001",
        "freelancer_id": "persona_aiml_01",
        "version": 1,
        "cover_letter": (
            "Hello, thank you for sharing this role. "
            "I have 18 years of experience writing Rust CUDA kernels that cut latency by 99.9%. "
            "In my project Multi-Agent Legal Contract Review Copilot on LangGraph + Postgres (ev_aiml_001), "
            "I built a LangGraph workflow with PostgresSaver."
        ),
        "screening_answers": {},
        "cited_evidence_ids": ["ev_aiml_001"],
        "proposed_rate": 3800.0,
        "currency": "USD",
        "connects_cost": 8,
    }

    # Revision 0 -> critic requests revision
    c1 = asyncio.run(critic_node(state))
    assert c1["critique"]["requires_revision"] is True
    assert c1["revision_count"] == 1

    # Simulate reaching max_revisions=2 with hallucination still present -> sanitize_proposal kicks in
    state["revision_count"] = 2
    c2 = asyncio.run(critic_node(state))
    assert c2["critique"]["requires_revision"] is False
    assert "99.9%" not in c2["proposal_draft"]["cover_letter"]
    assert "ev_aiml_001" in c2["proposal_draft"]["cover_letter"]
    assert c2["verification_report"]["grounding_score"] >= 0.85
