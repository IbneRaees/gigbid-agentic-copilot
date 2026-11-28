"""Compiled LangGraph `BidGraph`: Parallel Analysis, Bounded Critique, Dry-Run & HITL Interrupt."""

from __future__ import annotations

import time
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.graphs.checkpointer import get_sync_or_local_checkpointer
from agent.marketplace_port import build_marketplace_adapter
from agent.memory.promotion_rules import promote_from_bid_outcome
from agent.memory.store import GigBidMemoryStore
from agent.nodes._runtime import get_mcp_client_manager, record_mcp_audit
from agent.nodes.analyst import analyst_node
from agent.nodes.compliance import compliance_node
from agent.nodes.critic import critic_node
from agent.nodes.fit_matcher import fit_matcher_node
from agent.nodes.pricing import calculate_marketplace_fee, pricing_node
from agent.nodes.risk_assessor import risk_assessor_node
from agent.nodes.scout import scout_node
from agent.nodes.writer import writer_node
from agent.state import BidGraphState, make_budget_delta, make_trace_event
from gigsim.models import ProposalDraftRequest, ProposalSubmitRequest
from gigsim.store import GigSimStore


async def _execute_marketplace_proposal_tool(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    marketplace_mode: str,
    trace_id: str,
    freelancer_id: str,
) -> dict[str, Any]:
    """Call Marketplace MCP write tool (`create_proposal_draft` / `submit_proposal`) with safe fallback."""
    mcp_mgr = get_mcp_client_manager(trace_id=trace_id, user_id=freelancer_id)
    if mcp_mgr is not None:
        try:
            async with mcp_mgr as mcp:
                return await mcp.call_tool("marketplace", tool_name, arguments)
        except Exception:
            pass

    # Fallback 1: Direct MarketplacePort adapter (GigSim HTTP or UpworkLive draft-only)
    try:
        adapter = build_marketplace_adapter(marketplace_mode)
        if tool_name == "create_proposal_draft":
            res = await adapter.create_proposal_draft(**arguments)
        else:
            res = await adapter.submit_proposal(**arguments)
        dumped = res.model_dump(mode="json")
        record_mcp_audit(
            server_name="marketplace",
            tool_name=tool_name,
            raw_args=arguments,
            result_summary=f"adapter_fallback status={dumped.get('status')}",
            status="dry_run" if arguments.get("dry_run") else "ok",
            trace_id=trace_id,
            user_id=freelancer_id,
        )
        return dumped
    except Exception:
        pass

    # Fallback 2: In-process GigSimStore (or UpworkLive draft-only) for isolated offline tests
    if marketplace_mode == "upwork_live":
        from mcp_servers.marketplace.adapters.upwork_live import UpworkLiveMarketplaceAdapter

        upwork = UpworkLiveMarketplaceAdapter()
        res = (
            await upwork.create_proposal_draft(**arguments)
            if tool_name == "create_proposal_draft"
            else await upwork.submit_proposal(**arguments)
        )
        return res.model_dump(mode="json")

    store = GigSimStore()
    if tool_name == "create_proposal_draft":
        req = ProposalDraftRequest(**arguments)
        dumped = store.create_proposal_draft(req).model_dump(mode="json")
    else:
        req = ProposalSubmitRequest(**arguments)
        dumped = store.submit_proposal(req).model_dump(mode="json")

    record_mcp_audit(
        server_name="marketplace",
        tool_name=tool_name,
        raw_args=arguments,
        result_summary=f"inprocess_gigsim status={dumped.get('status')}, connects={dumped.get('connects_charged')}",
        status="dry_run" if arguments.get("dry_run") else "ok",
        trace_id=trace_id,
        user_id=freelancer_id,
    )
    return dumped


def _route_after_pricing(state: BidGraphState) -> str:
    decision = state.get("bid_decision") or {}
    if decision.get("action") != "bid":
        return "finalize_skip"
    return "writer"


def _route_after_critic(state: BidGraphState) -> str:
    critique = state.get("critique") or {}
    revision_count = int(state.get("revision_count", 0))
    max_revisions = int((state.get("budgets") or {}).get("max_revisions", 2))
    if critique.get("requires_revision") and revision_count <= max_revisions:
        # Critic incremented revision_count when requesting a revision; allow up to max_revisions drafts
        if int((state.get("proposal_draft") or {}).get("version", 1)) <= max_revisions:
            return "writer"
    return "compliance"


def _route_after_compliance(state: BidGraphState) -> str:
    verdict = state.get("compliance_verdict") or {}
    if verdict.get("blocked") or not verdict.get("passed", False):
        return "finalize_blocked"
    return "prepare_approval"


async def finalize_skip_node(state: BidGraphState) -> dict[str, Any]:
    """Record episodic skip decision & optional risk pattern memory when job is skipped."""
    t0 = time.perf_counter()
    decision = state.get("bid_decision") or {}
    risk = state.get("risk_report") or {}
    raw_job = state.get("raw_job") or {}
    freelancer_id = state.get("freelancer_id", "persona_aiml_01")

    tags = ["skip"]
    if risk.get("is_scam") or risk.get("requests_off_platform") or risk.get("requests_free_work"):
        tags.append("policy_trap")

    store = GigBidMemoryStore()
    promo = promote_from_bid_outcome(
        store=store,
        freelancer_id=freelancer_id,
        job_id=state["job_id"],
        client_id=raw_job.get("client_id"),
        outcome="skipped",
        summary=str(decision.get("rationale", "Skipped by Bid-or-Skip gate")),
        grounding_score=1.0,
        compliance_passed=True,
        evidence_ids=[],
        tags=tags,
    )

    duration_ms = (time.perf_counter() - t0) * 1000.0
    event = make_trace_event(
        node="finalize_skip",
        phase="skipped",
        duration_ms=duration_ms,
        summary=f"Run ended with SKIP: {decision.get('rationale', '')}",
    )
    return {
        "promoted_memories": [m.model_dump(mode="json") for m in promo.promoted_items],
        "node_timings": {"finalize_skip": round(duration_ms, 2)},
        "trace_events": [event],
        "status": "skipped",
    }


async def finalize_blocked_node(state: BidGraphState) -> dict[str, Any]:
    """Terminate cleanly if Compliance Guard blocked the proposal."""
    t0 = time.perf_counter()
    verdict = state.get("compliance_verdict") or {}
    duration_ms = (time.perf_counter() - t0) * 1000.0
    event = make_trace_event(
        node="finalize_blocked",
        phase="error",
        duration_ms=duration_ms,
        summary=f"Run blocked by Compliance Guard: {verdict.get('block_reason')}",
    )
    return {
        "node_timings": {"finalize_blocked": round(duration_ms, 2)},
        "trace_events": [event],
        "status": "blocked_by_compliance",
    }


async def prepare_approval_node(state: BidGraphState) -> dict[str, Any]:
    """Create non-spending draft and run `submit_proposal(dry_run=True)` preview before interrupt."""
    t0 = time.perf_counter()
    draft = state.get("proposal_draft") or {}
    job_id = str(draft.get("job_id") or state["job_id"])
    freelancer_id = str(draft.get("freelancer_id") or state.get("freelancer_id", "persona_aiml_01"))
    marketplace_mode = state.get("marketplace_mode", "gigsim")
    trace_id = state.get("trace_id", "trace-local")
    run_id = state.get("run_id", "run")

    base_args = {
        "job_id": job_id,
        "freelancer_id": freelancer_id,
        "cover_letter": str(draft.get("cover_letter", "")),
        "screening_answers": dict(draft.get("screening_answers") or {}),
        "proposed_rate": float(draft.get("proposed_rate", 1000.0)),
        "currency": str(draft.get("currency", "USD")),
        "cited_evidence_ids": list(draft.get("cited_evidence_ids") or []),
    }

    draft_res = await _execute_marketplace_proposal_tool(
        tool_name="create_proposal_draft",
        arguments={
            "idempotency_key": f"draft:{run_id}:{job_id}",
            "dry_run": False,
            **base_args,
        },
        marketplace_mode=marketplace_mode,
        trace_id=trace_id,
        freelancer_id=freelancer_id,
    )

    preview_res = await _execute_marketplace_proposal_tool(
        tool_name="submit_proposal",
        arguments={
            "idempotency_key": f"submit:{run_id}:{job_id}",
            "dry_run": True,
            **base_args,
        },
        marketplace_mode=marketplace_mode,
        trace_id=trace_id,
        freelancer_id=freelancer_id,
    )

    # Ensure dry_run_preview summary always has Connects & fee numbers
    if not preview_res.get("dry_run_preview"):
        rate = base_args["proposed_rate"]
        fee = calculate_marketplace_fee(rate)
        c_cost = int((state.get("job_spec") or {}).get("connects_cost", 6))
        bal = int((state.get("connects_balance") or {}).get("balance", 80))
        preview_res["dry_run_preview"] = {
            "job_id": job_id,
            "job_title": (state.get("job_spec") or {}).get("title", job_id),
            "freelancer_id": freelancer_id,
            "proposed_rate": rate,
            "currency": base_args["currency"],
            "estimated_fee_usd": fee,
            "estimated_net_usd": round(rate - fee, 2),
            "connects_cost": c_cost,
            "current_connects_balance": bal,
            "remaining_connects_after": max(0, bal - c_cost),
            "policy_passed": True,
            "policy_warnings": [],
        }

    duration_ms = (time.perf_counter() - t0) * 1000.0
    event = make_trace_event(
        node="prepare_approval",
        phase="completed",
        duration_ms=duration_ms,
        summary=(
            f"Prepared dry_run preview (connects_charged=0, "
            f"status={preview_res.get('status')}); awaiting human interrupt."
        ),
        verification_score=(state.get("verification_report") or {}).get("grounding_score"),
        rag_evidence_ids=base_args["cited_evidence_ids"],
    )
    return {
        "draft_record": draft_res,
        "dry_run_preview": preview_res,
        "budgets": make_budget_delta(steps=1),
        "node_timings": {"prepare_approval": round(duration_ms, 2)},
        "trace_events": [event],
        "status": "awaiting_human_approval",
    }


async def human_approval_gate_node(state: BidGraphState) -> dict[str, Any]:
    """Pause via LangGraph `interrupt()` for human approval; submit only when approved."""
    t0 = time.perf_counter()
    draft = dict(state.get("proposal_draft") or {})
    preview = state.get("dry_run_preview") or {}
    ver_report = state.get("verification_report") or {}
    compliance = state.get("compliance_verdict") or {}
    price_quote = state.get("price_quote") or {}
    job_spec = state.get("job_spec") or {}

    interrupt_payload = {
        "action": "submit_proposal",
        "run_id": state.get("run_id"),
        "thread_id": state.get("thread_id"),
        "job_id": state["job_id"],
        "job_title": job_spec.get("title", state["job_id"]),
        "freelancer_id": state.get("freelancer_id", "persona_aiml_01"),
        "marketplace_mode": state.get("marketplace_mode", "gigsim"),
        "proposal_draft": draft,
        "price_quote": price_quote,
        "grounding_score": ver_report.get("grounding_score", 1.0),
        "compliance_passed": compliance.get("passed", True),
        "dry_run_preview": preview,
    }

    # Execution pauses here until resumed with Command(resume={"approved": True/False, ...})
    human_decision = interrupt(interrupt_payload)
    if not isinstance(human_decision, dict):
        human_decision = {"approved": bool(human_decision)}

    approved = bool(human_decision.get("approved", False))
    if not approved:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        reason = str(human_decision.get("reason", "Rejected by human reviewer."))
        event = make_trace_event(
            node="human_approval_gate",
            phase="resumed",
            duration_ms=duration_ms,
            summary=f"Human REJECTED proposal submission: {reason}",
        )
        return {
            "human_approval": {"approved": False, "reason": reason},
            "node_timings": {"human_approval_gate": round(duration_ms, 2)},
            "trace_events": [event],
            "status": "rejected_by_human",
        }

    # Allow human edits to cover_letter, screening_answers, or proposed_rate prior to submission
    edited_letter = str(human_decision.get("cover_letter") or draft.get("cover_letter", ""))
    edited_answers = dict(
        human_decision.get("screening_answers") or draft.get("screening_answers") or {}
    )
    edited_rate = float(human_decision.get("proposed_rate") or draft.get("proposed_rate", 1000.0))

    draft["cover_letter"] = edited_letter
    draft["screening_answers"] = edited_answers
    draft["proposed_rate"] = edited_rate

    run_id = state.get("run_id", "run")
    job_id = str(draft.get("job_id") or state["job_id"])
    freelancer_id = str(draft.get("freelancer_id") or state.get("freelancer_id", "persona_aiml_01"))
    marketplace_mode = state.get("marketplace_mode", "gigsim")
    trace_id = state.get("trace_id", "trace-local")

    sub_res = await _execute_marketplace_proposal_tool(
        tool_name="submit_proposal",
        arguments={
            "idempotency_key": f"submit:{run_id}:{job_id}",
            "job_id": job_id,
            "freelancer_id": freelancer_id,
            "cover_letter": edited_letter,
            "screening_answers": edited_answers,
            "proposed_rate": edited_rate,
            "currency": str(draft.get("currency", "USD")),
            "cited_evidence_ids": list(draft.get("cited_evidence_ids") or []),
            "dry_run": False,
        },
        marketplace_mode=marketplace_mode,
        trace_id=trace_id,
        freelancer_id=freelancer_id,
    )

    sim_outcome = str(
        sub_res.get("simulated_client_outcome") or sub_res.get("status") or "submitted"
    )
    mapped_outcome = (
        "won"
        if sim_outcome == "won"
        else ("interview" if sim_outcome in {"interview", "replied"} else "approved")
    )
    promo = promote_from_bid_outcome(
        store=GigBidMemoryStore(),
        freelancer_id=freelancer_id,
        job_id=job_id,
        client_id=(state.get("raw_job") or {}).get("client_id"),
        outcome=mapped_outcome,
        summary=f"Approved & submitted proposal for {job_spec.get('title', job_id)} at {draft.get('currency', 'USD')} {edited_rate:,.0f}",
        grounding_score=float(ver_report.get("grounding_score", 1.0)),
        compliance_passed=bool(compliance.get("passed", True)),
        evidence_ids=list(draft.get("cited_evidence_ids") or []),
        screening_answers=edited_answers,
        tags=[str(job_spec.get("domain_category", "general"))],
        metadata={"domain_category": job_spec.get("domain_category", "general")},
    )

    duration_ms = (time.perf_counter() - t0) * 1000.0
    event = make_trace_event(
        node="human_approval_gate",
        phase="resumed",
        duration_ms=duration_ms,
        summary=(
            f"Human APPROVED & submitted proposal | status={sub_res.get('status')} "
            f"| connects_charged={sub_res.get('connects_charged', 0)}"
        ),
        verification_score=ver_report.get("grounding_score"),
        rag_evidence_ids=list(draft.get("cited_evidence_ids") or []),
    )
    return {
        "proposal_draft": draft,
        "human_approval": {"approved": True, **human_decision},
        "submission_result": sub_res,
        "promoted_memories": [m.model_dump(mode="json") for m in promo.promoted_items],
        "budgets": make_budget_delta(steps=1),
        "node_timings": {"human_approval_gate": round(duration_ms, 2)},
        "trace_events": [event],
        "status": str(sub_res.get("status", "submitted")),
    }


def build_bid_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    """Build and compile the GigBid Bid-or-Skip + Proposal LangGraph workflow."""
    builder = StateGraph(BidGraphState)

    builder.add_node("scout", scout_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("risk_assessor", risk_assessor_node)
    builder.add_node("fit_matcher", fit_matcher_node)
    builder.add_node("pricing", pricing_node)
    builder.add_node("finalize_skip", finalize_skip_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("compliance", compliance_node)
    builder.add_node("finalize_blocked", finalize_blocked_node)
    builder.add_node("prepare_approval", prepare_approval_node)
    builder.add_node("human_approval_gate", human_approval_gate_node)

    # Scout -> Parallel Fan-Out (Analyst + Risk Assessor + Fit Matcher) -> Fan-In to Pricing
    builder.add_edge(START, "scout")
    builder.add_edge("scout", "analyst")
    builder.add_edge("scout", "risk_assessor")
    builder.add_edge("scout", "fit_matcher")
    builder.add_edge(["analyst", "risk_assessor", "fit_matcher"], "pricing")

    builder.add_conditional_edges(
        "pricing",
        _route_after_pricing,
        {"finalize_skip": "finalize_skip", "writer": "writer"},
    )
    builder.add_edge("finalize_skip", END)

    # Bounded Critique Loop (max_revisions <= 2)
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"writer": "writer", "compliance": "compliance"},
    )

    builder.add_conditional_edges(
        "compliance",
        _route_after_compliance,
        {"finalize_blocked": "finalize_blocked", "prepare_approval": "prepare_approval"},
    )
    builder.add_edge("finalize_blocked", END)

    # Dry-run preview -> Mandatory HITL interrupt -> Submit
    builder.add_edge("prepare_approval", "human_approval_gate")
    builder.add_edge("human_approval_gate", END)

    saver = checkpointer or get_sync_or_local_checkpointer()
    return builder.compile(checkpointer=saver)


__all__ = ["build_bid_graph"]
