"""Typed LangGraph state schemas, reducers, budgets, and trace builders for GigBid."""

from __future__ import annotations

import operator
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict


def _merge_budgets(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer that safely merges parallel budget updates (summing deltas, keeping caps)."""
    base = dict(left or {})
    incoming = dict(right or {})
    if not incoming:
        return base

    merged = dict(base)
    for key in ("steps_used", "tokens_used", "prompt_tokens", "completion_tokens"):
        if key in incoming:
            # Incoming updates carry deltas when prefixed with _delta or explicit cumulative max
            if incoming.get("_is_delta"):
                merged[key] = int(base.get(key, 0)) + int(incoming[key])
            else:
                merged[key] = max(int(base.get(key, 0)), int(incoming[key]))
    if "estimated_cost_usd" in incoming:
        if incoming.get("_is_delta"):
            merged["estimated_cost_usd"] = round(
                float(base.get("estimated_cost_usd", 0.0)) + float(incoming["estimated_cost_usd"]),
                6,
            )
        else:
            merged["estimated_cost_usd"] = max(
                float(base.get("estimated_cost_usd", 0.0)),
                float(incoming["estimated_cost_usd"]),
            )

    for cap_key in ("max_steps", "max_tokens", "max_revisions", "max_budget_usd"):
        if cap_key in incoming:
            merged[cap_key] = incoming[cap_key]
    if "revision_count" in incoming:
        merged["revision_count"] = max(
            int(base.get("revision_count", 0)),
            int(incoming["revision_count"]),
        )
    merged.pop("_is_delta", None)
    return merged


def _merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer merging non-overlapping dictionary updates from parallel branches."""
    merged = dict(left or {})
    if right:
        merged.update(right)
    return merged


def _append_unique_strs(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer appending strings while preserving order and deduplicating."""
    seen: set[str] = set()
    out: list[str] = []
    for item in [*(left or []), *(right or [])]:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class TraceEvent(TypedDict, total=False):
    """Structured trace event emitted by every node for SSE streaming and auditability."""

    event_id: str
    node: str
    phase: Literal["started", "completed", "skipped", "interrupted", "resumed", "error"]
    timestamp: str
    duration_ms: float
    summary: str
    handoff_type: str | None
    handoff_payload: dict[str, Any] | None
    rag_evidence_ids: list[str]
    verification_score: float | None
    model_used: str | None


class BidGraphState(TypedDict, total=False):
    """Canonical LangGraph state for the GigBid Bid-or-Skip & Proposal Workflow."""

    # Identity & routing
    run_id: str
    thread_id: str
    trace_id: str
    job_id: str
    freelancer_id: str
    marketplace_mode: str
    model_profile: str
    rag_variant: str

    # Raw marketplace & profile context (populated by Scout)
    raw_job: dict[str, Any]
    wrapped_job_text: str
    client_history: dict[str, Any]
    freelancer_profile: dict[str, Any]
    connects_balance: dict[str, Any]
    semantic_memories: list[dict[str, Any]]

    # Parallel analysis fan-out outputs (Pydantic-serialized dicts)
    job_spec: dict[str, Any]
    analyst_report: dict[str, Any]
    risk_report: dict[str, Any]
    fit_report: dict[str, Any]
    retrieved_evidence: list[dict[str, Any]]
    policy_evidence: list[dict[str, Any]]

    # Pricing & Bid/Skip Gate
    bid_decision: dict[str, Any]
    price_quote: dict[str, Any]

    # Proposal drafting, verification & compliance loop
    proposal_draft: dict[str, Any]
    verification_report: dict[str, Any]
    critique: dict[str, Any]
    compliance_verdict: dict[str, Any]
    revision_count: int

    # Pre-approval dry-run preview & human interrupt
    draft_record: dict[str, Any]
    dry_run_preview: dict[str, Any]
    human_approval: dict[str, Any]
    submission_result: dict[str, Any]
    promoted_memories: list[dict[str, Any]]

    # Reducers for parallel observability, budgets, and errors
    budgets: Annotated[dict[str, Any], _merge_budgets]
    node_timings: Annotated[dict[str, float], _merge_dicts]
    trace_events: Annotated[list[dict[str, Any]], operator.add]
    audit_entries: Annotated[list[dict[str, Any]], operator.add]
    cited_evidence_ids: Annotated[list[str], _append_unique_strs]
    errors: Annotated[list[str], operator.add]

    # Terminal status
    status: str


class ConversationGraphState(TypedDict, total=False):
    """Canonical LangGraph state for the Client Reply & Calendar Liaison Workflow."""

    run_id: str
    thread_id: str
    trace_id: str
    job_id: str
    client_id: str
    freelancer_id: str
    marketplace_mode: str
    model_profile: str

    # Thread context
    thread_record: dict[str, Any]
    messages: list[dict[str, Any]]
    latest_client_message: str
    wrapped_client_message: str
    job_context: dict[str, Any]
    freelancer_profile: dict[str, Any]
    price_quote: dict[str, Any]

    # Analysis & Context gathering
    reply_intent: dict[str, Any]
    retrieved_evidence: list[dict[str, Any]]
    policy_evidence: list[dict[str, Any]]
    available_slots: list[dict[str, Any]]
    selected_slot: str | None

    # Draft, compliance, dry-run & approval
    reply_draft: dict[str, Any]
    compliance_verdict: dict[str, Any]
    send_preview: dict[str, Any]
    hold_preview: dict[str, Any]
    human_approval: dict[str, Any]
    send_result: dict[str, Any]
    hold_result: dict[str, Any]
    email_draft_result: dict[str, Any]
    promoted_memories: list[dict[str, Any]]

    # Observability & budgets
    budgets: Annotated[dict[str, Any], _merge_budgets]
    node_timings: Annotated[dict[str, float], _merge_dicts]
    trace_events: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    status: str


def make_trace_event(
    *,
    node: str,
    phase: Literal["started", "completed", "skipped", "interrupted", "resumed", "error"],
    summary: str,
    duration_ms: float = 0.0,
    handoff_type: str | None = None,
    handoff_payload: dict[str, Any] | None = None,
    rag_evidence_ids: list[str] | None = None,
    verification_score: float | None = None,
    model_used: str | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable trace event dictionary."""
    return {
        "event_id": f"evt_{uuid.uuid4().hex[:10]}",
        "node": node,
        "phase": phase,
        "timestamp": datetime.now(UTC).isoformat(),
        "duration_ms": round(duration_ms, 2),
        "summary": summary,
        "handoff_type": handoff_type,
        "handoff_payload": handoff_payload,
        "rag_evidence_ids": list(rag_evidence_ids or []),
        "verification_score": verification_score,
        "model_used": model_used,
    }


def make_budget_delta(
    *,
    steps: int = 1,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
) -> dict[str, Any]:
    """Create a delta budget update compatible with `_merge_budgets`."""
    return {
        "_is_delta": True,
        "steps_used": steps,
        "tokens_used": prompt_tokens + completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": estimated_cost_usd,
    }


def create_initial_bid_state(
    *,
    job_id: str,
    freelancer_id: str = "persona_aiml_01",
    thread_id: str | None = None,
    marketplace_mode: str | None = None,
    model_profile: str | None = None,
    rag_variant: str = "v4",
    raw_job: dict[str, Any] | None = None,
    client_history: dict[str, Any] | None = None,
    freelancer_profile: dict[str, Any] | None = None,
    connects_balance: dict[str, Any] | None = None,
    max_steps: int = 24,
    max_tokens: int = 24000,
    max_revisions: int = 2,
    max_budget_usd: float = 5.0,
) -> BidGraphState:
    """Initialize a clean, checkpoint-safe `BidGraphState`."""
    run_uid = uuid.uuid4().hex[:12]
    state: BidGraphState = {
        "run_id": f"bid_{run_uid}",
        "thread_id": thread_id or f"bid_{job_id}_{run_uid}",
        "trace_id": f"trace-{run_uid}",
        "job_id": job_id,
        "freelancer_id": freelancer_id,
        "marketplace_mode": marketplace_mode or "gigsim",
        "model_profile": model_profile or "open",
        "rag_variant": rag_variant,
        "revision_count": 0,
        "budgets": {
            "max_steps": max_steps,
            "steps_used": 0,
            "max_tokens": max_tokens,
            "tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "max_revisions": max_revisions,
            "revision_count": 0,
            "estimated_cost_usd": 0.0,
            "max_budget_usd": max_budget_usd,
        },
        "node_timings": {},
        "trace_events": [],
        "audit_entries": [],
        "cited_evidence_ids": [],
        "errors": [],
        "semantic_memories": [],
        "promoted_memories": [],
        "status": "initialized",
    }
    if raw_job is not None:
        state["raw_job"] = raw_job
    if client_history is not None:
        state["client_history"] = client_history
    if freelancer_profile is not None:
        state["freelancer_profile"] = freelancer_profile
    if connects_balance is not None:
        state["connects_balance"] = connects_balance
    return state


def create_initial_conversation_state(
    *,
    thread_id: str,
    freelancer_id: str = "persona_aiml_01",
    job_id: str | None = None,
    client_id: str | None = None,
    incoming_message: str | None = None,
    marketplace_mode: str | None = None,
    model_profile: str | None = None,
    max_steps: int = 16,
    max_tokens: int = 16000,
    max_budget_usd: float = 5.0,
) -> ConversationGraphState:
    """Initialize a clean, checkpoint-safe `ConversationGraphState`."""
    run_uid = uuid.uuid4().hex[:12]
    state: ConversationGraphState = {
        "run_id": f"conv_{run_uid}",
        "thread_id": thread_id,
        "trace_id": f"trace-conv-{run_uid}",
        "freelancer_id": freelancer_id,
        "marketplace_mode": marketplace_mode or "gigsim",
        "model_profile": model_profile or "open",
        "messages": [],
        "retrieved_evidence": [],
        "policy_evidence": [],
        "available_slots": [],
        "promoted_memories": [],
        "budgets": {
            "max_steps": max_steps,
            "steps_used": 0,
            "max_tokens": max_tokens,
            "tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "max_revisions": 0,
            "revision_count": 0,
            "estimated_cost_usd": 0.0,
            "max_budget_usd": max_budget_usd,
        },
        "node_timings": {},
        "trace_events": [],
        "errors": [],
        "status": "initialized",
    }
    if job_id is not None:
        state["job_id"] = job_id
    if client_id is not None:
        state["client_id"] = client_id
    if incoming_message is not None:
        state["latest_client_message"] = incoming_message
    return state


__all__ = [
    "BidGraphState",
    "ConversationGraphState",
    "TraceEvent",
    "create_initial_bid_state",
    "create_initial_conversation_state",
    "make_budget_delta",
    "make_trace_event",
]
