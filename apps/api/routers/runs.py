"""LangGraph Run Execution, SSE Streaming & HITL Approval Router."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from agent.graphs.bid_graph import build_bid_graph
from agent.graphs.checkpointer import get_sync_or_local_checkpointer
from agent.graphs.conversation_graph import build_conversation_graph
from agent.llm import resolve_model_profile
from agent.state import create_initial_bid_state, create_initial_conversation_state
from apps.api.auth import verify_optional_api_token
from apps.api.schemas import (
    ApprovalRequest,
    BidRunRequest,
    ConversationRunRequest,
    RunSummary,
)
from apps.api.streaming import get_run_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["Agent Runs & HITL"])

_SHARED_CHECKPOINTER = get_sync_or_local_checkpointer()
_BID_GRAPH = build_bid_graph(checkpointer=_SHARED_CHECKPOINTER)
_CONV_GRAPH = build_conversation_graph(checkpointer=_SHARED_CHECKPOINTER)


def _extract_interrupt_payload(state_dict: dict[str, Any]) -> dict[str, Any] | None:
    raw_int = state_dict.get("__interrupt__")
    if not raw_int:
        return None
    first = raw_int[0]
    val = getattr(first, "value", first)
    return dict(val) if isinstance(val, dict) else {"value": val}


def _snapshot_from_values(values: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "thread_id",
        "job_id",
        "freelancer_id",
        "status",
        "job_spec",
        "analyst_report",
        "risk_report",
        "fit_report",
        "bid_decision",
        "price_quote",
        "proposal_draft",
        "verification_report",
        "critique",
        "compliance_verdict",
        "dry_run_preview",
        "human_approval",
        "submission_result",
        "reply_intent",
        "reply_draft",
        "available_slots",
        "selected_slot",
        "send_preview",
        "hold_preview",
        "send_result",
        "hold_result",
        "promoted_memories",
        "budgets",
        "node_timings",
        "trace_events",
        "cited_evidence_ids",
    )
    return {k: values[k] for k in keys if k in values}


async def _drive_graph_stream(
    *,
    graph: Any,
    graph_input: Any,
    thread_id: str,
) -> dict[str, Any]:
    """Stream LangGraph node updates to SSE subscribers and capture interrupts."""
    registry = get_run_registry()
    config = {"configurable": {"thread_id": thread_id}}
    last_interrupt: dict[str, Any] | None = None

    try:
        async for mode, chunk in graph.astream(
            graph_input, config=config, stream_mode=["updates", "values"]
        ):
            if mode == "updates" and isinstance(chunk, dict):
                for node_name, delta in chunk.items():
                    if node_name == "__interrupt__":
                        raw_val = delta[0] if isinstance(delta, (list, tuple)) and delta else delta
                        payload = getattr(raw_val, "value", raw_val)
                        last_interrupt = (
                            dict(payload) if isinstance(payload, dict) else {"value": payload}
                        )
                        registry.update_run(
                            thread_id,
                            status="awaiting_human_approval",
                            awaiting_approval=True,
                            interrupt_payload=last_interrupt,
                        )
                        registry.publish_event(
                            thread_id,
                            "interrupt",
                            {"node": node_name, "interrupt_payload": last_interrupt},
                        )
                        continue

                    if isinstance(delta, dict):
                        events = delta.get("trace_events") or []
                        latest_evt = events[-1] if events else {}
                        registry.publish_event(
                            thread_id,
                            "node_update",
                            {
                                "node": node_name,
                                "status": delta.get("status"),
                                "trace_event": latest_evt,
                                "handoff_type": latest_evt.get("handoff_type"),
                                "handoff_payload": latest_evt.get("handoff_payload"),
                                "rag_evidence_ids": latest_evt.get("rag_evidence_ids", []),
                                "verification_score": latest_evt.get("verification_score"),
                                "budgets": delta.get("budgets"),
                            },
                        )
            elif mode == "values" and isinstance(chunk, dict):
                snap = _snapshot_from_values(chunk)
                int_payload = _extract_interrupt_payload(chunk)
                if int_payload is not None:
                    last_interrupt = int_payload
                registry.update_run(
                    thread_id,
                    status=str(chunk.get("status", "running")),
                    awaiting_approval=last_interrupt is not None,
                    interrupt_payload=last_interrupt,
                    state_snapshot=snap,
                )

        # Fetch final checkpoint state for authoritative status & interrupt inspection
        snap_state = await graph.aget_state(config)
        values = dict(getattr(snap_state, "values", {}) or {})
        tasks = getattr(snap_state, "tasks", ()) or ()
        pending_interrupt = last_interrupt
        for t in tasks:
            ints = getattr(t, "interrupts", ()) or ()
            if ints:
                val = getattr(ints[0], "value", ints[0])
                pending_interrupt = dict(val) if isinstance(val, dict) else {"value": val}
                break

        final_status = (
            "awaiting_human_approval"
            if pending_interrupt is not None
            else str(values.get("status", "completed"))
        )
        updated = registry.update_run(
            thread_id,
            status=final_status,
            awaiting_approval=pending_interrupt is not None,
            interrupt_payload=pending_interrupt,
            state_snapshot=_snapshot_from_values(values),
        )
        registry.publish_event(
            thread_id,
            "run_completed",
            {
                "status": final_status,
                "awaiting_approval": pending_interrupt is not None,
                "interrupt_payload": pending_interrupt,
                "state_snapshot": _snapshot_from_values(values),
            },
        )
        return updated or {}
    except Exception as exc:
        logger.exception("Graph execution failed for thread_id=%s", thread_id)
        registry.update_run(thread_id, status="error", awaiting_approval=False)
        registry.publish_event(thread_id, "run_error", {"error": str(exc)})
        return registry.get_run(thread_id) or {}


@router.get("", response_model=list[RunSummary])
async def list_runs(
    pending_approval: bool = Query(default=False),
    _: str = Depends(verify_optional_api_token),
) -> list[dict[str, Any]]:
    """List recent runs or filter to the Human-in-the-Loop approval queue."""
    return get_run_registry().list_runs(only_pending_approval=pending_approval)


@router.post("/bid", response_model=RunSummary)
async def start_bid_run(
    body: BidRunRequest,
    wait: bool = Query(
        default=False, description="Wait for graph to reach interrupt or terminal state"
    ),
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """Start a BidGraph workflow run and stream updates over SSE."""
    profile = body.model_profile or resolve_model_profile()
    mode = body.marketplace_mode or os.getenv("MARKETPLACE_MODE", "gigsim")
    state = create_initial_bid_state(
        job_id=body.job_id,
        freelancer_id=body.freelancer_id,
        marketplace_mode=mode,
        model_profile=profile,
        rag_variant=body.rag_variant,
    )
    thread_id = state["thread_id"]
    registry = get_run_registry()
    record = registry.create_run(
        run_id=state["run_id"],
        thread_id=thread_id,
        graph_type="bid",
        job_id=body.job_id,
        freelancer_id=body.freelancer_id,
    )
    registry.publish_event(
        thread_id,
        "run_started",
        {"run_id": state["run_id"], "job_id": body.job_id, "graph_type": "bid"},
    )

    if wait:
        return await _drive_graph_stream(graph=_BID_GRAPH, graph_input=state, thread_id=thread_id)

    asyncio.create_task(
        _drive_graph_stream(graph=_BID_GRAPH, graph_input=state, thread_id=thread_id)
    )
    return record


@router.post("/conversation", response_model=RunSummary)
async def start_conversation_run(
    body: ConversationRunRequest,
    wait: bool = Query(default=False),
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """Start a ConversationGraph workflow run for an active client thread."""
    profile = body.model_profile or resolve_model_profile()
    mode = body.marketplace_mode or os.getenv("MARKETPLACE_MODE", "gigsim")
    state = create_initial_conversation_state(
        thread_id=body.thread_id,
        freelancer_id=body.freelancer_id,
        incoming_message=body.incoming_message,
        marketplace_mode=mode,
        model_profile=profile,
    )
    # Give each LangGraph execution a unique checkpointer thread key while preserving marketplace thread_id in state
    lg_thread_id = f"conv_{body.thread_id}_{state['run_id']}"
    registry = get_run_registry()
    record = registry.create_run(
        run_id=state["run_id"],
        thread_id=lg_thread_id,
        graph_type="conversation",
        job_id=None,
        freelancer_id=body.freelancer_id,
    )
    registry.publish_event(
        lg_thread_id,
        "run_started",
        {
            "run_id": state["run_id"],
            "marketplace_thread_id": body.thread_id,
            "graph_type": "conversation",
        },
    )

    if wait:
        return await _drive_graph_stream(
            graph=_CONV_GRAPH, graph_input=state, thread_id=lg_thread_id
        )

    asyncio.create_task(
        _drive_graph_stream(graph=_CONV_GRAPH, graph_input=state, thread_id=lg_thread_id)
    )
    return record


@router.get("/{thread_id}", response_model=RunSummary)
async def get_run_detail(
    thread_id: str,
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """Get full state snapshot and pending interrupt payload for a run."""
    run = get_run_registry().get_run(thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{thread_id}' not found.")
    return run


@router.get("/{thread_id}/events")
async def get_run_events(
    thread_id: str,
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """Return buffered trace events for a run (non-streaming snapshot)."""
    run = get_run_registry().get_run(thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{thread_id}' not found.")
    return {"thread_id": thread_id, "events": get_run_registry().get_events(thread_id)}


@router.get("/{thread_id}/stream")
async def stream_run_events(
    thread_id: str,
    _: str = Depends(verify_optional_api_token),
) -> EventSourceResponse:
    """Server-Sent Events (SSE) stream for live node transitions, handoffs, RAG citations & interrupts."""
    registry = get_run_registry()
    if registry.get_run(thread_id) is None:
        raise HTTPException(status_code=404, detail=f"Run '{thread_id}' not found.")
    return EventSourceResponse(registry.subscribe(thread_id))


@router.post("/{thread_id}/approve", response_model=RunSummary)
async def approve_interrupted_run(
    thread_id: str,
    body: ApprovalRequest,
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """Resume an interrupted Bid or Conversation run with human approval/edits or rejection."""
    registry = get_run_registry()
    run = registry.get_run(thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{thread_id}' not found.")
    if not run.get("awaiting_approval"):
        raise HTTPException(
            status_code=409,
            detail=f"Run '{thread_id}' is not awaiting human approval (status={run.get('status')}).",
        )

    graph = _BID_GRAPH if run["graph_type"] == "bid" else _CONV_GRAPH
    resume_payload = body.model_dump(exclude_none=True)
    registry.update_run(thread_id, status="resuming", awaiting_approval=False)
    registry.publish_event(
        thread_id,
        "human_decision",
        {"approved": body.approved, "reason": body.reason},
    )
    return await _drive_graph_stream(
        graph=graph,
        graph_input=Command(resume=resume_payload),
        thread_id=thread_id,
    )
