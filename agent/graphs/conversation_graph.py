"""Compiled LangGraph `ConversationGraph`: Client Reply Classification, Calendar Hold & HITL Interrupt."""

from __future__ import annotations

import time
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.graphs.checkpointer import get_sync_or_local_checkpointer
from agent.marketplace_port import build_marketplace_adapter
from agent.memory.promotion_rules import promote_from_conversation
from agent.memory.store import GigBidMemoryStore
from agent.nodes._runtime import get_mcp_client_manager, get_workspace_store, record_mcp_audit
from agent.nodes.client_liaison import (
    classify_intent_node,
    draft_reply_node,
    gather_context_node,
    load_thread_node,
)
from agent.state import ConversationGraphState, make_budget_delta, make_trace_event
from gigsim.models import SendMessageRequest
from gigsim.store import GigSimStore


async def _call_workspace_write(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    trace_id: str,
    freelancer_id: str,
) -> dict[str, Any]:
    mcp_mgr = get_mcp_client_manager(trace_id=trace_id, user_id=freelancer_id)
    if mcp_mgr is not None:
        try:
            async with mcp_mgr as mcp:
                return await mcp.call_tool("workspace", tool_name, arguments)
        except Exception:
            pass

    store = get_workspace_store()
    if tool_name == "create_tentative_interview_hold":
        if arguments.get("dry_run"):
            res = {
                "hold_id": f"dryrun_hold_{arguments['job_id']}",
                "thread_id": arguments["thread_id"],
                "job_id": arguments["job_id"],
                "freelancer_id": arguments.get("freelancer_id", freelancer_id),
                "title": arguments.get("title", "Client Technical Interview"),
                "slot_start_iso": arguments["slot_start_iso"],
                "duration_minutes": arguments.get("duration_minutes", 30),
                "status": "dry_run_ok",
                "slot_available": arguments["slot_start_iso"] not in store.get_booked_starts(),
            }
        else:
            res = store.create_hold(
                idempotency_key=arguments["idempotency_key"],
                thread_id=arguments["thread_id"],
                job_id=arguments["job_id"],
                freelancer_id=arguments.get("freelancer_id", freelancer_id),
                title=arguments.get("title", "Client Technical Interview"),
                slot_start_iso=arguments["slot_start_iso"],
                duration_minutes=int(arguments.get("duration_minutes", 30)),
            )
    else:
        if arguments.get("dry_run"):
            res = {
                "draft_id": "dryrun_email_draft",
                "status": "dry_run_ok",
                "auto_sent": False,
            }
        else:
            res = store.create_draft(
                idempotency_key=arguments["idempotency_key"],
                thread_id=arguments.get("thread_id"),
                freelancer_id=arguments.get("freelancer_id", freelancer_id),
                recipient_alias=arguments["recipient_alias"],
                subject=arguments["subject"],
                body=arguments["body"],
            )

    record_mcp_audit(
        server_name="workspace",
        tool_name=tool_name,
        raw_args=arguments,
        result_summary=f"workspace_fallback status={res.get('status')}",
        status="dry_run" if arguments.get("dry_run") else "ok",
        trace_id=trace_id,
        user_id=freelancer_id,
    )
    return res


async def _call_marketplace_send_message(
    *,
    arguments: dict[str, Any],
    marketplace_mode: str,
    trace_id: str,
    freelancer_id: str,
) -> dict[str, Any]:
    mcp_mgr = get_mcp_client_manager(trace_id=trace_id, user_id=freelancer_id)
    if mcp_mgr is not None:
        try:
            async with mcp_mgr as mcp:
                return await mcp.call_tool("marketplace", "send_message", arguments)
        except Exception:
            pass

    try:
        adapter = build_marketplace_adapter(marketplace_mode)
        res = await adapter.send_message(**arguments)
        dumped = res.model_dump(mode="json")
        record_mcp_audit(
            server_name="marketplace",
            tool_name="send_message",
            raw_args=arguments,
            result_summary=f"adapter_fallback status={dumped.get('status')}",
            status="dry_run" if arguments.get("dry_run") else "ok",
            trace_id=trace_id,
            user_id=freelancer_id,
        )
        return dumped
    except Exception:
        pass

    if marketplace_mode == "upwork_live":
        from mcp_servers.marketplace.adapters.upwork_live import UpworkLiveMarketplaceAdapter

        upwork = UpworkLiveMarketplaceAdapter()
        return (await upwork.send_message(**arguments)).model_dump(mode="json")

    store = GigSimStore()
    req = SendMessageRequest(
        idempotency_key=arguments["idempotency_key"],
        freelancer_id=arguments["freelancer_id"],
        text=arguments["text"],
        proposed_slots=arguments.get("proposed_slots", []),
        dry_run=bool(arguments.get("dry_run", False)),
    )
    dumped = store.post_thread_message(arguments["thread_id"], req).model_dump(mode="json")
    dumped.setdefault("status", "dry_run_ok" if arguments.get("dry_run") else "sent")
    record_mcp_audit(
        server_name="marketplace",
        tool_name="send_message",
        raw_args=arguments,
        result_summary=f"inprocess_gigsim status={dumped.get('status')}",
        status="dry_run" if arguments.get("dry_run") else "ok",
        trace_id=trace_id,
        user_id=freelancer_id,
    )
    return dumped


async def prepare_send_preview_node(state: ConversationGraphState) -> dict[str, Any]:
    """Run `dry_run=True` previews for `send_message` and `create_tentative_interview_hold`."""
    t0 = time.perf_counter()
    thread_id = state["thread_id"]
    job_id = state.get("job_id", "job_001")
    freelancer_id = state.get("freelancer_id", "persona_aiml_01")
    marketplace_mode = state.get("marketplace_mode", "gigsim")
    trace_id = state.get("trace_id", "trace-conv")
    run_id = state.get("run_id", "conv")
    reply_draft = state.get("reply_draft") or {}
    selected_slot = state.get("selected_slot")
    intent = state.get("reply_intent") or {}

    send_preview = await _call_marketplace_send_message(
        arguments={
            "idempotency_key": f"msg:{run_id}:{thread_id}",
            "thread_id": thread_id,
            "freelancer_id": freelancer_id,
            "text": str(reply_draft.get("reply_text", "")),
            "proposed_slots": list(reply_draft.get("proposed_slots") or []),
            "dry_run": True,
        },
        marketplace_mode=marketplace_mode,
        trace_id=trace_id,
        freelancer_id=freelancer_id,
    )

    hold_preview: dict[str, Any] = {}
    if intent.get("requires_calendar_slots") and selected_slot:
        hold_preview = await _call_workspace_write(
            tool_name="create_tentative_interview_hold",
            arguments={
                "idempotency_key": f"hold:{run_id}:{thread_id}",
                "thread_id": thread_id,
                "job_id": job_id,
                "slot_start_iso": selected_slot,
                "duration_minutes": 30,
                "title": f"Interview: {(state.get('job_context') or {}).get('title', job_id)[:50]}",
                "freelancer_id": freelancer_id,
                "dry_run": True,
            },
            trace_id=trace_id,
            freelancer_id=freelancer_id,
        )

    duration_ms = (time.perf_counter() - t0) * 1000.0
    event = make_trace_event(
        node="prepare_send_preview",
        phase="completed",
        duration_ms=duration_ms,
        summary=(
            f"Prepared dry_run message preview (hold_preview={bool(hold_preview)}); "
            "awaiting human approval."
        ),
    )
    return {
        "send_preview": send_preview,
        "hold_preview": hold_preview,
        "budgets": make_budget_delta(steps=1),
        "node_timings": {"prepare_send_preview": round(duration_ms, 2)},
        "trace_events": [event],
        "status": "awaiting_human_approval",
    }


async def conversation_approval_gate_node(state: ConversationGraphState) -> dict[str, Any]:
    """Pause via `interrupt()` before executing `send_message` or `create_tentative_interview_hold`."""
    t0 = time.perf_counter()
    thread_id = state["thread_id"]
    job_id = state.get("job_id", "job_001")
    freelancer_id = state.get("freelancer_id", "persona_aiml_01")
    marketplace_mode = state.get("marketplace_mode", "gigsim")
    trace_id = state.get("trace_id", "trace-conv")
    run_id = state.get("run_id", "conv")

    reply_draft = dict(state.get("reply_draft") or {})
    intent_obj = state.get("reply_intent") or {}
    available_slots = state.get("available_slots") or []
    selected_slot = state.get("selected_slot")

    interrupt_payload = {
        "action": "send_message",
        "run_id": run_id,
        "thread_id": thread_id,
        "job_id": job_id,
        "freelancer_id": freelancer_id,
        "reply_intent": intent_obj,
        "reply_draft": reply_draft,
        "available_slots": available_slots,
        "selected_slot": selected_slot,
        "send_preview": state.get("send_preview") or {},
        "hold_preview": state.get("hold_preview") or {},
        "compliance_verdict": state.get("compliance_verdict") or {},
    }

    human_decision = interrupt(interrupt_payload)
    if not isinstance(human_decision, dict):
        human_decision = {"approved": bool(human_decision)}

    approved = bool(human_decision.get("approved", False))
    if not approved:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        reason = str(human_decision.get("reason", "Rejected by human reviewer."))
        event = make_trace_event(
            node="conversation_approval_gate",
            phase="resumed",
            duration_ms=duration_ms,
            summary=f"Human REJECTED client reply: {reason}",
        )
        return {
            "human_approval": {"approved": False, "reason": reason},
            "node_timings": {"conversation_approval_gate": round(duration_ms, 2)},
            "trace_events": [event],
            "status": "rejected_by_human",
        }

    final_text = str(human_decision.get("reply_text") or reply_draft.get("reply_text", ""))
    chosen_slot = human_decision.get("selected_slot") or selected_slot
    reply_draft["reply_text"] = final_text

    send_res = await _call_marketplace_send_message(
        arguments={
            "idempotency_key": f"msg:{run_id}:{thread_id}",
            "thread_id": thread_id,
            "freelancer_id": freelancer_id,
            "text": final_text,
            "proposed_slots": list(reply_draft.get("proposed_slots") or []),
            "dry_run": False,
        },
        marketplace_mode=marketplace_mode,
        trace_id=trace_id,
        freelancer_id=freelancer_id,
    )

    hold_res: dict[str, Any] = {}
    if chosen_slot and intent_obj.get("requires_calendar_slots"):
        hold_res = await _call_workspace_write(
            tool_name="create_tentative_interview_hold",
            arguments={
                "idempotency_key": f"hold:{run_id}:{thread_id}",
                "thread_id": thread_id,
                "job_id": job_id,
                "slot_start_iso": chosen_slot,
                "duration_minutes": 30,
                "title": f"Interview: {(state.get('job_context') or {}).get('title', job_id)[:50]}",
                "freelancer_id": freelancer_id,
                "dry_run": False,
            },
            trace_id=trace_id,
            freelancer_id=freelancer_id,
        )

    email_draft_res: dict[str, Any] = {}
    if human_decision.get("create_email_draft"):
        email_draft_res = await _call_workspace_write(
            tool_name="create_email_draft",
            arguments={
                "idempotency_key": f"email:{run_id}:{thread_id}",
                "thread_id": thread_id,
                "freelancer_id": freelancer_id,
                "recipient_alias": str(state.get("client_id", "client")),
                "subject": f"Interview Summary — {(state.get('job_context') or {}).get('title', job_id)[:60]}",
                "body": final_text,
                "dry_run": False,
            },
            trace_id=trace_id,
            freelancer_id=freelancer_id,
        )

    promo = promote_from_conversation(
        store=GigBidMemoryStore(),
        freelancer_id=freelancer_id,
        job_id=job_id,
        thread_id=thread_id,
        client_id=str(state.get("client_id", "cli_001")),
        intent=str(intent_obj.get("intent", "general_followup")),
        client_message=str(state.get("latest_client_message", "")),
        reply_text=final_text,
        compliance_passed=bool((state.get("compliance_verdict") or {}).get("passed", True)),
        evidence_ids=list(reply_draft.get("cited_evidence_ids") or []),
        proposed_counter_rate=intent_obj.get("proposed_counter_rate"),
    )

    duration_ms = (time.perf_counter() - t0) * 1000.0
    event = make_trace_event(
        node="conversation_approval_gate",
        phase="resumed",
        duration_ms=duration_ms,
        summary=(
            f"Human APPROVED reply | send_status={send_res.get('status')} "
            f"| hold_created={bool(hold_res)}"
        ),
    )
    return {
        "reply_draft": reply_draft,
        "selected_slot": chosen_slot,
        "human_approval": {"approved": True, **human_decision},
        "send_result": send_res,
        "hold_result": hold_res,
        "email_draft_result": email_draft_res,
        "promoted_memories": [m.model_dump(mode="json") for m in promo.promoted_items],
        "budgets": make_budget_delta(steps=1),
        "node_timings": {"conversation_approval_gate": round(duration_ms, 2)},
        "trace_events": [event],
        "status": str(send_res.get("status", "sent")),
    }


def build_conversation_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    """Build and compile the GigBid Conversation & Calendar Liaison LangGraph workflow."""
    builder = StateGraph(ConversationGraphState)

    builder.add_node("load_thread", load_thread_node)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("gather_context", gather_context_node)
    builder.add_node("draft_reply", draft_reply_node)
    builder.add_node("prepare_send_preview", prepare_send_preview_node)
    builder.add_node("conversation_approval_gate", conversation_approval_gate_node)

    builder.add_edge(START, "load_thread")
    builder.add_edge("load_thread", "classify_intent")
    builder.add_edge("classify_intent", "gather_context")
    builder.add_edge("gather_context", "draft_reply")
    builder.add_edge("draft_reply", "prepare_send_preview")
    builder.add_edge("prepare_send_preview", "conversation_approval_gate")
    builder.add_edge("conversation_approval_gate", END)

    saver = checkpointer or get_sync_or_local_checkpointer()
    return builder.compile(checkpointer=saver)


__all__ = ["build_conversation_graph"]
