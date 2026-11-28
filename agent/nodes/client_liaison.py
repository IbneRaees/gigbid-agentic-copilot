"""Client Liaison Nodes: classifies `ClientReplyIntent`, checks Workspace MCP slots, drafts reply."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from agent.handoffs import (
    ClientReplyDraft,
    ClientReplyIntent,
    ComplianceVerdict,
    WarningSignal,
)
from agent.llm import ModelRouter
from agent.marketplace_port import build_marketplace_adapter
from agent.nodes._runtime import get_mcp_client_manager, get_workspace_store
from agent.nodes.compliance import run_compliance_checks
from agent.nodes.scout import load_freelancer_profile
from agent.state import ConversationGraphState, make_budget_delta, make_trace_event
from rag.retriever import GigBidRetriever

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

_OFF_PLATFORM_RE = re.compile(
    r"\b(whatsapp|telegram|discord|skype|signal|paypal|zelle|wise|usdt|crypto|avoid\s+fees|"
    r"outside\s+the\s+platform|paying\s+through\s+the\s+platform)\b|"
    r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|"
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    re.IGNORECASE,
)
_INTERVIEW_RE = re.compile(
    r"\b(interview|call|meeting|zoom|huddle|time\s+slots?|free\s+for\s+a|schedule|tomorrow|thursday)\b",
    re.IGNORECASE,
)
_RATE_RE = re.compile(
    r"\b(budget\s+ceiling|lower|discount|rate|fixed\s+rather\s+than|\$[\d,]+)\b",
    re.IGNORECASE,
)
_OFFER_RE = re.compile(
    r"\b(send\s+the\s+.*offer|contract\s+offer|ready\s+to\s+hire|accept\s+offer|starting\s+monday)\b",
    re.IGNORECASE,
)
_REJECT_RE = re.compile(
    r"\b(went\s+with\s+another|not\s+moving\s+forward|position\s+filled|declined|unfortunately)\b",
    re.IGNORECASE,
)
_SCOPE_RE = re.compile(
    r"\b(clarifying\s+scope|target\s+hardware|deliverable|narrow\s+milestone|vram|latency)\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)")

_SHARED_RETRIEVER: GigBidRetriever | None = None


def _get_retriever() -> GigBidRetriever:
    global _SHARED_RETRIEVER
    if _SHARED_RETRIEVER is None:
        _SHARED_RETRIEVER = GigBidRetriever(auto_build=True)
    return _SHARED_RETRIEVER


def _wrap_client_msg(text: str) -> str:
    return f"<UNTRUSTED_CLIENT_MSG>\n{text}\n</UNTRUSTED_CLIENT_MSG>"


async def _load_thread_data(
    *,
    thread_id: str,
    marketplace_mode: str,
    trace_id: str,
    freelancer_id: str,
) -> dict[str, Any]:
    mcp_mgr = get_mcp_client_manager(trace_id=trace_id, user_id=freelancer_id)
    if mcp_mgr is not None:
        try:
            async with mcp_mgr as mcp:
                res = await mcp.call_tool("marketplace", "list_messages", {"thread_id": thread_id})
                threads = res.get("threads", [])
                if threads:
                    return dict(threads[0])
        except Exception:
            pass

    try:
        adapter = build_marketplace_adapter(marketplace_mode)
        threads = await adapter.list_messages(thread_id=thread_id)
        if threads:
            return threads[0].model_dump(mode="json")
    except Exception:
        pass

    msg_path = DATA_DIR / "messages.json"
    if msg_path.exists():
        for t in json.loads(msg_path.read_text(encoding="utf-8")):
            if t.get("thread_id") == thread_id:
                return dict(t)
    return {
        "thread_id": thread_id,
        "job_id": "job_001",
        "client_id": "cli_001",
        "freelancer_id": freelancer_id,
        "status": "replied",
        "messages": [],
    }


def _load_job_record(job_id: str) -> dict[str, Any]:
    jobs_path = DATA_DIR / "jobs.json"
    if jobs_path.exists():
        for j in json.loads(jobs_path.read_text(encoding="utf-8")):
            if j.get("job_id") == job_id:
                return dict(j)
    return {"job_id": job_id, "title": "Project Discussion", "required_skills": []}


async def load_thread_node(state: ConversationGraphState) -> dict[str, Any]:
    """Load conversation thread, job context, and freelancer profile."""
    t0 = time.perf_counter()
    thread_id = state["thread_id"]
    freelancer_id = state.get("freelancer_id", "persona_aiml_01")
    marketplace_mode = state.get("marketplace_mode", "gigsim")
    trace_id = state.get("trace_id", "trace-conv")

    thread_rec = dict(state.get("thread_record") or {})
    if not thread_rec:
        thread_rec = await _load_thread_data(
            thread_id=thread_id,
            marketplace_mode=marketplace_mode,
            trace_id=trace_id,
            freelancer_id=freelancer_id,
        )

    messages = list(state.get("messages") or thread_rec.get("messages") or [])
    latest_msg = str(state.get("latest_client_message") or "")
    if not latest_msg:
        client_msgs = [m for m in messages if m.get("sender") == "client"]
        latest_msg = (
            str(client_msgs[-1]["text"])
            if client_msgs
            else (str(messages[-1]["text"]) if messages else "")
        )

    job_id = str(state.get("job_id") or thread_rec.get("job_id") or "job_001")
    client_id = str(state.get("client_id") or thread_rec.get("client_id") or "cli_001")
    resolved_freelancer = str(thread_rec.get("freelancer_id") or freelancer_id)

    job_ctx = state.get("job_context") or _load_job_record(job_id)
    profile = state.get("freelancer_profile") or load_freelancer_profile(resolved_freelancer)

    duration_ms = (time.perf_counter() - t0) * 1000.0
    event = make_trace_event(
        node="load_thread",
        phase="completed",
        duration_ms=duration_ms,
        summary=f"Loaded thread {thread_id} (job={job_id}, {len(messages)} messages)",
    )
    return {
        "thread_record": thread_rec,
        "messages": messages,
        "latest_client_message": latest_msg,
        "wrapped_client_message": _wrap_client_msg(latest_msg),
        "job_id": job_id,
        "client_id": client_id,
        "freelancer_id": resolved_freelancer,
        "job_context": job_ctx,
        "freelancer_profile": profile,
        "budgets": make_budget_delta(steps=1),
        "node_timings": {"load_thread": round(duration_ms, 2)},
        "trace_events": [event],
        "status": "thread_loaded",
    }


def _heuristic_classify_intent(
    *,
    thread_id: str,
    job_id: str,
    client_id: str,
    client_msg: str,
) -> ClientReplyIntent:
    warnings: list[WarningSignal] = []
    scam_flag = False
    requires_calendar = False
    requires_pricing = False
    counter_rate: float | None = None

    off_match = _OFF_PLATFORM_RE.search(client_msg)
    if off_match:
        scam_flag = True
        warnings.append(
            WarningSignal(
                code="POL-001_002_OFF_PLATFORM",
                severity="critical",
                description="Client message requests off-platform contact or payment.",
                evidence_snippet=off_match.group(0),
            )
        )
        intent = "scam_or_offplatform"
        sentiment = "negative"
    elif _REJECT_RE.search(client_msg):
        intent = "rejection"
        sentiment = "negative"
    elif _OFFER_RE.search(client_msg):
        intent = "offer_signal"
        sentiment = "positive"
        money_hits = _MONEY_RE.findall(client_msg)
        if money_hits:
            counter_rate = float(money_hits[0].replace(",", ""))
    elif _INTERVIEW_RE.search(client_msg):
        intent = "interview_request"
        sentiment = "positive"
        requires_calendar = True
    elif _RATE_RE.search(client_msg) and (
        "rather than" in client_msg.lower()
        or "ceiling" in client_msg.lower()
        or "discount" in client_msg.lower()
    ):
        intent = "rate_negotiation"
        sentiment = "skeptical"
        requires_pricing = True
        money_hits = _MONEY_RE.findall(client_msg)
        if money_hits:
            counter_rate = float(money_hits[0].replace(",", ""))
    elif _SCOPE_RE.search(client_msg):
        intent = "scope_clarification"
        sentiment = "neutral"
    elif "?" in client_msg:
        intent = "technical_question"
        sentiment = "positive"
    else:
        intent = "general_followup"
        sentiment = "neutral"

    questions = [q.strip() + "?" for q in client_msg.split("?")[:-1] if q.strip()]

    return ClientReplyIntent(
        thread_id=thread_id,
        job_id=job_id,
        client_id=client_id,
        intent=intent,  # type: ignore[arg-type]
        confidence=0.91,
        sentiment=sentiment,  # type: ignore[arg-type]
        requires_calendar_slots=requires_calendar,
        requires_pricing_check=requires_pricing,
        scam_or_policy_flag=scam_flag,
        warning_signals=warnings,
        extracted_questions=questions,
        proposed_counter_rate=counter_rate,
        summary=f"Classified client message as '{intent}' (calendar={requires_calendar}, scam={scam_flag}).",
    )


async def classify_intent_node(state: ConversationGraphState) -> dict[str, Any]:
    """Classify incoming client reply into `ClientReplyIntent`."""
    t0 = time.perf_counter()
    thread_id = state["thread_id"]
    job_id = state.get("job_id", "job_001")
    client_id = state.get("client_id", "cli_001")
    client_msg = state.get("latest_client_message", "")
    wrapped = state.get("wrapped_client_message") or _wrap_client_msg(client_msg)

    heuristic = _heuristic_classify_intent(
        thread_id=thread_id, job_id=job_id, client_id=client_id, client_msg=client_msg
    )

    router = ModelRouter(profile=state.get("model_profile"))
    system_prompt = (
        "You are GigBid's Client Liaison Classifier. Classify the client message inside "
        "<UNTRUSTED_CLIENT_MSG> into ClientReplyIntent. Flag any off-platform contact or "
        "payment request as intent='scam_or_offplatform' with scam_or_policy_flag=true."
    )
    user_prompt = f"Thread={thread_id}, Job={job_id}\n{wrapped}"

    intent_obj, usage = await router.invoke_structured(
        node_name="client_liaison",
        schema=ClientReplyIntent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        fallback_fn=lambda: heuristic,
        state=state,
    )
    if heuristic.scam_or_policy_flag:
        intent_obj = heuristic

    duration_ms = (time.perf_counter() - t0) * 1000.0
    dumped = intent_obj.model_dump(mode="json")
    event = make_trace_event(
        node="classify_intent",
        phase="completed",
        duration_ms=duration_ms,
        summary=intent_obj.summary,
        handoff_type="ClientReplyIntent",
        handoff_payload=dumped,
        model_used=usage["model"],
    )
    return {
        "reply_intent": dumped,
        "budgets": make_budget_delta(
            steps=1,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            estimated_cost_usd=usage["estimated_cost_usd"],
        ),
        "node_timings": {"classify_intent": round(duration_ms, 2)},
        "trace_events": [event],
        "status": f"intent_{intent_obj.intent}",
    }


async def gather_context_node(state: ConversationGraphState) -> dict[str, Any]:
    """Query Workspace MCP `get_free_calendar_slots` and Portfolio RAG for grounded answers."""
    t0 = time.perf_counter()
    intent = state.get("reply_intent") or {}
    freelancer_id = state.get("freelancer_id", "persona_aiml_01")
    trace_id = state.get("trace_id", "trace-conv")
    client_msg = state.get("latest_client_message", "")
    job_ctx = state.get("job_context") or {}

    slots: list[dict[str, Any]] = []
    if intent.get("requires_calendar_slots"):
        mcp_mgr = get_mcp_client_manager(trace_id=trace_id, user_id=freelancer_id)
        if mcp_mgr is not None:
            try:
                async with mcp_mgr as mcp:
                    res = await mcp.call_tool(
                        "workspace",
                        "get_free_calendar_slots",
                        {"days_ahead": 3, "slot_minutes": 30},
                    )
                    slots = list(res.get("slots", []))
            except Exception:
                pass
        if not slots:
            store = get_workspace_store()
            slots = store.get_free_slots(days_ahead=3, slot_minutes=30)

    retriever = _get_retriever()
    ev_chunks = retriever.retrieve_portfolio(
        query=f"{job_ctx.get('title', '')} {client_msg}",
        k=3,
        freelancer_id=freelancer_id,
    )
    ev_dicts = [c.model_dump(mode="json") for c in ev_chunks if c.doc_type != "freelancer_profile"]
    if not ev_dicts:
        ev_dicts = [c.model_dump(mode="json") for c in ev_chunks]

    ev_ids = [str(c.get("evidence_id")) for c in ev_dicts if c.get("evidence_id")]
    selected_slot = slots[0]["slot_start_iso"] if slots else None

    duration_ms = (time.perf_counter() - t0) * 1000.0
    event = make_trace_event(
        node="gather_context",
        phase="completed",
        duration_ms=duration_ms,
        summary=f"Retrieved {len(ev_dicts)} portfolio items and {len(slots)} free calendar slots.",
        rag_evidence_ids=ev_ids,
    )
    return {
        "retrieved_evidence": ev_dicts,
        "available_slots": slots[:5],
        "selected_slot": selected_slot,
        "budgets": make_budget_delta(steps=1),
        "node_timings": {"gather_context": round(duration_ms, 2)},
        "trace_events": [event],
    }


def _heuristic_reply_draft(
    *,
    thread_id: str,
    job_id: str,
    intent_obj: dict[str, Any],
    slots: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    freelancer_profile: dict[str, Any],
) -> ClientReplyDraft:
    intent = str(intent_obj.get("intent", "general_followup"))
    ev_ids = [str(e["evidence_id"]) for e in evidence if e.get("evidence_id")][:3]
    top_ev = evidence[0] if evidence else None
    ev_ref = (
        f"Based on {top_ev.get('title')} ({top_ev.get('evidence_id')})"
        if top_ev
        else "Based on my verified project experience"
    )
    proposed_slots = [str(s["slot_start_iso"]) for s in slots[:3]]

    if intent == "scam_or_offplatform":
        reply_text = (
            "Thank you for reaching out. To stay compliant with marketplace safety and escrow policies "
            "(POL-001/POL-002), I keep all project discussions, technical calls, and milestone payments "
            "exclusively within the platform. Please let me know if you would like to proceed via "
            "official platform milestones."
        )
        proposed_slots = []
        ev_ids = []
    elif intent == "interview_request":
        slot_labels = ", ".join(s.get("label", s["slot_start_iso"]) for s in slots[:3])
        tech_note = ""
        if intent_obj.get("extracted_questions") and top_ev:
            summary_snip = str(top_ev.get("content", "")).split(".")[0]
            tech_note = f" Regarding your technical question: {ev_ref}, {summary_snip}. "
        reply_text = (
            f"Thank you for the invitation.{tech_note}"
            f"I am available for a 30-minute architecture call at any of these UTC slots: "
            f"{slot_labels or 'tomorrow at 14:00 UTC or 16:00 UTC'}. "
            f"Please let me know which slot works best and I will lock it in."
        )
    elif intent == "rate_negotiation":
        counter = intent_obj.get("proposed_counter_rate")
        floor = float(freelancer_profile.get("min_fixed_budget_usd", 750.0))
        if counter and float(counter) >= floor:
            reply_text = (
                f"Thanks for sharing your Q4 budget parameters. Adjusting Milestone 3 to Python SDK "
                f"delivery without a custom UI works cleanly at ${float(counter):,.0f} fixed, while keeping "
                f"full retrieval benchmarking and verification checks intact ({', '.join(ev_ids) or 'verified workflow'})."
            )
        else:
            reply_text = (
                f"Thanks for clarifying your budget. To preserve production testing and verification quality, "
                f"we can phase Milestone 1 at ${max(floor, 1500.0):,.0f} first and defer secondary features to Milestone 2."
            )
    elif intent == "offer_signal":
        reply_text = (
            "That timeline and Milestone 1 structure work great for my schedule. "
            "Please go ahead and send the milestone contract on the platform so we can kick off immediately."
        )
    elif intent in {"technical_question", "scope_clarification"}:
        detail = (
            str(top_ev.get("parent_content") or top_ev.get("content") or "").split(".")[0]
            if top_ev
            else ""
        )
        reply_text = (
            f"Great question. {ev_ref}, {detail or 'we validate performance with deterministic CI benchmarks'}. "
            f"Happy to share the benchmark breakdown on a brief platform call if helpful."
        )
    else:
        reply_text = "Thank you for the update! Please let me know how you would like to proceed with the next milestone."

    return ClientReplyDraft(
        thread_id=thread_id,
        job_id=job_id,
        reply_text=reply_text,
        proposed_slots=proposed_slots,
        cited_evidence_ids=ev_ids,
        compliance_passed=True,
    )


async def draft_reply_node(state: ConversationGraphState) -> dict[str, Any]:
    """Draft grounded client reply & enforce compliance before human approval."""
    t0 = time.perf_counter()
    thread_id = state["thread_id"]
    job_id = state.get("job_id", "job_001")
    intent_obj = state.get("reply_intent") or {}
    slots = state.get("available_slots") or []
    evidence = state.get("retrieved_evidence") or []
    profile = state.get("freelancer_profile") or {}

    heuristic = _heuristic_reply_draft(
        thread_id=thread_id,
        job_id=job_id,
        intent_obj=intent_obj,
        slots=slots,
        evidence=evidence,
        freelancer_profile=profile,
    )

    router = ModelRouter(profile=state.get("model_profile"))
    if intent_obj.get("intent") == "scam_or_offplatform":
        draft = heuristic
        usage = {
            "model": "deterministic-compliance-refusal",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    else:
        system_prompt = (
            "You are GigBid's Client Liaison. Draft a concise, professional reply citing only "
            "retrieved evidence_ids and provided calendar slots. NEVER share personal email/phone "
            "or agree to off-platform payment (POL-001/POL-002)."
        )
        user_prompt = (
            f"Intent: {intent_obj}\n"
            f"Available Slots: {slots[:3]}\n"
            f"Evidence: {evidence}\n"
            f"Client Message: {state.get('wrapped_client_message')}"
        )
        draft, usage = await router.invoke_structured(
            node_name="client_liaison",
            schema=ClientReplyDraft,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_fn=lambda: heuristic,
            state=state,
        )

    # Compliance check on reply text
    comp: ComplianceVerdict = run_compliance_checks(
        job_id=job_id,
        cover_letter=draft.reply_text,
        screening_answers={},
    )
    final_reply_text = comp.sanitized_cover_letter or draft.reply_text
    draft = draft.model_copy(
        update={
            "thread_id": thread_id,
            "job_id": job_id,
            "reply_text": final_reply_text,
            "compliance_passed": comp.passed,
            "requires_human_approval": True,
        }
    )

    duration_ms = (time.perf_counter() - t0) * 1000.0
    dumped = draft.model_dump(mode="json")
    event = make_trace_event(
        node="draft_reply",
        phase="completed",
        duration_ms=duration_ms,
        summary=(
            f"Drafted client reply (slots={len(draft.proposed_slots)}, "
            f"compliance_passed={comp.passed})"
        ),
        handoff_type="ClientReplyDraft",
        handoff_payload=dumped,
        rag_evidence_ids=draft.cited_evidence_ids,
        model_used=usage["model"],
    )
    return {
        "reply_draft": dumped,
        "compliance_verdict": comp.model_dump(mode="json"),
        "budgets": make_budget_delta(
            steps=1,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            estimated_cost_usd=usage["estimated_cost_usd"],
        ),
        "node_timings": {"draft_reply": round(duration_ms, 2)},
        "trace_events": [event],
        "status": "reply_drafted",
    }


__all__ = [
    "load_thread_node",
    "classify_intent_node",
    "gather_context_node",
    "draft_reply_node",
]
