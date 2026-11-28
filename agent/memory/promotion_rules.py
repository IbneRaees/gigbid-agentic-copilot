"""Episodic-to-Semantic Memory Promotion Rules for GigBid.

Enforces strict grounding and compliance gates before any interaction artifact
is promoted into reusable semantic memory:
  - PROMO-001: Reject promotion when `grounding_score < 0.85` or `not compliance_passed`.
  - PROMO-002: Promote winning/interview proposal structures citing verified `evidence_id`s.
  - PROMO-003: Promote verified screening Q&A snippets for reuse across similar jobs.
  - PROMO-004: Promote client preferences and risk patterns discovered in threads.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from agent.memory.store import (
    EpisodicEvent,
    GigBidMemoryStore,
    SemanticMemoryItem,
    make_memory_id,
)


class PromotionDecision(BaseModel):
    """Result of applying episodic-to-semantic promotion rules."""

    rule_ids: list[str] = Field(default_factory=list)
    promoted: bool
    reason: str
    promoted_items: list[SemanticMemoryItem] = Field(default_factory=list)


def evaluate_bid_promotion(
    event: EpisodicEvent,
    *,
    min_grounding_score: float = 0.85,
) -> PromotionDecision:
    """Evaluate whether a bid workflow outcome qualifies for semantic memory promotion."""
    # PROMO-004: Scam / policy traps always promote a defensive risk pattern
    if event.event_type == "skip_decision" and "policy_trap" in event.tags:
        risk_item = SemanticMemoryItem(
            memory_id=make_memory_id(
                "mem_risk", f"{event.client_id or event.job_id}:{event.summary[:80]}"
            ),
            freelancer_id=event.freelancer_id,
            memory_type="risk_pattern",
            title=f"Blocked risk pattern for {event.client_id or event.job_id}",
            content=event.summary,
            evidence_ids=[],
            tags=list(dict.fromkeys([*event.tags, "compliance_block"])),
            confidence=0.95,
            source_event_ids=[event.event_id],
        )
        return PromotionDecision(
            rule_ids=["PROMO-004"],
            promoted=True,
            reason="Recorded defensive risk pattern from policy-blocked job post.",
            promoted_items=[risk_item],
        )

    # PROMO-001: Safety & Grounding Gate
    if not event.compliance_passed:
        return PromotionDecision(
            rule_ids=["PROMO-001"],
            promoted=False,
            reason="Blocked promotion: compliance check did not pass.",
            promoted_items=[],
        )
    if event.grounding_score < min_grounding_score:
        return PromotionDecision(
            rule_ids=["PROMO-001"],
            promoted=False,
            reason=(
                f"Blocked promotion: grounding_score={event.grounding_score:.2f} "
                f"below threshold {min_grounding_score:.2f}."
            ),
            promoted_items=[],
        )

    promoted_items: list[SemanticMemoryItem] = []
    applied_rules: list[str] = []

    # PROMO-002: Winning / Interview / High-Grounding Approved Proposals
    if event.outcome in {"won", "interview", "approved"} and event.evidence_ids:
        confidence = (
            0.94 if event.outcome == "won" else (0.88 if event.outcome == "interview" else 0.82)
        )
        domain = str(event.metadata.get("domain_category", "general"))
        pattern_item = SemanticMemoryItem(
            memory_id=make_memory_id(
                "mem_win",
                f"{event.freelancer_id}:{domain}:{'-'.join(sorted(event.evidence_ids[:3]))}",
            ),
            freelancer_id=event.freelancer_id,
            memory_type="winning_pattern",
            title=f"Grounded {domain} proposal pattern ({event.outcome})",
            content=(
                f"Outcome={event.outcome} | Grounding={event.grounding_score:.2f} | "
                f"Evidence={', '.join(event.evidence_ids)} | {event.summary}"
            ),
            evidence_ids=list(event.evidence_ids),
            tags=list(dict.fromkeys([*event.tags, domain, event.outcome])),
            confidence=confidence,
            source_event_ids=[event.event_id],
        )
        promoted_items.append(pattern_item)
        applied_rules.append("PROMO-002")

    # PROMO-003: Reusable grounded screening Q&A snippets
    if event.screening_answers and event.evidence_ids:
        for question, answer in list(event.screening_answers.items())[:3]:
            if len(answer.strip()) < 20:
                continue
            qa_item = SemanticMemoryItem(
                memory_id=make_memory_id(
                    "mem_qa", f"{event.freelancer_id}:{question.strip().lower()[:100]}"
                ),
                freelancer_id=event.freelancer_id,
                memory_type="reusable_qa",
                title=question[:120],
                content=answer.strip(),
                evidence_ids=list(event.evidence_ids),
                tags=list(dict.fromkeys([*event.tags, "screening_qa"])),
                confidence=min(0.95, max(0.78, event.grounding_score)),
                source_event_ids=[event.event_id],
            )
            promoted_items.append(qa_item)
        if promoted_items and "PROMO-003" not in applied_rules:
            applied_rules.append("PROMO-003")

    if not promoted_items:
        return PromotionDecision(
            rule_ids=["PROMO-001"],
            promoted=False,
            reason=f"Outcome '{event.outcome}' did not meet promotion criteria.",
            promoted_items=[],
        )

    return PromotionDecision(
        rule_ids=applied_rules,
        promoted=True,
        reason=f"Promoted {len(promoted_items)} semantic memory item(s) via {applied_rules}.",
        promoted_items=promoted_items,
    )


def promote_from_bid_outcome(
    *,
    store: GigBidMemoryStore,
    freelancer_id: str,
    job_id: str,
    client_id: str | None,
    outcome: str,
    summary: str,
    grounding_score: float,
    compliance_passed: bool,
    evidence_ids: list[str],
    screening_answers: dict[str, str] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    min_grounding_score: float = 0.85,
) -> PromotionDecision:
    """Record an episodic bid event and promote qualifying items into semantic memory."""
    event = EpisodicEvent(
        event_id=f"ep_bid_{uuid.uuid4().hex[:10]}",
        freelancer_id=freelancer_id,
        job_id=job_id,
        client_id=client_id,
        event_type="skip_decision" if outcome == "skipped" else "proposal_outcome",
        outcome=outcome,  # type: ignore[arg-type]
        summary=summary,
        grounding_score=grounding_score,
        compliance_passed=compliance_passed,
        evidence_ids=evidence_ids,
        screening_answers=screening_answers or {},
        tags=tags or [],
        metadata=metadata or {},
    )
    store.add_episodic(event)

    decision = evaluate_bid_promotion(event, min_grounding_score=min_grounding_score)
    persisted: list[SemanticMemoryItem] = []
    if decision.promoted:
        for item in decision.promoted_items:
            persisted.append(store.upsert_semantic(item))
    return decision.model_copy(update={"promoted_items": persisted})


def promote_from_conversation(
    *,
    store: GigBidMemoryStore,
    freelancer_id: str,
    job_id: str,
    thread_id: str,
    client_id: str,
    intent: str,
    client_message: str,
    reply_text: str,
    compliance_passed: bool,
    evidence_ids: list[str] | None = None,
    proposed_counter_rate: float | None = None,
) -> PromotionDecision:
    """Record a conversation episode and promote client preferences or risk patterns."""
    outcome: Any = (
        "interview"
        if intent == "interview_request"
        else ("rejected" if intent == "rejection" else "replied")
    )
    event = EpisodicEvent(
        event_id=f"ep_conv_{uuid.uuid4().hex[:10]}",
        freelancer_id=freelancer_id,
        job_id=job_id,
        thread_id=thread_id,
        client_id=client_id,
        event_type="client_feedback",
        outcome=outcome,
        summary=f"Intent={intent}: {client_message[:180]}",
        grounding_score=1.0 if compliance_passed else 0.0,
        compliance_passed=compliance_passed,
        evidence_ids=evidence_ids or [],
        tags=["conversation", intent],
        metadata={"proposed_counter_rate": proposed_counter_rate},
    )
    store.add_episodic(event)

    if intent == "scam_or_offplatform":
        risk_item = store.upsert_semantic(
            SemanticMemoryItem(
                memory_id=make_memory_id("mem_client_risk", f"{client_id}:offplatform"),
                freelancer_id=freelancer_id,
                memory_type="risk_pattern",
                title=f"Client {client_id} requested off-platform contact/payment",
                content=f"Thread {thread_id} flagged for POL-001/POL-002 violation: {client_message[:200]}",
                evidence_ids=[],
                tags=["scam_or_offplatform", client_id],
                confidence=0.98,
                source_event_ids=[event.event_id],
            )
        )
        return PromotionDecision(
            rule_ids=["PROMO-004"],
            promoted=True,
            reason="Promoted client off-platform risk pattern.",
            promoted_items=[risk_item],
        )

    if not compliance_passed:
        return PromotionDecision(
            rule_ids=["PROMO-001"],
            promoted=False,
            reason="Blocked conversation promotion due to compliance failure.",
            promoted_items=[],
        )

    if intent in {"rate_negotiation", "scope_clarification", "offer_signal", "technical_question"}:
        pref_detail = (
            f"Budget ceiling ${proposed_counter_rate:,.0f}. " if proposed_counter_rate else ""
        ) + client_message[:220]
        pref_item = store.upsert_semantic(
            SemanticMemoryItem(
                memory_id=make_memory_id("mem_pref", f"{client_id}:{job_id}:{intent}"),
                freelancer_id=freelancer_id,
                memory_type="client_preference",
                title=f"Client {client_id} preference ({intent})",
                content=pref_detail,
                evidence_ids=evidence_ids or [],
                tags=[intent, client_id, job_id],
                confidence=0.86,
                source_event_ids=[event.event_id],
            )
        )
        return PromotionDecision(
            rule_ids=["PROMO-004"],
            promoted=True,
            reason=f"Promoted client preference from '{intent}' thread.",
            promoted_items=[pref_item],
        )

    return PromotionDecision(
        rule_ids=[],
        promoted=False,
        reason=f"No semantic promotion rule triggered for intent '{intent}'.",
        promoted_items=[],
    )


__all__ = [
    "PromotionDecision",
    "evaluate_bid_promotion",
    "promote_from_bid_outcome",
    "promote_from_conversation",
]
