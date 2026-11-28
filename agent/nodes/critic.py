"""Verifier & Critic Node: runs `ClaimVerifier.verify()` and gates bounded revisions (`<= 2`)."""

from __future__ import annotations

import time
from typing import Any

from agent.handoffs import ClaimCheck, Critique
from agent.llm import ModelRouter
from agent.state import BidGraphState, make_trace_event
from rag.retriever import EvidenceChunk
from rag.verify import ClaimVerifier

_SHARED_VERIFIER: ClaimVerifier | None = None


def _get_verifier() -> ClaimVerifier:
    global _SHARED_VERIFIER
    if _SHARED_VERIFIER is None:
        _SHARED_VERIFIER = ClaimVerifier(min_grounding_score=0.85)
    return _SHARED_VERIFIER


def _to_evidence_chunks(raw_chunks: list[dict[str, Any]]) -> list[EvidenceChunk]:
    out: list[EvidenceChunk] = []
    for item in raw_chunks:
        try:
            out.append(EvidenceChunk.model_validate(item))
        except Exception:
            continue
    return out


async def critic_node(state: BidGraphState) -> dict[str, Any]:
    """Verify every claim in `ProposalDraft`; trigger revision or `sanitize_proposal()`."""
    t0 = time.perf_counter()
    router = ModelRouter(profile=state.get("model_profile"))
    router.check_budget(state)

    draft = dict(state.get("proposal_draft") or {})
    freelancer_profile = state.get("freelancer_profile") or {}
    evidence_chunks = _to_evidence_chunks(state.get("retrieved_evidence") or [])
    revision_count = int(state.get("revision_count", 0))
    max_revisions = int((state.get("budgets") or {}).get("max_revisions", 2))

    verifier = _get_verifier()
    cover_letter = str(draft.get("cover_letter", ""))
    screening_answers = draft.get("screening_answers") or {}

    v_report = verifier.verify(
        cover_letter=cover_letter,
        evidence_chunks=evidence_chunks,
        freelancer_profile=freelancer_profile,
        screening_answers=screening_answers,
    )

    sanitized_applied = False
    # If unsupported claims persist when revision cap is reached, strip them deterministically
    if (not v_report.passed or v_report.grounding_score < 0.85) and revision_count >= max_revisions:
        sanitized_letter = verifier.sanitize_proposal(cover_letter, v_report)
        if sanitized_letter.strip():
            cover_letter = sanitized_letter
            draft["cover_letter"] = cover_letter
            draft["word_count"] = len(cover_letter.split())
            sanitized_applied = True
            v_report = verifier.verify(
                cover_letter=cover_letter,
                evidence_chunks=evidence_chunks,
                freelancer_profile=freelancer_profile,
                screening_answers=screening_answers,
            )

    # Convert ClaimVerificationResult -> ProposalDraft ClaimCheck entries
    claim_checks: list[dict[str, Any]] = []
    for c in v_report.claims:
        if c.status == "GENERIC":
            continue
        mapped_status = "supported" if c.status == "SUPPORTED" else "unsupported"
        claim_checks.append(
            ClaimCheck(
                claim_text=c.claim_text,
                status=mapped_status,  # type: ignore[arg-type]
                evidence_id=c.matched_evidence_ids[0] if c.matched_evidence_ids else None,
                support_score=c.confidence,
                revision_note=None if mapped_status == "supported" else c.reason,
            ).model_dump(mode="json")
        )
    draft["claim_checks"] = claim_checks

    grounding_10 = round(v_report.grounding_score * 10.0, 2)
    word_count = len(cover_letter.split())
    clarity_10 = 9.2 if 40 <= word_count <= 260 else 7.8
    cited_cnt = len(draft.get("cited_evidence_ids") or [])
    personalization_10 = round(min(9.8, 7.0 + 0.8 * cited_cnt), 2)
    overall_10 = round((0.55 * grounding_10 + 0.25 * personalization_10 + 0.20 * clarity_10), 2)

    needs_revision = (
        not v_report.passed or v_report.grounding_score < 0.85
    ) and revision_count < max_revisions

    issues = list(v_report.unsupported_claims)
    instructions = (
        [f"Remove or ground unsupported claim: {u}" for u in v_report.unsupported_claims]
        if needs_revision
        else []
    )
    if sanitized_applied:
        issues.append(
            "Max revisions reached; unsupported sentences removed via sanitize_proposal()."
        )

    critique = Critique(
        job_id=str(draft.get("job_id") or state.get("job_id", "unknown")),
        draft_version=int(draft.get("version", 1)),
        passed=(not needs_revision and v_report.passed),
        overall_score=overall_10,
        grounding_score=grounding_10,
        personalization_score=personalization_10,
        clarity_score=clarity_10,
        unsupported_claims=v_report.unsupported_claims,
        issues=issues,
        revision_instructions=instructions,
        requires_revision=needs_revision,
    )

    next_revision_count = revision_count + 1 if needs_revision else revision_count
    duration_ms = (time.perf_counter() - t0) * 1000.0
    critique_dict = critique.model_dump(mode="json")
    ver_dict = v_report.model_dump(mode="json")

    event = make_trace_event(
        node="critic",
        phase="completed",
        duration_ms=duration_ms,
        summary=(
            f"Grounding={v_report.grounding_score:.2f} ({v_report.supported_count}/"
            f"{v_report.verifiable_claims} supported) | requires_revision={needs_revision} "
            f"| sanitized={sanitized_applied}"
        ),
        handoff_type="Critique",
        handoff_payload=critique_dict,
        verification_score=v_report.grounding_score,
        rag_evidence_ids=list(draft.get("cited_evidence_ids") or []),
    )

    return {
        "proposal_draft": draft,
        "verification_report": ver_dict,
        "critique": critique_dict,
        "revision_count": next_revision_count,
        "budgets": {
            "_is_delta": True,
            "steps_used": 1,
            "revision_count": next_revision_count,
        },
        "node_timings": {"critic": round(duration_ms, 2)},
        "trace_events": [event],
        "status": "revision_requested" if needs_revision else "critique_passed",
    }


__all__ = ["critic_node"]
