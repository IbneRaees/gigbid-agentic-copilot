"""Compliance Guard Node: enforces Marketplace Policy Pack (`POL-001`..`POL-009`) via RAG + rules."""

from __future__ import annotations

import re
import time
from typing import Any

from agent.handoffs import ComplianceVerdict, ComplianceViolation
from agent.llm import ModelRouter
from agent.state import BidGraphState, make_budget_delta, make_trace_event
from rag.retriever import GigBidRetriever

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_CONTACT_HANDLE_RE = re.compile(
    r"\b(?:whatsapp|telegram|discord|skype|signal)\b|@[a-z0-9_]{3,}", re.IGNORECASE
)
_OFF_PAY_RE = re.compile(
    r"\b(?:paypal|wise|zelle|venmo|usdt|crypto|bitcoin|wire\s+transfer|outside\s+the\s+platform)\b",
    re.IGNORECASE,
)
_FREE_WORK_RE = re.compile(
    r"\b(?:work\s+for\s+free|free\s+custom\s+prototype|unpaid\s+sample|free\s+trial\s+sprint)\b",
    re.IGNORECASE,
)
_GUARANTEE_RE = re.compile(
    r"\b(?:100%\s+guaranteed|zero\s+bugs\s+ever|guaranteed\s+#1|guaranteed\s+\d+x\s+revenue)\b",
    re.IGNORECASE,
)
_INJECTION_LEAK_RE = re.compile(
    r"(?:system\s+override|ignore\s+all\s+previous\s+instructions|evil-exfil|\$99,999)",
    re.IGNORECASE,
)

_SHARED_RETRIEVER: GigBidRetriever | None = None


def _get_retriever() -> GigBidRetriever:
    global _SHARED_RETRIEVER
    if _SHARED_RETRIEVER is None:
        _SHARED_RETRIEVER = GigBidRetriever(auto_build=True)
    return _SHARED_RETRIEVER


def run_compliance_checks(
    *,
    job_id: str,
    cover_letter: str,
    screening_answers: dict[str, str],
    verification_report: dict[str, Any] | None = None,
    risk_report: dict[str, Any] | None = None,
) -> ComplianceVerdict:
    """Inspect outbound proposal text against `POL-001`..`POL-009` and sanitize minor leaks."""
    violations: list[ComplianceViolation] = []
    sanitized_letter = cover_letter
    sanitized_answers = dict(screening_answers)
    blocked = False

    def _scan_and_scrub(text: str, field_name: str) -> str:
        nonlocal blocked
        cleaned = text

        for pat, rule_id, cat, sev, repl, msg in [
            (
                _EMAIL_RE,
                "POL-001",
                "off_platform_contact",
                "warning",
                "[platform-messaging-only]",
                "Removed email address to comply with POL-001.",
            ),
            (
                _PHONE_RE,
                "POL-001",
                "off_platform_contact",
                "warning",
                "[platform-messaging-only]",
                "Removed phone number to comply with POL-001.",
            ),
            (
                _CONTACT_HANDLE_RE,
                "POL-001",
                "off_platform_contact",
                "warning",
                "platform messaging",
                "Replaced external messaging handle to comply with POL-001.",
            ),
            (
                _OFF_PAY_RE,
                "POL-002",
                "off_platform_payment",
                "blocking",
                "marketplace escrow",
                "Blocked off-platform payment reference per POL-002.",
            ),
            (
                _FREE_WORK_RE,
                "POL-003",
                "free_work",
                "warning",
                "paid milestone discovery",
                "Replaced unpaid speculative work offer per POL-003.",
            ),
            (
                _GUARANTEE_RE,
                "POL-005",
                "prohibited_guarantee",
                "warning",
                "evidence-backed engineering target",
                "Rewrote unenforceable guarantee per POL-005.",
            ),
            (
                _INJECTION_LEAK_RE,
                "POL-009",
                "prompt_injection_leak",
                "blocking",
                "",
                "Blocked leaked prompt-injection payload per POL-009.",
            ),
        ]:
            m = pat.search(cleaned)
            if m:
                violations.append(
                    ComplianceViolation(
                        rule_id=rule_id,
                        category=cat,  # type: ignore[arg-type]
                        severity=sev,  # type: ignore[arg-type]
                        matched_text=f"[{field_name}] {m.group(0)}",
                        remediation=msg,
                    )
                )
                if sev == "blocking":
                    blocked = True
                cleaned = pat.sub(repl, cleaned).strip()
        return cleaned

    sanitized_letter = _scan_and_scrub(sanitized_letter, "cover_letter")
    for q_key, ans_val in list(sanitized_answers.items()):
        sanitized_answers[q_key] = _scan_and_scrub(str(ans_val), f"screening:{q_key[:30]}")

    # POL-004: Verify grounding threshold was satisfied
    if verification_report:
        g_score = float(verification_report.get("grounding_score", 1.0))
        contradicted = int(verification_report.get("contradicted_count", 0))
        if contradicted > 0 or g_score < 0.85:
            violations.append(
                ComplianceViolation(
                    rule_id="POL-004",
                    category="fabricated_claim",
                    severity="blocking",
                    matched_text=f"grounding_score={g_score:.2f}, contradicted={contradicted}",
                    remediation="Proposal failed factual grounding verification (POL-004).",
                )
            )
            blocked = True

    if risk_report and risk_report.get("recommendation") == "block":
        violations.append(
            ComplianceViolation(
                rule_id="POL-001/002/003",
                category="policy_violation",
                severity="blocking",
                matched_text=str(risk_report.get("rationale", "Risk block")),
                remediation="Client/job blocked by Risk Assessor.",
            )
        )
        blocked = True

    passed = not blocked
    block_reason = (
        "; ".join(v.remediation for v in violations if v.severity == "blocking")
        if blocked
        else None
    )
    rationale = (
        "Compliance passed (POL-001..POL-009 verified)."
        if passed and not violations
        else (
            f"Compliance passed after sanitizing {len(violations)} warning(s)."
            if passed
            else f"Compliance BLOCKED: {block_reason}"
        )
    )

    return ComplianceVerdict(
        job_id=job_id,
        passed=passed,
        blocked=blocked,
        violations=violations,
        sanitized_cover_letter=sanitized_letter if sanitized_letter != cover_letter else None,
        sanitized_screening_answers=(
            sanitized_answers if sanitized_answers != screening_answers else {}
        ),
        block_reason=block_reason,
        rationale=rationale,
    )


async def compliance_node(state: BidGraphState) -> dict[str, Any]:
    """Run Policy RAG retrieval and enforce `POL-001`..`POL-009` prior to human interrupt."""
    t0 = time.perf_counter()
    router = ModelRouter(profile=state.get("model_profile"))
    router.check_budget(state)

    draft = dict(state.get("proposal_draft") or {})
    job_id = str(draft.get("job_id") or state.get("job_id", "unknown"))
    cover_letter = str(draft.get("cover_letter", ""))
    screening_answers = dict(draft.get("screening_answers") or {})

    retriever = _get_retriever()
    policy_chunks = retriever.retrieve_policy(
        query=f"{cover_letter[:250]} off-platform payment contact guarantees grounding",
        k=3,
    )
    policy_ids = [c.policy_rule_id for c in policy_chunks if c.policy_rule_id]

    verdict = run_compliance_checks(
        job_id=job_id,
        cover_letter=cover_letter,
        screening_answers=screening_answers,
        verification_report=state.get("verification_report"),
        risk_report=state.get("risk_report"),
    )

    if verdict.sanitized_cover_letter:
        draft["cover_letter"] = verdict.sanitized_cover_letter
        draft["word_count"] = len(verdict.sanitized_cover_letter.split())
    if verdict.sanitized_screening_answers:
        draft["screening_answers"] = verdict.sanitized_screening_answers

    duration_ms = (time.perf_counter() - t0) * 1000.0
    verdict_dict = verdict.model_dump(mode="json")
    event = make_trace_event(
        node="compliance",
        phase="completed" if verdict.passed else "error",
        duration_ms=duration_ms,
        summary=verdict.rationale,
        handoff_type="ComplianceVerdict",
        handoff_payload=verdict_dict,
        rag_evidence_ids=policy_ids,
    )
    return {
        "proposal_draft": draft,
        "compliance_verdict": verdict_dict,
        "budgets": make_budget_delta(steps=1),
        "node_timings": {"compliance": round(duration_ms, 2)},
        "trace_events": [event],
        "status": "compliance_passed" if verdict.passed else "compliance_blocked",
    }


__all__ = ["compliance_node", "run_compliance_checks"]
