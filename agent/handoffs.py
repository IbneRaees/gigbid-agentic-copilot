"""Strongly typed Pydantic v2 hand-off contracts for GigBid LangGraph nodes.

All inter-agent decisions, parallel branch outputs, critique loops, compliance
verdicts, and human-approval payloads flow through these schemas so every
state transition is validated, JSON-serializable for Postgres checkpointing,
and compatible with `agent/state.py` reducers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HandOffBase(BaseModel):
    """Base model enforcing strict validation while ignoring extra LLM keys."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class BudgetRange(HandOffBase):
    """Structured budget representation matching starter-kit conventions."""

    min: float = Field(..., ge=0.0, description="Minimum budget in currency units")
    max: float = Field(..., ge=0.0, description="Maximum budget in currency units")
    currency: str = Field(default="USD", min_length=3, max_length=6)
    budget_type: Literal["fixed", "hourly"] = Field(default="fixed")

    @model_validator(mode="after")
    def validate_min_le_max(self) -> BudgetRange:
        if self.max < self.min:
            self.max = self.min
        return self


class JobSpec(HandOffBase):
    """Structured extraction of a raw marketplace job posting."""

    job_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    deliverables: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    budget: BudgetRange
    duration_days: int = Field(default=14, ge=1)
    experience_level: Literal["entry", "intermediate", "expert"] = "intermediate"
    screening_questions: list[str] = Field(default_factory=list)
    hidden_needs: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    domain_category: str = Field(default="general")
    connects_cost: int = Field(default=6, ge=0)


class AnalystReport(HandOffBase):
    """Output of the Job Analyst node summarizing scope, effort, and clarity."""

    job_id: str = Field(..., min_length=1)
    job_spec: JobSpec
    scope_clarity_score: float = Field(..., ge=0.0, le=1.0)
    complexity_tier: Literal["low", "medium", "high", "enterprise"] = "medium"
    estimated_hours: float = Field(..., gt=0.0)
    key_success_criteria: list[str] = Field(default_factory=list)
    extracted_constraints: list[str] = Field(default_factory=list)
    injection_detected: bool = Field(default=False)
    sanitized_notes: str = Field(default="")


class WarningSignal(HandOffBase):
    """Individual risk, scam, policy, or prompt-injection indicator."""

    code: str = Field(..., min_length=1)
    severity: Literal["low", "medium", "high", "critical"]
    description: str = Field(..., min_length=1)
    evidence_snippet: str = Field(default="")


class RiskReport(HandOffBase):
    """Output of the Risk Assessor node evaluating client trust and safety."""

    job_id: str = Field(..., min_length=1)
    client_id: str = Field(..., min_length=1)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high", "critical"]
    client_trust_score: float = Field(..., ge=0.0, le=1.0)
    is_scam: bool = Field(default=False)
    has_prompt_injection: bool = Field(default=False)
    requests_free_work: bool = Field(default=False)
    requests_off_platform: bool = Field(default=False)
    warning_signals: list[WarningSignal] = Field(default_factory=list)
    recommendation: Literal["proceed", "caution", "block"] = "proceed"
    rationale: str = Field(..., min_length=1)


class RequirementMatch(HandOffBase):
    """Mapping from a single job requirement to verifiable portfolio evidence."""

    requirement: str = Field(..., min_length=1)
    matched: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    supporting_snippet: str = Field(default="")
    gap_note: str | None = None


class FitReport(HandOffBase):
    """Output of the Fit Matcher node grounded in Portfolio MCP evidence."""

    job_id: str = Field(..., min_length=1)
    freelancer_id: str = Field(..., min_length=1)
    fit_score: float = Field(..., ge=0.0, le=1.0)
    skill_coverage: float = Field(..., ge=0.0, le=1.0)
    domain_relevance: float = Field(..., ge=0.0, le=1.0)
    requirement_matches: list[RequirementMatch] = Field(default_factory=list)
    matched_evidence_ids: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=1)


class BidDecision(HandOffBase):
    """Output of the Bid-or-Skip decision gate combining Analyst, Risk, and Fit."""

    job_id: str = Field(..., min_length=1)
    action: Literal["bid", "skip", "watch"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    connects_cost: int = Field(..., ge=0)
    expected_win_probability: float = Field(..., ge=0.0, le=1.0)
    expected_value_usd: float
    primary_reasons: list[str] = Field(default_factory=list)
    blocking_risks: list[str] = Field(default_factory=list)
    auto_skipped: bool = Field(default=False)
    rationale: str = Field(..., min_length=1)


class PriceQuote(HandOffBase):
    """Output of the Pricing Strategist node with fee and Connects ROI analysis."""

    job_id: str = Field(..., min_length=1)
    pricing_model: Literal["fixed", "hourly"]
    recommended_amount: float = Field(..., gt=0.0)
    min_acceptable_amount: float = Field(..., gt=0.0)
    max_competitive_amount: float = Field(..., gt=0.0)
    currency: str = Field(default="USD", min_length=3, max_length=6)
    estimated_hours: float = Field(..., gt=0.0)
    effective_hourly_rate: float = Field(..., gt=0.0)
    marketplace_fee_usd: float = Field(default=0.0, ge=0.0)
    net_freelancer_earnings_usd: float = Field(default=0.0, ge=0.0)
    connects_cost: int = Field(default=6, ge=0)
    connects_roi_ratio: float = Field(default=0.0, ge=0.0)
    connects_worth_it: bool = Field(default=True)
    market_benchmark_median: float | None = None
    milestones: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = Field(..., min_length=1)


class ClaimCheck(HandOffBase):
    """Verification status of an individual factual claim inside a proposal draft."""

    claim_text: str = Field(..., min_length=1)
    status: Literal["supported", "partly_supported", "unsupported"]
    evidence_id: str | None = None
    support_score: float = Field(..., ge=0.0, le=1.0)
    revision_note: str | None = None


class ProposalDraft(HandOffBase):
    """Structured proposal draft produced by Proposal Writer and verified by RAG."""

    job_id: str = Field(..., min_length=1)
    freelancer_id: str = Field(..., min_length=1)
    version: int = Field(default=1, ge=1)
    cover_letter: str = Field(..., min_length=1)
    screening_answers: dict[str, str] = Field(default_factory=dict)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    word_count: int = Field(default=0, ge=0)
    proposed_rate: float = Field(..., gt=0.0)
    currency: str = Field(default="USD", min_length=3, max_length=6)
    connects_cost: int = Field(default=6, ge=0)

    @model_validator(mode="after")
    def compute_word_count(self) -> ProposalDraft:
        if self.word_count == 0 and self.cover_letter:
            self.word_count = len(self.cover_letter.split())
        return self


class Critique(HandOffBase):
    """Output of the Verifier & Critic node gating revision loops (max 2)."""

    job_id: str = Field(..., min_length=1)
    draft_version: int = Field(default=1, ge=1)
    passed: bool
    overall_score: float = Field(..., ge=0.0, le=10.0)
    grounding_score: float = Field(..., ge=0.0, le=10.0)
    personalization_score: float = Field(..., ge=0.0, le=10.0)
    clarity_score: float = Field(..., ge=0.0, le=10.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)
    requires_revision: bool = Field(default=False)


class ComplianceViolation(HandOffBase):
    """Single rule violation detected by the Compliance Guard."""

    rule_id: str = Field(..., min_length=1)
    category: Literal[
        "off_platform_contact",
        "off_platform_payment",
        "fabricated_claim",
        "prohibited_guarantee",
        "free_work",
        "policy_violation",
        "prompt_injection_leak",
    ]
    severity: Literal["warning", "blocking"]
    matched_text: str = Field(..., min_length=1)
    remediation: str = Field(..., min_length=1)


class ComplianceVerdict(HandOffBase):
    """Final policy check before human approval interrupt."""

    job_id: str = Field(..., min_length=1)
    passed: bool
    blocked: bool = Field(default=False)
    violations: list[ComplianceViolation] = Field(default_factory=list)
    sanitized_cover_letter: str | None = None
    sanitized_screening_answers: dict[str, str] = Field(default_factory=dict)
    policy_version: str = Field(default="2026.1")
    block_reason: str | None = None
    rationale: str = Field(..., min_length=1)


class ClientReplyIntent(HandOffBase):
    """Intent classification output for incoming client messages in Conversation Graph."""

    thread_id: str = Field(..., min_length=1)
    job_id: str = Field(..., min_length=1)
    client_id: str = Field(..., min_length=1)
    intent: Literal[
        "technical_question",
        "rate_negotiation",
        "scope_clarification",
        "interview_request",
        "offer_signal",
        "scam_or_offplatform",
        "rejection",
        "general_followup",
    ]
    confidence: float = Field(..., ge=0.0, le=1.0)
    sentiment: Literal["positive", "neutral", "skeptical", "negative"] = "neutral"
    requires_calendar_slots: bool = Field(default=False)
    requires_pricing_check: bool = Field(default=False)
    scam_or_policy_flag: bool = Field(default=False)
    warning_signals: list[WarningSignal] = Field(default_factory=list)
    extracted_questions: list[str] = Field(default_factory=list)
    proposed_counter_rate: float | None = None
    summary: str = Field(..., min_length=1)


class ClientReplyDraft(HandOffBase):
    """Suggested client reply and optional interview slots awaiting human approval."""

    thread_id: str = Field(..., min_length=1)
    job_id: str = Field(..., min_length=1)
    reply_text: str = Field(..., min_length=1)
    proposed_slots: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    compliance_passed: bool = Field(default=True)
    requires_human_approval: Literal[True] = True
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


__all__ = [
    "BudgetRange",
    "JobSpec",
    "AnalystReport",
    "WarningSignal",
    "RiskReport",
    "RequirementMatch",
    "FitReport",
    "BidDecision",
    "PriceQuote",
    "ClaimCheck",
    "ProposalDraft",
    "Critique",
    "ComplianceViolation",
    "ComplianceVerdict",
    "ClientReplyIntent",
    "ClientReplyDraft",
]