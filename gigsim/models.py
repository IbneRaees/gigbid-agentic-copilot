"""Strongly typed Pydantic v2 models for the GigSim Marketplace Simulator."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GigSimBase(BaseModel):
    """Base model with whitespace stripping and strict schema validation."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class HealthResponse(GigSimBase):
    """Service health and database seed status."""

    status: Literal["ok"] = "ok"
    service: str = "gigsim"
    version: str = "0.2.0"
    jobs_count: int
    clients_count: int
    threads_count: int


class BudgetInfo(GigSimBase):
    """Job budget structure aligned with GigBid handoff contracts."""

    min: float = Field(..., ge=0.0)
    max: float = Field(..., ge=0.0)
    currency: str = Field(default="USD")
    budget_type: Literal["fixed", "hourly"] = "fixed"


class JobRecord(GigSimBase):
    """Marketplace job posting returned by GigSim."""

    job_id: str
    client_id: str
    title: str
    description: str
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    budget: BudgetInfo
    duration_days: int = 14
    experience_level: str = "intermediate"
    screening_questions: list[str] = Field(default_factory=list)
    connects_cost: int = Field(default=6, ge=0)
    posted_at: str
    domain_category: str = "general"
    data_provenance: str = "synthetic"


class JobListResponse(GigSimBase):
    """Paginated/filtered job search response."""

    total: int
    limit: int
    offset: int
    jobs: list[JobRecord]


class ClientRecord(GigSimBase):
    """Client profile and historical hiring metrics."""

    client_id: str
    name: str
    country: str
    payment_verified: bool
    rating: float
    reviews_count: int
    jobs_posted: int
    hire_rate: float
    total_spent_usd: float
    avg_hourly_paid: float
    member_since: str
    past_reviews: list[str] = Field(default_factory=list)
    data_provenance: str = "synthetic"


class ConnectsLedgerEntry(GigSimBase):
    """Immutable audit record of Connects balance changes."""

    entry_id: str
    freelancer_id: str
    delta: int
    balance_after: int
    reason: str
    job_id: str | None = None
    proposal_id: str | None = None
    created_at: str


class ConnectsBalanceResponse(GigSimBase):
    """Current Connects balance and recent ledger entries for a freelancer."""

    freelancer_id: str
    balance: int
    weekly_budget: int
    recent_ledger: list[ConnectsLedgerEntry] = Field(default_factory=list)


class ProposalDraftRequest(GigSimBase):
    """Payload to create or update a non-submitted proposal draft."""

    idempotency_key: str = Field(..., min_length=4, max_length=128)
    job_id: str = Field(..., min_length=1)
    freelancer_id: str = Field(default="persona_aiml_01", min_length=1)
    cover_letter: str = Field(..., min_length=20, max_length=5000)
    screening_answers: dict[str, str] = Field(default_factory=dict)
    proposed_rate: float = Field(..., gt=0.0)
    currency: str = Field(default="USD", min_length=3, max_length=6)
    cited_evidence_ids: list[str] = Field(default_factory=list)


class ProposalSubmitRequest(GigSimBase):
    """Payload to preview (dry_run=True) or submit (dry_run=False) a proposal."""

    idempotency_key: str = Field(..., min_length=4, max_length=128)
    job_id: str = Field(..., min_length=1)
    freelancer_id: str = Field(default="persona_aiml_01", min_length=1)
    cover_letter: str = Field(..., min_length=20, max_length=5000)
    screening_answers: dict[str, str] = Field(default_factory=dict)
    proposed_rate: float = Field(..., gt=0.0)
    currency: str = Field(default="USD", min_length=3, max_length=6)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    dry_run: bool = Field(
        default=False,
        description="When true, validates policy and calculates Connects cost without mutating state.",
    )


class DryRunPreview(GigSimBase):
    """Detailed pre-flight summary shown on the GigBid human approval screen."""

    job_id: str
    job_title: str
    freelancer_id: str
    proposed_rate: float
    currency: str
    estimated_fee_usd: float
    estimated_net_usd: float
    connects_cost: int
    current_connects_balance: int
    remaining_connects_after: int
    policy_passed: bool
    policy_warnings: list[str] = Field(default_factory=list)


class ProposalResponse(GigSimBase):
    """Response returned by draft, dry-run, or live proposal submission."""

    proposal_id: str
    job_id: str
    freelancer_id: str
    status: Literal["draft", "dry_run_ok", "submitted", "viewed", "replied", "interview"]
    proposed_rate: float
    currency: str
    connects_charged: int
    connects_remaining: int
    thread_id: str | None = None
    simulated_client_outcome: str | None = None
    dry_run_preview: DryRunPreview | None = None
    created_at: str


class MessageItem(GigSimBase):
    """Single message within a client-freelancer conversation thread."""

    message_id: str
    thread_id: str
    sender: Literal["freelancer", "client", "system"]
    text: str
    timestamp: str


class ThreadResponse(GigSimBase):
    """Conversation thread metadata and chronological message history."""

    thread_id: str
    job_id: str
    client_id: str
    freelancer_id: str
    status: str
    messages: list[MessageItem] = Field(default_factory=list)


class SendMessageRequest(GigSimBase):
    """Request payload to post a human-approved message to a client thread."""

    idempotency_key: str = Field(..., min_length=4, max_length=128)
    freelancer_id: str = Field(default="persona_aiml_01", min_length=1)
    text: str = Field(..., min_length=2, max_length=4000)
    proposed_slots: list[str] = Field(default_factory=list)


class SendMessageResponse(GigSimBase):
    """Response after posting a message, including any simulated client follow-up."""

    thread_id: str
    sent_message: MessageItem
    simulated_client_reply: MessageItem | None = None
    thread_status: str