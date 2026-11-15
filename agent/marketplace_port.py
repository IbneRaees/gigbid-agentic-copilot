"""Abstract Marketplace Port (`MarketplacePort`) and canonical DTOs for GigBid.

Decouples LangGraph orchestration nodes and MCP servers from specific
marketplace backends (`gigsim` simulator vs. `upwork_live` API/MCP bridge).
All models use Pydantic v2 and align strictly with `agent/handoffs.py`
and `gigsim/models.py`.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent.handoffs import BudgetRange


class PortModel(BaseModel):
    """Base Pydantic v2 model for MarketplacePort contracts."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class MarketplaceJob(PortModel):
    """Canonical marketplace job listing returned by any MarketplacePort."""

    job_id: str = Field(..., min_length=1)
    client_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    budget: BudgetRange
    duration_days: int = Field(default=14, ge=1)
    experience_level: str = Field(default="intermediate")
    screening_questions: list[str] = Field(default_factory=list)
    connects_cost: int = Field(default=6, ge=0)
    posted_at: str = Field(default="")
    domain_category: str = Field(default="general")
    data_provenance: str = Field(default="synthetic")


class MarketplaceJobSearchResult(PortModel):
    """Paginated job search response."""

    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    jobs: list[MarketplaceJob] = Field(default_factory=list)
    source_mode: Literal["gigsim", "upwork_live"] = "gigsim"


class MarketplaceClientHistory(PortModel):
    """Client verification and historical hiring signals."""

    client_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    country: str = Field(default="Unknown")
    payment_verified: bool = Field(default=False)
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    reviews_count: int = Field(default=0, ge=0)
    jobs_posted: int = Field(default=0, ge=0)
    hire_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    total_spent_usd: float = Field(default=0.0, ge=0.0)
    avg_hourly_paid: float = Field(default=0.0, ge=0.0)
    member_since: str = Field(default="")
    past_reviews: list[str] = Field(default_factory=list)
    data_provenance: str = Field(default="synthetic")


class ConnectsLedgerRecord(PortModel):
    """Single Connects balance transaction record."""

    entry_id: str
    freelancer_id: str
    delta: int
    balance_after: int
    reason: str
    job_id: str | None = None
    proposal_id: str | None = None
    created_at: str


class MarketplaceConnectsBalance(PortModel):
    """Current Connects balance and transaction history."""

    freelancer_id: str
    balance: int = Field(..., ge=0)
    weekly_budget: int = Field(default=60, ge=0)
    recent_ledger: list[ConnectsLedgerRecord] = Field(default_factory=list)
    source_mode: Literal["gigsim", "upwork_live"] = "gigsim"


class MarketplaceMessage(PortModel):
    """Single message in a client-freelancer thread."""

    message_id: str
    thread_id: str
    sender: Literal["freelancer", "client", "system"]
    text: str
    timestamp: str


class MarketplaceThread(PortModel):
    """Client conversation thread with chronological messages."""

    thread_id: str
    job_id: str
    client_id: str
    freelancer_id: str
    status: str
    messages: list[MarketplaceMessage] = Field(default_factory=list)


class ProposalPreviewSummary(PortModel):
    """Pre-flight Connects, service fee, and policy check summary."""

    job_id: str
    job_title: str
    freelancer_id: str
    proposed_rate: float
    currency: str = "USD"
    estimated_fee_usd: float = 0.0
    estimated_net_usd: float = 0.0
    connects_cost: int = 0
    current_connects_balance: int = 0
    remaining_connects_after: int = 0
    policy_passed: bool = True
    policy_warnings: list[str] = Field(default_factory=list)


class MarketplaceProposalResult(PortModel):
    """Result of creating a proposal draft, dry-run preview, or submission."""

    proposal_id: str
    job_id: str
    freelancer_id: str
    status: Literal[
        "draft",
        "dry_run_ok",
        "manual_review_required",
        "submitted",
        "viewed",
        "replied",
        "interview",
    ]
    proposed_rate: float
    currency: str = "USD"
    connects_charged: int = Field(default=0, ge=0)
    connects_remaining: int = Field(default=0, ge=0)
    thread_id: str | None = None
    simulated_client_outcome: str | None = None
    dry_run_preview: ProposalPreviewSummary | None = None
    requires_manual_submission: bool = Field(
        default=False,
        description="Always True in upwork_live mode so human completes final submission.",
    )
    manual_review_url: str | None = None
    guardrail_notice: str | None = None
    created_at: str


class MarketplaceMessageResult(PortModel):
    """Result of drafting, previewing, or sending a message in a client thread."""

    thread_id: str
    status: Literal["sent", "dry_run_ok", "draft_only_manual_review"]
    sent_message: MarketplaceMessage | None = None
    simulated_client_reply: MarketplaceMessage | None = None
    thread_status: str = "active"
    requires_manual_send: bool = Field(default=False)
    manual_review_url: str | None = None
    guardrail_notice: str | None = None
    draft_payload: dict[str, Any] | None = None


@runtime_checkable
class MarketplacePort(Protocol):
    """Abstract port decoupling GigBid agents/MCP tools from marketplace backends."""

    async def search_jobs(
        self,
        *,
        q: str | None = None,
        skill: str | None = None,
        min_budget: float | None = None,
        max_budget: float | None = None,
        domain: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> MarketplaceJobSearchResult:
        """Search marketplace jobs using structured filters."""
        ...

    async def get_job(self, job_id: str) -> MarketplaceJob:
        """Retrieve full details for a single job posting."""
        ...

    async def get_client_history(self, client_id: str) -> MarketplaceClientHistory:
        """Retrieve client verification, spend, hire rate, and review history."""
        ...

    async def get_connects_balance(
        self, freelancer_id: str = "persona_aiml_01"
    ) -> MarketplaceConnectsBalance:
        """Retrieve current Connects balance and recent ledger transactions."""
        ...

    async def list_messages(
        self, thread_id: str | None = None
    ) -> list[MarketplaceThread]:
        """List all conversation threads or fetch a specific thread's messages."""
        ...

    async def create_proposal_draft(
        self,
        *,
        idempotency_key: str,
        job_id: str,
        freelancer_id: str,
        cover_letter: str,
        screening_answers: dict[str, str],
        proposed_rate: float,
        currency: str = "USD",
        cited_evidence_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> MarketplaceProposalResult:
        """Create or preview a non-submitted proposal draft without spending Connects."""
        ...

    async def submit_proposal(
        self,
        *,
        idempotency_key: str,
        job_id: str,
        freelancer_id: str,
        cover_letter: str,
        screening_answers: dict[str, str],
        proposed_rate: float,
        currency: str = "USD",
        cited_evidence_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> MarketplaceProposalResult:
        """Submit (or dry-run preview) a proposal according to marketplace policy."""
        ...

    async def send_message(
        self,
        *,
        idempotency_key: str,
        thread_id: str,
        freelancer_id: str,
        text: str,
        proposed_slots: list[str] | None = None,
        dry_run: bool = False,
    ) -> MarketplaceMessageResult:
        """Send (or dry-run preview) a human-approved message in a client thread."""
        ...


def build_marketplace_adapter(mode: str | None = None) -> MarketplacePort:
    """Instantiate the configured MarketplacePort adapter (`gigsim` or `upwork_live`)."""
    resolved_mode = (mode or os.getenv("MARKETPLACE_MODE", "gigsim")).strip().lower()
    if resolved_mode == "upwork_live":
        from mcp_servers.marketplace.adapters.upwork_live import (
            UpworkLiveMarketplaceAdapter,
        )

        return UpworkLiveMarketplaceAdapter()
    if resolved_mode == "gigsim":
        from mcp_servers.marketplace.adapters.gigsim import GigSimMarketplaceAdapter

        return GigSimMarketplaceAdapter()
    raise ValueError(
        f"Unsupported MARKETPLACE_MODE='{resolved_mode}'. Expected 'gigsim' or 'upwork_live'."
    )


__all__ = [
    "MarketplaceJob",
    "MarketplaceJobSearchResult",
    "MarketplaceClientHistory",
    "ConnectsLedgerRecord",
    "MarketplaceConnectsBalance",
    "MarketplaceMessage",
    "MarketplaceThread",
    "ProposalPreviewSummary",
    "MarketplaceProposalResult",
    "MarketplaceMessageResult",
    "MarketplacePort",
    "build_marketplace_adapter",
]