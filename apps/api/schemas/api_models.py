"""Pydantic v2 API request and response schemas for GigBid FastAPI server."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BidRunRequest(BaseModel):
    """Request payload to launch a `BidGraph` run for a target job."""

    job_id: str = Field(..., min_length=1)
    freelancer_id: str = Field(default="persona_aiml_01", min_length=1)
    rag_variant: Literal["v0", "v1", "v2", "v3", "v4"] = Field(default="v4")
    model_profile: Literal["offline_demo", "open", "paid"] | None = None
    marketplace_mode: Literal["gigsim", "upwork_live"] | None = None
    use_ROI_optimizer: bool = Field(default=True)


class ConversationRunRequest(BaseModel):
    """Request payload to launch a `ConversationGraph` run for a client thread."""

    thread_id: str = Field(..., min_length=1)
    freelancer_id: str = Field(default="persona_aiml_01", min_length=1)
    incoming_message: str | None = None
    model_profile: Literal["offline_demo", "open", "paid"] | None = None
    marketplace_mode: Literal["gigsim", "upwork_live"] | None = None


class ApprovalRequest(BaseModel):
    """Human-in-the-Loop resume payload for interrupted Bid or Conversation runs."""

    approved: bool
    reason: str | None = None
    cover_letter: str | None = None
    screening_answers: dict[str, str] | None = None
    proposed_rate: float | None = Field(default=None, gt=0.0)
    reply_text: str | None = None
    selected_slot: str | None = None
    create_email_draft: bool = False


class RunSummary(BaseModel):
    """Compact summary of a Bid or Conversation workflow run."""

    run_id: str
    thread_id: str
    graph_type: Literal["bid", "conversation"]
    job_id: str | None = None
    freelancer_id: str
    status: str
    awaiting_approval: bool = False
    created_at: str
    updated_at: str
    interrupt_payload: dict[str, Any] | None = None
    state_snapshot: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    app: str = "GigBid API"
    version: str = "0.1.0"
    marketplace_mode: str
    model_profile: str
    active_runs: int = 0
    pending_approvals: int = 0
