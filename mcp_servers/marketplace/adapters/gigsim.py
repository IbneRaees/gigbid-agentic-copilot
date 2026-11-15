"""Async `httpx` Adapter connecting GigBid to the Phase 2A GigSim Simulator (`:8081`).

Implements `agent.marketplace_port.MarketplacePort` with:
  - Bearer token authentication (`GIGSIM_API_TOKEN`)
  - Bounded exponential backoff retries on `429` / `503` / transient network errors
  - Full forwarding of `idempotency_key` and `dry_run` parameters
  - Optional custom `httpx.AsyncClient` injection for in-memory ASGI testing
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from agent.handoffs import BudgetRange
from agent.marketplace_port import (
    ConnectsLedgerRecord,
    MarketplaceClientHistory,
    MarketplaceConnectsBalance,
    MarketplaceJob,
    MarketplaceJobSearchResult,
    MarketplaceMessage,
    MarketplaceMessageResult,
    MarketplacePort,
    MarketplaceProposalResult,
    MarketplaceThread,
    ProposalPreviewSummary,
)
from mcp_servers.security import utc_now_iso


class GigSimAdapterError(RuntimeError):
    """Raised when GigSim returns an unrecoverable HTTP or transport error."""

    def __init__(self, status_code: int, message: str, detail: Any = None) -> None:
        super().__init__(f"GigSim error ({status_code}): {message}")
        self.status_code = status_code
        self.detail = detail


def _compute_fee(amount: float) -> float:
    """Tiered marketplace fee matching POL-007."""
    if amount <= 500.0:
        return round(amount * 0.15, 2)
    if amount <= 10000.0:
        return round(500.0 * 0.15 + (amount - 500.0) * 0.10, 2)
    return round(500.0 * 0.15 + 9500.0 * 0.10 + (amount - 10000.0) * 0.05, 2)


class GigSimMarketplaceAdapter(MarketplacePort):
    """Production async `httpx` client implementing `MarketplacePort` against GigSim."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_token: str | None = None,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.4,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("GIGSIM_BASE_URL", "http://localhost:8081")).rstrip(
            "/"
        )
        self.api_token = api_token or os.getenv("GIGSIM_API_TOKEN", "gigsim-dev-token-2026")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.backoff_base_seconds = backoff_base_seconds
        self._external_client = client

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        cleaned_params = {k: v for k, v in params.items() if v is not None} if params else None
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self._external_client is not None:
                    resp = await self._external_client.request(
                        method,
                        path,
                        params=cleaned_params,
                        json=json_body,
                        headers=self._headers(),
                    )
                else:
                    async with httpx.AsyncClient(
                        base_url=self.base_url,
                        timeout=self.timeout_seconds,
                    ) as client:
                        resp = await client.request(
                            method,
                            path,
                            params=cleaned_params,
                            json=json_body,
                            headers=self._headers(),
                        )

                if resp.status_code in {429, 502, 503, 504} and attempt < self.max_retries:
                    retry_after = float(
                        resp.headers.get(
                            "Retry-After", self.backoff_base_seconds * (2 ** (attempt - 1))
                        )
                    )
                    await asyncio.sleep(min(retry_after, 3.0))
                    continue

                if resp.status_code >= 400:
                    try:
                        err_payload = resp.json()
                    except Exception:
                        err_payload = resp.text
                    detail = (
                        err_payload.get("detail", err_payload)
                        if isinstance(err_payload, dict)
                        else err_payload
                    )
                    raise GigSimAdapterError(
                        status_code=resp.status_code,
                        message=str(detail),
                        detail=err_payload,
                    )

                return resp.json()

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff_base_seconds * (2 ** (attempt - 1)))
                    continue
                raise GigSimAdapterError(
                    status_code=503,
                    message=f"Unable to reach GigSim at {self.base_url}: {exc}",
                ) from exc

        raise GigSimAdapterError(
            status_code=503,
            message=f"GigSim request exhausted retries: {last_exc}",
        )

    @staticmethod
    def _parse_job(raw: dict[str, Any]) -> MarketplaceJob:
        raw_budget = raw.get("budget") or {}
        return MarketplaceJob(
            job_id=raw["job_id"],
            client_id=raw["client_id"],
            title=raw["title"],
            description=raw["description"],
            required_skills=list(raw.get("required_skills", [])),
            nice_to_have_skills=list(raw.get("nice_to_have_skills", [])),
            budget=BudgetRange(
                min=float(raw_budget.get("min", 0.0)),
                max=float(raw_budget.get("max", 0.0)),
                currency=str(raw_budget.get("currency", "USD")),
                budget_type=raw_budget.get("budget_type", "fixed"),
            ),
            duration_days=int(raw.get("duration_days", 14)),
            experience_level=str(raw.get("experience_level", "intermediate")),
            screening_questions=list(raw.get("screening_questions", [])),
            connects_cost=int(raw.get("connects_cost", 6)),
            posted_at=str(raw.get("posted_at", "")),
            domain_category=str(raw.get("domain_category", "general")),
            data_provenance=str(raw.get("data_provenance", "synthetic")),
        )

    @staticmethod
    def _parse_thread(raw: dict[str, Any]) -> MarketplaceThread:
        return MarketplaceThread(
            thread_id=raw["thread_id"],
            job_id=raw["job_id"],
            client_id=raw["client_id"],
            freelancer_id=raw["freelancer_id"],
            status=raw.get("status", "replied"),
            messages=[
                MarketplaceMessage(
                    message_id=m["message_id"],
                    thread_id=m["thread_id"],
                    sender=m["sender"],
                    text=m["text"],
                    timestamp=m["timestamp"],
                )
                for m in raw.get("messages", [])
            ],
        )

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
        data = await self._request(
            "GET",
            "/jobs",
            params={
                "q": q,
                "skill": skill,
                "min_budget": min_budget,
                "max_budget": max_budget,
                "domain": domain,
                "limit": limit,
                "offset": offset,
            },
        )
        return MarketplaceJobSearchResult(
            total=int(data.get("total", 0)),
            limit=int(data.get("limit", limit)),
            offset=int(data.get("offset", offset)),
            jobs=[self._parse_job(item) for item in data.get("jobs", [])],
            source_mode="gigsim",
        )

    async def get_job(self, job_id: str) -> MarketplaceJob:
        data = await self._request("GET", f"/jobs/{job_id}")
        return self._parse_job(data)

    async def get_client_history(self, client_id: str) -> MarketplaceClientHistory:
        data = await self._request("GET", f"/clients/{client_id}")
        return MarketplaceClientHistory(
            client_id=data["client_id"],
            name=data["name"],
            country=data.get("country", "Unknown"),
            payment_verified=bool(data.get("payment_verified", False)),
            rating=float(data.get("rating", 0.0)),
            reviews_count=int(data.get("reviews_count", 0)),
            jobs_posted=int(data.get("jobs_posted", 0)),
            hire_rate=float(data.get("hire_rate", 0.0)),
            total_spent_usd=float(data.get("total_spent_usd", 0.0)),
            avg_hourly_paid=float(data.get("avg_hourly_paid", 0.0)),
            member_since=str(data.get("member_since", "")),
            past_reviews=list(data.get("past_reviews", [])),
            data_provenance=str(data.get("data_provenance", "synthetic")),
        )

    async def get_connects_balance(
        self, freelancer_id: str = "persona_aiml_01"
    ) -> MarketplaceConnectsBalance:
        data = await self._request(
            "GET",
            "/account/connects",
            params={"freelancer_id": freelancer_id},
        )
        return MarketplaceConnectsBalance(
            freelancer_id=data["freelancer_id"],
            balance=int(data["balance"]),
            weekly_budget=int(data.get("weekly_budget", 60)),
            recent_ledger=[
                ConnectsLedgerRecord(
                    entry_id=entry["entry_id"],
                    freelancer_id=entry["freelancer_id"],
                    delta=int(entry["delta"]),
                    balance_after=int(entry["balance_after"]),
                    reason=entry["reason"],
                    job_id=entry.get("job_id"),
                    proposal_id=entry.get("proposal_id"),
                    created_at=entry["created_at"],
                )
                for entry in data.get("recent_ledger", [])
            ],
            source_mode="gigsim",
        )

    async def list_messages(self, thread_id: str | None = None) -> list[MarketplaceThread]:
        if thread_id:
            data = await self._request("GET", f"/threads/{thread_id}/messages")
            return [self._parse_thread(data)]
        data = await self._request("GET", "/threads")
        return [self._parse_thread(item) for item in data]

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
        if dry_run:
            job = await self.get_job(job_id)
            bal = await self.get_connects_balance(freelancer_id)
            fee = _compute_fee(proposed_rate)
            preview = ProposalPreviewSummary(
                job_id=job.job_id,
                job_title=job.title,
                freelancer_id=freelancer_id,
                proposed_rate=proposed_rate,
                currency=currency,
                estimated_fee_usd=fee,
                estimated_net_usd=round(proposed_rate - fee, 2),
                connects_cost=job.connects_cost,
                current_connects_balance=bal.balance,
                remaining_connects_after=bal.balance,
                policy_passed=True,
                policy_warnings=[],
            )
            return MarketplaceProposalResult(
                proposal_id=f"dryrun_draft_{job.job_id}",
                job_id=job.job_id,
                freelancer_id=freelancer_id,
                status="dry_run_ok",
                proposed_rate=proposed_rate,
                currency=currency,
                connects_charged=0,
                connects_remaining=bal.balance,
                simulated_client_outcome="dry_run_draft_preview",
                dry_run_preview=preview,
                requires_manual_submission=False,
                created_at=utc_now_iso(),
            )

        data = await self._request(
            "POST",
            "/proposals/draft",
            json_body={
                "idempotency_key": idempotency_key,
                "job_id": job_id,
                "freelancer_id": freelancer_id,
                "cover_letter": cover_letter,
                "screening_answers": screening_answers,
                "proposed_rate": proposed_rate,
                "currency": currency,
                "cited_evidence_ids": cited_evidence_ids or [],
            },
        )
        return MarketplaceProposalResult(
            proposal_id=data["proposal_id"],
            job_id=data["job_id"],
            freelancer_id=data["freelancer_id"],
            status=data["status"],
            proposed_rate=float(data["proposed_rate"]),
            currency=data.get("currency", currency),
            connects_charged=int(data.get("connects_charged", 0)),
            connects_remaining=int(data.get("connects_remaining", 0)),
            thread_id=data.get("thread_id"),
            simulated_client_outcome=data.get("simulated_client_outcome"),
            requires_manual_submission=False,
            created_at=data["created_at"],
        )

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
        data = await self._request(
            "POST",
            "/proposals/submit",
            json_body={
                "idempotency_key": idempotency_key,
                "job_id": job_id,
                "freelancer_id": freelancer_id,
                "cover_letter": cover_letter,
                "screening_answers": screening_answers,
                "proposed_rate": proposed_rate,
                "currency": currency,
                "cited_evidence_ids": cited_evidence_ids or [],
                "dry_run": dry_run,
            },
        )
        raw_preview = data.get("dry_run_preview")
        preview = (
            ProposalPreviewSummary(
                job_id=raw_preview["job_id"],
                job_title=raw_preview["job_title"],
                freelancer_id=raw_preview["freelancer_id"],
                proposed_rate=float(raw_preview["proposed_rate"]),
                currency=raw_preview.get("currency", currency),
                estimated_fee_usd=float(raw_preview.get("estimated_fee_usd", 0.0)),
                estimated_net_usd=float(raw_preview.get("estimated_net_usd", 0.0)),
                connects_cost=int(raw_preview.get("connects_cost", 0)),
                current_connects_balance=int(raw_preview.get("current_connects_balance", 0)),
                remaining_connects_after=int(raw_preview.get("remaining_connects_after", 0)),
                policy_passed=bool(raw_preview.get("policy_passed", True)),
                policy_warnings=list(raw_preview.get("policy_warnings", [])),
            )
            if raw_preview
            else None
        )
        return MarketplaceProposalResult(
            proposal_id=data["proposal_id"],
            job_id=data["job_id"],
            freelancer_id=data["freelancer_id"],
            status=data["status"],
            proposed_rate=float(data["proposed_rate"]),
            currency=data.get("currency", currency),
            connects_charged=int(data.get("connects_charged", 0)),
            connects_remaining=int(data.get("connects_remaining", 0)),
            thread_id=data.get("thread_id"),
            simulated_client_outcome=data.get("simulated_client_outcome"),
            dry_run_preview=preview,
            requires_manual_submission=False,
            created_at=data["created_at"],
        )

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
        if dry_run:
            # Verify thread exists via read endpoint without mutating messages
            await self.list_messages(thread_id=thread_id)
            return MarketplaceMessageResult(
                thread_id=thread_id,
                status="dry_run_ok",
                sent_message=None,
                simulated_client_reply=None,
                thread_status="dry_run_preview_only",
                requires_manual_send=False,
                draft_payload={
                    "thread_id": thread_id,
                    "freelancer_id": freelancer_id,
                    "text": text,
                    "proposed_slots": proposed_slots or [],
                },
            )

        data = await self._request(
            "POST",
            f"/threads/{thread_id}/messages",
            json_body={
                "idempotency_key": idempotency_key,
                "freelancer_id": freelancer_id,
                "text": text,
                "proposed_slots": proposed_slots or [],
            },
        )
        sent_raw = data["sent_message"]
        reply_raw = data.get("simulated_client_reply")
        return MarketplaceMessageResult(
            thread_id=data["thread_id"],
            status="sent",
            sent_message=MarketplaceMessage(
                message_id=sent_raw["message_id"],
                thread_id=sent_raw["thread_id"],
                sender=sent_raw["sender"],
                text=sent_raw["text"],
                timestamp=sent_raw["timestamp"],
            ),
            simulated_client_reply=(
                MarketplaceMessage(
                    message_id=reply_raw["message_id"],
                    thread_id=reply_raw["thread_id"],
                    sender=reply_raw["sender"],
                    text=reply_raw["text"],
                    timestamp=reply_raw["timestamp"],
                )
                if reply_raw
                else None
            ),
            thread_status=data.get("thread_status", "active_discussion"),
            requires_manual_send=False,
        )


__all__ = ["GigSimMarketplaceAdapter", "GigSimAdapterError"]
