"""Upwork Live Marketplace Adapter (`MARKETPLACE_MODE=upwork_live`).

Connects to Upwork's official GraphQL API / Official Upwork MCP Server
(`UPWORK_MCP_URL=https://mcp.upwork.com/mcp`) using OAuth 2.1 credentials
configured in `.env`.

CRITICAL LIVE-MODE SAFETY GUARDRAIL (POL-008):
  - `create_proposal_draft`, `submit_proposal`, and `send_message` NEVER
    execute live proposal submissions or send external client messages.
  - Even when `dry_run=False`, `submit_proposal` and `send_message` strictly
    create a platform/local draft record, spend 0 Connects, and return
    `requires_manual_submission=True` along with a manual review URL and
    copy-ready payload so the human freelancer completes the final click on
    Upwork directly.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import httpx

from agent.handoffs import BudgetRange
from agent.marketplace_port import (
    MarketplaceClientHistory,
    MarketplaceConnectsBalance,
    MarketplaceJob,
    MarketplaceJobSearchResult,
    MarketplaceMessageResult,
    MarketplacePort,
    MarketplaceProposalResult,
    MarketplaceThread,
    ProposalPreviewSummary,
)
from mcp_servers.security import utc_now_iso


def _compute_upwork_fee(amount: float) -> float:
    """Compute Upwork tiered service fee (POL-007)."""
    if amount <= 500.0:
        return round(amount * 0.15, 2)
    if amount <= 10000.0:
        return round(500.0 * 0.15 + (amount - 500.0) * 0.10, 2)
    return round(500.0 * 0.15 + 9500.0 * 0.10 + (amount - 10000.0) * 0.05, 2)


class UpworkLiveMarketplaceAdapter(MarketplacePort):
    """OAuth 2.1 Upwork API / MCP adapter with hard-enforced draft-only writes."""

    DRAFT_ONLY_NOTICE = (
        "UPWORK_LIVE DRAFT-ONLY GUARDRAIL (POL-008): Automated live submission and "
        "outbound messaging are disabled in upwork_live mode. A verified draft has "
        "been prepared for your manual review and submission on Upwork."
    )

    def __init__(
        self,
        *,
        mcp_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        access_token: str | None = None,
        timeout_seconds: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.mcp_url = (
            mcp_url or os.getenv("UPWORK_MCP_URL", "https://mcp.upwork.com/mcp")
        ).rstrip("/")
        self.client_id = client_id or os.getenv("UPWORK_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("UPWORK_CLIENT_SECRET", "")
        self.access_token = access_token or os.getenv("UPWORK_ACCESS_TOKEN", "")
        self.timeout_seconds = timeout_seconds
        self._external_client = client

    def _is_authenticated(self) -> bool:
        return bool(self.access_token.strip())

    def _auth_headers(self) -> dict[str, str]:
        if not self._is_authenticated():
            raise PermissionError(
                "Upwork Live mode requires a valid OAuth 2.1 access token "
                "(connect via /api/auth/upwork/callback or set UPWORK_ACCESS_TOKEN)."
            )
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _call_upwork_mcp_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke a read-only tool on Upwork's Streamable HTTP MCP endpoint."""
        headers = self._auth_headers()
        rpc_payload = {
            "jsonrpc": "2.0",
            "id": f"upwork-read-{tool_name}",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        if self._external_client is not None:
            resp = await self._external_client.post(self.mcp_url, json=rpc_payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(self.mcp_url, json=rpc_payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"Upwork MCP error: {body['error']}")
        return body.get("result", {})

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
        if not self._is_authenticated():
            return MarketplaceJobSearchResult(
                total=0,
                limit=limit,
                offset=offset,
                jobs=[],
                source_mode="upwork_live",
            )

        query_parts = [p for p in (q, skill, domain) if p]
        result = await self._call_upwork_mcp_tool(
            "search_jobs",
            {
                "query": " ".join(query_parts) or "Python",
                "min_budget": min_budget,
                "max_budget": max_budget,
                "limit": limit,
                "offset": offset,
            },
        )
        raw_jobs = result.get("jobs", []) if isinstance(result, dict) else []
        parsed: list[MarketplaceJob] = []
        for item in raw_jobs:
            b = item.get("budget") or {}
            parsed.append(
                MarketplaceJob(
                    job_id=str(item.get("id") or item.get("job_id")),
                    client_id=str(item.get("client_id") or "upwork_client"),
                    title=str(item.get("title", "Upwork Job")),
                    description=str(item.get("description", "")),
                    required_skills=list(item.get("skills", [])),
                    budget=BudgetRange(
                        min=float(b.get("min", 0.0)),
                        max=float(b.get("max", b.get("min", 0.0))),
                        currency=str(b.get("currency", "USD")),
                        budget_type=b.get("budget_type", "fixed"),
                    ),
                    connects_cost=int(item.get("connects_cost", 8)),
                    posted_at=str(item.get("posted_at", utc_now_iso())),
                    data_provenance="upwork_live",
                )
            )
        return MarketplaceJobSearchResult(
            total=int(result.get("total", len(parsed)))
            if isinstance(result, dict)
            else len(parsed),
            limit=limit,
            offset=offset,
            jobs=parsed,
            source_mode="upwork_live",
        )

    async def get_job(self, job_id: str) -> MarketplaceJob:
        if not self._is_authenticated():
            return MarketplaceJob(
                job_id=job_id,
                client_id="upwork_unconnected",
                title=f"Upwork Job {job_id} (OAuth Token Pending)",
                description="Connect Upwork OAuth 2.1 in Settings to fetch live job metadata.",
                budget=BudgetRange(min=1000.0, max=2000.0, currency="USD", budget_type="fixed"),
                connects_cost=8,
                posted_at=utc_now_iso(),
                data_provenance="upwork_live",
            )
        result = await self._call_upwork_mcp_tool("get_job_details", {"job_id": job_id})
        b = result.get("budget") or {}
        return MarketplaceJob(
            job_id=job_id,
            client_id=str(result.get("client_id", "upwork_client")),
            title=str(result.get("title", f"Upwork Job {job_id}")),
            description=str(result.get("description", "")),
            required_skills=list(result.get("skills", [])),
            budget=BudgetRange(
                min=float(b.get("min", 0.0)),
                max=float(b.get("max", b.get("min", 0.0))),
                currency=str(b.get("currency", "USD")),
                budget_type=b.get("budget_type", "fixed"),
            ),
            connects_cost=int(result.get("connects_cost", 8)),
            posted_at=str(result.get("posted_at", utc_now_iso())),
            data_provenance="upwork_live",
        )

    async def get_client_history(self, client_id: str) -> MarketplaceClientHistory:
        if not self._is_authenticated():
            return MarketplaceClientHistory(
                client_id=client_id,
                name="Upwork Client (OAuth Token Pending)",
                payment_verified=False,
                data_provenance="upwork_live",
            )
        result = await self._call_upwork_mcp_tool("get_client_profile", {"client_id": client_id})
        return MarketplaceClientHistory(
            client_id=client_id,
            name=str(result.get("name", "Upwork Client")),
            country=str(result.get("country", "Unknown")),
            payment_verified=bool(result.get("payment_verified", False)),
            rating=float(result.get("rating", 0.0)),
            reviews_count=int(result.get("reviews_count", 0)),
            jobs_posted=int(result.get("jobs_posted", 0)),
            hire_rate=float(result.get("hire_rate", 0.0)),
            total_spent_usd=float(result.get("total_spent_usd", 0.0)),
            avg_hourly_paid=float(result.get("avg_hourly_paid", 0.0)),
            member_since=str(result.get("member_since", "")),
            past_reviews=list(result.get("past_reviews", [])),
            data_provenance="upwork_live",
        )

    async def get_connects_balance(
        self, freelancer_id: str = "persona_aiml_01"
    ) -> MarketplaceConnectsBalance:
        if not self._is_authenticated():
            return MarketplaceConnectsBalance(
                freelancer_id=freelancer_id,
                balance=0,
                weekly_budget=60,
                recent_ledger=[],
                source_mode="upwork_live",
            )
        result = await self._call_upwork_mcp_tool("get_connects_balance", {})
        return MarketplaceConnectsBalance(
            freelancer_id=freelancer_id,
            balance=int(result.get("balance", 0)),
            weekly_budget=int(result.get("weekly_budget", 60)),
            recent_ledger=[],
            source_mode="upwork_live",
        )

    async def list_messages(self, thread_id: str | None = None) -> list[MarketplaceThread]:
        if not self._is_authenticated():
            return []
        result = await self._call_upwork_mcp_tool(
            "list_messages", {"thread_id": thread_id} if thread_id else {}
        )
        raw_threads = result.get("threads", []) if isinstance(result, dict) else []
        return [
            MarketplaceThread(
                thread_id=str(th.get("thread_id", "")),
                job_id=str(th.get("job_id", "")),
                client_id=str(th.get("client_id", "")),
                freelancer_id=str(th.get("freelancer_id", "upwork_user")),
                status=str(th.get("status", "active")),
                messages=[],
            )
            for th in raw_threads
        ]

    def _build_draft_only_proposal_result(
        self,
        *,
        idempotency_key: str,
        job_id: str,
        freelancer_id: str,
        proposed_rate: float,
        currency: str,
        dry_run: bool,
    ) -> MarketplaceProposalResult:
        """Prepare a draft-only response that NEVER submits live or spends Connects."""
        digest = hashlib.sha256(f"{idempotency_key}:{job_id}:{freelancer_id}".encode()).hexdigest()[
            :10
        ]
        fee = _compute_upwork_fee(proposed_rate)
        preview = ProposalPreviewSummary(
            job_id=job_id,
            job_title=f"Upwork Job {job_id}",
            freelancer_id=freelancer_id,
            proposed_rate=proposed_rate,
            currency=currency,
            estimated_fee_usd=fee,
            estimated_net_usd=round(proposed_rate - fee, 2),
            connects_cost=8,
            current_connects_balance=0,
            remaining_connects_after=0,
            policy_passed=True,
            policy_warnings=[self.DRAFT_ONLY_NOTICE],
        )
        return MarketplaceProposalResult(
            proposal_id=f"upwork_draft_{digest}",
            job_id=job_id,
            freelancer_id=freelancer_id,
            status="dry_run_ok" if dry_run else "manual_review_required",
            proposed_rate=proposed_rate,
            currency=currency,
            connects_charged=0,
            connects_remaining=0,
            simulated_client_outcome="draft_only_manual_review",
            dry_run_preview=preview,
            requires_manual_submission=True,
            manual_review_url=f"https://www.upwork.com/ab/proposals/job/{job_id}/apply/",
            guardrail_notice=self.DRAFT_ONLY_NOTICE,
            created_at=utc_now_iso(),
        )

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
        return self._build_draft_only_proposal_result(
            idempotency_key=idempotency_key,
            job_id=job_id,
            freelancer_id=freelancer_id,
            proposed_rate=proposed_rate,
            currency=currency,
            dry_run=dry_run,
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
        """HARD GUARDRAIL: Never execute live submission on Upwork; always return draft + manual URL."""
        return self._build_draft_only_proposal_result(
            idempotency_key=idempotency_key,
            job_id=job_id,
            freelancer_id=freelancer_id,
            proposed_rate=proposed_rate,
            currency=currency,
            dry_run=dry_run,
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
        """HARD GUARDRAIL: Never dispatch outbound messages in upwork_live mode; prepare draft only."""
        return MarketplaceMessageResult(
            thread_id=thread_id,
            status="dry_run_ok" if dry_run else "draft_only_manual_review",
            sent_message=None,
            simulated_client_reply=None,
            thread_status="manual_review_required",
            requires_manual_send=True,
            manual_review_url=f"https://www.upwork.com/messages/rooms/{thread_id}",
            guardrail_notice=self.DRAFT_ONLY_NOTICE,
            draft_payload={
                "idempotency_key": idempotency_key,
                "thread_id": thread_id,
                "freelancer_id": freelancer_id,
                "text": text,
                "proposed_slots": proposed_slots or [],
            },
        )


__all__ = ["UpworkLiveMarketplaceAdapter"]
