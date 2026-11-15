"""Marketplace MCP Server (`port 8092`) built with the official MCP Python SDK.

Exposes 5 Read tools and 3 Write tools:
  Read Tools (`readOnlyHint=True, destructiveHint=False, idempotentHint=True`):
    1. `search_jobs`
    2. `get_job`
    3. `get_client_history`
    4. `get_connects_balance`
    5. `list_messages`

  Write Tools (require `idempotency_key: str` and `dry_run: bool = False`):
    6. `create_proposal_draft` (`readOnlyHint=False, destructiveHint=False, idempotentHint=True`)
    7. `submit_proposal` (`readOnlyHint=False, destructiveHint=True, idempotentHint=True`)
    8. `send_message` (`readOnlyHint=False, destructiveHint=True, idempotentHint=True`)

Dynamically routes calls to `GigSimMarketplaceAdapter` (`MARKETPLACE_MODE=gigsim`)
or `UpworkLiveMarketplaceAdapter` (`MARKETPLACE_MODE=upwork_live`, draft-only writes),
enforces Bearer authentication + sliding-window rate limits on HTTP transport, and
records every invocation to `logs/mcp_audit.jsonl`.
"""

from __future__ import annotations

import os
import time
from typing import Any

import uvicorn

from agent.marketplace_port import MarketplacePort, build_marketplace_adapter
from mcp_servers.security import (
    MCPAuditLogger,
    MCPIdempotencyStore,
    MCPServer,
    SlidingWindowRateLimiter,
    build_secured_mcp_asgi_app,
    make_tool_annotations,
)


def build_marketplace_mcp_server(
    *,
    adapter: MarketplacePort | None = None,
    mode: str | None = None,
    audit_logger: MCPAuditLogger | None = None,
    idempotency_store: MCPIdempotencyStore | None = None,
) -> MCPServer:
    """Construct and configure the GigBid Marketplace MCP server instance."""
    resolved_mode = (mode or os.getenv("MARKETPLACE_MODE", "gigsim")).strip().lower()
    audit = audit_logger or MCPAuditLogger()
    idem = idempotency_store or MCPIdempotencyStore()

    def _get_adapter() -> MarketplacePort:
        return adapter if adapter is not None else build_marketplace_adapter(resolved_mode)

    mcp = MCPServer("gigbid-marketplace-mcp")

    # ------------------------------------------------------------------
    # 5 READ TOOLS
    # ------------------------------------------------------------------

    @mcp.tool(
        name="search_jobs",
        description=(
            "Search marketplace job listings by keyword query, required skill, budget range, "
            "or domain category. Read-only operation."
        ),
        annotations=make_tool_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def search_jobs(
        q: str | None = None,
        skill: str | None = None,
        min_budget: float | None = None,
        max_budget: float | None = None,
        domain: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_args = {
            "q": q,
            "skill": skill,
            "min_budget": min_budget,
            "max_budget": max_budget,
            "domain": domain,
            "limit": limit,
            "offset": offset,
        }
        try:
            res = await _get_adapter().search_jobs(
                q=q,
                skill=skill,
                min_budget=min_budget,
                max_budget=max_budget,
                domain=domain,
                limit=limit,
                offset=offset,
            )
            dumped = res.model_dump(mode="json")
            audit.record(
                server_name="marketplace",
                tool_name="search_jobs",
                raw_args=raw_args,
                result_summary=f"total={res.total}, returned={len(res.jobs)}, mode={res.source_mode}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return dumped
        except Exception as exc:
            audit.record(
                server_name="marketplace",
                tool_name="search_jobs",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    @mcp.tool(
        name="get_job",
        description="Fetch complete job details, screening questions, budget, and Connects cost for a job_id.",
        annotations=make_tool_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def get_job(job_id: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_args = {"job_id": job_id}
        try:
            res = await _get_adapter().get_job(job_id)
            dumped = res.model_dump(mode="json")
            audit.record(
                server_name="marketplace",
                tool_name="get_job",
                raw_args=raw_args,
                result_summary=f"job_id={res.job_id}, title={res.title[:60]}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return dumped
        except Exception as exc:
            audit.record(
                server_name="marketplace",
                tool_name="get_job",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    @mcp.tool(
        name="get_client_history",
        description=(
            "Retrieve client payment verification status, total spend, hire rate, rating, "
            "and past freelancer reviews for risk assessment."
        ),
        annotations=make_tool_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def get_client_history(client_id: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_args = {"client_id": client_id}
        try:
            res = await _get_adapter().get_client_history(client_id)
            dumped = res.model_dump(mode="json")
            audit.record(
                server_name="marketplace",
                tool_name="get_client_history",
                raw_args=raw_args,
                result_summary=(
                    f"client_id={res.client_id}, verified={res.payment_verified}, "
                    f"hire_rate={res.hire_rate:.2f}"
                ),
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return dumped
        except Exception as exc:
            audit.record(
                server_name="marketplace",
                tool_name="get_client_history",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    @mcp.tool(
        name="get_connects_balance",
        description="Retrieve the freelancer's current Connects balance, weekly budget, and recent ledger entries.",
        annotations=make_tool_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def get_connects_balance(
        freelancer_id: str = "persona_aiml_01",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_args = {"freelancer_id": freelancer_id}
        try:
            res = await _get_adapter().get_connects_balance(freelancer_id=freelancer_id)
            dumped = res.model_dump(mode="json")
            audit.record(
                server_name="marketplace",
                tool_name="get_connects_balance",
                raw_args=raw_args,
                result_summary=f"freelancer_id={res.freelancer_id}, balance={res.balance}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return dumped
        except Exception as exc:
            audit.record(
                server_name="marketplace",
                tool_name="get_connects_balance",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    @mcp.tool(
        name="list_messages",
        description="List active client conversation threads or fetch messages for a specific thread_id.",
        annotations=make_tool_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def list_messages(thread_id: str | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_args = {"thread_id": thread_id}
        try:
            threads = await _get_adapter().list_messages(thread_id=thread_id)
            dumped = {
                "total_threads": len(threads),
                "threads": [t.model_dump(mode="json") for t in threads],
            }
            audit.record(
                server_name="marketplace",
                tool_name="list_messages",
                raw_args=raw_args,
                result_summary=f"threads_returned={len(threads)}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return dumped
        except Exception as exc:
            audit.record(
                server_name="marketplace",
                tool_name="list_messages",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    # ------------------------------------------------------------------
    # 3 WRITE TOOLS (Require `idempotency_key` and `dry_run`)
    # ------------------------------------------------------------------

    @mcp.tool(
        name="create_proposal_draft",
        description=(
            "Save a non-submitted proposal draft without spending Connects. Requires "
            "'idempotency_key' and supports 'dry_run=True' for zero-side-effect preview."
        ),
        annotations=make_tool_annotations(
            read_only=False, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def create_proposal_draft(
        idempotency_key: str,
        job_id: str,
        cover_letter: str,
        proposed_rate: float,
        freelancer_id: str = "persona_aiml_01",
        screening_answers: dict[str, str] | None = None,
        currency: str = "USD",
        cited_evidence_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        payload = {
            "job_id": job_id,
            "freelancer_id": freelancer_id,
            "cover_letter": cover_letter,
            "screening_answers": screening_answers or {},
            "proposed_rate": proposed_rate,
            "currency": currency,
            "cited_evidence_ids": cited_evidence_ids or [],
        }
        raw_args = {"idempotency_key": idempotency_key, "dry_run": dry_run, **payload}
        try:
            if not dry_run:
                cached = idem.get_cached(
                    server_name="marketplace",
                    tool_name="create_proposal_draft",
                    idempotency_key=idempotency_key,
                    payload=payload,
                )
                if cached is not None:
                    audit.record(
                        server_name="marketplace",
                        tool_name="create_proposal_draft",
                        raw_args=raw_args,
                        result_summary=f"idempotent_replay proposal_id={cached.get('proposal_id')}",
                        status="idempotent_replay",
                        duration_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                    return cached

            res = await _get_adapter().create_proposal_draft(
                idempotency_key=idempotency_key,
                job_id=job_id,
                freelancer_id=freelancer_id,
                cover_letter=cover_letter,
                screening_answers=screening_answers or {},
                proposed_rate=proposed_rate,
                currency=currency,
                cited_evidence_ids=cited_evidence_ids or [],
                dry_run=dry_run,
            )
            dumped = res.model_dump(mode="json")
            if not dry_run:
                idem.save(
                    server_name="marketplace",
                    tool_name="create_proposal_draft",
                    idempotency_key=idempotency_key,
                    payload=payload,
                    response=dumped,
                )
            audit.record(
                server_name="marketplace",
                tool_name="create_proposal_draft",
                raw_args=raw_args,
                result_summary=f"status={res.status}, proposal_id={res.proposal_id}",
                status="dry_run" if dry_run else "ok",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return dumped
        except Exception as exc:
            audit.record(
                server_name="marketplace",
                tool_name="create_proposal_draft",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    @mcp.tool(
        name="submit_proposal",
        description=(
            "Preview (dry_run=True) or submit (dry_run=False) a human-approved proposal. "
            "Requires 'idempotency_key'. In 'upwork_live' mode, strictly enforces draft-only "
            "output with manual review URL and never spends Connects or submits live."
        ),
        annotations=make_tool_annotations(
            read_only=False, destructive=True, idempotent=True, open_world=False
        ),
    )
    async def submit_proposal(
        idempotency_key: str,
        job_id: str,
        cover_letter: str,
        proposed_rate: float,
        freelancer_id: str = "persona_aiml_01",
        screening_answers: dict[str, str] | None = None,
        currency: str = "USD",
        cited_evidence_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        payload = {
            "job_id": job_id,
            "freelancer_id": freelancer_id,
            "cover_letter": cover_letter,
            "screening_answers": screening_answers or {},
            "proposed_rate": proposed_rate,
            "currency": currency,
            "cited_evidence_ids": cited_evidence_ids or [],
        }
        raw_args = {"idempotency_key": idempotency_key, "dry_run": dry_run, **payload}
        try:
            if not dry_run:
                cached = idem.get_cached(
                    server_name="marketplace",
                    tool_name="submit_proposal",
                    idempotency_key=idempotency_key,
                    payload=payload,
                )
                if cached is not None:
                    audit.record(
                        server_name="marketplace",
                        tool_name="submit_proposal",
                        raw_args=raw_args,
                        result_summary=f"idempotent_replay proposal_id={cached.get('proposal_id')}",
                        status="idempotent_replay",
                        duration_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                    return cached

            res = await _get_adapter().submit_proposal(
                idempotency_key=idempotency_key,
                job_id=job_id,
                freelancer_id=freelancer_id,
                cover_letter=cover_letter,
                screening_answers=screening_answers or {},
                proposed_rate=proposed_rate,
                currency=currency,
                cited_evidence_ids=cited_evidence_ids or [],
                dry_run=dry_run,
            )
            dumped = res.model_dump(mode="json")
            if not dry_run:
                idem.save(
                    server_name="marketplace",
                    tool_name="submit_proposal",
                    idempotency_key=idempotency_key,
                    payload=payload,
                    response=dumped,
                )
            audit.record(
                server_name="marketplace",
                tool_name="submit_proposal",
                raw_args=raw_args,
                result_summary=(
                    f"status={res.status}, connects_charged={res.connects_charged}, "
                    f"remaining={res.connects_remaining}"
                ),
                status="dry_run" if dry_run else "ok",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return dumped
        except Exception as exc:
            audit.record(
                server_name="marketplace",
                tool_name="submit_proposal",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    @mcp.tool(
        name="send_message",
        description=(
            "Preview (dry_run=True) or send (dry_run=False) a human-approved reply to an active "
            "client conversation thread. Requires 'idempotency_key'. In 'upwork_live' mode, "
            "strictly prepares a draft for manual review without dispatching externally."
        ),
        annotations=make_tool_annotations(
            read_only=False, destructive=True, idempotent=True, open_world=False
        ),
    )
    async def send_message(
        idempotency_key: str,
        thread_id: str,
        text: str,
        freelancer_id: str = "persona_aiml_01",
        proposed_slots: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        payload = {
            "thread_id": thread_id,
            "freelancer_id": freelancer_id,
            "text": text,
            "proposed_slots": proposed_slots or [],
        }
        raw_args = {"idempotency_key": idempotency_key, "dry_run": dry_run, **payload}
        try:
            if not dry_run:
                cached = idem.get_cached(
                    server_name="marketplace",
                    tool_name="send_message",
                    idempotency_key=idempotency_key,
                    payload=payload,
                )
                if cached is not None:
                    audit.record(
                        server_name="marketplace",
                        tool_name="send_message",
                        raw_args=raw_args,
                        result_summary=f"idempotent_replay thread_id={thread_id}",
                        status="idempotent_replay",
                        duration_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                    return cached

            res = await _get_adapter().send_message(
                idempotency_key=idempotency_key,
                thread_id=thread_id,
                freelancer_id=freelancer_id,
                text=text,
                proposed_slots=proposed_slots or [],
                dry_run=dry_run,
            )
            dumped = res.model_dump(mode="json")
            if not dry_run:
                idem.save(
                    server_name="marketplace",
                    tool_name="send_message",
                    idempotency_key=idempotency_key,
                    payload=payload,
                    response=dumped,
                )
            audit.record(
                server_name="marketplace",
                tool_name="send_message",
                raw_args=raw_args,
                result_summary=f"status={res.status}, thread_id={res.thread_id}",
                status="dry_run" if dry_run else "ok",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return dumped
        except Exception as exc:
            audit.record(
                server_name="marketplace",
                tool_name="send_message",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    return mcp


def create_app(
    *,
    adapter: MarketplacePort | None = None,
    mode: str | None = None,
    expected_token: str | None = None,
    rate_limit_rpm: int | None = None,
    rate_limiter: SlidingWindowRateLimiter | None = None,
    audit_logger: MCPAuditLogger | None = None,
    idempotency_store: MCPIdempotencyStore | None = None,
):
    """Create the secured Starlette ASGI application for Marketplace MCP."""
    server = build_marketplace_mcp_server(
        adapter=adapter,
        mode=mode,
        audit_logger=audit_logger,
        idempotency_store=idempotency_store,
    )
    return build_secured_mcp_asgi_app(
        server,
        server_name="marketplace",
        expected_token=expected_token,
        rate_limit_rpm=rate_limit_rpm,
        rate_limiter=rate_limiter,
    )


def main() -> None:
    host = os.getenv("MARKETPLACE_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MARKETPLACE_MCP_PORT", "8092"))
    asgi_app = create_app()
    uvicorn.run(asgi_app, host=host, port=port)


if __name__ == "__main__":
    main()
