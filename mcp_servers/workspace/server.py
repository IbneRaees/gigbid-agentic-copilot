"""Workspace MCP Server (`port 8093`) for Calendar Slot Discovery & Draft-Only Communications.

Exposes 3 Tools:
  1. `get_free_calendar_slots` (`readOnlyHint=True, destructiveHint=False, idempotentHint=True`)
  2. `create_tentative_interview_hold` (`readOnlyHint=False, destructiveHint=False, idempotentHint=True`)
     - Requires `idempotency_key: str` and `dry_run: bool = False`.
  3. `create_email_draft` (`readOnlyHint=False, destructiveHint=False, idempotentHint=True`)
     - Requires `idempotency_key: str` and `dry_run: bool = False`.
     - Strictly creates a local draft (never sends live email).

Persists tentative holds and email drafts cleanly to local SQLite (`gigsim/workspace.db`),
enforces Bearer authentication + sliding-window rate limits, and records JSONL audit logs.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn

from mcp_servers.security import (
    MCPAuditLogger,
    MCPIdempotencyStore,
    MCPServer,
    SlidingWindowRateLimiter,
    build_secured_mcp_asgi_app,
    make_tool_annotations,
    utc_now_iso,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_WORKSPACE_DB_PATH = Path(
    os.getenv("WORKSPACE_DB_PATH", str(ROOT_DIR / "gigsim" / "workspace.db"))
)


class WorkspaceStore:
    """SQLite persistence layer for tentative interview holds and email drafts."""

    def __init__(self, db_path: Path = DEFAULT_WORKSPACE_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS calendar_holds (
                    hold_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    freelancer_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    slot_start_iso TEXT NOT NULL,
                    slot_end_iso TEXT NOT NULL,
                    duration_minutes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS email_drafts (
                    draft_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    freelancer_id TEXT NOT NULL,
                    recipient_alias TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def get_booked_starts(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT slot_start_iso FROM calendar_holds WHERE status != 'cancelled'"
            ).fetchall()
        return {str(r["slot_start_iso"]) for r in rows}

    def get_free_slots(
        self,
        *,
        start_date: str | None = None,
        days_ahead: int = 3,
        slot_minutes: int = 30,
        timezone_name: str = "UTC",
    ) -> list[dict[str, Any]]:
        if start_date:
            base_day = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            if base_day.tzinfo is None:
                base_day = base_day.replace(tzinfo=UTC)
        else:
            now = datetime.now(UTC)
            base_day = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        booked = self.get_booked_starts()
        candidate_hours_utc = (14, 16, 18)  # 14:00, 16:00, 18:00 UTC business slots
        slots: list[dict[str, Any]] = []

        for day_offset in range(max(1, min(days_ahead, 14))):
            day = base_day + timedelta(days=day_offset)
            for hr in candidate_hours_utc:
                slot_start = day.replace(hour=hr, minute=0, second=0, microsecond=0)
                slot_end = slot_start + timedelta(minutes=slot_minutes)
                start_iso = slot_start.isoformat().replace("+00:00", "Z")
                if start_iso in booked:
                    continue
                slots.append(
                    {
                        "slot_start_iso": start_iso,
                        "slot_end_iso": slot_end.isoformat().replace("+00:00", "Z"),
                        "duration_minutes": slot_minutes,
                        "timezone": timezone_name,
                        "label": slot_start.strftime("%a %Y-%m-%d %H:%M UTC"),
                    }
                )
        return slots

    def create_hold(
        self,
        *,
        idempotency_key: str,
        thread_id: str,
        job_id: str,
        freelancer_id: str,
        title: str,
        slot_start_iso: str,
        duration_minutes: int,
    ) -> dict[str, Any]:
        start_dt = datetime.fromisoformat(slot_start_iso.replace("Z", "+00:00"))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        norm_start = start_dt.isoformat().replace("+00:00", "Z")
        norm_end = end_dt.isoformat().replace("+00:00", "Z")

        if norm_start in self.get_booked_starts():
            raise ValueError(f"Calendar slot '{norm_start}' already has a tentative hold.")

        hold_id = f"hold_{hashlib.sha256(idempotency_key.encode()).hexdigest()[:10]}"
        created_at = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO calendar_holds (
                    hold_id, thread_id, job_id, freelancer_id, title,
                    slot_start_iso, slot_end_iso, duration_minutes, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hold_id,
                    thread_id,
                    job_id,
                    freelancer_id,
                    title,
                    norm_start,
                    norm_end,
                    duration_minutes,
                    "tentative_hold",
                    created_at,
                ),
            )

        return {
            "hold_id": hold_id,
            "thread_id": thread_id,
            "job_id": job_id,
            "freelancer_id": freelancer_id,
            "title": title,
            "slot_start_iso": norm_start,
            "slot_end_iso": norm_end,
            "duration_minutes": duration_minutes,
            "status": "tentative_hold",
            "created_at": created_at,
        }

    def create_draft(
        self,
        *,
        idempotency_key: str,
        thread_id: str | None,
        freelancer_id: str,
        recipient_alias: str,
        subject: str,
        body: str,
    ) -> dict[str, Any]:
        draft_id = f"draft_email_{hashlib.sha256(idempotency_key.encode()).hexdigest()[:10]}"
        created_at = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO email_drafts (
                    draft_id, thread_id, freelancer_id, recipient_alias,
                    subject, body, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft_id,
                    thread_id,
                    freelancer_id,
                    recipient_alias,
                    subject,
                    body,
                    "draft_saved_never_sent",
                    created_at,
                ),
            )

        return {
            "draft_id": draft_id,
            "thread_id": thread_id,
            "freelancer_id": freelancer_id,
            "recipient_alias": recipient_alias,
            "subject": subject,
            "body": body,
            "status": "draft_saved_never_sent",
            "auto_sent": False,
            "created_at": created_at,
        }

    def count_records(self) -> tuple[int, int]:
        with self._connect() as conn:
            holds = conn.execute("SELECT COUNT(*) FROM calendar_holds").fetchone()[0]
            drafts = conn.execute("SELECT COUNT(*) FROM email_drafts").fetchone()[0]
        return int(holds), int(drafts)


def build_workspace_mcp_server(
    *,
    db_path: Path = DEFAULT_WORKSPACE_DB_PATH,
    audit_logger: MCPAuditLogger | None = None,
    idempotency_store: MCPIdempotencyStore | None = None,
) -> MCPServer:
    """Construct and configure the GigBid Workspace MCP server."""
    store = WorkspaceStore(db_path=db_path)
    audit = audit_logger or MCPAuditLogger()
    idem = idempotency_store or MCPIdempotencyStore(
        db_path=Path(db_path).parent / "workspace_idempotency.db"
    )
    mcp = MCPServer("gigbid-workspace-mcp")

    @mcp.tool(
        name="get_free_calendar_slots",
        description=(
            "Query free 30-minute interview slots on the freelancer's calendar over the next "
            "N days, excluding existing tentative holds. Read-only operation."
        ),
        annotations=make_tool_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def get_free_calendar_slots(
        start_date: str | None = None,
        days_ahead: int = 3,
        slot_minutes: int = 30,
        timezone_name: str = "UTC",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_args = {
            "start_date": start_date,
            "days_ahead": days_ahead,
            "slot_minutes": slot_minutes,
            "timezone_name": timezone_name,
        }
        try:
            slots = store.get_free_slots(
                start_date=start_date,
                days_ahead=days_ahead,
                slot_minutes=slot_minutes,
                timezone_name=timezone_name,
            )
            result = {
                "timezone": timezone_name,
                "total_available": len(slots),
                "slots": slots,
            }
            audit.record(
                server_name="workspace",
                tool_name="get_free_calendar_slots",
                raw_args=raw_args,
                result_summary=f"available_slots={len(slots)}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return result
        except Exception as exc:
            audit.record(
                server_name="workspace",
                tool_name="get_free_calendar_slots",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    @mcp.tool(
        name="create_tentative_interview_hold",
        description=(
            "Create a tentative calendar hold for a proposed client interview slot. "
            "Requires 'idempotency_key' and supports 'dry_run=True' to preview without persisting."
        ),
        annotations=make_tool_annotations(
            read_only=False, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def create_tentative_interview_hold(
        idempotency_key: str,
        thread_id: str,
        job_id: str,
        slot_start_iso: str,
        duration_minutes: int = 30,
        title: str = "Client Technical Interview",
        freelancer_id: str = "persona_aiml_01",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        payload = {
            "thread_id": thread_id,
            "job_id": job_id,
            "slot_start_iso": slot_start_iso,
            "duration_minutes": duration_minutes,
            "title": title,
            "freelancer_id": freelancer_id,
        }
        raw_args = {"idempotency_key": idempotency_key, "dry_run": dry_run, **payload}
        try:
            if not dry_run:
                cached = idem.get_cached(
                    server_name="workspace",
                    tool_name="create_tentative_interview_hold",
                    idempotency_key=idempotency_key,
                    payload=payload,
                )
                if cached is not None:
                    audit.record(
                        server_name="workspace",
                        tool_name="create_tentative_interview_hold",
                        raw_args=raw_args,
                        result_summary=f"idempotent_replay hold_id={cached.get('hold_id')}",
                        status="idempotent_replay",
                        duration_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                    return cached

            if dry_run:
                start_dt = datetime.fromisoformat(slot_start_iso.replace("Z", "+00:00"))
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=UTC)
                norm_start = start_dt.isoformat().replace("+00:00", "Z")
                is_free = norm_start not in store.get_booked_starts()
                preview = {
                    "hold_id": f"dryrun_hold_{job_id}",
                    "thread_id": thread_id,
                    "job_id": job_id,
                    "freelancer_id": freelancer_id,
                    "title": title,
                    "slot_start_iso": norm_start,
                    "duration_minutes": duration_minutes,
                    "status": "dry_run_ok",
                    "slot_available": is_free,
                }
                audit.record(
                    server_name="workspace",
                    tool_name="create_tentative_interview_hold",
                    raw_args=raw_args,
                    result_summary=f"dry_run slot_available={is_free}",
                    status="dry_run",
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                )
                return preview

            created = store.create_hold(
                idempotency_key=idempotency_key,
                thread_id=thread_id,
                job_id=job_id,
                freelancer_id=freelancer_id,
                title=title,
                slot_start_iso=slot_start_iso,
                duration_minutes=duration_minutes,
            )
            idem.save(
                server_name="workspace",
                tool_name="create_tentative_interview_hold",
                idempotency_key=idempotency_key,
                payload=payload,
                response=created,
            )
            audit.record(
                server_name="workspace",
                tool_name="create_tentative_interview_hold",
                raw_args=raw_args,
                result_summary=f"created hold_id={created['hold_id']}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return created
        except Exception as exc:
            audit.record(
                server_name="workspace",
                tool_name="create_tentative_interview_hold",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    @mcp.tool(
        name="create_email_draft",
        description=(
            "Create a local follow-up or scheduling summary email draft for human review. "
            "Strictly draft-only (never sends email). Requires 'idempotency_key' and supports 'dry_run'."
        ),
        annotations=make_tool_annotations(
            read_only=False, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def create_email_draft(
        idempotency_key: str,
        recipient_alias: str,
        subject: str,
        body: str,
        thread_id: str | None = None,
        freelancer_id: str = "persona_aiml_01",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        payload = {
            "thread_id": thread_id,
            "freelancer_id": freelancer_id,
            "recipient_alias": recipient_alias,
            "subject": subject,
            "body": body,
        }
        raw_args = {"idempotency_key": idempotency_key, "dry_run": dry_run, **payload}
        try:
            if not dry_run:
                cached = idem.get_cached(
                    server_name="workspace",
                    tool_name="create_email_draft",
                    idempotency_key=idempotency_key,
                    payload=payload,
                )
                if cached is not None:
                    audit.record(
                        server_name="workspace",
                        tool_name="create_email_draft",
                        raw_args=raw_args,
                        result_summary=f"idempotent_replay draft_id={cached.get('draft_id')}",
                        status="idempotent_replay",
                        duration_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                    return cached

            if dry_run:
                preview = {
                    "draft_id": "dryrun_email_draft",
                    "thread_id": thread_id,
                    "freelancer_id": freelancer_id,
                    "recipient_alias": recipient_alias,
                    "subject": subject,
                    "word_count": len(body.split()),
                    "status": "dry_run_ok",
                    "auto_sent": False,
                }
                audit.record(
                    server_name="workspace",
                    tool_name="create_email_draft",
                    raw_args=raw_args,
                    result_summary="dry_run email draft preview",
                    status="dry_run",
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                )
                return preview

            created = store.create_draft(
                idempotency_key=idempotency_key,
                thread_id=thread_id,
                freelancer_id=freelancer_id,
                recipient_alias=recipient_alias,
                subject=subject,
                body=body,
            )
            idem.save(
                server_name="workspace",
                tool_name="create_email_draft",
                idempotency_key=idempotency_key,
                payload=payload,
                response=created,
            )
            audit.record(
                server_name="workspace",
                tool_name="create_email_draft",
                raw_args=raw_args,
                result_summary=f"created draft_id={created['draft_id']}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return created
        except Exception as exc:
            audit.record(
                server_name="workspace",
                tool_name="create_email_draft",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    return mcp


def create_app(
    *,
    db_path: Path = DEFAULT_WORKSPACE_DB_PATH,
    expected_token: str | None = None,
    rate_limit_rpm: int | None = None,
    rate_limiter: SlidingWindowRateLimiter | None = None,
    audit_logger: MCPAuditLogger | None = None,
    idempotency_store: MCPIdempotencyStore | None = None,
):
    """Create the secured Starlette ASGI application for Workspace MCP."""
    server = build_workspace_mcp_server(
        db_path=db_path,
        audit_logger=audit_logger,
        idempotency_store=idempotency_store,
    )
    return build_secured_mcp_asgi_app(
        server,
        server_name="workspace",
        expected_token=expected_token,
        rate_limit_rpm=rate_limit_rpm,
        rate_limiter=rate_limiter,
    )


def main() -> None:
    host = os.getenv("WORKSPACE_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("WORKSPACE_MCP_PORT", "8093"))
    asgi_app = create_app()
    uvicorn.run(asgi_app, host=host, port=port)


if __name__ == "__main__":
    main()