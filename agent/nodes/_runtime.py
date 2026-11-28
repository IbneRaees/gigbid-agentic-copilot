"""Cross-platform runtime helpers for lazy MCP loading, audit logging, and workspace fallback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        s = _BEARER_RE.sub(r"\1[REDACTED_TOKEN]", value)
        s = _EMAIL_RE.sub("[REDACTED_EMAIL]", s)
        s = _PHONE_RE.sub("[REDACTED_PHONE]", s)
        return s[:497] + "..." if len(s) > 500 else s
    if isinstance(value, dict):
        return {str(k): _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def is_mcp_enabled() -> bool:
    return os.getenv("GIGBID_USE_MCP", "true").lower() not in {"0", "false", "no"}


def get_mcp_client_manager(*, trace_id: str, user_id: str) -> Any | None:
    """Lazily instantiate `GigBidMCPClientManager` if enabled and `langchain_mcp_adapters` is present."""
    if not is_mcp_enabled():
        return None
    try:
        from agent.mcp_client import GigBidMCPClientManager

        return GigBidMCPClientManager(include_github=False, trace_id=trace_id, user_id=user_id)
    except Exception:
        return None


def record_mcp_audit(
    *,
    server_name: str,
    tool_name: str,
    raw_args: dict[str, Any],
    result_summary: str,
    status: str = "ok",
    trace_id: str = "local-trace",
    user_id: str = "persona_aiml_01",
) -> None:
    """Record an audit entry using `MCPAuditLogger` or an identical JSONL writer."""
    try:
        from mcp_servers.security import MCPAuditLogger

        MCPAuditLogger().record(
            server_name=server_name,
            tool_name=tool_name,
            raw_args=raw_args,
            result_summary=result_summary,
            status=status,
            trace_id=trace_id,
            user_id=user_id,
        )
        return
    except Exception:
        pass

    log_path = Path(os.getenv("MCP_AUDIT_LOG_PATH", str(ROOT_DIR / "logs" / "mcp_audit.jsonl")))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "trace_id": trace_id,
        "user_id": user_id,
        "server_name": server_name,
        "tool_name": tool_name,
        "status": status,
        "duration_ms": 0.0,
        "sanitized_args": _scrub(raw_args),
        "result_summary": _scrub(result_summary),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


class LocalWorkspaceFallback:
    """SQLite-compatible workspace store matching `mcp_servers/workspace/server.py`."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(
            db_path or os.getenv("WORKSPACE_DB_PATH", str(ROOT_DIR / "gigsim" / "workspace.db"))
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def get_booked_starts(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT slot_start_iso FROM calendar_holds WHERE status != 'cancelled'"
            ).fetchall()
        return {str(r["slot_start_iso"]) for r in rows}

    def get_free_slots(
        self, *, days_ahead: int = 3, slot_minutes: int = 30, timezone_name: str = "UTC"
    ) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        base_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        booked = self.get_booked_starts()
        slots: list[dict[str, Any]] = []
        for day_offset in range(max(1, min(days_ahead, 14))):
            day = base_day + timedelta(days=day_offset)
            for hr in (14, 16, 18):
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
        hold_id = f"hold_{hashlib.sha256(idempotency_key.encode()).hexdigest()[:10]}"
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO calendar_holds (
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
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO email_drafts (
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


def get_workspace_store() -> Any:
    try:
        from mcp_servers.workspace.server import WorkspaceStore

        return WorkspaceStore()
    except Exception:
        return LocalWorkspaceFallback()


__all__ = [
    "get_mcp_client_manager",
    "get_workspace_store",
    "is_mcp_enabled",
    "record_mcp_audit",
]
