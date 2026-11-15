"""Shared Security, Rate Limiting, Idempotency & JSONL Audit Logging for GigBid MCP Servers.

Implements:
  - SDK version-agnostic `MCPServer` / `ToolAnnotations` builders (`mcp` 1.x and 2.x)
  - Bearer-token authentication middleware for Streamable HTTP / SSE transports
  - Per-user sliding-window rate limiter
  - Persistent SQLite + in-memory idempotency-key deduplicator for write tools
  - Append-only JSONL audit logger (`logs/mcp_audit.jsonl`) with PII/secret redaction
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # Backward compatibility for mcp < 2.0.0
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[assignment]

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_LOG_PATH = Path(
    os.getenv("MCP_AUDIT_LOG_PATH", str(ROOT_DIR / "logs" / "mcp_audit.jsonl"))
)
DEFAULT_IDEMPOTENCY_DB_PATH = Path(
    os.getenv("MCP_IDEMPOTENCY_DB_PATH", str(ROOT_DIR / "gigsim" / "mcp_idempotency.db"))
)

# Context variables propagated from HTTP middleware or direct Python calls
current_trace_id: ContextVar[str] = ContextVar("mcp_trace_id", default="local-trace")
current_user_id: ContextVar[str] = ContextVar("mcp_user_id", default="persona_aiml_01")

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)
_SENSITIVE_KEYS = {
    "token",
    "api_key",
    "secret",
    "password",
    "authorization",
    "access_token",
    "refresh_token",
}


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def make_tool_annotations(
    *,
    read_only: bool,
    destructive: bool,
    idempotent: bool,
    open_world: bool = False,
) -> ToolAnnotations:
    """Construct `mcp.types.ToolAnnotations` compatible with both mcp 1.x and 2.x."""
    model_fields = getattr(ToolAnnotations, "model_fields", {})
    if "read_only_hint" in model_fields:
        return ToolAnnotations(
            read_only_hint=read_only,
            destructive_hint=destructive,
            idempotent_hint=idempotent,
            open_world_hint=open_world,
        )
    return ToolAnnotations(
        readOnlyHint=read_only,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=open_world,
    )


def sanitize_value(value: Any, key_hint: str = "") -> Any:
    """Recursively scrub secrets and PII from arguments before writing audit logs."""
    if key_hint.lower() in _SENSITIVE_KEYS:
        return "[REDACTED_SECRET]"
    if isinstance(value, str):
        scrubbed = _BEARER_RE.sub(r"\1[REDACTED_TOKEN]", value)
        scrubbed = _EMAIL_RE.sub("[REDACTED_EMAIL]", scrubbed)
        scrubbed = _PHONE_RE.sub("[REDACTED_PHONE]", scrubbed)
        if len(scrubbed) > 500:
            return scrubbed[:497] + "..."
        return scrubbed
    if isinstance(value, dict):
        return {str(k): sanitize_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(item) for item in value]
    return value


class MCPAuditLogger:
    """Append-only JSONL audit logger recording every MCP tool invocation."""

    def __init__(self, log_path: Path = DEFAULT_AUDIT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        server_name: str,
        tool_name: str,
        raw_args: dict[str, Any],
        result_summary: str,
        status: str = "ok",
        duration_ms: float = 0.0,
        trace_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        entry = {
            "timestamp": utc_now_iso(),
            "trace_id": trace_id or current_trace_id.get(),
            "user_id": user_id or current_user_id.get(),
            "server_name": server_name,
            "tool_name": tool_name,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "sanitized_args": sanitize_value(raw_args),
            "result_summary": sanitize_value(result_summary),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry


class SlidingWindowRateLimiter:
    """Thread/async-safe per-user sliding-window rate limiter."""

    def __init__(self, max_requests_per_minute: int = 120) -> None:
        self.max_rpm = max(1, int(max_requests_per_minute))
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, identity: str) -> tuple[bool, int]:
        """Return `(allowed, retry_after_seconds)` for the given user/token identity."""
        now = time.monotonic()
        window = self._buckets[identity]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.max_rpm:
            retry_after = max(1, int(60.0 - (now - window[0])) + 1)
            return False, retry_after
        window.append(now)
        return True, 0

    def reset(self) -> None:
        """Clear all rate-limit buckets (useful in tests)."""
        self._buckets.clear()


class MCPIdempotencyStore:
    """SQLite-backed idempotency store preventing duplicate write side effects."""

    def __init__(self, db_path: Path = DEFAULT_IDEMPOTENCY_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_idempotency (
                    scoped_key TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get_cached(
        self, *, server_name: str, tool_name: str, idempotency_key: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return cached response dict if key was already executed with identical args.

        Raises `ValueError` if `idempotency_key` was reused with different arguments.
        """
        if not idempotency_key or not idempotency_key.strip():
            raise ValueError("Write tools require a non-empty 'idempotency_key'.")

        scoped_key = f"{server_name}:{tool_name}:{idempotency_key.strip()}"
        req_hash = self._hash_payload(payload)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_hash, response_json FROM mcp_idempotency WHERE scoped_key = ?",
                (scoped_key,),
            ).fetchone()

        if row is None:
            return None
        if row["payload_hash"] != req_hash:
            raise ValueError(
                f"Idempotency conflict: key '{idempotency_key}' was already used on '{tool_name}' "
                f"with different arguments."
            )
        return json.loads(row["response_json"])

    def save(
        self,
        *,
        server_name: str,
        tool_name: str,
        idempotency_key: str,
        payload: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        scoped_key = f"{server_name}:{tool_name}:{idempotency_key.strip()}"
        req_hash = self._hash_payload(payload)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO mcp_idempotency (
                    scoped_key, tool_name, payload_hash, response_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    scoped_key,
                    tool_name,
                    req_hash,
                    json.dumps(response, ensure_ascii=False, default=str),
                    utc_now_iso(),
                ),
            )


class BearerAuthAndRateLimitMiddleware(BaseHTTPMiddleware):
    """HTTP transport middleware enforcing Bearer auth, sliding-window rate limits, and trace IDs."""

    def __init__(
        self,
        app: Any,
        *,
        expected_token: str,
        rate_limiter: SlidingWindowRateLimiter,
        server_name: str,
    ) -> None:
        super().__init__(app)
        self.expected_token = expected_token
        self.rate_limiter = rate_limiter
        self.server_name = server_name

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Unauthenticated health probe for Docker Compose healthchecks
        if request.url.path == "/health":
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "detail": f"Missing Bearer token for {self.server_name} MCP server.",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        provided_token = auth_header[7:].strip()
        if not secrets.compare_digest(provided_token, self.expected_token):
            return JSONResponse(
                status_code=403,
                content={
                    "error": "forbidden",
                    "detail": f"Invalid Bearer token for {self.server_name} MCP server.",
                },
            )

        user_id = request.headers.get("x-user-id", "persona_aiml_01").strip() or "persona_aiml_01"
        trace_id = request.headers.get("x-trace-id", f"trace-{int(time.time() * 1000)}").strip()

        allowed, retry_after = self.rate_limiter.allow(f"{user_id}:{provided_token[:8]}")
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": f"Rate limit exceeded for user '{user_id}' on {self.server_name} MCP.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        tok_trace = current_trace_id.set(trace_id)
        tok_user = current_user_id.set(user_id)
        try:
            response = await call_next(request)
            response.headers["X-Trace-Id"] = trace_id
            return response
        finally:
            current_trace_id.reset(tok_trace)
            current_user_id.reset(tok_user)


def build_secured_mcp_asgi_app(
    mcp_server: MCPServer,
    *,
    server_name: str,
    expected_token: str | None = None,
    rate_limit_rpm: int | None = None,
    rate_limiter: SlidingWindowRateLimiter | None = None,
) -> Starlette:
    """Wrap an MCP server's Streamable HTTP ASGI app with Bearer auth, rate limits, and `/health`."""
    token = expected_token or os.getenv("MCP_SERVER_TOKEN") or os.getenv(
        "GIGSIM_API_TOKEN", "gigsim-dev-token-2026"
    )
    rpm = rate_limit_rpm or int(os.getenv("MCP_RATE_LIMIT_RPM", "120"))
    limiter = rate_limiter or SlidingWindowRateLimiter(max_requests_per_minute=rpm)

    async def health(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "server": server_name,
                "transport": "streamable-http",
                "timestamp": utc_now_iso(),
            }
        )

    mcp_http_app = mcp_server.streamable_http_app()
    starlette_app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Mount("/", app=mcp_http_app),
        ],
        lifespan=getattr(mcp_http_app.router, "lifespan_context", None),
    )
    starlette_app.add_middleware(
        BearerAuthAndRateLimitMiddleware,
        expected_token=token,
        rate_limiter=limiter,
        server_name=server_name,
    )
    return starlette_app


__all__ = [
    "MCPServer",
    "MCPAuditLogger",
    "SlidingWindowRateLimiter",
    "MCPIdempotencyStore",
    "BearerAuthAndRateLimitMiddleware",
    "build_secured_mcp_asgi_app",
    "make_tool_annotations",
    "sanitize_value",
    "utc_now_iso",
    "current_trace_id",
    "current_user_id",
]