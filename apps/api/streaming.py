"""In-memory + LangGraph Run Registry & SSE Event Streaming Bridge for GigBid API."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class RunRegistry:
    """Thread-safe registry tracking active/completed LangGraph runs and SSE subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)

    def create_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        graph_type: str,
        job_id: str | None,
        freelancer_id: str,
    ) -> dict[str, Any]:
        now = _utc_now()
        record = {
            "run_id": run_id,
            "thread_id": thread_id,
            "graph_type": graph_type,
            "job_id": job_id,
            "freelancer_id": freelancer_id,
            "status": "running",
            "awaiting_approval": False,
            "created_at": now,
            "updated_at": now,
            "interrupt_payload": None,
            "state_snapshot": {},
        }
        with self._lock:
            self._runs[thread_id] = record
        return dict(record)

    def update_run(
        self,
        thread_id: str,
        *,
        status: str | None = None,
        awaiting_approval: bool | None = None,
        interrupt_payload: dict[str, Any] | None = None,
        state_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            rec = self._runs.get(thread_id)
            if rec is None:
                return None
            if status is not None:
                rec["status"] = status
            if awaiting_approval is not None:
                rec["awaiting_approval"] = awaiting_approval
            if interrupt_payload is not None or awaiting_approval is False:
                rec["interrupt_payload"] = interrupt_payload
            if state_snapshot is not None:
                rec["state_snapshot"] = state_snapshot
            rec["updated_at"] = _utc_now()
            return dict(rec)

    def get_run(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._runs.get(thread_id)
            return dict(rec) if rec is not None else None

    def list_runs(self, *, only_pending_approval: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            items = [dict(r) for r in self._runs.values()]
        if only_pending_approval:
            items = [r for r in items if r.get("awaiting_approval")]
        items.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
        return items

    def publish_event(self, thread_id: str, event_type: str, data: dict[str, Any]) -> None:
        envelope = {
            "event": event_type,
            "timestamp": _utc_now(),
            "thread_id": thread_id,
            "data": data,
        }
        with self._lock:
            self._events[thread_id].append(envelope)
            queues = list(self._subscribers.get(thread_id, []))
        for q in queues:
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                pass

    def get_events(self, thread_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events.get(thread_id, []))

    async def subscribe(self, thread_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield historical SSE events first, then stream live updates until terminal status."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        with self._lock:
            history = list(self._events.get(thread_id, []))
            self._subscribers[thread_id].append(queue)
            current_run = dict(self._runs[thread_id]) if thread_id in self._runs else None

        try:
            for past in history:
                yield {
                    "event": past["event"],
                    "data": json.dumps(past, ensure_ascii=False, default=str),
                }

            if current_run and current_run.get("status") in {
                "skipped",
                "blocked_by_compliance",
                "submitted",
                "sent",
                "rejected_by_human",
                "error",
                "manual_review_required",
                "draft_only_manual_review",
            }:
                yield {
                    "event": "stream_end",
                    "data": json.dumps(
                        {"thread_id": thread_id, "status": current_run.get("status")},
                        default=str,
                    ),
                }
                return

            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield {
                        "event": msg["event"],
                        "data": json.dumps(msg, ensure_ascii=False, default=str),
                    }
                    if msg["event"] in {"run_completed", "run_error", "stream_end"}:
                        break
                except TimeoutError:
                    yield {
                        "event": "ping",
                        "data": json.dumps({"timestamp": _utc_now(), "thread_id": thread_id}),
                    }
                    run_now = self.get_run(thread_id)
                    if run_now and run_now.get("status") in {
                        "skipped",
                        "blocked_by_compliance",
                        "submitted",
                        "sent",
                        "rejected_by_human",
                        "error",
                    }:
                        break
        finally:
            with self._lock:
                subs = self._subscribers.get(thread_id, [])
                if queue in subs:
                    subs.remove(queue)


_REGISTRY = RunRegistry()


def get_run_registry() -> RunRegistry:
    """Singleton accessor for the GigBid `RunRegistry`."""
    return _REGISTRY


__all__ = ["RunRegistry", "get_run_registry"]
