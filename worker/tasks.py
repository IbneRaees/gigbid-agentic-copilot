"""Celery / Redis Background Worker Tasks for GigBid.

Handles scheduled job scouting, RAG index rebuilds, and client-thread polling.
Strictly enforces `AUTO_SEND_ENABLED=false`: background tasks may prepare
drafts up to the LangGraph `interrupt()` gate, NEVER submit proposals or send
client replies without explicit human approval.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from typing import Any

from celery import Celery

from agent.graphs.bid_graph import build_bid_graph
from agent.graphs.checkpointer import get_sync_or_local_checkpointer
from agent.graphs.conversation_graph import build_conversation_graph
from agent.llm import resolve_model_profile
from agent.nodes.scout import scout_jobs
from agent.state import create_initial_bid_state, create_initial_conversation_state
from rag.index.hybrid_store import HybridIndexStore

logger = logging.getLogger(__name__)

BROKER_URL = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND") or os.getenv(
    "REDIS_URL", "redis://localhost:6379/1"
)

celery_app = Celery("gigbid", broker=BROKER_URL, backend=RESULT_BACKEND)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    beat_schedule={
        "scout-saved-searches-every-15m": {
            "task": "worker.tasks.scout_and_prepare_bids",
            "schedule": 900.0,
            "args": ("persona_aiml_01", 5),
        },
    },
)


def _run_async(coro: Any) -> Any:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def _scout_and_prepare_async(
    freelancer_id: str = "persona_aiml_01",
    limit: int = 5,
    rag_variant: str = "v4",
) -> dict[str, Any]:
    """Scout jobs and run BidGraph up to `skip` or HITL `interrupt()` (never auto-submits)."""
    jobs = await scout_jobs(freelancer_id=freelancer_id, limit=limit)
    graph = build_bid_graph(checkpointer=get_sync_or_local_checkpointer())
    prepared: list[dict[str, Any]] = []
    skipped: list[str] = []

    for job in jobs[:limit]:
        job_id = str(job["job_id"])
        state = create_initial_bid_state(
            job_id=job_id,
            freelancer_id=freelancer_id,
            model_profile=resolve_model_profile(),
            rag_variant=rag_variant,
        )
        config = {"configurable": {"thread_id": state["thread_id"]}}
        res = await graph.ainvoke(state, config=config)
        if "__interrupt__" in res:
            prepared.append(
                {
                    "job_id": job_id,
                    "thread_id": state["thread_id"],
                    "status": "awaiting_human_approval",
                }
            )
        else:
            skipped.append(job_id)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "freelancer_id": freelancer_id,
        "scanned": len(jobs[:limit]),
        "awaiting_approval": prepared,
        "skipped": skipped,
        "auto_submitted": 0,  # Hard invariant: POL-008
    }


@celery_app.task(name="worker.tasks.scout_and_prepare_bids")
def scout_and_prepare_bids(
    freelancer_id: str = "persona_aiml_01",
    limit: int = 5,
    rag_variant: str = "v4",
) -> dict[str, Any]:
    """Celery task running BidGraph scouting up to the human approval gate."""
    logger.info("Running scout_and_prepare_bids for %s (limit=%d)", freelancer_id, limit)
    return _run_async(_scout_and_prepare_async(freelancer_id, limit, rag_variant))


@celery_app.task(name="worker.tasks.prepare_conversation_reply")
def prepare_conversation_reply(
    thread_id: str,
    freelancer_id: str = "persona_aiml_01",
) -> dict[str, Any]:
    """Celery task running ConversationGraph up to the human approval gate."""

    async def _run() -> dict[str, Any]:
        graph = build_conversation_graph(checkpointer=get_sync_or_local_checkpointer())
        state = create_initial_conversation_state(
            thread_id=thread_id,
            freelancer_id=freelancer_id,
            model_profile=resolve_model_profile(),
        )
        lg_tid = f"worker_conv_{thread_id}_{state['run_id']}"
        res = await graph.ainvoke(state, config={"configurable": {"thread_id": lg_tid}})
        return {
            "thread_id": thread_id,
            "lg_thread_id": lg_tid,
            "awaiting_approval": "__interrupt__" in res,
            "intent": (res.get("reply_intent") or {}).get("intent"),
            "auto_sent": False,
        }

    return _run_async(_run())


@celery_app.task(name="worker.tasks.rebuild_rag_index")
def rebuild_rag_index(
    data_dir: str = "data", index_dir: str = "rag/index/local_store"
) -> dict[str, Any]:
    """Rebuild the hybrid dense + BM25 portfolio/policy RAG index."""
    store = HybridIndexStore(index_dir=index_dir)
    meta = store.build_from_data(data_dir=data_dir)
    return {"status": "rebuilt", **meta}


__all__ = [
    "celery_app",
    "scout_and_prepare_bids",
    "prepare_conversation_reply",
    "rebuild_rag_index",
]
