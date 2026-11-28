"""Memory, Portfolio Evidence, MCP Audit Logs & Evaluation Results Router."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query

from agent.memory.store import GigBidMemoryStore
from apps.api.auth import verify_optional_api_token

router = APIRouter(tags=["Observability & Evals"])

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
EVALS_RESULTS_DIR = ROOT_DIR / "evals" / "results"


@router.get("/memory")
async def get_memory_and_portfolio(
    freelancer_id: str | None = Query(default=None),
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """Return promoted semantic memories, episodic events, and verified portfolio evidence_ids."""
    store = GigBidMemoryStore()
    semantic = [
        m.model_dump(mode="json")
        for m in store.list_semantic(freelancer_id=freelancer_id, limit=100)
    ]
    episodic = [
        e.model_dump(mode="json")
        for e in store.list_episodic(freelancer_id=freelancer_id, limit=100)
    ]

    portfolio_path = DATA_DIR / "portfolio_items.json"
    portfolio_items: list[dict[str, Any]] = []
    if portfolio_path.exists():
        raw_items = json.loads(portfolio_path.read_text(encoding="utf-8"))
        for item in raw_items:
            if freelancer_id and item.get("freelancer_id") != freelancer_id:
                continue
            portfolio_items.append(item)

    return {
        "stats": store.stats(),
        "semantic_memories": semantic,
        "episodic_events": episodic,
        "portfolio_items": portfolio_items,
    }


@router.get("/audit-logs")
async def get_mcp_audit_logs(
    server: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """Return PII-scrubbed MCP tool audit entries from `logs/mcp_audit.jsonl`."""
    log_path = Path(os.getenv("MCP_AUDIT_LOG_PATH", str(ROOT_DIR / "logs" / "mcp_audit.jsonl")))
    entries: list[dict[str, Any]] = []
    if await asyncio.to_thread(log_path.exists):
        raw_text = await asyncio.to_thread(log_path.read_text, encoding="utf-8")
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if server and entry.get("server_name") != server:
                continue
            if status and entry.get("status") != status:
                continue
            entries.append(entry)

    entries.reverse()
    return {
        "total": len(entries),
        "log_path": str(log_path),
        "entries": entries[:limit],
    }


@router.get("/evals/results")
async def get_eval_results(
    _: str = Depends(verify_optional_api_token),
) -> dict[str, Any]:
    """Return RAG comparison, full end-to-end eval, and extension ablation reports."""

    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    return {
        "rag_comparison": _read_json(EVALS_RESULTS_DIR / "rag_comparison.json"),
        "full_eval_report": _read_json(EVALS_RESULTS_DIR / "full_eval_report.json"),
        "extension_ablation": _read_json(EVALS_RESULTS_DIR / "extension_ablation.json"),
        "judge_calibration": _read_json(EVALS_RESULTS_DIR / "judge_calibration.json"),
    }
