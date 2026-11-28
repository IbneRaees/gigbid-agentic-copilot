"""Episodic & Semantic Memory Store for GigBid (JSON-backed with optional Postgres sync)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MEMORY_STORE_PATH = Path(
    os.getenv("GIGBID_MEMORY_PATH", str(ROOT_DIR / "data" / "processed" / "memory_store.json"))
)

_TOKEN_RE = re.compile(r"[a-z0-9_+#.\-]+", re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) > 1}


class EpisodicEvent(BaseModel):
    """Raw episodic interaction record captured from Bid or Conversation graph runs."""

    event_id: str
    freelancer_id: str
    job_id: str
    thread_id: str | None = None
    client_id: str | None = None
    event_type: Literal["proposal_outcome", "client_feedback", "qa_pair", "skip_decision"]
    outcome: Literal["won", "interview", "replied", "approved", "rejected", "skipped", "drafted"]
    summary: str
    grounding_score: float = Field(default=0.0, ge=0.0, le=1.0)
    compliance_passed: bool = True
    evidence_ids: list[str] = Field(default_factory=list)
    screening_answers: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now_iso)


class SemanticMemoryItem(BaseModel):
    """Distilled, reusable semantic memory promoted from verified episodic events."""

    memory_id: str
    freelancer_id: str
    memory_type: Literal["winning_pattern", "client_preference", "reusable_qa", "risk_pattern"]
    title: str
    content: str
    evidence_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_event_ids: list[str] = Field(default_factory=list)
    usage_count: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)


class GigBidMemoryStore:
    """Thread-safe persistent store managing episodic events and semantic memories."""

    _lock = threading.Lock()

    def __init__(self, store_path: str | Path | None = None) -> None:
        self.store_path = Path(store_path or DEFAULT_MEMORY_STORE_PATH)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._episodic: list[EpisodicEvent] = []
        self._semantic: list[SemanticMemoryItem] = []
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
            self._episodic = [EpisodicEvent.model_validate(e) for e in raw.get("episodic", [])]
            self._semantic = [SemanticMemoryItem.model_validate(s) for s in raw.get("semantic", [])]
        except Exception:
            self._episodic = []
            self._semantic = []

    def _save(self) -> None:
        payload = {
            "updated_at": _utc_now_iso(),
            "episodic": [e.model_dump(mode="json") for e in self._episodic],
            "semantic": [s.model_dump(mode="json") for s in self._semantic],
        }
        tmp_path = self.store_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self.store_path)

    def add_episodic(self, event: EpisodicEvent) -> EpisodicEvent:
        """Persist an episodic event (deduplicating by `event_id`)."""
        with self._lock:
            self._episodic = [e for e in self._episodic if e.event_id != event.event_id]
            self._episodic.append(event)
            self._episodic = self._episodic[-500:]
            self._save()
        return event

    def upsert_semantic(self, item: SemanticMemoryItem) -> SemanticMemoryItem:
        """Insert or reinforce an existing semantic memory item."""
        with self._lock:
            for idx, existing in enumerate(self._semantic):
                if existing.memory_id == item.memory_id:
                    merged_sources = list(
                        dict.fromkeys([*existing.source_event_ids, *item.source_event_ids])
                    )
                    merged_evidence = list(
                        dict.fromkeys([*existing.evidence_ids, *item.evidence_ids])
                    )
                    updated = existing.model_copy(
                        update={
                            "content": item.content,
                            "confidence": max(existing.confidence, item.confidence),
                            "source_event_ids": merged_sources,
                            "evidence_ids": merged_evidence,
                            "usage_count": existing.usage_count + 1,
                            "updated_at": _utc_now_iso(),
                        }
                    )
                    self._semantic[idx] = updated
                    self._save()
                    return updated
            self._semantic.append(item)
            self._semantic = self._semantic[-200:]
            self._save()
        return item

    def query_semantic(
        self,
        *,
        freelancer_id: str | None = None,
        query: str = "",
        memory_type: str | None = None,
        limit: int = 5,
    ) -> list[SemanticMemoryItem]:
        """Retrieve relevant semantic memories ranked by lexical overlap and confidence."""
        q_tokens = _tokenize(query)
        scored: list[tuple[float, SemanticMemoryItem]] = []
        with self._lock:
            items = list(self._semantic)

        for item in items:
            if freelancer_id and item.freelancer_id != freelancer_id:
                continue
            if memory_type and item.memory_type != memory_type:
                continue
            doc_tokens = _tokenize(f"{item.title} {item.content} {' '.join(item.tags)}")
            overlap = len(q_tokens & doc_tokens) / max(1, len(q_tokens)) if q_tokens else 0.5
            score = (0.65 * overlap) + (0.35 * item.confidence)
            scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], -pair[1].usage_count))
        return [item for _, item in scored[: max(1, limit)]]

    def list_episodic(
        self,
        *,
        freelancer_id: str | None = None,
        limit: int = 50,
    ) -> list[EpisodicEvent]:
        with self._lock:
            items = list(self._episodic)
        if freelancer_id:
            items = [e for e in items if e.freelancer_id == freelancer_id]
        return list(reversed(items[-limit:]))

    def list_semantic(
        self,
        *,
        freelancer_id: str | None = None,
        memory_type: str | None = None,
        limit: int = 50,
    ) -> list[SemanticMemoryItem]:
        with self._lock:
            items = list(self._semantic)
        if freelancer_id:
            items = [s for s in items if s.freelancer_id == freelancer_id]
        if memory_type:
            items = [s for s in items if s.memory_type == memory_type]
        return list(reversed(items[-limit:]))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "episodic_count": len(self._episodic),
                "semantic_count": len(self._semantic),
                "store_path": str(self.store_path),
            }

    def clear(self) -> None:
        with self._lock:
            self._episodic = []
            self._semantic = []
            self._save()


def make_memory_id(prefix: str, key: str) -> str:
    """Create a deterministic memory identifier."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


__all__ = [
    "EpisodicEvent",
    "SemanticMemoryItem",
    "GigBidMemoryStore",
    "make_memory_id",
]
