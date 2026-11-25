"""Unified GigBidRetriever routing variants (V0-V4) and namespaces (portfolio vs policy)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from rag.index.hybrid_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    HybridIndexStore,
)
from rag.ingest.pipeline import DocumentChunk, NamespaceType, ProvenanceType
from rag.variants import VARIANT_REGISTRY

logger = logging.getLogger(__name__)

VariantId = Literal["v0", "v1", "v2", "v3", "v4"]


class EvidenceChunk(BaseModel):
    """Typed evidence unit returned by GigBidRetriever and consumed by agent nodes & ClaimVerifier."""

    chunk_id: str
    parent_id: str
    evidence_id: str | None = None
    namespace: NamespaceType
    doc_type: str
    title: str
    section_title: str = ""
    content: str = Field(..., description="Retrieved chunk content (or full parent text in V4)")
    parent_content: str = Field(..., description="Complete parent case study or policy rule text")
    freelancer_id: str | None = None
    skills: list[str] = Field(default_factory=list)
    outcome_status: str | None = None
    policy_rule_id: str | None = None
    data_provenance: ProvenanceType = "synthetic"
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    variant_id: str = "v4"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Structured telemetry and payload wrapper for a retrieval call."""

    query: str
    namespace: str
    variant_id: str
    chunks: list[EvidenceChunk]
    latency_ms: float
    total_indexed: int
    filters_applied: dict[str, Any] = Field(default_factory=dict)


def _normalize_variant_name(raw_variant: str | None) -> VariantId:
    if not raw_variant:
        return "v4"
    val = str(raw_variant).strip().lower()
    if val.startswith("v0"):
        return "v0"
    if val.startswith("v1"):
        return "v1"
    if val.startswith("v2"):
        return "v2"
    if val.startswith("v3"):
        return "v3"
    if val.startswith("v4"):
        return "v4"
    return "v4"


def _load_rag_config(config_path: str | Path = "agent/config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        raw_text = path.read_text(encoding="utf-8")
        if yaml is not None:
            loaded = yaml.safe_load(raw_text) or {}
            return loaded.get("rag", {}) if isinstance(loaded, dict) else {}
        # Lightweight fallback parser if PyYAML is not installed
        rag_section: dict[str, Any] = {}
        in_rag = False
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not line.startswith(" ") and stripped.endswith(":"):
                in_rag = stripped == "rag:"
                continue
            if in_rag and ":" in stripped:
                k, v = stripped.split(":", 1)
                rag_section[k.strip()] = v.strip().strip("\"'")
        return rag_section
    except Exception as exc:
        logger.warning("Could not parse %s (%s); using default RAG config.", path, exc)
        return {}


class GigBidRetriever:
    """Single typed entrypoint for all portfolio and policy RAG retrieval across GigBid."""

    def __init__(
        self,
        config_path: str | Path = "agent/config.yaml",
        index_dir: str | Path | None = None,
        data_dir: str | Path = "data",
        default_variant: VariantId | None = None,
        auto_build: bool = True,
    ) -> None:
        rag_cfg = _load_rag_config(config_path)

        configured_variant = default_variant or _normalize_variant_name(
            rag_cfg.get("variant") or rag_cfg.get("active_variant") or "v4"
        )
        self.default_variant: VariantId = configured_variant

        resolved_index_dir = Path(index_dir or rag_cfg.get("index_dir") or "rag/index/local_store")
        embedding_model = str(rag_cfg.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
        reranker_model = str(rag_cfg.get("reranker_model") or DEFAULT_RERANKER_MODEL)
        self.default_top_k = int(rag_cfg.get("top_k") or 5)

        self.store = HybridIndexStore(
            index_dir=resolved_index_dir,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
        if not self.store.load() and auto_build:
            logger.info(
                "Local hybrid index not found at %s; building from %s...",
                resolved_index_dir,
                data_dir,
            )
            self.store.build_from_data(data_dir=data_dir)

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        namespace: NamespaceType | None = "portfolio",
        variant: VariantId | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Execute retrieval using the selected variant and namespace, logging latency telemetry."""
        t0 = time.perf_counter()
        active_variant: VariantId = _normalize_variant_name(variant or self.default_variant)
        top_k = max(1, k if k is not None else self.default_top_k)
        active_filters = dict(filters or {})

        variant_fn = VARIANT_REGISTRY[active_variant]
        raw_hits: list[tuple[DocumentChunk, dict[str, float]]] = variant_fn(
            store=self.store,
            query=query,
            k=top_k,
            namespace=namespace,
            filters=active_filters,
        )

        evidence_chunks: list[EvidenceChunk] = []
        for chunk, metrics in raw_hits:
            evidence_chunks.append(
                EvidenceChunk(
                    chunk_id=chunk.chunk_id,
                    parent_id=chunk.parent_id,
                    evidence_id=chunk.evidence_id,
                    namespace=chunk.namespace,
                    doc_type=chunk.doc_type,
                    title=chunk.title,
                    section_title=chunk.section_title,
                    content=chunk.text,
                    parent_content=chunk.parent_text,
                    freelancer_id=chunk.freelancer_id,
                    skills=chunk.skills,
                    outcome_status=chunk.outcome_status,
                    policy_rule_id=chunk.policy_rule_id,
                    data_provenance=chunk.data_provenance,
                    score=metrics["score"],
                    dense_score=metrics["dense_score"],
                    sparse_score=metrics["sparse_score"],
                    rerank_score=metrics["rerank_score"],
                    variant_id=active_variant,
                    metadata=chunk.metadata,
                )
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        logger.info(
            "RAG retrieve | variant=%s namespace=%s k=%d hits=%d latency_ms=%.2f evidence_ids=%s",
            active_variant,
            namespace,
            top_k,
            len(evidence_chunks),
            elapsed_ms,
            [c.evidence_id for c in evidence_chunks],
        )

        return RetrievalResult(
            query=query,
            namespace=namespace or "all",
            variant_id=active_variant,
            chunks=evidence_chunks,
            latency_ms=elapsed_ms,
            total_indexed=len(self.store.chunks),
            filters_applied=active_filters,
        )

    def retrieve_portfolio(
        self,
        query: str,
        k: int | None = None,
        freelancer_id: str | None = None,
        required_skills: list[str] | None = None,
        variant: VariantId | None = None,
    ) -> list[EvidenceChunk]:
        """Convenience wrapper for retrieving grounded freelancer portfolio evidence."""
        filters: dict[str, Any] = {}
        if freelancer_id:
            filters["freelancer_id"] = freelancer_id
        if required_skills:
            filters["required_skills"] = required_skills
        return self.retrieve(
            query=query,
            k=k,
            namespace="portfolio",
            variant=variant,
            filters=filters,
        ).chunks

    def retrieve_policy(
        self,
        query: str,
        k: int | None = None,
        variant: VariantId | None = None,
    ) -> list[EvidenceChunk]:
        """Convenience wrapper for retrieving marketplace policy rules for Compliance Guard."""
        return self.retrieve(
            query=query,
            k=k,
            namespace="policy",
            variant=variant,
            filters={},
        ).chunks
