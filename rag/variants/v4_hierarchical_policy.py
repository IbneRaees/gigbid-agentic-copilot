"""Variant V4: Hierarchical Parent-Document & Policy-Section Retrieval.

Characteristics:
  - Child-level Hybrid RRF recall finds pinpoint matches on specific problem statements,
    technical architectures, quantitative metrics, or individual policy clauses.
  - Expands matched child chunks to their complete parent document (`parent_text`: full case
    study with Problem + Solution + Measured Metrics + Skills, or full `POL-XXX` policy rule).
  - Deduplicates by `parent_id` so each returned item represents a complete, self-contained
    evidence unit.
  - Reranks the expanded parent contexts against the query using the cross-encoder
    (`cross-encoder/ms-marco-MiniLM-L-6-v2`) so compliance constraints and quantitative metrics
    are never severed mid-sentence.
"""

from __future__ import annotations

from typing import Any

from rag.index.hybrid_store import HybridIndexStore
from rag.ingest.pipeline import DocumentChunk
from rag.variants.v1_hybrid_rrf import retrieve_v1_hybrid_rrf


def retrieve_v4_hierarchical_policy(
    store: HybridIndexStore,
    query: str,
    k: int = 5,
    namespace: str | None = "portfolio",
    filters: dict[str, Any] | None = None,
    recall_k: int = 24,
) -> list[tuple[DocumentChunk, dict[str, float]]]:
    """Retrieve child chunks, collapse to unique parent contexts, and cross-encoder rerank."""
    if not store.is_built:
        return []

    child_hits = retrieve_v1_hybrid_rrf(
        store=store,
        query=query,
        k=max(recall_k, k * 4),
        namespace=namespace,
        filters=filters,
    )
    if not child_hits:
        return []

    # Group by parent_id, keeping highest-scoring child as representative & elevating text=parent_text
    parent_buckets: dict[str, tuple[DocumentChunk, dict[str, float]]] = {}
    for chunk, metrics in child_hits:
        pid = chunk.parent_id
        if pid not in parent_buckets or metrics["score"] > parent_buckets[pid][1]["score"]:
            expanded_chunk = chunk.model_copy(
                update={
                    "text": chunk.parent_text,
                    "section_title": f"Full Parent Context ({chunk.section_title})",
                }
            )
            parent_buckets[pid] = (expanded_chunk, dict(metrics))

    unique_parents = list(parent_buckets.values())
    pairs = [(query, parent_chunk.text) for parent_chunk, _ in unique_parents]
    ce_scores = store.reranker.predict(pairs)

    reranked_parents: list[tuple[DocumentChunk, dict[str, float]]] = []
    for (parent_chunk, metrics), ce_score in zip(unique_parents, ce_scores, strict=False):
        blended = 0.75 * ce_score + 0.25 * min(1.0, metrics["score"])
        reranked_parents.append(
            (
                parent_chunk,
                {
                    "score": round(blended, 6),
                    "dense_score": metrics["dense_score"],
                    "sparse_score": metrics["sparse_score"],
                    "rerank_score": round(ce_score, 6),
                },
            )
        )

    reranked_parents.sort(key=lambda item: item[1]["score"], reverse=True)
    return reranked_parents[: max(1, k)]
