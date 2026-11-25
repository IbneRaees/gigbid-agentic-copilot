"""Variant V0 (Baseline): Naive top-k dense cosine similarity.

Characteristics:
  - Pure dense dot-product over flat chunks.
  - No metadata filtering (except strict namespace boundary when requested by caller, or completely
    unfiltered when namespace is None).
  - No BM25 lexical search, no query expansion, and no neural reranking.
"""

from __future__ import annotations

from typing import Any

from rag.index.hybrid_store import HybridIndexStore
from rag.ingest.pipeline import DocumentChunk


def retrieve_v0_naive(
    store: HybridIndexStore,
    query: str,
    k: int = 5,
    namespace: str | None = "portfolio",
    filters: dict[str, Any] | None = None,
) -> list[tuple[DocumentChunk, dict[str, float]]]:
    """Return top-k chunks by raw dense cosine similarity."""
    if not store.is_built:
        return []

    dense = store.dense_scores(query)
    candidates: list[tuple[int, float]] = []
    for idx, chunk in enumerate(store.chunks):
        if namespace is not None and chunk.namespace != namespace:
            continue
        candidates.append((idx, float(dense[idx])))

    candidates.sort(key=lambda item: item[1], reverse=True)
    top = candidates[: max(1, k)]

    results: list[tuple[DocumentChunk, dict[str, float]]] = []
    for idx, score in top:
        results.append(
            (
                store.chunks[idx],
                {
                    "score": round(score, 6),
                    "dense_score": round(score, 6),
                    "sparse_score": 0.0,
                    "rerank_score": 0.0,
                },
            )
        )
    return results
