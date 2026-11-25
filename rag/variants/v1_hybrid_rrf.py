"""Variant V1: Hybrid Dense + BM25 Sparse Retrieval with Reciprocal Rank Fusion (RRF).

Characteristics:
  - Combines dense semantic cosine similarity and BM25 exact keyword ranking via RRF:
      RRF(d) = w_dense / (k_rrf + rank_dense(d)) + w_sparse / (k_rrf + rank_sparse(d))
  - Applies hard metadata filtering (`namespace`, `freelancer_id`, `doc_type`) and soft metadata
    boosting (`required_skills` overlap and `won`/`completed` outcome priority).
"""

from __future__ import annotations

from typing import Any

from rag.index.hybrid_store import HybridIndexStore
from rag.ingest.pipeline import DocumentChunk


def _matches_hard_filters(
    chunk: DocumentChunk,
    namespace: str | None,
    filters: dict[str, Any],
) -> bool:
    if namespace is not None and chunk.namespace != namespace:
        return False

    freelancer_id = filters.get("freelancer_id")
    if freelancer_id and chunk.namespace == "portfolio":
        if chunk.freelancer_id and chunk.freelancer_id != freelancer_id:
            return False

    doc_type = filters.get("doc_type")
    if doc_type:
        allowed_types = {doc_type} if isinstance(doc_type, str) else set(doc_type)
        if chunk.doc_type not in allowed_types:
            return False

    policy_rule_id = filters.get("policy_rule_id")
    if policy_rule_id and chunk.policy_rule_id != policy_rule_id:
        return False

    return True


def _soft_metadata_boost(chunk: DocumentChunk, filters: dict[str, Any]) -> float:
    boost = 0.0
    req_skills = filters.get("required_skills") or filters.get("skills") or []
    if req_skills and chunk.skills:
        req_lower = {str(s).strip().lower() for s in req_skills if str(s).strip()}
        chunk_lower = {s.lower() for s in chunk.skills}
        if req_lower:
            overlap = len(req_lower & chunk_lower) / len(req_lower)
            boost += 0.25 * overlap

    if chunk.outcome_status in {"won", "interviewed", "replied"}:
        boost += 0.08
    elif chunk.outcome_status == "completed":
        boost += 0.04

    return boost


def retrieve_v1_hybrid_rrf(
    store: HybridIndexStore,
    query: str,
    k: int = 5,
    namespace: str | None = "portfolio",
    filters: dict[str, Any] | None = None,
    rrf_k: int = 60,
    dense_weight: float = 0.55,
    sparse_weight: float = 0.45,
) -> list[tuple[DocumentChunk, dict[str, float]]]:
    """Fuse dense and sparse rankings via RRF with metadata filtering."""
    if not store.is_built:
        return []

    active_filters = filters or {}
    dense = store.dense_scores(query)
    sparse = store.sparse_scores(query)

    valid_indices = [
        idx
        for idx, chunk in enumerate(store.chunks)
        if _matches_hard_filters(chunk, namespace, active_filters)
    ]
    if not valid_indices:
        return []

    dense_ranked = sorted(valid_indices, key=lambda i: float(dense[i]), reverse=True)
    sparse_ranked = sorted(valid_indices, key=lambda i: float(sparse[i]), reverse=True)

    dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranked, start=1)}
    sparse_rank_map = {idx: rank for rank, idx in enumerate(sparse_ranked, start=1)}

    scored: list[tuple[int, float, float, float]] = []
    max_possible_rrf = (dense_weight + sparse_weight) / (rrf_k + 1)

    for idx in valid_indices:
        d_rank = dense_rank_map[idx]
        s_rank = sparse_rank_map[idx]
        raw_rrf = (dense_weight / (rrf_k + d_rank)) + (sparse_weight / (rrf_k + s_rank))
        norm_rrf = raw_rrf / max_possible_rrf
        boost = _soft_metadata_boost(store.chunks[idx], active_filters)
        final_score = norm_rrf + boost
        scored.append((idx, final_score, float(dense[idx]), float(sparse[idx])))

    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[: max(1, k)]

    return [
        (
            store.chunks[idx],
            {
                "score": round(final_score, 6),
                "dense_score": round(d_score, 6),
                "sparse_score": round(s_score, 6),
                "rerank_score": 0.0,
            },
        )
        for idx, final_score, d_score, s_score in top
    ]
