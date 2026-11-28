"""Variant V3: Two-Stage Hybrid RRF + Cross-Encoder Reranking + MMR Deduplication.

Characteristics:
  - Stage 1 (Recall): Retrieves top-20 candidates via Hybrid Dense + BM25 RRF (`retrieve_v1_hybrid_rrf`).
  - Stage 2 (Precision): Scores every `(query, chunk.text)` pair with `cross-encoder/ms-marco-MiniLM-L-6-v2`
    (CPU-optimized ~90 MB model).
  - Stage 3 (Diversity): Applies Maximal Marginal Relevance (MMR, lambda=0.75) with an explicit
    penalty on duplicate `evidence_id`s so top-k results surface distinct portfolio proof points
    rather than three chunks from the same project.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rag.index.hybrid_store import HybridIndexStore
from rag.ingest.pipeline import DocumentChunk
from rag.variants.v1_hybrid_rrf import retrieve_v1_hybrid_rrf


def _apply_mmr_diversity(
    store: HybridIndexStore,
    scored_candidates: list[tuple[DocumentChunk, dict[str, float]]],
    k: int,
    lambda_mult: float = 0.75,
) -> list[tuple[DocumentChunk, dict[str, float]]]:
    """Select top-k diverse chunks balancing relevance score and inter-chunk novelty."""
    if len(scored_candidates) <= 1:
        return scored_candidates[:k]

    chunk_to_idx = {c.chunk_id: i for i, c in enumerate(store.chunks)}
    cand_vecs: list[np.ndarray] = []
    for chunk, _ in scored_candidates:
        idx = chunk_to_idx.get(chunk.chunk_id)
        if idx is not None and idx < store.vectors.shape[0]:
            cand_vecs.append(store.vectors[idx])
        else:
            cand_vecs.append(store.embedder.encode([chunk.text])[0])

    selected_positions: list[int] = []
    selected_evidence_ids: set[str] = set()
    remaining_positions = list(range(len(scored_candidates)))

    while remaining_positions and len(selected_positions) < k:
        best_pos = -1
        best_mmr = -1e9

        for pos in remaining_positions:
            chunk, metrics = scored_candidates[pos]
            rel = metrics["score"]

            if not selected_positions:
                redundancy = 0.0
            else:
                sims = [
                    float(np.dot(cand_vecs[pos], cand_vecs[s_pos])) for s_pos in selected_positions
                ]
                redundancy = max(sims)

            # Mild penalty when citing an already-selected parent evidence_id
            same_parent_penalty = (
                0.12 if (chunk.evidence_id and chunk.evidence_id in selected_evidence_ids) else 0.0
            )
            mmr_val = lambda_mult * rel - (1.0 - lambda_mult) * redundancy - same_parent_penalty

            if mmr_val > best_mmr:
                best_mmr = mmr_val
                best_pos = pos

        selected_positions.append(best_pos)
        remaining_positions.remove(best_pos)
        ev_id = scored_candidates[best_pos][0].evidence_id
        if ev_id:
            selected_evidence_ids.add(ev_id)

    return [scored_candidates[pos] for pos in selected_positions]


def retrieve_v3_cross_encoder_rerank(
    store: HybridIndexStore,
    query: str,
    k: int = 5,
    namespace: str | None = "portfolio",
    filters: dict[str, Any] | None = None,
    recall_k: int = 20,
) -> list[tuple[DocumentChunk, dict[str, float]]]:
    """Two-stage Hybrid RRF recall + Cross-Encoder neural reranking + MMR diversity."""
    if not store.is_built:
        return []

    stage1 = retrieve_v1_hybrid_rrf(
        store=store,
        query=query,
        k=max(recall_k, k * 3),
        namespace=namespace,
        filters=filters,
    )
    if not stage1:
        return []

    pairs = [(query, chunk.text) for chunk, _ in stage1]
    ce_scores = store.reranker.predict(pairs)

    reranked: list[tuple[DocumentChunk, dict[str, float]]] = []
    for (chunk, metrics), ce_score in zip(stage1, ce_scores, strict=False):
        blended = 0.72 * ce_score + 0.28 * min(1.0, metrics["score"])
        updated = {
            "score": round(blended, 6),
            "dense_score": metrics["dense_score"],
            "sparse_score": metrics["sparse_score"],
            "rerank_score": round(ce_score, 6),
        }
        reranked.append((chunk, updated))

    reranked.sort(key=lambda item: item[1]["score"], reverse=True)
    return _apply_mmr_diversity(store, reranked, k=max(1, k))
