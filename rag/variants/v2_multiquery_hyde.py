"""Variant V2: Multi-Query Decomposition + Hypothetical Document Embedding (HyDE).

Characteristics:
  - Decomposes multi-requirement job descriptions or policy checks into focused sub-queries
    (splitting on clauses, bullet points, conjunctions, and extracted technical keywords).
  - Synthesizes a Hypothetical Document Embedding (HyDE) passage representing an ideal matching
    case study or policy clause (deterministically offline, or via local Ollama/OpenAI-compatible
    endpoint when `GIGBID_HYDE_LLM_URL` is explicitly configured).
  - Executes Hybrid RRF across the original query, sub-queries, and HyDE passage, fusing all
    sub-rankings via multi-query Reciprocal Rank Fusion.
"""

from __future__ import annotations

import os
import re
from typing import Any

from rag.index.hybrid_store import HybridIndexStore, tokenize_technical
from rag.ingest.pipeline import DocumentChunk
from rag.variants.v1_hybrid_rrf import retrieve_v1_hybrid_rrf


def decompose_query(query: str, max_subqueries: int = 4) -> list[str]:
    """Split a complex job post or requirement prompt into focused atomic sub-queries."""
    cleaned = query.strip()
    if not cleaned:
        return []

    raw_parts = re.split(
        r"[\n;•]+|\.\s+|\b(?:and|plus|with|including)\b", cleaned, flags=re.IGNORECASE
    )
    subqueries: list[str] = []
    seen: set[str] = set()

    for part in raw_parts:
        candidate = re.sub(r"\s+", " ", part).strip(" -*:,.")
        if len(candidate) >= 12 and candidate.lower() not in seen:
            seen.add(candidate.lower())
            subqueries.append(candidate)
        if len(subqueries) >= max_subqueries:
            break

    # Also construct a dense keyword-focused sub-query for multi-skill matching
    tech_tokens = [
        t
        for t in tokenize_technical(cleaned)
        if len(t) >= 3
        and t not in {"the", "and", "for", "with", "that", "this", "from", "need", "looking"}
    ]
    if tech_tokens:
        kw_query = " ".join(dict.fromkeys(tech_tokens[:12]))
        if kw_query.lower() not in seen:
            subqueries.append(kw_query)

    return subqueries[:max_subqueries]


def generate_hyde_passage(query: str, namespace: str | None = "portfolio") -> str:
    """Generate a hypothetical ideal matching document passage for HyDE retrieval."""
    # Optional live local Ollama / OpenAI-compatible expansion when explicitly opted in
    llm_url = os.getenv("GIGBID_HYDE_LLM_URL", "").strip()
    if llm_url:
        try:
            import httpx

            resp = httpx.post(
                llm_url,
                json={
                    "model": os.getenv("GIGBID_HYDE_MODEL", "qwen2.5:3b"),
                    "prompt": (
                        "Write a concise 2-sentence factual case-study summary demonstrating "
                        f"production delivery for: {query[:400]}"
                    ),
                    "stream": False,
                },
                timeout=3.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = str(data.get("response") or "").strip()
                if text:
                    return text
        except Exception:
            pass

    # Deterministic domain-calibrated HyDE synthesis (zero network latency)
    keywords = list(dict.fromkeys(tokenize_technical(query)))[:14]
    kw_phrase = ", ".join(keywords) if keywords else query[:160]
    if namespace == "policy":
        return (
            f"Marketplace Policy Rule and Compliance Clause governing {kw_phrase}. "
            f"Specifies mandatory platform requirements, prohibited circumvention behaviors, "
            f"Connects cost rules, and human approval enforcement."
        )
    return (
        f"Verified Case Study and Production Architecture delivering {kw_phrase}. "
        f"Implemented scalable backend services, measurable latency and throughput improvements, "
        f"automated testing, and verifiable client outcomes."
    )


def retrieve_v2_multiquery_hyde(
    store: HybridIndexStore,
    query: str,
    k: int = 5,
    namespace: str | None = "portfolio",
    filters: dict[str, Any] | None = None,
) -> list[tuple[DocumentChunk, dict[str, float]]]:
    """Run multi-query decomposition + HyDE expansion fused over Hybrid RRF."""
    if not store.is_built:
        return []

    subqueries = decompose_query(query)
    hyde_doc = generate_hyde_passage(query, namespace=namespace)

    # Weighted query list: primary query (1.0), HyDE passage (0.75), sub-queries (0.55 each)
    weighted_queries: list[tuple[str, float]] = [(query, 1.0), (hyde_doc, 0.75)]
    for sq in subqueries:
        if sq.lower() != query.lower():
            weighted_queries.append((sq, 0.55))

    candidate_pool_size = max(15, k * 4)
    fused_scores: dict[str, float] = {}
    best_metrics: dict[str, dict[str, float]] = {}
    chunk_lookup: dict[str, DocumentChunk] = {}

    rrf_k = 60
    for q_text, weight in weighted_queries:
        hits = retrieve_v1_hybrid_rrf(
            store=store,
            query=q_text,
            k=candidate_pool_size,
            namespace=namespace,
            filters=filters,
        )
        for rank, (chunk, metrics) in enumerate(hits, start=1):
            cid = chunk.chunk_id
            chunk_lookup[cid] = chunk
            fused_scores[cid] = fused_scores.get(cid, 0.0) + weight * (1.0 / (rrf_k + rank))
            if cid not in best_metrics or metrics["score"] > best_metrics[cid]["score"]:
                best_metrics[cid] = dict(metrics)

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[
        : max(1, k)
    ]
    top_fused = fused_scores[ranked_ids[0]] if ranked_ids else 1.0

    results: list[tuple[DocumentChunk, dict[str, float]]] = []
    for cid in ranked_ids:
        m = best_metrics[cid]
        norm_score = fused_scores[cid] / max(1e-9, top_fused)
        results.append(
            (
                chunk_lookup[cid],
                {
                    "score": round(norm_score, 6),
                    "dense_score": m["dense_score"],
                    "sparse_score": m["sparse_score"],
                    "rerank_score": 0.0,
                },
            )
        )
    return results
