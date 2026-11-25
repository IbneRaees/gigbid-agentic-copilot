"""Reproducible RAG Benchmark Suite comparing Variants V0 through V4 (Part B4/B5)."""

from __future__ import annotations

import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.retriever import GigBidRetriever, VariantId

VARIANTS: list[VariantId] = ["v0", "v1", "v2", "v3", "v4"]


def _build_eval_queries(retriever: GigBidRetriever) -> list[dict[str, Any]]:
    """Construct deterministic evaluation queries directly from indexed portfolio & policy items."""
    queries: list[dict[str, Any]] = []
    seen_parents: set[str] = set()

    # Select up to 5 distinct portfolio items across personas
    for chunk in retriever.store.chunks:
        if (
            chunk.namespace == "portfolio"
            and chunk.doc_type != "freelancer_profile"
            and chunk.evidence_id
            and chunk.parent_id not in seen_parents
        ):
            seen_parents.add(chunk.parent_id)
            skills = chunk.skills[:3]
            queries.append(
                {
                    "query_id": f"q_port_{len(queries) + 1:02d}",
                    "namespace": "portfolio",
                    "query": f"Looking for an engineer to build {chunk.title} using {', '.join(skills)}.",
                    "filters": {
                        "freelancer_id": chunk.freelancer_id,
                        "required_skills": skills,
                    },
                    "gold_evidence_ids": [chunk.evidence_id],
                    "gold_facts": [s.lower() for s in skills[:2]],
                }
            )
            if len(queries) >= 5:
                break

    # Select up to 3 distinct policy rules
    for chunk in retriever.store.chunks:
        if (
            chunk.namespace == "policy"
            and chunk.policy_rule_id
            and chunk.parent_id not in seen_parents
        ):
            seen_parents.add(chunk.parent_id)
            queries.append(
                {
                    "query_id": f"q_pol_{len(queries) + 1:02d}",
                    "namespace": "policy",
                    "query": f"Marketplace compliance rule regarding {chunk.title}",
                    "filters": {},
                    "gold_evidence_ids": [chunk.policy_rule_id],
                    "gold_facts": [w.lower() for w in chunk.title.split() if len(w) > 4][:2],
                }
            )
            if len(queries) >= 8:
                break

    return queries


def _compute_ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    dcg = 0.0
    seen: set[str] = set()
    for rank, ev_id in enumerate(retrieved_ids[:k], start=1):
        if ev_id in gold_ids and ev_id not in seen:
            dcg += 1.0 / math.log2(rank + 1)
            seen.add(ev_id)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def run_benchmark(
    k: int = 3,
    output_path: str | Path = "evals/results/rag_comparison.json",
    data_dir: str | Path = "data",
    index_dir: str | Path = "rag/index/local_store",
) -> dict[str, Any]:
    retriever = GigBidRetriever(index_dir=index_dir, data_dir=data_dir)
    eval_queries = _build_eval_queries(retriever)
    summary_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        precisions: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        fact_coverages: list[float] = []
        latencies_ms: list[float] = []

        for item in eval_queries:
            t0 = time.perf_counter()
            res = retriever.retrieve(
                query=item["query"],
                k=k,
                namespace=item["namespace"],
                variant=variant,
                filters=item.get("filters") or {},
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            retrieved_ids = [c.evidence_id or c.chunk_id for c in res.chunks]
            gold_set = set(item["gold_evidence_ids"])

            unique_hits = len(set(retrieved_ids[:k]) & gold_set)
            precisions.append(unique_hits / max(1, k))
            recalls.append(unique_hits / max(1, len(gold_set)))

            rr = 0.0
            for rank, eid in enumerate(retrieved_ids[:k], start=1):
                if eid in gold_set:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)
            ndcgs.append(_compute_ndcg_at_k(retrieved_ids, gold_set, k))

            combined_text = " ".join(c.content.lower() for c in res.chunks)
            gold_facts = item.get("gold_facts") or []
            if gold_facts:
                covered = sum(1 for fact in gold_facts if fact.lower() in combined_text)
                fact_coverages.append(covered / len(gold_facts))
            else:
                fact_coverages.append(1.0)

        sorted_lat = sorted(latencies_ms)
        p50 = statistics.median(sorted_lat)
        p95_idx = min(len(sorted_lat) - 1, math.ceil(0.95 * len(sorted_lat)) - 1)
        p95 = sorted_lat[p95_idx]

        summary_rows.append(
            {
                "variant": variant,
                f"precision_at_{k}": round(statistics.mean(precisions), 4),
                f"recall_at_{k}": round(statistics.mean(recalls), 4),
                "mrr": round(statistics.mean(mrrs), 4),
                f"ndcg_at_{k}": round(statistics.mean(ndcgs), 4),
                "grounding_fact_coverage": round(statistics.mean(fact_coverages), 4),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
            }
        )

    best_variant = max(
        summary_rows,
        key=lambda r: (r["mrr"], r[f"ndcg_at_{k}"], r["grounding_fact_coverage"]),
    )["variant"]

    report = {
        "benchmark": "GigBid Advanced RAG Variants (V0-V4)",
        "generated_at": datetime.now(UTC).isoformat(),
        "k": k,
        "num_queries": len(eval_queries),
        "embedding_model": retriever.store.embedding_model,
        "embedding_backend": retriever.store.embedder.backend_mode,
        "reranker_model": retriever.store.reranker_model,
        "reranker_backend": retriever.store.reranker.backend_mode,
        "recommended_variant": best_variant,
        "results": summary_rows,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_benchmark(k=3)
    print(f"Benchmark complete | Recommended variant: {report['recommended_variant']}")
    header = f"{'Variant':<8} {'P@3':<8} {'R@3':<8} {'MRR':<8} {'nDCG@3':<8} {'FactCov':<9} {'p50(ms)':<9} {'p95(ms)':<9}"
    print(header)
    print("-" * len(header))
    for row in report["results"]:
        print(
            f"{row['variant']:<8} "
            f"{row['precision_at_3']:<8.4f} "
            f"{row['recall_at_3']:<8.4f} "
            f"{row['mrr']:<8.4f} "
            f"{row['ndcg_at_3']:<8.4f} "
            f"{row['grounding_fact_coverage']:<9.4f} "
            f"{row['p50_latency_ms']:<9.2f} "
            f"{row['p95_latency_ms']:<9.2f}"
        )


if __name__ == "__main__":
    main()
