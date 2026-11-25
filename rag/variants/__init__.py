"""Configurable RAG retrieval variants (V0 Baseline through V4 Hierarchical Policy)."""

from rag.variants.v0_naive import retrieve_v0_naive
from rag.variants.v1_hybrid_rrf import retrieve_v1_hybrid_rrf
from rag.variants.v2_multiquery_hyde import retrieve_v2_multiquery_hyde
from rag.variants.v3_cross_encoder_rerank import retrieve_v3_cross_encoder_rerank
from rag.variants.v4_hierarchical_policy import retrieve_v4_hierarchical_policy

VARIANT_REGISTRY = {
    "v0": retrieve_v0_naive,
    "v1": retrieve_v1_hybrid_rrf,
    "v2": retrieve_v2_multiquery_hyde,
    "v3": retrieve_v3_cross_encoder_rerank,
    "v4": retrieve_v4_hierarchical_policy,
}

__all__ = [
    "VARIANT_REGISTRY",
    "retrieve_v0_naive",
    "retrieve_v1_hybrid_rrf",
    "retrieve_v2_multiquery_hyde",
    "retrieve_v3_cross_encoder_rerank",
    "retrieve_v4_hierarchical_policy",
]
