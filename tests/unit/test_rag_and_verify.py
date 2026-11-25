"""Unit and integration tests for GigBid Phase 3 Advanced RAG & Claim Verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.index.hybrid_store import HybridIndexStore
from rag.ingest.pipeline import IngestionPipeline
from rag.retriever import EvidenceChunk, GigBidRetriever
from rag.verify import ClaimSplitter, ClaimVerifier


@pytest.fixture()
def built_store(tmp_path: Path) -> HybridIndexStore:
    """Build a fresh HybridIndexStore in a temporary directory."""
    index_dir = tmp_path / "local_store"
    store = HybridIndexStore(index_dir=index_dir)
    meta = store.build_from_data(data_dir="data")
    assert meta["total_chunks"] > 0
    return store


def test_ingestion_pipeline_structure_and_provenance() -> None:
    pipeline = IngestionPipeline(data_dir="data")
    result = pipeline.run()

    assert result.total_chunks >= 8
    assert result.portfolio_chunks >= 4
    assert result.policy_chunks >= 3
    assert result.parent_documents >= 4

    for chunk in result.chunks:
        assert chunk.data_provenance in {"kaggle_real", "synthetic"}
        assert chunk.parent_id
        assert len(chunk.parent_text) >= len(chunk.text)
        if chunk.namespace == "policy":
            assert chunk.policy_rule_id is not None


def test_hybrid_index_persistence_and_reload(tmp_path: Path) -> None:
    index_dir = tmp_path / "persisted_store"
    store1 = HybridIndexStore(index_dir=index_dir)
    store1.build_from_data(data_dir="data")

    assert (index_dir / "vectors.npy").exists()
    assert (index_dir / "chunks.jsonl").exists()
    assert (index_dir / "bm25_corpus.json").exists()
    assert (index_dir / "index_meta.json").exists()

    store2 = HybridIndexStore(index_dir=index_dir)
    assert store2.load() is True
    assert len(store2.chunks) == len(store1.chunks)
    assert store2.vectors.shape == store1.vectors.shape


@pytest.mark.parametrize("variant", ["v0", "v1", "v2", "v3", "v4"])
def test_all_rag_variants_return_valid_evidence_chunks(
    built_store: HybridIndexStore,
    variant: str,
) -> None:
    sample_chunk = next(
        c
        for c in built_store.chunks
        if c.namespace == "portfolio" and c.doc_type != "freelancer_profile"
    )
    retriever = GigBidRetriever(index_dir=built_store.index_dir, default_variant=variant)  # type: ignore[arg-type]
    res = retriever.retrieve(
        query=f"{sample_chunk.title} {' '.join(sample_chunk.skills[:3])}",
        k=3,
        namespace="portfolio",
        filters={
            "freelancer_id": sample_chunk.freelancer_id,
            "required_skills": sample_chunk.skills[:2],
        },
    )

    assert res.variant_id == variant
    assert res.namespace == "portfolio"
    assert len(res.chunks) >= 1
    assert res.latency_ms >= 0.0

    for chunk in res.chunks:
        assert isinstance(chunk, EvidenceChunk)
        assert chunk.namespace == "portfolio"
        assert chunk.data_provenance in {"kaggle_real", "synthetic"}
        assert chunk.content.strip()


def test_v4_hierarchical_policy_retrieval_returns_complete_rule_context(
    built_store: HybridIndexStore,
) -> None:
    retriever = GigBidRetriever(index_dir=built_store.index_dir, default_variant="v4")
    policy_chunks = retriever.retrieve_policy(
        query="Can I share my personal email or WhatsApp or accept PayPal outside the platform?",
        k=2,
    )

    assert len(policy_chunks) >= 1
    top_rule = policy_chunks[0]
    assert top_rule.namespace == "policy"
    assert top_rule.policy_rule_id is not None
    assert top_rule.content == top_rule.parent_content
    assert any(
        kw in top_rule.content.lower()
        for kw in (
            "off-platform",
            "contact",
            "escrow",
            "payment",
            "paypal",
            "email",
            "circumvention",
        )
    )


def test_claim_verifier_passes_grounded_proposal_and_catches_hallucinations(
    built_store: HybridIndexStore,
) -> None:
    target_chunk = next(
        c
        for c in built_store.chunks
        if c.namespace == "portfolio" and c.doc_type != "freelancer_profile"
    )
    retriever = GigBidRetriever(index_dir=built_store.index_dir, default_variant="v4")
    evidence = retriever.retrieve_portfolio(
        query=f"{target_chunk.title} {' '.join(target_chunk.skills)}",
        k=4,
        freelancer_id=target_chunk.freelancer_id,
    )
    assert len(evidence) >= 1
    top_ev = evidence[0]
    top_skill = top_ev.skills[0] if top_ev.skills else "Python"

    freelancer_profile = {
        "freelancer_id": target_chunk.freelancer_id or "freelancer_01",
        "name": "Verified Engineer",
        "headline": "Senior Software & AI Systems Engineer",
        "bio": "6 years building production backend and AI systems.",
        "years_experience": 6,
        "skills": list({*top_ev.skills, "Python", "FastAPI", "PostgreSQL"}),
    }

    verifier = ClaimVerifier()

    # 1. Grounded proposal citing the actual retrieved evidence ID and verified skills
    grounded_letter = (
        "Hello, I reviewed your project requirements carefully. "
        f"In my project {top_ev.title} ({top_ev.evidence_id}), I delivered production work using {top_skill}. "
        f"I can apply this {top_skill} architecture directly to your workflow."
    )
    grounded_report = verifier.verify(
        cover_letter=grounded_letter,
        evidence_chunks=evidence,
        freelancer_profile=freelancer_profile,
    )
    assert grounded_report.contradicted_count == 0
    assert grounded_report.unsupported_count == 0
    assert grounded_report.grounding_score == 1.0
    assert grounded_report.passed is True

    # 2. Hallucinated proposal injecting fake 99.8% metric, fake 15 years experience, and unowned Rust/CUDA
    hallucinated_letter = (
        "Hello, I am excited to apply for your project. "
        "I have 15 years of experience architecting quantum CUDA kernels in Rust for NASA. "
        "My previous system reduced server costs by 99.8% across 50 million daily users."
    )
    bad_report = verifier.verify(
        cover_letter=hallucinated_letter,
        evidence_chunks=evidence,
        freelancer_profile=freelancer_profile,
    )
    assert bad_report.passed is False
    assert (bad_report.unsupported_count + bad_report.contradicted_count) >= 2
    assert bad_report.grounding_score < 0.5
    assert len(bad_report.unsupported_claims) >= 2

    # 3. Sanitize removes hallucinated sentences while keeping benign greeting
    sanitized = verifier.sanitize_proposal(hallucinated_letter, bad_report)
    assert "99.8%" not in sanitized
    assert "NASA" not in sanitized
    assert "Hello" in sanitized


def test_claim_splitter_distinguishes_generic_vs_metric_claims() -> None:
    splitter = ClaimSplitter()
    claims = splitter.split(
        cover_letter=(
            "Hi there, thank you for posting this role. "
            "We cut p95 ingestion latency from 45s to 2.8s using FastAPI and Celery. "
            "Looking forward to your reply."
        )
    )
    assert len(claims) == 3
    assert claims[0].claim_type == "GENERIC"
    assert claims[1].claim_type == "METRIC"
    assert "45s" in claims[1].extracted_numbers or "2.8s" in claims[1].extracted_numbers
    assert claims[2].claim_type == "GENERIC"
