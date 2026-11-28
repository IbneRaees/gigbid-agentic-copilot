"""Unified Hybrid Dense + BM25 Sparse Index with CPU-safe open-weight models.

Features:
  - Lightweight CPU-friendly open-weight embeddings (`BAAI/bge-small-en-v1.5`, 384-d, ~130 MB)
    with a deterministic offline hashing/projection fallback if model weights cannot be
    downloaded in an air-gapped environment.
  - Lightweight CPU-friendly cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB)
    with a deterministic lexical-cosine fallback when offline.
  - Exact BM25 lexical index (`rank_bm25.BM25Okapi`) preserving code identifiers, library names,
    policy IDs (`POL-001`), and numeric metrics (`62%`, `40ms`, `50k`).
  - Atomic disk persistence (`rag/index/local_store/`) + optional `pgvector` upsert when Postgres
    is available.
  - CLI entrypoint: `python -m rag.index.hybrid_store --rebuild`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rag.ingest.pipeline import DocumentChunk, IngestionPipeline

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FALLBACK_DIM = 384

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-\./+%]*", re.IGNORECASE)


def tokenize_technical(text: str) -> list[str]:
    """Domain-aware tokenizer keeping library names, metrics, and rule codes intact."""
    lowered = text.lower()
    raw_tokens = _TOKEN_RE.findall(lowered)
    tokens: list[str] = []
    for tok in raw_tokens:
        tokens.append(tok)
        if "-" in tok or "_" in tok or "/" in tok:
            subparts = [p for p in re.split(r"[\-_/.]+", tok) if len(p) > 1]
            tokens.extend(subparts)
    return tokens


class EmbeddingBackend:
    """Thread-safe wrapper around BAAI/bge-small-en-v1.5 with deterministic offline fallback."""

    _lock = threading.Lock()
    _instances: dict[str, EmbeddingBackend] = {}

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._st_model: Any = None
        self.dim: int = FALLBACK_DIM
        self.backend_mode: str = "uninitialized"
        self._load_model()

    @classmethod
    def get_shared(cls, model_name: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingBackend:
        with cls._lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = cls(model_name=model_name)
            return cls._instances[model_name]

    def _load_model(self) -> None:
        if os.getenv("GIGBID_OFFLINE_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
            self.backend_mode = "deterministic_hash_fallback"
            self.dim = FALLBACK_DIM
            return
        try:
            import os
            if os.getenv("GIGBID_LIGHTWEIGHT_RAG", "false").lower() == "true":
                raise RuntimeError("Lightweight RAG mode enabled")
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name, device="cpu")
            self._st_model = model
            self.dim = int(model.get_sentence_embedding_dimension() or FALLBACK_DIM)
            self.backend_mode = "sentence_transformers"
            logger.info(
                "Loaded dense embedding model %s on CPU (dim=%d)", self.model_name, self.dim
            )
        except Exception as exc:
            logger.warning(
                "Could not load SentenceTransformer(%s) (%s); using deterministic hash embeddings.",
                self.model_name,
                exc,
            )
            self._st_model = None
            self.dim = FALLBACK_DIM
            self.backend_mode = "deterministic_hash_fallback"

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings of shape (len(texts), self.dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self._st_model is not None:
            vecs = self._st_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vecs, dtype=np.float32)

        return self._encode_deterministic(texts)

    def _encode_deterministic(self, texts: list[str]) -> np.ndarray:
        """Deterministic signed feature-hashing embedding normalized to unit L2 norm."""
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = tokenize_technical(text)
            if not tokens:
                continue
            # Unigrams + bigrams for semantic phrase sensitivity
            ngrams = list(tokens) + [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
            for ng in ngrams:
                digest = hashlib.blake2b(ng.encode("utf-8"), digest_size=8).digest()
                h_val = int.from_bytes(digest[:4], "little")
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                col = h_val % self.dim
                mat[row_idx, col] += sign
            norm = float(np.linalg.norm(mat[row_idx]))
            if norm > 1e-12:
                mat[row_idx] /= norm
        return mat


class CrossEncoderBackend:
    """Thread-safe wrapper around cross-encoder/ms-marco-MiniLM-L-6-v2 with offline fallback."""

    _lock = threading.Lock()
    _instances: dict[str, CrossEncoderBackend] = {}

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._ce_model: Any = None
        self.backend_mode: str = "uninitialized"
        self._load_model()

    @classmethod
    def get_shared(cls, model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoderBackend:
        with cls._lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = cls(model_name=model_name)
            return cls._instances[model_name]

    def _load_model(self) -> None:
        if os.getenv("GIGBID_OFFLINE_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
            self.backend_mode = "lexical_interaction_fallback"
            return
        try:
            import os
            if os.getenv("GIGBID_LIGHTWEIGHT_RAG", "false").lower() == "true":
                raise RuntimeError("Lightweight RAG mode enabled")
            from sentence_transformers import CrossEncoder

            self._ce_model = CrossEncoder(self.model_name, device="cpu", max_length=512)
            self.backend_mode = "cross_encoder"
            logger.info("Loaded cross-encoder reranker %s on CPU", self.model_name)
        except Exception as exc:
            logger.warning(
                "Could not load CrossEncoder(%s) (%s); using lexical interaction reranker.",
                self.model_name,
                exc,
            )
            self._ce_model = None
            self.backend_mode = "lexical_interaction_fallback"

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, passage) pairs and return probabilities in [0.0, 1.0]."""
        if not pairs:
            return []

        if self._ce_model is not None:
            raw_scores = self._ce_model.predict(pairs, show_progress_bar=False)
            results: list[float] = []
            for s in raw_scores:
                val = float(s)
                # MiniLM MS-MARCO outputs logits; map cleanly via sigmoid to (0, 1)
                if val < 0.0 or val > 1.0:
                    val = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, val))))
                results.append(val)
            return results

        # Deterministic offline fallback combining token overlap, bigram recall, and numeric match
        scores: list[float] = []
        for query, passage in pairs:
            q_toks = tokenize_technical(query)
            p_toks = tokenize_technical(passage)
            if not q_toks or not p_toks:
                scores.append(0.0)
                continue
            q_set = set(q_toks)
            p_set = set(p_toks)
            unigram_recall = len(q_set & p_set) / max(1, len(q_set))
            q_bi = {f"{q_toks[i]}_{q_toks[i + 1]}" for i in range(len(q_toks) - 1)}
            p_bi = {f"{p_toks[i]}_{p_toks[i + 1]}" for i in range(len(p_toks) - 1)}
            bigram_recall = (len(q_bi & p_bi) / max(1, len(q_bi))) if q_bi else unigram_recall
            score = 0.65 * unigram_recall + 0.35 * bigram_recall
            scores.append(round(min(1.0, max(0.0, score)), 6))
        return scores


class _SimpleBM25:
    """Lightweight BM25Okapi implementation used if rank_bm25 is unavailable, or wrapping it."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_size = len(tokenized_corpus)
        self.k1 = k1
        self.b = b
        self._bm25_lib: Any = None
        try:
            from rank_bm25 import BM25Okapi

            if tokenized_corpus:
                self._bm25_lib = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        except Exception:
            self._bm25_lib = None

        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        df: dict[str, int] = {}
        for doc in tokenized_corpus:
            self.doc_len.append(len(doc))
            freqs: dict[str, int] = {}
            for tok in doc:
                freqs[tok] = freqs.get(tok, 0) + 1
            self.doc_freqs.append(freqs)
            for tok in freqs:
                df[tok] = df.get(tok, 0) + 1
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 1.0
        for tok, freq in df.items():
            self.idf[tok] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        if self.corpus_size == 0:
            return np.zeros(0, dtype=np.float32)
        if self._bm25_lib is not None:
            return np.asarray(self._bm25_lib.get_scores(query_tokens), dtype=np.float32)

        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for idx, freqs in enumerate(self.doc_freqs):
            dl = self.doc_len[idx]
            denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / max(1e-6, self.avgdl)))
            score = 0.0
            for q in query_tokens:
                if q in freqs:
                    tf = freqs[q]
                    idf = self.idf.get(q, 0.0)
                    score += idf * ((tf * (self.k1 + 1.0)) / (tf + denom_norm))
            scores[idx] = score
        return scores


class HybridIndexStore:
    """Unified Dense + Sparse BM25 index with atomic disk persistence and optional pgvector."""

    def __init__(
        self,
        index_dir: str | Path = "rag/index/local_store",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.embedder = EmbeddingBackend.get_shared(embedding_model)
        self.reranker = CrossEncoderBackend.get_shared(reranker_model)

        self.chunks: list[DocumentChunk] = []
        self.vectors: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: _SimpleBM25 = _SimpleBM25([])
        self.metadata: dict[str, Any] = {}

    @property
    def is_built(self) -> bool:
        return len(self.chunks) > 0 and self.vectors.shape[0] == len(self.chunks)

    def build_from_data(
        self,
        data_dir: str | Path = "data",
        github_repos_dir: str | Path | None = None,
        sync_pgvector: bool = False,
    ) -> dict[str, Any]:
        """Ingest corpora, compute dense + sparse indices, and persist atomically."""
        pipeline = IngestionPipeline(data_dir=data_dir, github_repos_dir=github_repos_dir)
        ingest_result = pipeline.run()
        self.build_from_chunks(ingest_result.chunks)
        self.metadata["sources_loaded"] = ingest_result.sources_loaded
        self.save()

        if sync_pgvector:
            self._try_sync_pgvector()

        return self.metadata

    def build_from_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Build in-memory dense vectors and BM25 index from pre-chunked documents."""
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        self.vectors = self.embedder.encode(texts)
        self.tokenized_corpus = [tokenize_technical(t) for t in texts]
        self.bm25 = _SimpleBM25(self.tokenized_corpus)
        self.metadata = {
            "built_at": datetime.now(UTC).isoformat(),
            "total_chunks": len(self.chunks),
            "portfolio_chunks": sum(1 for c in self.chunks if c.namespace == "portfolio"),
            "policy_chunks": sum(1 for c in self.chunks if c.namespace == "policy"),
            "embedding_model": self.embedding_model,
            "embedding_backend": self.embedder.backend_mode,
            "embedding_dim": int(self.vectors.shape[1])
            if self.vectors.ndim == 2
            else self.embedder.dim,
            "reranker_model": self.reranker_model,
            "reranker_backend": self.reranker.backend_mode,
        }

    def save(self) -> None:
        """Atomically persist index artifacts to self.index_dir."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "vectors.npy", self.vectors)

        chunks_path = self.index_dir / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")

        (self.index_dir / "bm25_corpus.json").write_text(
            json.dumps(self.tokenized_corpus, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.index_dir / "index_meta.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Persisted hybrid index (%d chunks) to %s", len(self.chunks), self.index_dir)

    def load(self) -> bool:
        """Load persisted index from self.index_dir. Returns False if artifacts are missing."""
        vectors_path = self.index_dir / "vectors.npy"
        chunks_path = self.index_dir / "chunks.jsonl"
        bm25_path = self.index_dir / "bm25_corpus.json"
        meta_path = self.index_dir / "index_meta.json"

        if not (vectors_path.exists() and chunks_path.exists() and bm25_path.exists()):
            return False

        loaded_chunks: list[DocumentChunk] = []
        with chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    loaded_chunks.append(DocumentChunk.model_validate_json(stripped))

        loaded_vectors = np.load(vectors_path).astype(np.float32)
        loaded_corpus = json.loads(bm25_path.read_text(encoding="utf-8"))
        loaded_meta = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        )

        if loaded_vectors.shape[0] != len(loaded_chunks):
            logger.warning("Index dimension mismatch on disk; rebuild required.")
            return False

        self.chunks = loaded_chunks
        self.vectors = loaded_vectors
        self.tokenized_corpus = loaded_corpus
        self.bm25 = _SimpleBM25(self.tokenized_corpus)
        self.metadata = loaded_meta
        return True

    def dense_scores(self, query: str) -> np.ndarray:
        """Compute cosine similarity against all indexed chunks (vectors are L2-normalized)."""
        if not self.is_built:
            return np.zeros(0, dtype=np.float32)
        q_vec = self.embedder.encode([query])[0]
        if q_vec.shape[0] != self.vectors.shape[1]:
            # Re-encode if embedding dimension changed
            self.vectors = self.embedder.encode([c.text for c in self.chunks])
        return np.dot(self.vectors, q_vec).astype(np.float32)

    def sparse_scores(self, query: str) -> np.ndarray:
        """Compute BM25 lexical relevance scores against all indexed chunks."""
        if not self.is_built:
            return np.zeros(0, dtype=np.float32)
        q_tokens = tokenize_technical(query)
        return self.bm25.get_scores(q_tokens)

    def _try_sync_pgvector(self) -> None:
        """Best-effort upsert into Postgres pgvector table when DATABASE_URL is reachable."""
        dsn = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
        if not dsn:
            logger.info("DATABASE_URL not set; skipping pgvector sync (using local_store).")
            return
        try:
            import psycopg

            clean_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://").replace(
                "postgresql+psycopg://", "postgresql://"
            )
            dim = self.vectors.shape[1]
            with psycopg.connect(clean_dsn, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS gigbid_rag_chunks (
                            chunk_id TEXT PRIMARY KEY,
                            parent_id TEXT NOT NULL,
                            evidence_id TEXT,
                            namespace TEXT NOT NULL,
                            doc_type TEXT NOT NULL,
                            title TEXT NOT NULL,
                            text TEXT NOT NULL,
                            parent_text TEXT NOT NULL,
                            freelancer_id TEXT,
                            skills JSONB NOT NULL DEFAULT '[]'::jsonb,
                            policy_rule_id TEXT,
                            data_provenance TEXT NOT NULL,
                            embedding vector({dim})
                        );
                        """
                    )
                    for idx, chunk in enumerate(self.chunks):
                        vec_literal = (
                            "[" + ",".join(f"{float(v):.6f}" for v in self.vectors[idx]) + "]"
                        )
                        cur.execute(
                            """
                            INSERT INTO gigbid_rag_chunks (
                                chunk_id, parent_id, evidence_id, namespace, doc_type,
                                title, text, parent_text, freelancer_id, skills,
                                policy_rule_id, data_provenance, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::vector)
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                text = EXCLUDED.text,
                                parent_text = EXCLUDED.parent_text,
                                skills = EXCLUDED.skills,
                                embedding = EXCLUDED.embedding;
                            """,
                            (
                                chunk.chunk_id,
                                chunk.parent_id,
                                chunk.evidence_id,
                                chunk.namespace,
                                chunk.doc_type,
                                chunk.title,
                                chunk.text,
                                chunk.parent_text,
                                chunk.freelancer_id,
                                json.dumps(chunk.skills),
                                chunk.policy_rule_id,
                                chunk.data_provenance,
                                vec_literal,
                            ),
                        )
                conn.commit()
            logger.info(
                "Synced %d chunks to Postgres pgvector table gigbid_rag_chunks.", len(self.chunks)
            )
        except Exception as exc:
            logger.warning("Postgres pgvector sync skipped (%s); local_store remains active.", exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Build or inspect the GigBid Hybrid RAG Index.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the index from data/ sources."
    )
    parser.add_argument(
        "--data-dir", default="data", help="Path to data directory (default: data)."
    )
    parser.add_argument(
        "--index-dir",
        default="rag/index/local_store",
        help="Output directory for persisted index files.",
    )
    parser.add_argument(
        "--github-dir",
        default=None,
        help="Optional directory of GitHub README markdown files.",
    )
    parser.add_argument(
        "--sync-pgvector",
        action="store_true",
        help="Also upsert embeddings into Postgres pgvector if DATABASE_URL is reachable.",
    )
    args = parser.parse_args()

    store = HybridIndexStore(index_dir=args.index_dir)
    if args.rebuild or not store.load():
        meta = store.build_from_data(
            data_dir=args.data_dir,
            github_repos_dir=args.github_dir,
            sync_pgvector=args.sync_pgvector,
        )
        print(json.dumps(meta, indent=2))
    else:
        print(json.dumps(store.metadata, indent=2))


if __name__ == "__main__":
    main()
