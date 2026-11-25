"""Structure-aware ingestion and chunking for GigBid RAG corpora."""

from rag.ingest.pipeline import (
    DocumentChunk,
    IngestionPipeline,
    IngestResult,
)

__all__ = [
    "DocumentChunk",
    "IngestionPipeline",
    "IngestResult",
]
