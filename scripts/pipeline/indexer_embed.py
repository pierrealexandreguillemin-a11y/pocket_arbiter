"""Embedding operations: model loading, encoding, CCH formatting, blob serialization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/embeddinggemma-300m-qat-q4_0-unquantized"
EMBEDDING_DIM = 768
BATCH_SIZE = 128


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Convert float32 numpy array to bytes for SQLite BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Convert SQLite BLOB back to float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


def load_model(model_id: str = DEFAULT_MODEL_ID) -> SentenceTransformer:
    """Load sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading model %s", model_id)
    model = SentenceTransformer(model_id)
    return model


def make_cch_title(
    source: str,
    section: str,
    source_titles: dict[str, str],
) -> str:
    """Build CCH title: 'Document Title > Section'.

    Args:
        source: PDF filename.
        section: Section heading or hierarchy.
        source_titles: Mapping source filename → display title.
    """
    doc_title = source_titles.get(source, source.replace(".pdf", ""))
    if section:
        return f"{doc_title} > {section}"
    return doc_title


def format_document(text: str, cch_title: str) -> str:
    """Format document with EmbeddingGemma prompt: 'title: {cch} | text: {text}'."""
    return f"title: {cch_title} | text: {text}"


def format_query(query: str) -> str:
    """Format query with EmbeddingGemma prompt: 'task: search result | query: {q}'."""
    return f"task: search result | query: {query}"


def embed_documents(
    texts: list[str],
    cch_titles: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Embed documents with CCH prefix, L2 normalized."""
    formatted = [format_document(t, c) for t, c in zip(texts, cch_titles, strict=True)]
    embeddings = model.encode(
        formatted,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


def embed_queries(
    queries: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Embed queries with search prompt, L2 normalized."""
    formatted = [format_query(q) for q in queries]
    embeddings = model.encode(
        formatted,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
    )
    return embeddings.astype(np.float32)
