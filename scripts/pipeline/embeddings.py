"""
Generation d'embeddings - Pocket Arbiter

Ce module genere des embeddings 768D avec EmbeddingGemma-300M-QAT-Q4
pour le retrieval semantique RAG.

ISO Reference:
    - ISO/IEC 12207 S7.3.3 - Implementation
    - ISO/IEC 25010 S4.2 - Performance efficiency (FA-01: Recall >= 80%)
    - ISO/IEC 42001 A.6.2.2 - Documentation modeles

Usage:
    python embeddings.py -i corpus/processed/chunks_fr.json -o corpus/processed/embeddings_fr.npy
"""

import argparse
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scripts.pipeline.embeddings_config import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIM,
    FALLBACK_EMBEDDING_DIM,
    FALLBACK_MODEL_ID,
    MODEL_ID,
    MODEL_ID_FULL,
    MRL_DIM_BALANCED,
    MRL_DIM_FAST,
    MRL_DIM_FULL,
    MRL_DIMS,
    PROMPT_CLASSIFICATION,
    PROMPT_CLUSTERING,
    PROMPT_DOCUMENT,
    PROMPT_DOCUMENT_WITH_TITLE,
    PROMPT_QA,
    PROMPT_QUERY,
    PROMPT_SIMILARITY,
    is_embeddinggemma_model,
    measure_performance,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "load_embedding_model",
    "embed_texts",
    "embed_query",
    "embed_documents",
    "embed_chunks",
    "generate_query_embedding",
    "generate_corpus_embeddings",
    "measure_performance",
    "is_embeddinggemma_model",
    "MODEL_ID",
    "MODEL_ID_FULL",
    "FALLBACK_MODEL_ID",
    "DEFAULT_BATCH_SIZE",
    "EMBEDDING_DIM",
    "FALLBACK_EMBEDDING_DIM",
    "MRL_DIMS",
    "MRL_DIM_FULL",
    "MRL_DIM_BALANCED",
    "MRL_DIM_FAST",
    "PROMPT_QUERY",
    "PROMPT_DOCUMENT",
    "PROMPT_DOCUMENT_WITH_TITLE",
    "PROMPT_QA",
    "PROMPT_CLASSIFICATION",
    "PROMPT_CLUSTERING",
    "PROMPT_SIMILARITY",
]


# --- Main Functions ---


def load_embedding_model(
    model_id: str = MODEL_ID,
    device: str | None = None,
    truncate_dim: int | None = None,
) -> "SentenceTransformer":
    """
    Charge le modele EmbeddingGemma-300M via sentence-transformers.

    Args:
        model_id: Identifiant HuggingFace du modele.
        device: Device cible (cpu/cuda). Auto-detect si None.
        truncate_dim: Dimension reduite MRL (256/512 pour gain memoire).

    Returns:
        Modele SentenceTransformer charge.

    Raises:
        RuntimeError: Si aucun modele ne peut etre charge.
    """
    from sentence_transformers import SentenceTransformer

    try:
        logger.info(f"Loading model: {model_id}")
        model = SentenceTransformer(
            model_id,
            device=device,
            truncate_dim=truncate_dim,
        )
        dim = model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded: {model_id} ({dim}D)")
        return model

    except Exception as e:
        logger.warning(f"Failed to load {model_id}: {e}")

        # Cascade de fallback: QAT -> Full -> E5
        if model_id == MODEL_ID:
            logger.info(f"Trying fallback: {MODEL_ID_FULL}")
            return load_embedding_model(MODEL_ID_FULL, device, truncate_dim)
        elif model_id == MODEL_ID_FULL:
            logger.info(f"Trying fallback: {FALLBACK_MODEL_ID}")
            return load_embedding_model(FALLBACK_MODEL_ID, device, truncate_dim)

        raise RuntimeError(f"Cannot load any embedding model: {e}") from e


def embed_texts(
    texts: list[str],
    model: "SentenceTransformer",
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Genere des embeddings pour une liste de textes.

    Args:
        texts: Liste des textes a encoder.
        model: Modele SentenceTransformer charge.
        batch_size: Taille de batch pour l'inference.
        show_progress: Afficher barre de progression.
        normalize: Normaliser les vecteurs (L2 norm = 1).

    Returns:
        np.ndarray de shape (N, dim) avec les embeddings.

    Raises:
        ValueError: Si texts est vide.
    """
    if not texts:
        raise ValueError("texts list cannot be empty")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    return np.array(embeddings)


def embed_query(
    query: str,
    model: "SentenceTransformer",
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode une requete utilisateur avec le prompt officiel Google.

    Args:
        query: Texte de la requete utilisateur.
        model: Modele SentenceTransformer charge.
        normalize: Normaliser le vecteur (L2 norm = 1).

    Returns:
        np.ndarray de shape (dim,) avec l'embedding.
    """
    # EmbeddingGemma: utiliser methode officielle encode_query
    if hasattr(model, "encode_query"):
        embedding = model.encode_query(
            query,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return np.array(embedding)

    # Fallback pour autres modeles: encode standard
    embedding = model.encode(
        [query],
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return np.array(embedding[0])


def embed_documents(
    documents: list[str],
    model: "SentenceTransformer",
    titles: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode des documents avec le prompt officiel Google.

    Args:
        documents: Liste des documents/chunks a encoder.
        model: Modele SentenceTransformer charge.
        titles: Liste optionnelle de titres (section/chapitre) pour chaque document.
                Si fourni, utilise "title: {title} | text: " au lieu de "title: none".
                Améliore la relevance de ~4% selon Google.
        batch_size: Taille de batch pour l'inference.
        show_progress: Afficher barre de progression.
        normalize: Normaliser les vecteurs (L2 norm = 1).

    Returns:
        np.ndarray de shape (N, dim) avec les embeddings.

    Raises:
        ValueError: Si documents est vide ou titles de mauvaise taille.
    """
    if not documents:
        raise ValueError("documents list cannot be empty")

    if titles is not None and len(titles) != len(documents):
        raise ValueError(
            f"titles length ({len(titles)}) != documents length ({len(documents)})"
        )

    # Si titles fournis, utiliser prompts manuels avec titles (meilleure relevance +4%)
    # Source: ai.google.dev/gemma/docs/embeddinggemma - "include titles when available"
    if titles is not None and is_embeddinggemma_model(
        getattr(model, "model_card_data", {}).get("base_model", MODEL_ID)
    ):
        docs_with_prompts = []
        for doc, title in zip(documents, titles):
            if title:
                prompt = PROMPT_DOCUMENT_WITH_TITLE.format(title=title)
            else:
                prompt = PROMPT_DOCUMENT  # Défaut Google: "title: none"
            docs_with_prompts.append(f"{prompt}{doc}")

        embeddings = model.encode(
            docs_with_prompts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return np.array(embeddings)

    # EmbeddingGemma sans titles: utiliser methode officielle encode_document
    if hasattr(model, "encode_document"):
        embeddings = model.encode_document(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return np.array(embeddings)

    # Fallback pour autres modeles: encode standard
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return np.array(embeddings)


# Module-level cache for query embedding model
_query_model_cache: dict[str, "SentenceTransformer"] = {}


def generate_query_embedding(
    query: str,
    model_id: str | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Genere un embedding pour une requete utilisateur.

    Utilise un cache module-level pour eviter de recharger le modele.

    Args:
        query: Texte de la requete.
        model_id: ID du modele (utilise MODEL_ID par defaut).
        normalize: Normaliser le vecteur (L2 norm = 1).

    Returns:
        np.ndarray de shape (dim,) avec l'embedding.
    """
    global _query_model_cache

    if model_id is None:
        model_id = MODEL_ID

    if model_id not in _query_model_cache:
        _query_model_cache[model_id] = load_embedding_model(model_id)

    model = _query_model_cache[model_id]
    return embed_query(query, model, normalize=normalize)


def _extract_chunk_title(chunk: dict) -> str:
    """
    Extrait un titre riche depuis un chunk pour l'embedding.

    Combine les champs disponibles pour maximiser la relevance (~4% boost):
    - section: "CHAPITRE Ier", "Titre VII", "Article 4.1"
    - article_num: numero d'article si disponible
    - source: nom du fichier PDF (fallback)
    - table_type: pour les tables summaries

    Examples:
        - "CHAPITRE Ier - Article 3" (section + article)
        - "Titre VII" (section only)
        - "Table LA-octobre2025" (table summary)
        - "Statuts FFE" (source filename cleaned)
    """
    section = (chunk.get("section") or "").strip()
    article_num = chunk.get("article_num")
    source = (chunk.get("source") or "").replace(".pdf", "").strip()
    table_type = chunk.get("table_type")

    # Table summaries: "Table {source}"
    if table_type:
        return f"Table {source}" if source else "Table"

    # Section + article: "CHAPITRE Ier - Article 3"
    if section and article_num:
        return f"{section} - Article {article_num}"

    # Section only
    if section:
        return section

    # Fallback: source filename (cleaned)
    return source


def embed_chunks(
    chunks: list[dict],
    model: "SentenceTransformer",
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_titles: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Genere des embeddings pour des chunks du pipeline.

    Args:
        chunks: Liste de chunks conformes a CHUNK_SCHEMA.md.
        model: Modele SentenceTransformer charge.
        batch_size: Taille de batch.
        use_titles: Extraire et utiliser titres riches dans prompt.
                    Combine section/article/source pour ~4% relevance boost.

    Returns:
        Tuple (embeddings array shape (N, dim), list of chunk IDs).

    Raises:
        ValueError: Si chunks est vide ou contient des chunks invalides.
    """
    if not chunks:
        raise ValueError("chunks list cannot be empty")

    texts = []
    titles = []
    ids = []

    for chunk in chunks:
        if "text" not in chunk or "id" not in chunk:
            raise ValueError("Invalid chunk format: missing 'text' or 'id'")
        texts.append(chunk["text"])
        ids.append(chunk["id"])
        if use_titles:
            titles.append(_extract_chunk_title(chunk))

    # Passer titles si use_titles activé (améliore relevance ~4%)
    embeddings = embed_documents(
        texts,
        model,
        titles=titles if use_titles else None,
        batch_size=batch_size,
    )

    return embeddings, ids


def generate_corpus_embeddings(
    input_file: Path,
    output_file: Path,
    model_id: str = MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    truncate_dim: int | None = None,
) -> dict:
    """
    Pipeline complet: charge chunks, genere embeddings, sauvegarde.

    Args:
        input_file: Fichier chunks JSON (chunks_fr.json).
        output_file: Fichier numpy de sortie (embeddings_fr.npy).
        model_id: ID du modele HuggingFace.
        batch_size: Taille de batch.
        truncate_dim: Dimension MRL reduite (optionnel).

    Returns:
        Rapport avec total_chunks, embedding_dim, time_s, output_file.

    Raises:
        FileNotFoundError: Si input_file n'existe pas.
    """
    from scripts.pipeline.utils import get_timestamp, load_json, save_json

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info(f"Loading chunks from: {input_file}")
    data = load_json(input_file)
    chunks = data.get("chunks", [])

    if not chunks:
        raise ValueError(f"No chunks found in {input_file}")

    logger.info(f"Found {len(chunks)} chunks")

    model = load_embedding_model(model_id, truncate_dim=truncate_dim)
    embedding_dim = model.get_sentence_embedding_dimension()

    logger.info(f"Generating {embedding_dim}D embeddings...")
    start_time = time.perf_counter()

    embeddings, ids = embed_chunks(chunks, model, batch_size)

    elapsed = time.perf_counter() - start_time
    logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.1f}s")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)
    logger.info(f"Saved embeddings: {output_file}")

    ids_file = output_file.with_suffix(".ids.json")
    save_json({"chunk_ids": ids, "total": len(ids), "model_id": model_id}, ids_file)

    report = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "model_id": model_id,
        "total_chunks": len(chunks),
        "embedding_dim": embedding_dim,
        "embeddings_shape": list(embeddings.shape),
        "time_seconds": round(elapsed, 2),
        "ms_per_chunk": round((elapsed / len(chunks)) * 1000, 2),
        "timestamp": get_timestamp(),
    }

    report_file = output_file.with_suffix(".report.json")
    save_json(report, report_file)

    return report


# --- CLI ---


def main() -> None:
    """Point d'entree CLI pour la generation d'embeddings."""
    parser = argparse.ArgumentParser(
        description="Generation d'embeddings pour Pocket Arbiter",
    )

    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Fichier chunks JSON"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Fichier numpy de sortie"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=MODEL_ID,
        help=f"ID modele (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE, help="Taille batch"
    )
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=None,
        choices=MRL_DIMS,
        help=f"Dimension MRL (Matryoshka): {MRL_DIMS}. 256 recommandé mobile, 768 default",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Logs detailles")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {args.model}")

    report = generate_corpus_embeddings(
        args.input, args.output, args.model, args.batch_size, args.truncate_dim
    )

    logger.info("=" * 50)
    logger.info(f"Chunks: {report['total_chunks']}")
    logger.info(f"Dimension: {report['embedding_dim']}D")
    logger.info(f"Time: {report['time_seconds']}s ({report['ms_per_chunk']}ms/chunk)")


if __name__ == "__main__":
    main()
