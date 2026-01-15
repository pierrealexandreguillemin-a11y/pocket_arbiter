"""
Generation d'embeddings - Pocket Arbiter

Ce module genere des embeddings 768D avec EmbeddingGemma-300M-QAT-Q4
pour le retrieval semantique RAG.

Modele optimal ISO:
    google/embeddinggemma-300m-qat-q4_0-unquantized
    - QAT (Quantization Aware Training) = perte qualite minimale
    - Taille: ~229MB (conforme ISO 42001 cible 200MB)
    - MTEB: 69.31 (EN) / 60.62 (Multilingual)

Implementation conforme fabricant Google:
    - encode_query(): Prompt "task: search result | query: {content}"
    - encode_document(): Prompt "title: none | text: {content}"
    - WARNING: float16 NON SUPPORTE (utiliser float32 ou bfloat16)
    - Source: https://huggingface.co/blog/embeddinggemma

ISO Reference:
    - ISO/IEC 12207 S7.3.3 - Implementation
    - ISO/IEC 25010 S4.2 - Performance efficiency (FA-01: Recall >= 80%)
    - ISO/IEC 25010 PR-01 - RAM < 500MB
    - ISO/IEC 42001 A.6.2.2 - Documentation modeles

Dependencies:
    - sentence-transformers >= 3.0.0
    - torch >= 2.2.0
    - numpy >= 1.24.0

Usage:
    python embeddings.py -i corpus/processed/chunks_fr.json -o corpus/processed/embeddings_fr.npy
    python embeddings.py -i corpus/processed/chunks_intl.json -o corpus/processed/embeddings_intl.npy

Example (API officielle Google):
    >>> from scripts.pipeline.embeddings import load_embedding_model, embed_query, embed_documents
    >>> model = load_embedding_model()
    >>> query_emb = embed_query("règle toucher-jouer", model)
    >>> doc_embs = embed_documents(["Article 4.1...", "Article 4.2..."], model)
    >>> query_emb.shape, doc_embs.shape
    ((768,), (2, 768))
"""

import argparse
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --- Constants ---
# ISO 42001 A.6.2.2 - Documentation modeles
# ISO 25010 PR-01 - RAM < 500MB, PR-04 - Stockage < 200MB

# Modele principal: QAT Q4 (Quantization Aware Training)
# - Taille: ~229MB (conforme ISO 42001 §5.1)
# - MTEB: 69.31 EN / 60.62 Multilingual
# - Source: https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized
MODEL_ID = "google/embeddinggemma-300m-qat-q4_0-unquantized"

# Fallback: modele complet F32 (si QAT non disponible)
MODEL_ID_FULL = "google/embeddinggemma-300m"

# Fallback leger pour tests CI/CD
FALLBACK_MODEL_ID = "intfloat/multilingual-e5-base"

EMBEDDING_DIM = 768
FALLBACK_EMBEDDING_DIM = 768
DEFAULT_BATCH_SIZE = 32

# --- Prompts Officiels Google (OBLIGATOIRES pour performance optimale) ---
# Source: https://huggingface.co/blog/embeddinggemma
# WARNING: EmbeddingGemma NE SUPPORTE PAS float16 (utiliser float32 ou bfloat16)

PROMPT_QUERY = "task: search result | query: "
PROMPT_DOCUMENT = "title: none | text: "
PROMPT_QA = "task: question answering | query: "
PROMPT_CLASSIFICATION = "task: classification | query: "
PROMPT_CLUSTERING = "task: clustering | query: "
PROMPT_SIMILARITY = "task: sentence similarity | query: "


# --- Main Functions ---


def is_embeddinggemma_model(model_id: str) -> bool:
    """Verifie si le modele est un EmbeddingGemma (necessite prompts speciaux)."""
    return "embeddinggemma" in model_id.lower()


def load_embedding_model(
    model_id: str = MODEL_ID,
    device: str | None = None,
    truncate_dim: int | None = None,
) -> "SentenceTransformer":
    """
    Charge le modele EmbeddingGemma-300M via sentence-transformers.

    Tente de charger le modele principal, avec fallback automatique
    vers un modele alternatif en cas d'echec.

    Args:
        model_id: Identifiant HuggingFace du modele.
        device: Device cible (cpu/cuda). Auto-detect si None.
        truncate_dim: Dimension reduite MRL (256/512 pour gain memoire).

    Returns:
        Modele SentenceTransformer charge.

    Raises:
        RuntimeError: Si aucun modele ne peut etre charge.

    Example:
        >>> model = load_embedding_model()
        >>> model.get_sentence_embedding_dimension()
        768
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
    Genere des embeddings pour une liste de textes (methode generique).

    Pour EmbeddingGemma, preferer embed_query() ou embed_documents()
    qui appliquent automatiquement les prompts officiels Google.

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

    Example:
        >>> embeddings = embed_texts(["Hello", "World"], model)
        >>> embeddings.shape
        (2, 768)
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

    Utilise encode_query() pour EmbeddingGemma (prompt automatique),
    ou ajoute manuellement le prompt pour les autres modeles.

    ISO 42001: Methode conforme aux recommandations fabricant.

    Args:
        query: Texte de la requete utilisateur.
        model: Modele SentenceTransformer charge.
        normalize: Normaliser le vecteur (L2 norm = 1).

    Returns:
        np.ndarray de shape (dim,) avec l'embedding.

    Example:
        >>> emb = embed_query("règle toucher-jouer", model)
        >>> emb.shape
        (768,)
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
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode des documents avec le prompt officiel Google.

    Utilise encode_document() pour EmbeddingGemma (prompt automatique),
    ou encode standard pour les autres modeles.

    ISO 42001: Methode conforme aux recommandations fabricant.

    Args:
        documents: Liste des documents/chunks a encoder.
        model: Modele SentenceTransformer charge.
        batch_size: Taille de batch pour l'inference.
        show_progress: Afficher barre de progression.
        normalize: Normaliser les vecteurs (L2 norm = 1).

    Returns:
        np.ndarray de shape (N, dim) avec les embeddings.

    Raises:
        ValueError: Si documents est vide.

    Example:
        >>> embs = embed_documents(["Article 4.1...", "Article 4.2..."], model)
        >>> embs.shape
        (2, 768)
    """
    if not documents:
        raise ValueError("documents list cannot be empty")

    # EmbeddingGemma: utiliser methode officielle encode_document
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
    Pour EmbeddingGemma, utilise le prompt officiel Google via encode_query().

    ISO 42001: Methode conforme aux recommandations fabricant.

    Args:
        query: Texte de la requete.
        model_id: ID du modele (utilise MODEL_ID par defaut).
        normalize: Normaliser le vecteur (L2 norm = 1).

    Returns:
        np.ndarray de shape (dim,) avec l'embedding.

    Example:
        >>> emb = generate_query_embedding("toucher-jouer echecs")
        >>> emb.shape
        (768,)
    """
    global _query_model_cache

    if model_id is None:
        model_id = MODEL_ID

    # Load or reuse cached model
    if model_id not in _query_model_cache:
        _query_model_cache[model_id] = load_embedding_model(model_id)

    model = _query_model_cache[model_id]

    # Utiliser embed_query pour appliquer les prompts officiels
    return embed_query(query, model, normalize=normalize)


def embed_chunks(
    chunks: list[dict],
    model: "SentenceTransformer",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, list[str]]:
    """
    Genere des embeddings pour des chunks du pipeline.

    Extrait le texte de chaque chunk, genere les embeddings avec
    les prompts officiels Google (encode_document), et retourne
    egalement les IDs pour validation.

    ISO 42001: Methode conforme aux recommandations fabricant.

    Args:
        chunks: Liste de chunks conformes a CHUNK_SCHEMA.md.
        model: Modele SentenceTransformer charge.
        batch_size: Taille de batch.

    Returns:
        Tuple (embeddings array shape (N, dim), list of chunk IDs).

    Raises:
        ValueError: Si chunks est vide ou contient des chunks invalides.

    Example:
        >>> embeddings, ids = embed_chunks(chunks, model)
        >>> len(ids) == embeddings.shape[0]
        True
    """
    if not chunks:
        raise ValueError("chunks list cannot be empty")

    texts = []
    ids = []

    for chunk in chunks:
        if "text" not in chunk or "id" not in chunk:
            raise ValueError("Invalid chunk format: missing 'text' or 'id'")
        texts.append(chunk["text"])
        ids.append(chunk["id"])

    # Utiliser embed_documents pour appliquer les prompts officiels Google
    embeddings = embed_documents(texts, model, batch_size)

    return embeddings, ids


def measure_performance(
    model: "SentenceTransformer",
    sample_texts: list[str],
    n_iterations: int = 10,
) -> dict:
    """
    Mesure les performances du modele (latence, throughput).

    Args:
        model: Modele charge.
        sample_texts: Textes de test.
        n_iterations: Nombre d'iterations pour moyenne.

    Returns:
        dict avec ms_per_text, texts_per_second, total_time_s.

    Example:
        >>> perf = measure_performance(model, ["Hello world"] * 10)
        >>> perf["ms_per_text"] < 100
        True
    """
    if not sample_texts:
        return {"ms_per_text": 0.0, "texts_per_second": 0.0, "total_time_s": 0.0}

    # Warmup
    _ = model.encode(sample_texts[:1], show_progress_bar=False)

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.encode(sample_texts, show_progress_bar=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    ms_per_text = (avg_time / len(sample_texts)) * 1000

    return {
        "ms_per_text": round(ms_per_text, 2),
        "texts_per_second": round(len(sample_texts) / avg_time, 2),
        "total_time_s": round(avg_time, 3),
    }


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

    # Load chunks
    logger.info(f"Loading chunks from: {input_file}")
    data = load_json(input_file)
    chunks = data.get("chunks", [])

    if not chunks:
        raise ValueError(f"No chunks found in {input_file}")

    logger.info(f"Found {len(chunks)} chunks")

    # Load model
    model = load_embedding_model(model_id, truncate_dim=truncate_dim)
    embedding_dim = model.get_sentence_embedding_dimension()

    # Generate embeddings
    logger.info(f"Generating {embedding_dim}D embeddings...")
    start_time = time.perf_counter()

    embeddings, ids = embed_chunks(chunks, model, batch_size)

    elapsed = time.perf_counter() - start_time
    logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.1f}s")

    # Save embeddings
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)
    logger.info(f"Saved embeddings: {output_file}")

    # Save IDs mapping for validation
    ids_file = output_file.with_suffix(".ids.json")
    save_json({"chunk_ids": ids, "total": len(ids)}, ids_file)

    # Build report
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

    # Save report
    report_file = output_file.with_suffix(".report.json")
    save_json(report, report_file)

    return report


# --- CLI ---


def main() -> None:
    """Point d'entree CLI pour la generation d'embeddings."""
    parser = argparse.ArgumentParser(
        description="Generation d'embeddings pour Pocket Arbiter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python embeddings.py -i corpus/processed/chunks_fr.json -o corpus/processed/embeddings_fr.npy
    python embeddings.py -i corpus/processed/chunks_intl.json -o corpus/processed/embeddings_intl.npy
    python embeddings.py -i chunks.json -o embeddings.npy --model intfloat/multilingual-e5-small
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Fichier chunks JSON (ex: chunks_fr.json)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Fichier numpy de sortie (ex: embeddings_fr.npy)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=MODEL_ID,
        help=f"ID modele HuggingFace (default: {MODEL_ID})",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Taille batch (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=None,
        help="Dimension MRL reduite (256 pour gain memoire)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Afficher logs detailles",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {args.model}")

    report = generate_corpus_embeddings(
        args.input,
        args.output,
        args.model,
        args.batch_size,
        args.truncate_dim,
    )

    logger.info("=" * 50)
    logger.info(f"Chunks: {report['total_chunks']}")
    logger.info(f"Dimension: {report['embedding_dim']}D")
    logger.info(f"Time: {report['time_seconds']}s ({report['ms_per_chunk']}ms/chunk)")
    logger.info(f"Output: {report['output_file']}")


if __name__ == "__main__":
    main()
