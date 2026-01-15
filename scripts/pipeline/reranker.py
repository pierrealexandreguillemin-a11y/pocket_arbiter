"""
Reranker Module - Pocket Arbiter

Cross-encoder reranking pour ameliorer precision retrieval.
+20-35% recall selon benchmarks (Pinecone, BGE).

ISO Reference:
    - ISO/IEC 25010 - Performance efficiency (Recall >= 90%)
    - ISO/IEC 42001 - AI traceability (scores de confiance)

Usage:
    >>> from scripts.pipeline.reranker import load_reranker, rerank
    >>> reranker = load_reranker()
    >>> reranked = rerank("query", chunks, reranker, top_k=5)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# Modeles recommandes (multilingual FR support)
# bge-reranker-v2-m3: Best multilingual, 600M params
# ms-marco-MiniLM-L-6-v2: Leger, rapide (EN only)
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
FALLBACK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker(
    model_name: str = DEFAULT_MODEL,
    max_length: int = 512,
    use_fallback: bool = True,
) -> "CrossEncoder":
    """
    Charge le modele de reranking cross-encoder.

    Args:
        model_name: Nom du modele HuggingFace.
        max_length: Longueur max des inputs (tronque si depasse).
        use_fallback: Si True, essaie le modele fallback en cas d'erreur.

    Returns:
        Instance CrossEncoder chargee.

    Raises:
        OSError: Si le modele ne peut pas etre charge.

    Example:
        >>> reranker = load_reranker()
        >>> reranker = load_reranker("BAAI/bge-reranker-v2-m3")
    """
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(model_name, max_length=max_length)
    except OSError as e:
        if use_fallback and model_name != FALLBACK_MODEL:
            print(
                f"Warning: Could not load {model_name}, using fallback {FALLBACK_MODEL}"
            )
            return CrossEncoder(FALLBACK_MODEL, max_length=max_length)
        raise OSError(f"Could not load reranker model {model_name}: {e}") from e


def rerank(
    query: str,
    chunks: list[dict],
    model: "CrossEncoder",
    top_k: int = 5,
    content_key: str = "text",
) -> list[dict]:
    """
    Rerank chunks avec cross-encoder.

    Le cross-encoder evalue la pertinence de chaque paire (query, chunk)
    et retrie les resultats par score decroissant.

    Args:
        query: Question utilisateur.
        chunks: Liste de chunks avec cle `content_key`.
        model: CrossEncoder charge via load_reranker().
        top_k: Nombre de resultats finaux.
        content_key: Cle pour le contenu du chunk (defaut "text").

    Returns:
        Top-k chunks rerankes avec score "rerank_score" ajoute.

    Example:
        >>> chunks = [{"text": "...", "page": 41}, {"text": "...", "page": 42}]
        >>> reranked = rerank("toucher-jouer?", chunks, model, top_k=5)
        >>> reranked[0]["rerank_score"]
        0.95
    """
    if not chunks:
        return []

    if len(chunks) == 1:
        # Single chunk - return as-is with score
        chunks[0]["rerank_score"] = 1.0
        return chunks

    # Build query-document pairs
    pairs = [[query, c.get(content_key, "")] for c in chunks]

    # Score with cross-encoder
    scores = model.predict(pairs)

    # Add scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # Sort by score descending
    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    return ranked[:top_k]


def rerank_with_scores(
    query: str,
    chunks: list[dict],
    model: "CrossEncoder",
    top_k: int = 5,
    content_key: str = "text",
) -> tuple[list[dict], list[float]]:
    """
    Rerank chunks et retourne scores separement.

    Variante de rerank() qui retourne les scores dans une liste separee
    pour faciliter l'analyse et le debugging.

    Args:
        query: Question utilisateur.
        chunks: Liste de chunks avec cle `content_key`.
        model: CrossEncoder charge.
        top_k: Nombre de resultats finaux.
        content_key: Cle pour le contenu du chunk.

    Returns:
        Tuple (chunks rerankes, scores correspondants).

    Example:
        >>> chunks, scores = rerank_with_scores("query", chunks, model)
        >>> scores[0]  # Score du top chunk
        0.95
    """
    reranked = rerank(query, chunks, model, top_k, content_key)
    scores = [c.get("rerank_score", 0.0) for c in reranked]
    return reranked, scores


# =============================================================================
# CLI for manual testing
# =============================================================================


def main() -> None:
    """CLI pour test manuel du reranker."""
    import argparse

    parser = argparse.ArgumentParser(description="Test reranker Pocket Arbiter")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Modele de reranking (defaut: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Quelle est la regle du toucher-jouer ?",
        help="Question de test",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        nargs="+",
        default=[
            "Article 4.1: Le joueur qui touche une piece doit la jouer.",
            "Le roque est un coup special du roi.",
            "La pendule doit etre placee du cote droit de l'arbitre.",
        ],
        help="Chunks de test",
    )

    args = parser.parse_args()

    print(f"Loading reranker: {args.model}")
    reranker = load_reranker(args.model)

    # Create chunk dicts
    chunks = [{"text": text, "id": i} for i, text in enumerate(args.chunks)]

    print(f"\nQuery: {args.query}")
    print(f"Chunks: {len(chunks)}")

    reranked = rerank(args.query, chunks, reranker, top_k=len(chunks))

    print("\n--- Reranked Results ---")
    for i, chunk in enumerate(reranked):
        print(f"{i+1}. [score={chunk['rerank_score']:.4f}] {chunk['text'][:60]}...")


if __name__ == "__main__":
    main()
