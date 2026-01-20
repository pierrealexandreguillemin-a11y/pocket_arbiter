"""
Evaluation du modele fine-tune - Pocket Arbiter

Compare le recall du modele fine-tune vs le modele de base.

ISO Reference: ISO/IEC 25010 S4.2, ISO/IEC 29119

Usage:
    python -m scripts.training.evaluate_finetuned \
        --model models/embeddinggemma-chess-fr \
        --db corpus/processed/corpus_fr_v3.db \
        --questions tests/data/gold_standard_fr.json
"""

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.pipeline.tests.test_recall import benchmark_recall
from scripts.pipeline.utils import get_timestamp, save_json

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_TOP_K = 5
DEFAULT_TOLERANCE = 2


def load_finetuned_model(model_path: Path) -> "SentenceTransformer":
    """Charge un modele fine-tune depuis un repertoire local."""
    from sentence_transformers import SentenceTransformer

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    model = SentenceTransformer(str(model_path))
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded: {dim}D")
    return model


def evaluate_finetuned_model(
    model: "SentenceTransformer",
    db_path: Path,
    questions_file: Path,
    top_k: int = DEFAULT_TOP_K,
    tolerance: int = DEFAULT_TOLERANCE,
    use_hybrid: bool = True,
) -> dict:
    """
    Evalue un modele sur le gold standard.

    Args:
        model: Modele d'embeddings.
        db_path: Base de donnees corpus.
        questions_file: Fichier questions gold standard.
        top_k: Nombre de resultats.
        tolerance: Tolerance pages.
        use_hybrid: Utiliser recherche hybride.

    Returns:
        Resultats du benchmark.
    """
    logger.info(f"Evaluating on {questions_file}")
    result = benchmark_recall(
        db_path=db_path,
        questions_file=questions_file,
        model=model,
        top_k=top_k,
        use_hybrid=use_hybrid,
        tolerance=tolerance,
    )
    return result


def compare_models(
    base_model_id: str,
    finetuned_path: Path,
    db_path: Path,
    questions_file: Path,
    top_k: int = DEFAULT_TOP_K,
    tolerance: int = DEFAULT_TOLERANCE,
) -> dict:
    """
    Compare le modele de base et le modele fine-tune.

    Args:
        base_model_id: ID du modele de base.
        finetuned_path: Chemin du modele fine-tune.
        db_path: Base de donnees corpus.
        questions_file: Fichier questions.
        top_k: Nombre de resultats.
        tolerance: Tolerance pages.

    Returns:
        Comparaison des deux modeles.
    """
    from scripts.pipeline.embeddings import load_embedding_model

    # Evaluer modele de base
    logger.info(f"Evaluating base model: {base_model_id}")
    base_model = load_embedding_model(base_model_id)
    base_result = evaluate_finetuned_model(
        base_model, db_path, questions_file, top_k, tolerance
    )

    # Evaluer modele fine-tune
    logger.info(f"Evaluating finetuned model: {finetuned_path}")
    ft_model = load_finetuned_model(finetuned_path)
    ft_result = evaluate_finetuned_model(
        ft_model, db_path, questions_file, top_k, tolerance
    )

    # Calcul delta
    delta = ft_result["recall_mean"] - base_result["recall_mean"]

    return {
        "base_model": {
            "model_id": base_model_id,
            "recall_mean": base_result["recall_mean"],
            "recall_std": base_result["recall_std"],
        },
        "finetuned_model": {
            "model_path": str(finetuned_path),
            "recall_mean": ft_result["recall_mean"],
            "recall_std": ft_result["recall_std"],
        },
        "improvement": {
            "delta_recall": round(delta, 4),
            "delta_percent": round(delta * 100, 2),
            "improved": delta > 0,
        },
        "config": {
            "top_k": top_k,
            "tolerance": tolerance,
            "questions_file": str(questions_file),
        },
        "timestamp": get_timestamp(),
    }


def main() -> None:
    """Point d'entree CLI."""
    parser = argparse.ArgumentParser(description="Evaluate finetuned model")
    parser.add_argument(
        "--model", "-m", type=Path, required=True, help="Chemin du modele fine-tune"
    )
    parser.add_argument(
        "--db", "-d", type=Path, required=True, help="Base SQLite corpus"
    )
    parser.add_argument(
        "--questions",
        "-q",
        type=Path,
        required=True,
        help="Fichier questions gold standard",
    )
    parser.add_argument(
        "--base-model", type=str, default=None, help="Modele de base pour comparaison"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.base_model:
        # Comparaison
        result = compare_models(
            base_model_id=args.base_model,
            finetuned_path=args.model,
            db_path=args.db,
            questions_file=args.questions,
            top_k=args.top_k,
            tolerance=args.tolerance,
        )
        logger.info("=" * 50)
        logger.info("COMPARISON RESULTS")
        logger.info(
            f"Base model recall:      {result['base_model']['recall_mean']:.2%}"
        )
        logger.info(
            f"Finetuned model recall: {result['finetuned_model']['recall_mean']:.2%}"
        )
        logger.info(
            f"Improvement:            {result['improvement']['delta_percent']:+.2f}%"
        )
    else:
        # Evaluation simple
        model = load_finetuned_model(args.model)
        result = evaluate_finetuned_model(
            model, args.db, args.questions, args.top_k, args.tolerance
        )
        result["model_path"] = str(args.model)
        result["timestamp"] = get_timestamp()

        logger.info("=" * 50)
        logger.info(f"Recall@{args.top_k}: {result['recall_mean']:.2%}")
        logger.info(f"Std: {result['recall_std']:.2%}")

    # Sauvegarder
    output = args.output or args.model / "evaluation_report.json"
    save_json(result, output)
    logger.info(f"Report saved: {output}")


if __name__ == "__main__":
    main()
