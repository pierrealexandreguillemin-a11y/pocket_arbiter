"""
Training module - Fine-tuning EmbeddingGemma pour Pocket Arbiter

Ce module contient les outils pour fine-tuner le modele d'embeddings
sur le domaine echecs/arbitrage FR.

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 80%)
    - ISO/IEC 42001 A.6.2.2 - Documentation modeles

Modules:
    - generate_synthetic_data: Generation de paires (query, chunk)
    - finetune_embeddinggemma: Fine-tuning avec sentence-transformers
    - evaluate_finetuned: Evaluation recall sur gold standard
"""

__all__ = [
    "generate_questions_for_chunk",
    "generate_synthetic_dataset",
    "finetune_embeddinggemma",
    "get_training_args",
    "evaluate_finetuned_model",
    "compare_models",
]


def __getattr__(name: str):
    """Lazy imports pour eviter chargement au demarrage."""
    if name == "generate_questions_for_chunk":
        from scripts.training.generate_synthetic_data import (
            generate_questions_for_chunk,
        )

        return generate_questions_for_chunk

    if name == "generate_synthetic_dataset":
        from scripts.training.generate_synthetic_data import generate_synthetic_dataset

        return generate_synthetic_dataset

    if name == "finetune_embeddinggemma":
        from scripts.training.finetune_embeddinggemma import finetune_embeddinggemma

        return finetune_embeddinggemma

    if name == "get_training_args":
        from scripts.training.finetune_embeddinggemma import get_training_args

        return get_training_args

    if name == "evaluate_finetuned_model":
        from scripts.training.evaluate_finetuned import evaluate_finetuned_model

        return evaluate_finetuned_model

    if name == "compare_models":
        from scripts.training.evaluate_finetuned import compare_models

        return compare_models

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
