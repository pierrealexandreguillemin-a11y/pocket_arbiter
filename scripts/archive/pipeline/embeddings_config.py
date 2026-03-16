"""
Embeddings Configuration - Pocket Arbiter

Constants, prompts et utilitaires pour le module embeddings.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Documentation modeles
    - ISO/IEC 25010 PR-01 - RAM < 500MB
"""

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# --- Constants ---
# ISO 42001 A.6.2.2 - Documentation modeles
# ISO 25010 PR-01 - RAM < 500MB, PR-04 - Stockage < 200MB

# Modele principal: EmbeddingGemma 300M QAT (Quantization-Aware Training)
# - Utilise unquantized weights mais entraine pour quantization
# - Pipeline coherence: corpus → fine-tuning QLoRA → TFLite deployment
# - Source: https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized
# NOTE: QAT requis pour eviter distribution shift entre corpus et production
MODEL_ID = "google/embeddinggemma-300m-qat-q4_0-unquantized"

# Full precision (meilleur recall mais inconsistant avec deployment)
MODEL_ID_FULL = "google/embeddinggemma-300m"

# Alias pour compatibilite (QAT est maintenant le defaut)
MODEL_ID_QAT = MODEL_ID

# Fallback leger pour tests CI/CD
FALLBACK_MODEL_ID = "google/embeddinggemma-300m"

EMBEDDING_DIM = 768
FALLBACK_EMBEDDING_DIM = 768
# Google recommande 128-256 pour stabilité gradients et in-batch negatives
# Source: ai.google.dev/gemma/docs/embeddinggemma/fine-tuning
DEFAULT_BATCH_SIZE = 128

# --- Matryoshka Representation Learning (MRL) ---
# EmbeddingGemma supporte MRL: dimensions 768 → 512 → 256 → 128
# Truncation sans re-training, perte <2% accuracy
# Source: arxiv.org/abs/2509.20354, huggingface.co/blog/embeddinggemma
MRL_DIMS = [768, 512, 256, 128]  # Dimensions supportées (nested)
MRL_DIM_FULL = 768  # Meilleure qualité
MRL_DIM_BALANCED = 256  # Équilibre qualité/performance (recommandé mobile)
MRL_DIM_FAST = 128  # Minimum, +6x compression vs 768D

# --- Prompts Officiels Google (OBLIGATOIRES pour performance optimale) ---
# Source: https://huggingface.co/blog/embeddinggemma
# Source: ai.google.dev/gemma/docs/embeddinggemma/inference
# WARNING: EmbeddingGemma NE SUPPORTE PAS float16 (utiliser float32 ou bfloat16)

PROMPT_QUERY = "task: search result | query: "
# Document prompt - défaut Google = "title: none"
# Avec titre (+4% relevance): "title: {actual_title} | text: "
PROMPT_DOCUMENT = "title: none | text: "  # Défaut Google officiel
PROMPT_DOCUMENT_WITH_TITLE = "title: {title} | text: "  # Optionnel si titre dispo
PROMPT_QA = "task: question answering | query: "
PROMPT_CLASSIFICATION = "task: classification | query: "
PROMPT_CLUSTERING = "task: clustering | query: "
PROMPT_SIMILARITY = "task: sentence similarity | query: "


def is_embeddinggemma_model(model_id: str) -> bool:
    """Verifie si le modele est un EmbeddingGemma (necessite prompts speciaux)."""
    return "embeddinggemma" in model_id.lower()


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
