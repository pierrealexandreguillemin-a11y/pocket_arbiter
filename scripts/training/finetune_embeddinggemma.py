"""
Fine-tuning EmbeddingGemma pour Pocket Arbiter

Fine-tune le modele d'embeddings sur le domaine echecs/arbitrage FR.

ISO Reference: ISO/IEC 42001 A.6.2.2, ISO/IEC 25010 S4.2

Usage:
    python -m scripts.training.finetune_embeddinggemma \
        --triplets data/training/triplets_training.jsonl \
        --output models/embeddinggemma-chess-fr
"""

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from scripts.pipeline.utils import get_timestamp, save_json

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default hyperparameters (CPU-safe)
DEFAULT_MODEL_ID = "google/embeddinggemma-300m-qat-q4_0-unquantized"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_RATIO = 0.1
MAX_RAM_PERCENT = 85


def load_triplets_jsonl(input_path: Path) -> list[dict]:
    """Charge les triplets depuis un fichier JSONL."""
    if not input_path.exists():
        raise FileNotFoundError(f"Triplets file not found: {input_path}")
    triplets = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triplets.append(json.loads(line))
    return triplets


def check_resources() -> None:
    """Verifie les ressources systeme (RAM, CPU)."""
    mem = psutil.virtual_memory()
    if mem.percent > MAX_RAM_PERCENT:
        raise MemoryError(f"RAM usage {mem.percent}% > {MAX_RAM_PERCENT}%")
    logger.info(f"RAM usage: {mem.percent}%")


def get_training_args(
    output_dir: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    use_cpu: bool = True,
) -> dict:
    """Retourne les arguments d'entrainement pour SentenceTransformerTrainer."""
    return {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 4,  # Simule batch=16 sur CPU
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "fp16": not use_cpu,  # Pas de mixed precision sur CPU
        "dataloader_num_workers": 0,  # Evite multiprocessing issues Windows
        "logging_steps": 50,
        "save_strategy": "epoch",
        "report_to": "none",
        "max_grad_norm": 1.0,  # Stabilite
    }


def finetune_embeddinggemma(
    triplets: list[dict],
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    use_cpu: bool = True,
    monitor_resources: bool = True,
) -> "SentenceTransformer":
    """
    Fine-tune EmbeddingGemma avec MultipleNegativesRankingLoss.

    Args:
        triplets: Liste de triplets {anchor, positive, negative}.
        output_dir: Repertoire de sortie.
        model_id: ID du modele de base.
        epochs: Nombre d'epochs.
        batch_size: Taille de batch.
        learning_rate: Learning rate.
        use_cpu: Forcer CPU.
        monitor_resources: Surveiller RAM.

    Returns:
        Modele fine-tune.
    """
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    if monitor_resources:
        check_resources()

    # Charger le modele
    logger.info(f"Loading model: {model_id}")
    device = "cpu" if use_cpu else None
    model = SentenceTransformer(model_id, device=device)

    # Preparer le dataset
    logger.info(f"Preparing dataset with {len(triplets)} triplets")
    dataset = Dataset.from_list(triplets)

    # Loss function
    loss = MultipleNegativesRankingLoss(model)

    # Arguments d'entrainement
    args_dict = get_training_args(
        str(output_dir), epochs, batch_size, learning_rate, use_cpu=use_cpu
    )
    args = SentenceTransformerTrainingArguments(**args_dict)

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )

    # Entrainement
    logger.info("Starting training...")
    trainer.train()

    # Sauvegarder
    logger.info(f"Saving model to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    return model


def main() -> None:
    """Point d'entree CLI."""
    parser = argparse.ArgumentParser(description="Fine-tune EmbeddingGemma")
    parser.add_argument("--triplets", "-t", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--epochs", "-e", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    triplets = load_triplets_jsonl(args.triplets)
    logger.info(f"Loaded {len(triplets)} triplets")

    model = finetune_embeddinggemma(
        triplets=triplets,
        output_dir=args.output,
        model_id=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_cpu=not args.use_gpu,
    )

    # Rapport
    report = {
        "model_id": args.model,
        "output_dir": str(args.output),
        "total_triplets": len(triplets),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "timestamp": get_timestamp(),
    }
    save_json(report, args.output / "training_report.json")

    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
