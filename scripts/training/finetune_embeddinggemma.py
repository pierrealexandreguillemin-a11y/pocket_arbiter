"""
Fine-tuning EmbeddingGemma avec LoRA/PEFT pour Pocket Arbiter

Fine-tune le modele d'embeddings sur le domaine echecs/arbitrage FR.
Utilise PEFT (Parameter-Efficient Fine-Tuning) pour reduire la memoire
et accelerer l'entrainement.

ISO Reference: ISO/IEC 42001 A.6.2.2, ISO/IEC 25010 S4.2
Source: https://sbert.net/examples/sentence_transformer/training/peft/

Usage:
    python -m scripts.training.finetune_embeddinggemma \
        --triplets data/training/unified/triplets_train.jsonl \
        --output models/embeddinggemma-chess-fr \
        --use-lora  # Recommande

    # Full fine-tuning (legacy, plus lent)
    python -m scripts.training.finetune_embeddinggemma \
        --triplets data/training/unified/triplets_train.jsonl \
        --output models/embeddinggemma-chess-fr-full \
        --no-lora
"""

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
from transformers import TrainerCallback

from scripts.pipeline.utils import get_timestamp, save_json

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default hyperparameters
DEFAULT_MODEL_ID = "google/embeddinggemma-300m-qat-q4_0-unquantized"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8  # LoRA permet batch plus grand
DEFAULT_LEARNING_RATE = 2e-4  # LoRA utilise LR plus eleve
DEFAULT_WARMUP_RATIO = 0.1
MAX_RAM_PERCENT = 85
RAM_CHECK_INTERVAL = 10  # Check RAM every N steps

# LoRA hyperparameters (sentence-transformers defaults)
LORA_R = 64  # Rank of LoRA matrices
LORA_ALPHA = 128  # Scaling factor
LORA_DROPOUT = 0.1


class RAMMonitorCallback(TrainerCallback):
    """Callback for continuous RAM monitoring during training (ISO 25010)."""

    def __init__(self, max_ram_percent: float = MAX_RAM_PERCENT) -> None:
        self.max_ram_percent = max_ram_percent
        self.last_check_step = 0

    def on_step_end(self, args, state, control, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Check RAM usage every N steps."""
        if state.global_step - self.last_check_step >= RAM_CHECK_INTERVAL:
            mem = psutil.virtual_memory()
            if mem.percent > self.max_ram_percent:
                logger.error(f"RAM {mem.percent}% > {self.max_ram_percent}%! Stopping.")
                control.should_training_stop = True
            elif mem.percent > self.max_ram_percent - 10:
                logger.warning(f"RAM warning: {mem.percent}%")
            self.last_check_step = state.global_step


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
    use_lora: bool = True,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
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
        use_lora: Utiliser LoRA/PEFT (recommande).
        lora_r: Rank des matrices LoRA.
        lora_alpha: Facteur de scaling LoRA.
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

    # Ajouter LoRA adapter si demande
    if use_lora:
        from peft import LoraConfig, TaskType

        logger.info(f"Adding LoRA adapter (r={lora_r}, alpha={lora_alpha})")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=LORA_DROPOUT,
        )
        model.add_adapter(peft_config)

        # Compter les parametres entrainables
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    else:
        logger.info("Full fine-tuning (no LoRA)")

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

    # Callbacks for monitoring
    callbacks: list[TrainerCallback] = []
    if monitor_resources:
        callbacks.append(RAMMonitorCallback(MAX_RAM_PERCENT))

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        callbacks=callbacks,
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
    parser = argparse.ArgumentParser(
        description="Fine-tune EmbeddingGemma avec LoRA/PEFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LoRA fine-tuning (recommande)
  python -m scripts.training.finetune_embeddinggemma \\
      --triplets data/training/unified/triplets_train.jsonl \\
      --output models/embeddinggemma-chess-fr

  # Full fine-tuning (legacy)
  python -m scripts.training.finetune_embeddinggemma \\
      --triplets data/training/unified/triplets_train.jsonl \\
      --output models/embeddinggemma-chess-fr-full \\
      --no-lora
        """,
    )
    parser.add_argument("--triplets", "-t", type=Path, required=True,
                        help="Fichier JSONL des triplets")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Repertoire de sortie du modele")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL_ID,
                        help=f"Model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--epochs", "-e", type=int, default=DEFAULT_EPOCHS,
                        help=f"Nombre d'epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")

    # LoRA options
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument("--use-lora", action="store_true", default=True,
                            help="Utiliser LoRA/PEFT (default)")
    lora_group.add_argument("--no-lora", action="store_true",
                            help="Full fine-tuning sans LoRA")
    parser.add_argument("--lora-r", type=int, default=LORA_R,
                        help=f"LoRA rank (default: {LORA_R})")
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA,
                        help=f"LoRA alpha (default: {LORA_ALPHA})")

    parser.add_argument("--use-gpu", action="store_true", help="Utiliser GPU si disponible")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determiner si LoRA
    use_lora = not args.no_lora

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
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Rapport
    report = {
        "model_id": args.model,
        "output_dir": str(args.output),
        "total_triplets": len(triplets),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "use_lora": use_lora,
        "lora_r": args.lora_r if use_lora else None,
        "lora_alpha": args.lora_alpha if use_lora else None,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "timestamp": get_timestamp(),
    }
    save_json(report, args.output / "training_report.json")

    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {args.output}")
    if use_lora:
        logger.info(f"LoRA adapter: r={args.lora_r}, alpha={args.lora_alpha}")


if __name__ == "__main__":
    main()
