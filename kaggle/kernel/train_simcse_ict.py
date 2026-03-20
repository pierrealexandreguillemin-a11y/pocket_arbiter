"""SimCSE + ICT self-supervised fine-tuning for EmbeddingGemma-300M.

Kaggle T4 script — self-contained, no local imports.
Reads data from /kaggle/input/pocket-arbiter-training-data/
Saves checkpoints to /kaggle/working/ (downloadable as output)

Standards:
    SimCSE: Gao et al. EMNLP 2021 (arXiv:2104.08821)
    ICT: Lee et al. ACL 2019 (arXiv:1906.00300)
    LoRA: Hu et al. ICLR 2022 (arXiv:2106.09685)
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys

import numpy as np
import torch

# === Install deps ===
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "sentence-transformers==5.2.0",
        "peft>=0.14",
    ]
)

from datasets import Dataset  # noqa: E402
from peft import LoraConfig, TaskType  # noqa: E402
from sentence_transformers import (  # noqa: E402
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import (  # noqa: E402
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.training_args import BatchSamplers  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# === Config (single source of truth) ===

SEED = 42
MODEL_ID = "google/embeddinggemma-300m"

# Kaggle paths
INPUT_DIR = "/kaggle/input/pocket-arbiter-training-data"
OUTPUT_DIR = "/kaggle/working"
SIMCSE_PAIRS_PATH = os.path.join(INPUT_DIR, "simcse_pairs.jsonl")
ICT_PAIRS_PATH = os.path.join(INPUT_DIR, "ict_pairs.jsonl")
SIMCSE_CHECKPOINT = os.path.join(OUTPUT_DIR, "embeddinggemma-simcse")
ICT_CHECKPOINT = os.path.join(OUTPUT_DIR, "embeddinggemma-simcse-ict")

LORA_CONFIG = {
    "rank": 8,
    "alpha": 8,
    "dropout": 0.1,
    "target_modules": ["q_proj", "v_proj"],
}

SIMCSE_CONFIG = {
    "batch_size": 64,
    "lr": 2e-5,
    "epochs": 3,
    "temperature": 0.05,
    "mini_batch_size": 16,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
}

ICT_CONFIG = {
    "batch_size": 64,
    "lr": 1e-5,
    "epochs": 5,
    "masking_rate": 0.9,
    "mini_batch_size": 16,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
}


# === Functions ===


def set_seed(seed: int) -> None:
    """Set all seeds for reproducibility (MLOps standard)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_pairs(path: str) -> Dataset:
    """Load JSONL pairs as HuggingFace Dataset."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return Dataset.from_dict(
        {
            "anchor": [r["query"] for r in rows],
            "positive": [r["document"] for r in rows],
        }
    )


def create_model_with_lora(model_id: str) -> SentenceTransformer:
    """Load EmbeddingGemma and attach LoRA adapters."""
    model = SentenceTransformer(model_id)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_CONFIG["rank"],
        lora_alpha=LORA_CONFIG["alpha"],
        lora_dropout=LORA_CONFIG["dropout"],
        target_modules=LORA_CONFIG["target_modules"],
    )
    model.add_adapter(peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total
    )
    return model


def train_stage(
    model: SentenceTransformer,
    dataset: Dataset,
    config: dict,
    output_dir: str,
    stage_name: str,
) -> SentenceTransformer:
    """Train one stage (SimCSE or ICT)."""
    logger.info("=== Stage: %s (%d examples) ===", stage_name, len(dataset))

    temperature = config.get("temperature", 0.05)
    scale = 1.0 / temperature
    loss = CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=config["mini_batch_size"],
        scale=scale,
    )

    # SimCSE: BATCH_SAMPLER (NO_DUPLICATES conflicts with identical pairs)
    # ICT: NO_DUPLICATES (standard for contrastive)
    sampler = (
        BatchSamplers.BATCH_SAMPLER
        if stage_name == "SimCSE"
        else BatchSamplers.NO_DUPLICATES
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        # fp32 default (T4=Turing, no bf16. fp16 risky per model card. fp32 safe.)
        # ~4-6 GB VRAM in fp32, fits T4 16GB.
        seed=SEED,
        batch_sampler=sampler,
        logging_steps=5,
        logging_nan_inf_filter=True,
        save_strategy="epoch",
        load_best_model_at_end=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()
    model.save(output_dir)
    logger.info("Checkpoint saved to %s", output_dir)
    return model


# === Main ===

if __name__ == "__main__":
    set_seed(SEED)

    # Verify input files exist
    for path in [SIMCSE_PAIRS_PATH, ICT_PAIRS_PATH]:
        assert os.path.exists(path), f"Missing: {path}"
    logger.info("Input files verified")

    # Load data
    simcse_data = load_pairs(SIMCSE_PAIRS_PATH)
    ict_data = load_pairs(ICT_PAIRS_PATH)
    logger.info("SimCSE: %d pairs, ICT: %d pairs", len(simcse_data), len(ict_data))

    # Create model with LoRA
    model = create_model_with_lora(MODEL_ID)

    # Stage 1: SimCSE
    model = train_stage(model, simcse_data, SIMCSE_CONFIG, SIMCSE_CHECKPOINT, "SimCSE")

    # Stage 2: ICT (continues from SimCSE LoRA)
    model = train_stage(model, ict_data, ICT_CONFIG, ICT_CHECKPOINT, "ICT")

    logger.info("=== Training complete ===")
    logger.info("SimCSE checkpoint: %s", SIMCSE_CHECKPOINT)
    logger.info("ICT checkpoint: %s", ICT_CHECKPOINT)

    # List output files for download
    for ckpt in [SIMCSE_CHECKPOINT, ICT_CHECKPOINT]:
        if os.path.isdir(ckpt):
            files = os.listdir(ckpt)
            logger.info("%s: %d files (%s)", ckpt, len(files), ", ".join(files[:5]))
