"""SimCSE + ICT self-supervised fine-tuning for EmbeddingGemma-300M.

This is the LOCAL REFERENCE copy. The actual training script is:
    kaggle/kernel/train_simcse_ict.py (self-contained, Kaggle paths)

This file imports from scripts.training.config for local testing ONLY.
Training runs on Kaggle T4, not locally.

Standards:
    SimCSE: Gao et al. EMNLP 2021 (arXiv:2104.08821)
    ICT: Lee et al. ACL 2019 (arXiv:1906.00300)
    LoRA: Hu et al. ICLR 2022 (arXiv:2106.09685)
"""

from __future__ import annotations

import json
import logging
import random

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from scripts.training.config import (
    ICT_CHECKPOINT,
    ICT_CONFIG,
    ICT_PAIRS_PATH,
    LORA_CONFIG,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    QUERY_PROMPT,
    SEED,
    SIMCSE_CHECKPOINT,
    SIMCSE_CONFIG,
    SIMCSE_PAIRS_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    logger.info(
        "max_seq_length: %d -> %d (aligned with build pipeline)",
        model.max_seq_length,
        MAX_SEQ_LENGTH,
    )
    model.max_seq_length = MAX_SEQ_LENGTH
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

    # scale = 1/temperature. SimCSE paper: temp=0.05 → scale=20.0 (= CachedMNRL default).
    # ICT has no temperature in config → falls back to 0.05 → same scale.
    temperature = config.get("temperature", 0.05)
    scale = 1.0 / temperature
    loss = CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=config["mini_batch_size"],  # 16 (default 32, reduced for fp32)
        scale=scale,  # 20.0 (SimCSE paper temp 0.05)
    )

    # NO_DUPLICATES for both stages (sbert.net recommendation for contrastive loss)
    sampler = BatchSamplers.NO_DUPLICATES

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        # fp32 default. T4=Turing (compute 7.5), no bf16. Model card forbids fp16.
        seed=SEED,
        batch_sampler=sampler,
        logging_steps=1,
        logging_nan_inf_filter=True,
        save_strategy="epoch",
        load_best_model_at_end=False,
        # Prompt on anchor (query) ONLY — positive is pre-formatted in JSONL
        # with CCH title: "title: {cch} | text: {chunk}"
        prompts={
            "anchor": QUERY_PROMPT,
        },
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )
    train_result = trainer.train()

    # Validate training completed
    final_loss = train_result.training_loss
    logger.info("Final training loss: %.4f", final_loss)
    assert not np.isnan(final_loss), "FATAL: Training loss is NaN"
    assert not np.isinf(final_loss), "FATAL: Training loss is Inf"

    # Merge LoRA into base model before saving (CRITICAL for standalone loading)
    transformer = model[0]
    if hasattr(transformer, "auto_model") and hasattr(
        transformer.auto_model, "merge_and_unload"
    ):
        logger.info("Merging LoRA adapters into base model...")
        transformer.auto_model = transformer.auto_model.merge_and_unload()
        # CRITICAL: reset peft flag so saved model loads as plain transformer
        # Without this, Stage 2 has 0 trainable params (issue #3246)
        if hasattr(transformer.auto_model, "_hf_peft_config_loaded"):
            transformer.auto_model._hf_peft_config_loaded = False
        logger.info("LoRA merged successfully")

    model.save(output_dir)
    logger.info("Checkpoint saved to %s", output_dir)

    # Validate checkpoint produces valid embeddings (with prompt, like inference)
    test_model = SentenceTransformer(output_dir)
    test_model.max_seq_length = MAX_SEQ_LENGTH
    test_query = QUERY_PROMPT + "Test query validation"
    test_emb = test_model.encode([test_query], normalize_embeddings=True)
    assert test_emb.shape == (1, 768), f"FATAL: Expected (1, 768), got {test_emb.shape}"
    assert not np.any(np.isnan(test_emb)), "FATAL: Checkpoint produces NaN embeddings"
    norm = np.linalg.norm(test_emb[0])
    assert 0.99 < norm < 1.01, f"FATAL: Embedding not normalized (norm={norm:.4f})"
    logger.info("Embedding validation PASS: shape=%s, norm=%.4f", test_emb.shape, norm)
    del test_model

    return model


def main() -> None:
    """Run two-stage self-supervised fine-tuning."""
    set_seed(SEED)

    simcse_data = load_pairs(SIMCSE_PAIRS_PATH)
    ict_data = load_pairs(ICT_PAIRS_PATH)
    logger.info("SimCSE: %d pairs, ICT: %d pairs", len(simcse_data), len(ict_data))

    model = create_model_with_lora(MODEL_ID)

    # Stage 1: SimCSE
    model = train_stage(model, simcse_data, SIMCSE_CONFIG, SIMCSE_CHECKPOINT, "SimCSE")

    # Stage 2: ICT (continues from SimCSE LoRA)
    model = train_stage(model, ict_data, ICT_CONFIG, ICT_CHECKPOINT, "ICT")

    logger.info("=== Training complete ===")
    logger.info("SimCSE checkpoint: %s", SIMCSE_CHECKPOINT)
    logger.info("ICT checkpoint: %s", ICT_CHECKPOINT)


if __name__ == "__main__":
    main()
