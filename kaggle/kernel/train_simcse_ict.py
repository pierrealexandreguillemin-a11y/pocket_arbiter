"""SimCSE + ICT self-supervised fine-tuning for EmbeddingGemma-300M.

Kaggle T4 script — self-contained, production-grade, zero failure tolerance.

Input:  /kaggle/input/pocket-arbiter-training-data/{simcse,ict}_pairs.jsonl
Output: /kaggle/working/embeddinggemma-{simcse,simcse-ict}/

Standards:
    SimCSE: Gao et al. EMNLP 2021 (arXiv:2104.08821)
    ICT: Lee et al. ACL 2019 (arXiv:1906.00300)
    LoRA: Hu et al. ICLR 2022 (arXiv:2106.09685)
    Microsoft ML Production Checklist
    Kaggle best practices (seeds, idempotent, self-contained)
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

# ============================================================
# PHASE 0: Environment setup
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("=== PHASE 0: Environment setup ===")

# 0a. GPU validation
assert (
    torch.cuda.is_available()
), "FATAL: No GPU detected. This script requires Kaggle T4."
GPU_NAME = torch.cuda.get_device_name(0)
GPU_VRAM_MB = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
logger.info("GPU: %s (%.0f MB VRAM)", GPU_NAME, GPU_VRAM_MB)
assert GPU_VRAM_MB >= 14000, f"FATAL: Need >= 14 GB VRAM, got {GPU_VRAM_MB:.0f} MB"
assert (
    "bf16" not in GPU_NAME.lower() or "A100" in GPU_NAME
), "INFO: T4 Turing — fp32 mode (no bf16)"

# 0b. Install pinned deps
DEPS = ["sentence-transformers==5.2.0", "peft>=0.14"]
logger.info("Installing: %s", DEPS)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + DEPS)

# 0c. Import after install
# 0d. Version validation
import peft  # noqa: E402
import sentence_transformers  # noqa: E402
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

logger.info("sentence-transformers==%s", sentence_transformers.__version__)
logger.info("peft==%s", peft.__version__)
logger.info("torch==%s (CUDA %s)", torch.__version__, torch.version.cuda)
assert sentence_transformers.__version__.startswith(
    "5."
), f"FATAL: Need sentence-transformers 5.x, got {sentence_transformers.__version__}"

# ============================================================
# CONFIG (single source of truth, matches scripts/training/config.py)
# ============================================================

SEED = 42
MODEL_ID = "google/embeddinggemma-300m"

INPUT_DIR = "/kaggle/input/pocket-arbiter-training-data"
OUTPUT_DIR = "/kaggle/working"
SIMCSE_PAIRS_PATH = os.path.join(INPUT_DIR, "simcse_pairs.jsonl")
ICT_PAIRS_PATH = os.path.join(INPUT_DIR, "ict_pairs.jsonl")
SIMCSE_CHECKPOINT = os.path.join(OUTPUT_DIR, "embeddinggemma-simcse")
ICT_CHECKPOINT = os.path.join(OUTPUT_DIR, "embeddinggemma-simcse-ict")

# Query prompt must match indexer_embed.py format_query
# NO document prompt: documents are pre-formatted in JSONL with CCH title
# ("title: {cch} | text: {text}") by ict_data.format_document()
QUERY_PROMPT = "task: search result | query: "

# Pipeline desktop uses seq_length=2048 (EmbeddingGemma default).
# Chunks median 390 tokens, max 623 — all fit in 2048, NOT in 256.
# TFLite Android uses 256 but that's a DEPLOYMENT concern (gate T1), not training.
# Training MUST match the build pipeline (2048) to learn full chunk representations.
MAX_SEQ_LENGTH = 2048

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


# ============================================================
# FUNCTIONS
# ============================================================


def set_seed(seed: int) -> None:
    """Set ALL seeds for full reproducibility (MLOps standard)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(
        "Seeds set: %d (random, numpy, torch, cuda, cudnn, PYTHONHASHSEED)", seed
    )


def load_and_validate_pairs(path: str, name: str) -> Dataset:
    """Load JSONL pairs, validate every row, return HuggingFace Dataset."""
    assert os.path.exists(path), f"FATAL: Missing {path}"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    # Data validation (ISO 42001 A.6.2.3)
    assert len(rows) > 0, f"FATAL: {name} is empty"
    for i, r in enumerate(rows):
        assert "query" in r, f"FATAL: {name} row {i} missing 'query'"
        assert "document" in r, f"FATAL: {name} row {i} missing 'document'"
        assert len(r["query"]) > 0, f"FATAL: {name} row {i} empty query"
        assert len(r["document"]) > 0, f"FATAL: {name} row {i} empty document"

    q_lens = [len(r["query"]) for r in rows]
    d_lens = [len(r["document"]) for r in rows]
    logger.info(
        "%s: %d pairs, query len [%d/%d/%d], doc len [%d/%d/%d] (min/median/max)",
        name,
        len(rows),
        min(q_lens),
        sorted(q_lens)[len(q_lens) // 2],
        max(q_lens),
        min(d_lens),
        sorted(d_lens)[len(d_lens) // 2],
        max(d_lens),
    )
    return Dataset.from_dict(
        {
            "anchor": [r["query"] for r in rows],
            "positive": [r["document"] for r in rows],
        }
    )


def create_and_validate_model(model_id: str) -> SentenceTransformer:
    """Load model, attach LoRA, validate architecture."""
    logger.info("Loading model: %s", model_id)
    model = SentenceTransformer(model_id)

    # Align seq_length with build pipeline (2048, NOT TFLite 256)
    logger.info(
        "max_seq_length: %d -> %d (aligned with build pipeline)",
        model.max_seq_length,
        MAX_SEQ_LENGTH,
    )
    model.max_seq_length = MAX_SEQ_LENGTH

    # Validate pooling (must be mean_tokens for EmbeddingGemma)
    pooling = model[1]
    assert hasattr(
        pooling, "pooling_mode_mean_tokens"
    ), f"FATAL: Expected Pooling module at index 1, got {type(pooling)}"
    assert (
        pooling.pooling_mode_mean_tokens
    ), "FATAL: EmbeddingGemma must use mean_tokens pooling"
    logger.info("Pooling: mean_tokens (verified)")

    # Attach LoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_CONFIG["rank"],
        lora_alpha=LORA_CONFIG["alpha"],
        lora_dropout=LORA_CONFIG["dropout"],
        target_modules=LORA_CONFIG["target_modules"],
    )
    model.add_adapter(peft_config)

    # Validate trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    logger.info("Params: %d trainable / %d total (%.2f%%)", trainable, total, pct)
    assert trainable > 0, "FATAL: No trainable parameters after LoRA"
    assert (
        pct < 5
    ), f"FATAL: Too many trainable params ({pct:.1f}%), expected <5% with LoRA"

    # Log VRAM after model load
    vram_used = torch.cuda.memory_allocated() / 1024 / 1024
    vram_total = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
    logger.info(
        "VRAM after model load: %.0f / %.0f MB (%.1f%%)",
        vram_used,
        vram_total,
        100 * vram_used / vram_total,
    )
    assert (
        vram_used < vram_total * 0.8
    ), f"FATAL: VRAM usage {vram_used:.0f} MB > 80% of {vram_total:.0f} MB"

    return model


def train_stage(
    model: SentenceTransformer,
    dataset: Dataset,
    config: dict,
    output_dir: str,
    stage_name: str,
) -> SentenceTransformer:
    """Train one stage with full validation."""
    logger.info("=" * 60)
    logger.info(
        "STAGE: %s (%d examples, %d epochs)", stage_name, len(dataset), config["epochs"]
    )
    logger.info("=" * 60)

    # scale = 1/temperature. SimCSE paper: temp=0.05 → scale=20.0 (= CachedMNRL default).
    # ICT has no temperature in config → falls back to 0.05 → same scale.
    # Explicit pass to document the choice, even though it matches default.
    temperature = config.get("temperature", 0.05)
    scale = 1.0 / temperature
    loss = CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=config[
            "mini_batch_size"
        ],  # 16 (default 32, reduced for fp32 VRAM)
        scale=scale,  # 20.0 (SimCSE paper temp 0.05)
    )

    # NO_DUPLICATES for both stages: ensures no in-batch negatives are duplicates
    # of anchor/positive. For SimCSE (identical pairs), this correctly deduplicates
    # across the batch while preserving the (anchor_i, positive_i) training pair.
    # BATCH_SAMPLER would allow self-negatives that corrupt the contrastive loss.
    sampler = BatchSamplers.NO_DUPLICATES

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        # fp32 default. T4=Turing compute 7.5, no bf16. Model card forbids fp16.
        seed=SEED,
        batch_sampler=sampler,
        logging_steps=1,
        logging_nan_inf_filter=True,
        save_strategy="epoch",
        load_best_model_at_end=False,
        # Prompt on anchor (query) ONLY — positive is pre-formatted in JSONL
        # with CCH title: "title: {cch} | text: {chunk}"
        # Adding a prompt on positive would double-format it.
        prompts={
            "anchor": QUERY_PROMPT,
        },
    )

    # Train
    t0 = time.time()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )
    train_result = trainer.train()
    elapsed = time.time() - t0
    logger.info("Training time: %.1f sec (%.1f min)", elapsed, elapsed / 60)

    # Validate training completed
    final_loss = train_result.training_loss
    logger.info("Final training loss: %.4f", final_loss)
    assert not np.isnan(final_loss), "FATAL: Training loss is NaN"
    assert not np.isinf(final_loss), "FATAL: Training loss is Inf"
    assert final_loss < 100, f"FATAL: Training loss suspiciously high: {final_loss}"

    # Merge LoRA into base model before saving (CRITICAL for standalone loading)
    # Without merge, checkpoint contains adapter files only — unusable by load_model()
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
    else:
        logger.warning("merge_and_unload not available — saving with adapters")

    # Save merged checkpoint (standalone, no peft dependency needed to load)
    model.save(output_dir)
    logger.info("Checkpoint saved: %s", output_dir)

    # Validate checkpoint files exist
    expected_files = ["config.json", "model.safetensors"]
    for fname in expected_files:
        # sentence-transformers saves in subdirs, check recursively
        found = any(
            f == fname for dirpath, dirs, files in os.walk(output_dir) for f in files
        )
        if not found:
            logger.warning(
                "Expected file %s not found in %s (non-fatal)", fname, output_dir
            )

    # Count output files
    total_files = sum(len(files) for _, _, files in os.walk(output_dir))
    total_size = (
        sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(output_dir)
            for f in fns
        )
        / 1024
        / 1024
    )
    logger.info("Checkpoint: %d files, %.1f MB total", total_files, total_size)
    assert total_files > 0, f"FATAL: No files in checkpoint {output_dir}"
    assert total_size > 1, f"FATAL: Checkpoint suspiciously small ({total_size:.1f} MB)"

    # Validate checkpoint produces valid embeddings (with prompt, like inference)
    logger.info("Validating checkpoint embeddings...")
    test_model = SentenceTransformer(output_dir)
    test_model.max_seq_length = MAX_SEQ_LENGTH
    test_query = QUERY_PROMPT + "Test query validation"
    test_emb = test_model.encode([test_query], normalize_embeddings=True)
    assert test_emb.shape == (1, 768), f"FATAL: Expected (1, 768), got {test_emb.shape}"
    assert not np.any(np.isnan(test_emb)), "FATAL: Checkpoint produces NaN embeddings"
    assert not np.any(np.isinf(test_emb)), "FATAL: Checkpoint produces Inf embeddings"
    norm = np.linalg.norm(test_emb[0])
    assert 0.99 < norm < 1.01, f"FATAL: Embedding not normalized (norm={norm:.4f})"
    logger.info("Embedding validation PASS: shape=%s, norm=%.4f", test_emb.shape, norm)
    del test_model  # free VRAM

    # VRAM status
    vram_used = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("VRAM after %s: %.0f MB", stage_name, vram_used)

    return model


def validate_outputs() -> None:
    """Post-training: validate both checkpoints exist and are valid."""
    logger.info("=== POST-TRAINING VALIDATION ===")
    for ckpt_path, name in [(SIMCSE_CHECKPOINT, "SimCSE"), (ICT_CHECKPOINT, "ICT")]:
        assert os.path.isdir(
            ckpt_path
        ), f"FATAL: {name} checkpoint missing: {ckpt_path}"
        files = []
        for dp, _, fns in os.walk(ckpt_path):
            for f in fns:
                fp = os.path.join(dp, f)
                size_mb = os.path.getsize(fp) / 1024 / 1024
                files.append((f, size_mb))
        logger.info("%s checkpoint (%s):", name, ckpt_path)
        for f, s in sorted(files):
            logger.info("  %s (%.1f MB)", f, s)
        total = sum(s for _, s in files)
        logger.info("  TOTAL: %d files, %.1f MB", len(files), total)
        assert len(files) > 0, f"FATAL: {name} checkpoint empty"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    t_start = time.time()

    # Phase 1: Reproducibility
    logger.info("=== PHASE 1: Reproducibility ===")
    set_seed(SEED)

    # Phase 2: Data validation
    logger.info("=== PHASE 2: Data validation ===")
    simcse_data = load_and_validate_pairs(SIMCSE_PAIRS_PATH, "SimCSE")
    ict_data = load_and_validate_pairs(ICT_PAIRS_PATH, "ICT")

    # Phase 3: Model setup + validation
    logger.info("=== PHASE 3: Model setup ===")
    model = create_and_validate_model(MODEL_ID)

    # Phase 4: Training
    logger.info("=== PHASE 4: Training ===")
    model = train_stage(model, simcse_data, SIMCSE_CONFIG, SIMCSE_CHECKPOINT, "SimCSE")
    model = train_stage(model, ict_data, ICT_CONFIG, ICT_CHECKPOINT, "ICT")

    # Phase 5: Output validation
    logger.info("=== PHASE 5: Output validation ===")
    validate_outputs()

    # Summary
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE — %.1f min total", elapsed / 60)
    logger.info("SimCSE: %s", SIMCSE_CHECKPOINT)
    logger.info("ICT: %s", ICT_CHECKPOINT)
    logger.info("GPU: %s", GPU_NAME)
    logger.info("=" * 60)
