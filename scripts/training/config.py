"""Training hyperparameters — single source of truth.

Sources:
- SimCSE: Gao et al. EMNLP 2021, Table D.1 (batch 64, temp 0.05)
- ICT/ORQA: Lee et al. ACL 2019, §7.3 + §9.2 (masking 90%)
- LoRA: Hu et al. ICLR 2022 (rank 8)
- sbert.net training overview (warmup 10%, weight_decay 0.01)
- FINETUNING_RESOURCES.md §4.1-4.2
"""

from __future__ import annotations

SEED = 42
MODEL_ID = "google/embeddinggemma-300m"
DB_PATH = "corpus/processed/corpus_v2_fr.db"

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

SIMCSE_CHECKPOINT = "models/embeddinggemma-simcse"
ICT_CHECKPOINT = "models/embeddinggemma-simcse-ict"
SIMCSE_PAIRS_PATH = "data/training/simcse_pairs.jsonl"
ICT_PAIRS_PATH = "data/training/ict_pairs.jsonl"
DATA_STATS_PATH = "data/training/data_stats.json"

# Pinned deps for Kaggle reproducibility
KAGGLE_DEPS = "sentence-transformers==5.2.0 peft>=0.14 torch>=2.2"
