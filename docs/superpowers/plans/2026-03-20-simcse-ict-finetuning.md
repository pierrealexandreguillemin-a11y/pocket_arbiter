# SimCSE + ICT Self-Supervised Fine-tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune EmbeddingGemma-300M with SimCSE + ICT (self-supervised, corpus-only) via LoRA on Kaggle T4, rebuild corpus DB, measure recall.

**Architecture:** Two-stage LoRA training — SimCSE (uniformity warmup) then ICT (retrieval task-specific). Data extracted from corpus DB (1116 enriched children). Training script versioned in git, executed on Kaggle T4.

**Tech Stack:** sentence-transformers==5.2.0, peft>=0.14, torch>=2.2, Kaggle T4 16GB, fp32 (T4=Turing, no bf16).

**Spec:** `docs/superpowers/specs/2026-03-20-simcse-ict-finetuning-design.md`

---

## Quality gates (bloquants)

| Gate | Condition | Bloque |
|------|-----------|--------|
| G1 Data | SimCSE = 1116 paires non-vides, ICT >= 1000 paires queries >20 chars | Task 3 |
| G2 Stats | Token lengths median 50-500, max < 2048, distribution loguee | Task 3 |
| G3 Sample | 30 pseudo-queries ICT verifiees manuellement | Task 3 |
| G4 Dry-run | Script tourne sans crash en CPU 1 epoch | Task 4 |
| G5 SimCSE recall | recall@5 >= 60.1% apres stage 1, sinon STOP | Task 6 |
| G6 ICT recall | recall@5 >= 60.1% apres stage 2, sinon rollback | Task 8 |
| G7 TFLite gate | recall@5 >= 65.1% (+5pp) pour justifier conversion | Task 9 |
| G8 Integrity | I1-I9 PASS sur DB rebuild | Task 7, Task 8 |

## Outputs par task

| Task | Output | Versioning |
|------|--------|------------|
| 1 | `scripts/training/config.py` + tests | git |
| 2 | `scripts/training/ict_data.py` + tests | git |
| 3 | `data/training/simcse_pairs.jsonl`, `ict_pairs.jsonl`, `data_stats.json` | git |
| 4 | `scripts/training/train_simcse_ict.py` | git |
| 5 | `models/embeddinggemma-simcse/`, `models/embeddinggemma-simcse-ict/` | HF Hub / download |
| 5b | `scripts/pipeline/recall.py` (model_id param) | git |
| 6 | `data/benchmarks/recall_post_simcse.json` | git |
| 7 | `corpus/processed/corpus_v2_fr.db` (rebuilt SimCSE) | DVC |
| 8 | `data/benchmarks/recall_post_ict.json`, `corpus_v2_fr.db` (rebuilt ICT) | git + DVC |
| 9 | `models/model_card.json`, `CLAUDE.md` (decision) | git |

## File Structure

| File | Action | SRP |
|------|--------|-----|
| `scripts/training/__init__.py` | CREATE | Package marker |
| `scripts/training/config.py` | CREATE | Hyperparams, seeds, paths — single source of truth |
| `scripts/training/ict_data.py` | CREATE | Data extraction + validation + stats |
| `scripts/training/train_simcse_ict.py` | CREATE | Training script (Kaggle-compatible, dry-run local) |
| `scripts/training/tests/__init__.py` | CREATE | Package marker |
| `scripts/training/tests/test_config.py` | CREATE | Config sanity tests |
| `scripts/training/tests/test_ict_data.py` | CREATE | Data extraction tests |
| `scripts/pipeline/recall.py` | MODIFY | Accept model_id param |
| `models/model_card.json` | UPDATE | Finetuning section |

---

### Task 1: Config module

**Files:**
- Create: `scripts/training/__init__.py`
- Create: `scripts/training/config.py`
- Create: `scripts/training/tests/__init__.py`
- Create: `scripts/training/tests/test_config.py`

**DoD:** Config importable, all tests PASS, hyperparams match spec.

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p scripts/training/tests data/training
```

- [ ] **Step 2: Write test_config.py**

```python
# scripts/training/tests/test_config.py
"""Tests for training config sanity."""

from scripts.training.config import (
    ICT_CONFIG,
    LORA_CONFIG,
    SEED,
    SIMCSE_CONFIG,
)


def test_seed_is_42():
    assert SEED == 42


def test_simcse_config_keys():
    required = {"batch_size", "lr", "epochs", "temperature", "mini_batch_size",
                "weight_decay", "max_grad_norm", "warmup_ratio"}
    assert required.issubset(set(SIMCSE_CONFIG.keys()))


def test_ict_config_keys():
    required = {"batch_size", "lr", "epochs", "masking_rate", "mini_batch_size",
                "weight_decay", "max_grad_norm", "warmup_ratio"}
    assert required.issubset(set(ICT_CONFIG.keys()))


def test_lora_config_keys():
    required = {"rank", "alpha", "dropout", "target_modules"}
    assert required.issubset(set(LORA_CONFIG.keys()))


def test_ict_lr_less_than_simcse():
    assert ICT_CONFIG["lr"] < SIMCSE_CONFIG["lr"]


def test_lora_alpha_equals_rank():
    assert LORA_CONFIG["alpha"] == LORA_CONFIG["rank"]


def test_masking_rate_90_percent():
    assert ICT_CONFIG["masking_rate"] == 0.9
```

- [ ] **Step 3: Run test — expect FAIL**

Run: `python -m pytest scripts/training/tests/test_config.py -v`

- [ ] **Step 4: Write config.py**

```python
# scripts/training/config.py
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
```

- [ ] **Step 5: Create __init__.py files (empty)**

- [ ] **Step 6: Run tests — expect PASS**

Run: `python -m pytest scripts/training/tests/test_config.py -v`

- [ ] **Step 7: Commit**

```bash
git add scripts/training/
git commit -m "feat(training): add config module (SimCSE+ICT+LoRA hyperparams)"
```

---

### Task 2: ICT data extraction + validation

**Files:**
- Create: `scripts/training/ict_data.py`
- Create: `scripts/training/tests/test_ict_data.py`

**DoD:** All data functions tested, extraction + validation + stats computation work.

- [ ] **Step 1: Write test_ict_data.py**

```python
# scripts/training/tests/test_ict_data.py
"""Tests for ICT data extraction."""

from scripts.training.config import SEED
from scripts.training.ict_data import (
    compute_data_stats,
    extract_random_sentence,
    generate_ict_pairs,
    generate_simcse_pairs,
    mask_sentence_from_chunk,
    save_pairs,
    validate_pairs,
)


def test_extract_random_sentence_returns_long():
    text = "Phrase courte. Deuxieme phrase assez longue pour passer le filtre minimum."
    s = extract_random_sentence(text, seed=SEED)
    assert s is not None
    assert len(s) > 20


def test_extract_random_sentence_none_if_all_short():
    assert extract_random_sentence("A. B. C.", seed=SEED) is None


def test_extract_random_sentence_reproducible():
    text = "Premiere phrase longue. Deuxieme phrase aussi longue. Troisieme pour varier."
    s1 = extract_random_sentence(text, seed=42)
    s2 = extract_random_sentence(text, seed=42)
    assert s1 == s2


def test_mask_sentence_removes_90_percent():
    chunk = "Phrase a masquer. Le reste du chunk continue ici."
    # Run 100 times, ~90 should mask
    masked = sum(
        1 for i in range(100)
        if "Phrase a masquer" not in mask_sentence_from_chunk(
            chunk, "Phrase a masquer.", mask_rate=0.9, seed=i
        )
    )
    assert 80 <= masked <= 100  # ~90% ± tolerance


def test_mask_sentence_preserves_at_zero():
    chunk = "Phrase gardee. Le reste."
    result = mask_sentence_from_chunk(chunk, "Phrase gardee.", mask_rate=0.0, seed=0)
    assert "Phrase gardee." in result


def test_generate_simcse_pairs():
    texts = ["Chunk texte numero un." for _ in range(10)]
    pairs = generate_simcse_pairs(texts)
    assert len(pairs) == 10
    assert pairs[0][0] == pairs[0][1]


def test_generate_ict_pairs_count():
    texts = [
        "Premiere phrase longue du chunk reglementaire. Deuxieme phrase du chunk."
        for _ in range(10)
    ]
    pairs = generate_ict_pairs(texts, seed=SEED)
    assert 0 < len(pairs) <= 10
    for query, chunk in pairs:
        assert len(query) > 20


def test_generate_ict_pairs_reproducible():
    texts = ["Phrase un longue. Phrase deux longue." for _ in range(10)]
    assert generate_ict_pairs(texts, seed=42) == generate_ict_pairs(texts, seed=42)


def test_validate_pairs_pass():
    pairs = [("Query longue suffisante ok", "Document texte complet")] * 10
    errors = validate_pairs(pairs, min_query_len=20)
    assert len(errors) == 0


def test_validate_pairs_fail_short():
    pairs = [("Short", "Document texte complet")]
    errors = validate_pairs(pairs, min_query_len=20)
    assert len(errors) == 1


def test_compute_data_stats():
    pairs = [("Query de test assez longue", "Document plus long que la query")] * 5
    stats = compute_data_stats(pairs)
    assert stats["count"] == 5
    assert "query_len_median" in stats
    assert "doc_len_median" in stats
    assert stats["query_len_min"] > 0


def test_save_pairs(tmp_path):
    pairs = [("query", "doc")]
    path = tmp_path / "test.jsonl"
    save_pairs(pairs, path)
    assert path.exists()
    import json
    with open(path) as f:
        row = json.loads(f.readline())
    assert row["query"] == "query"
    assert row["document"] == "doc"
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m pytest scripts/training/tests/test_ict_data.py -v`

- [ ] **Step 3: Write ict_data.py**

```python
# scripts/training/ict_data.py
"""Data extraction for SimCSE and ICT training.

SimCSE: (chunk_text, chunk_text) — dropout = augmentation.
ICT: (random_sentence, chunk_text) — ORQA standard §9.2 (90% masking).
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import statistics
from pathlib import Path

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u00DC])")
_MIN_SENTENCE_LEN = 20


def load_children_texts(db_path: str) -> list[str]:
    """Load all children texts from corpus DB."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT text FROM children ORDER BY id").fetchall()
    conn.close()
    return [r[0] for r in rows]


def extract_random_sentence(text: str, seed: int = 42) -> str | None:
    """Extract a random sentence (>20 chars) from text. ORQA standard."""
    sentences = _SENT_RE.split(text)
    valid = [s.strip() for s in sentences if len(s.strip()) > _MIN_SENTENCE_LEN]
    if not valid:
        return None
    return random.Random(seed).choice(valid)


def mask_sentence_from_chunk(
    chunk: str,
    sentence: str,
    mask_rate: float = 0.9,
    seed: int = 0,
) -> str:
    """Remove sentence from chunk with probability mask_rate (ORQA §9.2)."""
    if random.Random(seed).random() < mask_rate:
        return chunk.replace(sentence, "").strip()
    return chunk


def generate_simcse_pairs(texts: list[str]) -> list[tuple[str, str]]:
    """SimCSE pairs: (text, text) for dropout augmentation."""
    return [(t, t) for t in texts if t.strip()]


def generate_ict_pairs(
    texts: list[str],
    seed: int = 42,
) -> list[tuple[str, str]]:
    """ICT pairs: (random_sentence, chunk) with 90% masking."""
    pairs = []
    for i, text in enumerate(texts):
        sentence = extract_random_sentence(text, seed=seed + i)
        if sentence is None:
            continue
        masked = mask_sentence_from_chunk(text, sentence, seed=seed + i)
        if len(masked.strip()) < _MIN_SENTENCE_LEN:
            continue
        pairs.append((sentence, masked))
    return pairs


def validate_pairs(
    pairs: list[tuple[str, str]],
    min_query_len: int = _MIN_SENTENCE_LEN,
) -> list[str]:
    """Validate pairs. Returns list of error messages (empty = OK)."""
    errors = []
    for i, (query, doc) in enumerate(pairs):
        if len(query) <= min_query_len:
            errors.append(f"Pair {i}: query too short ({len(query)} chars)")
        if not doc.strip():
            errors.append(f"Pair {i}: empty document")
    return errors


def compute_data_stats(pairs: list[tuple[str, str]]) -> dict:
    """Compute data statistics for validation (ISO 42001 A.6.2.3)."""
    q_lens = [len(q) for q, _ in pairs]
    d_lens = [len(d) for _, d in pairs]
    return {
        "count": len(pairs),
        "query_len_min": min(q_lens) if q_lens else 0,
        "query_len_median": int(statistics.median(q_lens)) if q_lens else 0,
        "query_len_max": max(q_lens) if q_lens else 0,
        "query_len_p95": int(sorted(q_lens)[int(0.95 * len(q_lens))]) if q_lens else 0,
        "doc_len_min": min(d_lens) if d_lens else 0,
        "doc_len_median": int(statistics.median(d_lens)) if d_lens else 0,
        "doc_len_max": max(d_lens) if d_lens else 0,
        "doc_len_p95": int(sorted(d_lens)[int(0.95 * len(d_lens))]) if d_lens else 0,
    }


def save_pairs(pairs: list[tuple[str, str]], path: str | Path) -> None:
    """Save pairs as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for query, doc in pairs:
            json.dump({"query": query, "document": doc}, f, ensure_ascii=False)
            f.write("\n")
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `python -m pytest scripts/training/tests/test_ict_data.py -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/training/ict_data.py scripts/training/tests/test_ict_data.py
git commit -m "feat(training): add ICT data extraction with validation and stats"
```

---

### Task 3: Generate and validate training data

**Files:**
- Output: `data/training/simcse_pairs.jsonl`
- Output: `data/training/ict_pairs.jsonl`
- Output: `data/training/data_stats.json`

**DoD:** Gates G1 (counts), G2 (stats), G3 (sample 30) PASS.

- [ ] **Step 1: Generate pairs**

```bash
python -c "
import json
from scripts.training.ict_data import (
    load_children_texts, generate_simcse_pairs, generate_ict_pairs,
    validate_pairs, compute_data_stats, save_pairs,
)
from scripts.training.config import DB_PATH, SIMCSE_PAIRS_PATH, ICT_PAIRS_PATH, DATA_STATS_PATH, SEED

texts = load_children_texts(DB_PATH)
print(f'Loaded {len(texts)} children')

# SimCSE
simcse = generate_simcse_pairs(texts)
save_pairs(simcse, SIMCSE_PAIRS_PATH)
print(f'SimCSE: {len(simcse)} pairs')

# ICT
ict = generate_ict_pairs(texts, seed=SEED)
save_pairs(ict, ICT_PAIRS_PATH)
print(f'ICT: {len(ict)} pairs')

# G1: Count gate
assert len(simcse) == 1116, f'SimCSE count {len(simcse)} != 1116'
assert len(ict) >= 1000, f'ICT count {len(ict)} < 1000'
print('G1 PASS: counts OK')

# Validate
errors = validate_pairs(ict)
assert len(errors) == 0, f'Validation errors: {errors}'
print('G1b PASS: all queries > 20 chars')

# G2: Stats gate
simcse_stats = compute_data_stats(simcse)
ict_stats = compute_data_stats(ict)
stats = {'simcse': simcse_stats, 'ict': ict_stats}
with open(DATA_STATS_PATH, 'w') as f:
    json.dump(stats, f, indent=2)
print(f'SimCSE stats: {simcse_stats}')
print(f'ICT stats: {ict_stats}')
assert ict_stats['doc_len_max'] < 10000, 'Doc too long'
print('G2 PASS: stats OK')
"
```

- [ ] **Step 2: G3 — manual sample 30 pseudo-queries**

```bash
python -c "
import json
with open('data/training/ict_pairs.jsonl') as f:
    pairs = [json.loads(line) for line in f]
print(f'Total ICT pairs: {len(pairs)}')
print()
import random
random.seed(42)
sample = random.sample(pairs, min(30, len(pairs)))
for i, p in enumerate(sample):
    print(f'{i+1:2d}. Q: {p[\"query\"][:100]}')
    print(f'    D: {p[\"document\"][:80]}...')
    print()
"
```

**Manually verify:** pseudo-queries are diverse, non-trivial, related to chunk content.

- [ ] **Step 3: Commit**

```bash
git add data/training/simcse_pairs.jsonl data/training/ict_pairs.jsonl data/training/data_stats.json
git commit -m "feat(training): generate SimCSE+ICT pairs (G1-G3 PASS)"
```

---

### Task 4: Training script

**Files:**
- Create: `scripts/training/train_simcse_ict.py`

**DoD:** G4 dry-run PASS (CPU, 1 epoch, no crash).

- [ ] **Step 1: Write train_simcse_ict.py**

```python
# scripts/training/train_simcse_ict.py
"""SimCSE + ICT self-supervised fine-tuning for EmbeddingGemma-300M.

Usage (Kaggle T4):  python scripts/training/train_simcse_ict.py
Usage (local CPU):  python scripts/training/train_simcse_ict.py --dry-run

Standards:
    SimCSE: Gao et al. EMNLP 2021 (arXiv:2104.08821)
    ICT: Lee et al. ACL 2019 (arXiv:1906.00300)
    LoRA: Hu et al. ICLR 2022 (arXiv:2106.09685)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

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
    ICT_CONFIG,
    ICT_PAIRS_PATH,
    ICT_CHECKPOINT,
    LORA_CONFIG,
    MODEL_ID,
    SEED,
    SIMCSE_CONFIG,
    SIMCSE_PAIRS_PATH,
    SIMCSE_CHECKPOINT,
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
    return Dataset.from_dict({
        "anchor": [r["query"] for r in rows],
        "positive": [r["document"] for r in rows],
    })


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
    logger.info("Trainable: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)
    return model


def train_stage(
    model: SentenceTransformer,
    dataset: Dataset,
    config: dict,
    output_dir: str,
    stage_name: str,
    dry_run: bool = False,
) -> SentenceTransformer:
    """Train one stage (SimCSE or ICT)."""
    logger.info("=== Stage: %s (%d examples) ===", stage_name, len(dataset))

    temperature = config.get("temperature", 0.05)
    loss = CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=config["mini_batch_size"],
        temperature=temperature,
    )

    epochs = 1 if dry_run else config["epochs"]
    use_bf16 = torch.cuda.is_available() and not dry_run
    # SimCSE: use BATCH_SIZE sampler (NO_DUPLICATES may conflict with identical pairs)
    # ICT: use NO_DUPLICATES (standard for contrastive)
    sampler = (
        BatchSamplers.BATCH_SIZE if stage_name == "SimCSE"
        else BatchSamplers.NO_DUPLICATES
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        bf16=use_bf16,
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
    model.save(output_dir)  # sentence-transformers save (preserves pooling + modules.json)
    logger.info("Checkpoint saved to %s", output_dir)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="SimCSE + ICT fine-tuning")
    parser.add_argument("--dry-run", action="store_true", help="CPU, 1 epoch, quick test")
    args = parser.parse_args()

    set_seed(SEED)

    simcse_data = load_pairs(SIMCSE_PAIRS_PATH)
    ict_data = load_pairs(ICT_PAIRS_PATH)
    logger.info("SimCSE: %d pairs, ICT: %d pairs", len(simcse_data), len(ict_data))

    model = create_model_with_lora(MODEL_ID)

    # Stage 1: SimCSE
    model = train_stage(
        model, simcse_data, SIMCSE_CONFIG,
        SIMCSE_CHECKPOINT, "SimCSE", dry_run=args.dry_run,
    )

    # Stage 2: ICT (continues from SimCSE LoRA)
    model = train_stage(
        model, ict_data, ICT_CONFIG,
        ICT_CHECKPOINT, "ICT", dry_run=args.dry_run,
    )

    logger.info("=== Training complete ===")
    logger.info("SimCSE checkpoint: %s", SIMCSE_CHECKPOINT)
    logger.info("ICT checkpoint: %s", ICT_CHECKPOINT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: G4 — Local dry-run**

Run: `python scripts/training/train_simcse_ict.py --dry-run`
Expected: Runs without crash, logs "Training complete". Slow on CPU (~2 min, 1 epoch).

- [ ] **Step 3: Commit**

```bash
git add scripts/training/train_simcse_ict.py
git commit -m "feat(training): add SimCSE+ICT training script (LoRA, bf16, dry-run)"
```

---

### Task 5: Run training on Kaggle

**This task happens on Kaggle.**

- [ ] **Step 1: Upload to Kaggle**

1. Create Kaggle Dataset: `corpus_v2_fr.db`, `simcse_pairs.jsonl`, `ict_pairs.jsonl`
2. Create Notebook, enable GPU T4
3. Install: `pip install sentence-transformers==5.2.0 peft>=0.14`
4. Copy scripts from `scripts/training/`

- [ ] **Step 2: Run training**

```bash
python train_simcse_ict.py
```

Expected: ~10 min total. Two checkpoints saved.

- [ ] **Step 3: Download BOTH checkpoints**

Download `models/embeddinggemma-simcse/` AND `models/embeddinggemma-simcse-ict/`.

---

### Task 5b: Fix recall.py model_id

**CRITICAL:** `run_recall()` defaults to base model. Query embeddings mismatch.

**Files:**
- Modify: `scripts/pipeline/recall.py`

**DoD:** `run_recall` accepts `model_id` param, existing tests still PASS.

- [ ] **Step 1: Modify run_recall signature**

In `scripts/pipeline/recall.py:210`, change:

```python
def run_recall(
    db_path: Path | str,
    gs_path: Path | str,
    output_dir: Path | str = "data/benchmarks",
    model_id: str | None = None,
) -> dict:
```

And at line ~232, change `model = load_model()` to:

```python
    model = load_model(model_id) if model_id else load_model()
```

- [ ] **Step 2: Run existing tests**

Run: `python -m pytest scripts/pipeline/tests/ -v -m "not slow"`
Expected: ALL PASS (backward compatible, model_id defaults to None).

- [ ] **Step 3: Commit**

```bash
git add scripts/pipeline/recall.py
git commit -m "feat(pipeline): recall.py accepts model_id for fine-tuned model eval"
```

---

### Task 6: Measure post-SimCSE recall (gate G5)

**DoD:** G5 PASS (recall@5 >= 60.1%), `recall_post_simcse.json` in git.

- [ ] **Step 1: DVC snapshot BEFORE rebuild**

```bash
python -m dvc add corpus/processed/corpus_v2_fr.db && python -m dvc push
```

- [ ] **Step 2: Dry-run (no embedding)**

```bash
python -c "
from pathlib import Path
from scripts.pipeline.chunker import chunk_document
import json, logging
logging.basicConfig(level=logging.INFO)
docling_dir = Path('corpus/processed/docling_v2_fr')
total = sum(
    len(chunk_document(json.load(open(p))['markdown'], json.load(open(p))['source'],
        json.load(open(p)).get('heading_pages'))['children'])
    for p in sorted(docling_dir.glob('*.json'))
)
print(f'Children: {total} (expected 1116)')
assert total == 1116
"
```

- [ ] **Step 3: Rebuild with SimCSE model**

```bash
python -c "
from pathlib import Path
from scripts.pipeline.indexer import build_index
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
build_index(
    docling_dir=Path('corpus/processed/docling_v2_fr'),
    table_summaries_path=Path('corpus/processed/table_summaries_claude.json'),
    output_db=Path('corpus/processed/corpus_v2_fr.db'),
    model_id='models/embeddinggemma-simcse',
)
"
```

Expected: ~12 min, I1-I9 PASS (G8).

- [ ] **Step 4: Measure recall (ALL metrics)**

```bash
python -c "
from scripts.pipeline.recall import run_recall
r = run_recall(
    'corpus/processed/corpus_v2_fr.db',
    'tests/data/gold_standard_annales_fr_v8_adversarial.json',
    model_id='models/embeddinggemma-simcse',
)
g = r['global']
print(f'recall@1  = {g[\"recall@1\"]:.1%}')
print(f'recall@3  = {g[\"recall@3\"]:.1%}')
print(f'recall@5  = {g[\"recall@5\"]:.1%}')
print(f'recall@10 = {g[\"recall@10\"]:.1%}')
print(f'MRR       = {g[\"mrr\"]:.3f}')
print()
print(f'G5 (>=60.1%): {\"PASS — continue ICT\" if g[\"recall@5\"] >= 0.601 else \"FAIL — STOP\"}'  )
"
```

**GATE G5:** If recall@5 < 60.1% → STOP. `dvc checkout`, restore baseline. Do NOT proceed.

- [ ] **Step 5: Save and commit**

```bash
cp data/benchmarks/recall_baseline.json data/benchmarks/recall_post_simcse.json
git add data/benchmarks/recall_post_simcse.json
git commit -m "test(training): post-SimCSE recall (G5 evaluated)"
```

---

### Task 7: Rebuild with ICT model (if G5 PASS)

**DoD:** G8 PASS (I1-I9), DB rebuilt with ICT checkpoint.

- [ ] **Step 1: Rebuild with ICT model**

```bash
python -c "
from pathlib import Path
from scripts.pipeline.indexer import build_index
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
build_index(
    docling_dir=Path('corpus/processed/docling_v2_fr'),
    table_summaries_path=Path('corpus/processed/table_summaries_claude.json'),
    output_db=Path('corpus/processed/corpus_v2_fr.db'),
    model_id='models/embeddinggemma-simcse-ict',
)
"
```

Expected: ~12 min, I1-I9 PASS (G8).

---

### Task 8: Measure post-ICT recall (gate G6, G7)

**DoD:** G6 PASS, G7 evaluated, `recall_post_ict.json` in git, DVC snapshot.

- [ ] **Step 1: Measure recall (ALL metrics)**

```bash
python -c "
from scripts.pipeline.recall import run_recall
r = run_recall(
    'corpus/processed/corpus_v2_fr.db',
    'tests/data/gold_standard_annales_fr_v8_adversarial.json',
    model_id='models/embeddinggemma-simcse-ict',
)
g = r['global']
print(f'recall@1  = {g[\"recall@1\"]:.1%}')
print(f'recall@3  = {g[\"recall@3\"]:.1%}')
print(f'recall@5  = {g[\"recall@5\"]:.1%}')
print(f'recall@10 = {g[\"recall@10\"]:.1%}')
print(f'MRR       = {g[\"mrr\"]:.3f}')
print()
print(f'G6 rollback (>=60.1%): {\"PASS\" if g[\"recall@5\"] >= 0.601 else \"FAIL — ROLLBACK\"}'  )
print(f'G7 TFLite  (>=65.1%): {\"PASS\" if g[\"recall@5\"] >= 0.651 else \"FAIL — no TFLite conversion\"}'  )
"
```

- [ ] **Step 2: Save recall report**

```bash
cp data/benchmarks/recall_baseline.json data/benchmarks/recall_post_ict.json
```

- [ ] **Step 3: DVC snapshot**

```bash
python -m dvc add corpus/processed/corpus_v2_fr.db && python -m dvc push
```

- [ ] **Step 4: Commit**

```bash
git add data/benchmarks/recall_post_ict.json corpus/processed/corpus_v2_fr.db.dvc
git commit -m "test(training): post-ICT recall (G6/G7 evaluated)"
```

---

### Task 9: Model card + decision

**DoD:** Model card updated, CLAUDE.md cap updated, decision documented.

- [ ] **Step 1: Update model_card.json finetuning section**

Fill `null` fields with measured values from recall reports.

- [ ] **Step 2: Update CLAUDE.md cap**

Document the recall results and next step decision.

- [ ] **Step 3: Commit**

```bash
git add models/model_card.json CLAUDE.md
git commit -m "docs: SimCSE+ICT fine-tuning results and decision"
```

---

## Rollback procedures

| Situation | Action |
|-----------|--------|
| G5 FAIL (SimCSE degrades) | `dvc checkout corpus/processed/corpus_v2_fr.db.dvc` + `dvc pull` |
| G6 FAIL (ICT degrades) | Rebuild with SimCSE checkpoint only (Task 6 DB) |
| G8 FAIL (integrity) | Fix before proceeding, do NOT ship broken DB |
| Training NaN/crash | Check bf16 → fallback fp32, check LR, reduce epochs |
