# CPT + AdaptLLM Generation Fine-tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Gemma 3 270M IT on FFE chess regulation corpus via TAPT + AdaptLLM SFT to improve RAG generation quality.

**Architecture:** Two-phase pipeline — (1) TAPT: CLM full fine-tuning on corpus paragraphs, (2) AdaptLLM: SFT on regex-mined reading comprehension exercises. Mining and eval run locally, training runs on Kaggle T4.

**Tech Stack:** HuggingFace Transformers + Trainer, Kaggle T4 16GB, tiktoken (stats), DVC (versioning)

**Spec:** `docs/superpowers/specs/2026-03-21-cpt-adaptllm-generation-design.md`

**ADR:** `docs/adr/ADR-001-generation-model-selection.md` (Option A: Gemma 3 270M IT)

---

## File Map

| File | Responsibility | Created in |
|------|---------------|------------|
| `scripts/training/mine_reading_tasks.py` | Regex mining FR, 6 task types, stats output | Task 1 |
| `scripts/training/tests/test_mine_reading.py` | Tests mining on real corpus fixtures | Task 1 |
| `scripts/training/prepare_kaggle_dataset.py` | Corpus paragraphs JSONL + copy tasks for Kaggle | Task 2 |
| `scripts/training/tests/test_prepare_kaggle.py` | Tests dataset prep | Task 2 |
| `kaggle/kernel-generation/train_generation.py` | Self-contained Kaggle kernel: TAPT + SFT | Task 3 |
| `kaggle/kernel-generation/kernel-metadata.json` | Kaggle kernel config | Task 3 |
| `kaggle/dataset-generation/dataset-metadata.json` | Kaggle dataset config | Task 2 |
| `scripts/training/eval_generation.py` | Generate on 34Q+264Q, 3 output files, citation regex | Task 4 |
| `scripts/training/tests/test_eval_generation.py` | Tests GS loading, citation regex, prompt build | Task 4 |
| `scripts/training/generation_prompt.py` | System prompt RAG + prompt builder | Task 4 |

---

### Task 1: AdaptLLM Regex Mining Script (TDD)

**Files:**
- Create: `scripts/training/mine_reading_tasks.py`
- Create: `scripts/training/tests/test_mine_reading.py`

**Context:**
- Spec §Phase 2 defines 6 task types, regex patterns, and yield expectations
- Must read from `corpus/processed/docling_v2_fr/*.json` (field: `markdown`)
- Output: `data/training/reading_tasks.jsonl` + `data/training/mining_stats.json`
- Gate G2: yield >= 500 exercises
- Connector patterns: `FR_CONNECTORS` dict from spec line 159-166
- Completion restricted to 2-sentence radius around connectors (spec line 209-211)
- Mining stats MUST include distribution by type AND by document source (spec line 213-214)

- [ ] **Step 1: Write test fixtures**

Create `scripts/training/tests/test_mine_reading.py` with fixtures containing French regulatory text with known connectors. Test each of the 6 task types individually.

```python
"""Tests for AdaptLLM regex mining on French regulatory text."""

import pytest
from scripts.training.mine_reading_tasks import (
    mine_connectors,
    mine_summarization,
    mine_completion,
    format_exercise,
    compute_mining_stats,
)

FIXTURE_TEXT = """## Chapitre 3 : Forfaits

### Section 3.1 : Principes generaux

Le forfait est prononce lorsque le joueur ne se presente pas.
Par consequent, le joueur perd la partie par defaut.

Cependant, si le joueur previent l'arbitre, un delai peut etre accorde.

En application de l'article 6.7 des Lois des Echecs, l'arbitre peut
accorder un delai supplementaire sous reserve de circonstances exceptionnelles.

De plus, le reglement prevoit que le joueur doit signer la feuille de partie.
"""

FIXTURE_SOURCE = "R01_test.json"


class TestMineConnectors:
    def test_finds_nli_consequent(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        types = [r["task_type"] for r in results]
        assert "nli_consequent" in types

    def test_finds_nli_contrast(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        types = [r["task_type"] for r in results]
        assert "nli_contrast" in types

    def test_finds_reference(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        types = [r["task_type"] for r in results]
        assert "reference" in types

    def test_finds_conditional(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        types = [r["task_type"] for r in results]
        assert "conditional" in types

    def test_finds_addition(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        types = [r["task_type"] for r in results]
        assert "addition" in types

    def test_finds_causal(self):
        text = "Le joueur est elimine car il a depasse le temps."
        results = mine_connectors(text, "test.json")
        types = [r["task_type"] for r in results]
        assert "causal" in types

    def test_exercise_has_required_fields(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        assert len(results) > 0
        ex = results[0]
        assert "messages" in ex
        assert "task_type" in ex
        assert "source" in ex
        assert len(ex["messages"]) == 2
        assert ex["messages"][0]["role"] == "user"
        assert ex["messages"][1]["role"] == "assistant"


class TestMineSummarization:
    def test_extracts_from_headings(self):
        results = mine_summarization(FIXTURE_TEXT, FIXTURE_SOURCE)
        assert len(results) >= 2  # Chapitre 3 + Section 3.1

    def test_passage_is_section_content(self):
        results = mine_summarization(FIXTURE_TEXT, FIXTURE_SOURCE)
        assert any("forfait" in r["messages"][0]["content"].lower() for r in results)


class TestMineCompletion:
    def test_only_near_connectors(self):
        results = mine_completion(FIXTURE_TEXT, FIXTURE_SOURCE)
        # Completion only near connectors, not arbitrary sentences
        assert len(results) <= 10  # bounded by connector count * radius

    def test_has_masked_sentence(self):
        results = mine_completion(FIXTURE_TEXT, FIXTURE_SOURCE)
        if results:
            assert "___" in results[0]["messages"][0]["content"]


class TestComputeMiningStats:
    def test_stats_structure(self):
        exercises = [
            {"task_type": "nli_consequent", "source": "R01.json"},
            {"task_type": "nli_contrast", "source": "R01.json"},
            {"task_type": "summarization", "source": "LA.json"},
        ]
        stats = compute_mining_stats(exercises)
        assert "total" in stats
        assert "by_type" in stats
        assert "by_source" in stats
        assert stats["total"] == 3
        assert stats["by_type"]["nli_consequent"] == 1
        assert stats["by_source"]["LA.json"] == 1

    def test_la_bias_flag(self):
        """Spec: if >80% matches from LA, stats must flag it."""
        exercises = [{"task_type": "nli", "source": "LA.json"}] * 9 + [
            {"task_type": "nli", "source": "R01.json"}
        ]
        stats = compute_mining_stats(exercises)
        assert "la_bias_pct" in stats
        assert stats["la_bias_pct"] == 90.0
        assert stats["la_bias_warning"] is True

    def test_no_bias_flag_when_balanced(self):
        exercises = [{"task_type": "nli", "source": "R01.json"}] * 5 + [
            {"task_type": "nli", "source": "LA.json"}
        ] * 3
        stats = compute_mining_stats(exercises)
        assert stats["la_bias_warning"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/training/tests/test_mine_reading.py -v`
Expected: FAIL (ImportError — module not created yet)

- [ ] **Step 3: Implement `mine_reading_tasks.py`**

Create `scripts/training/mine_reading_tasks.py` implementing:
- `mine_connectors(text, source)` → list of exercises from FR_CONNECTORS regex
- `mine_summarization(text, source)` → exercises from markdown headings
- `mine_completion(text, source)` → exercises from sentences near connectors (2-phrase radius)
- `format_exercise(task_type, user_content, assistant_content, source)` → dict with messages
- `compute_mining_stats(exercises)` → dict with total, by_type, by_source, la_bias_pct, la_bias_warning (True if >80% from LA)
- `mine_document(text, source)` → all 6 types combined for one document
- `main()` CLI: `--input`, `--output`, `--stats`

Key implementation details:
- Each connector match → extract the sentence + surrounding paragraph as passage
- Summarization: heading text = answer, section text below = passage
- Completion: mask the connector phrase with `___`, answer = original phrase
- Messages format: `[{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]`
- Stats: count by task_type and by source document
- Gate G2 check: assert total >= 500 at the end

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/training/tests/test_mine_reading.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run mining on real corpus, verify yield**

Run: `python scripts/training/mine_reading_tasks.py --input corpus/processed/docling_v2_fr --output data/training/reading_tasks.jsonl --stats data/training/mining_stats.json`
Expected: >= 500 exercises, stats file created, no errors

Verify:
```bash
python -c "
import json
d = json.load(open('data/training/mining_stats.json'))
print(f'Total: {d[\"total\"]} (gate G2: {\"PASS\" if d[\"total\"] >= 500 else \"FAIL\"})')
print(f'By type: {d[\"by_type\"]}')
print(f'LA bias: {d[\"la_bias_pct\"]:.1f}% (warning: {d[\"la_bias_warning\"]})')
# Assert no type at 0 (spec DoD)
zero_types = [t for t, c in d['by_type'].items() if c == 0]
assert not zero_types, f'FATAL: Types with 0 exercises: {zero_types}'
print('All types have >0 exercises: OK')
"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/training/mine_reading_tasks.py scripts/training/tests/test_mine_reading.py data/training/reading_tasks.jsonl data/training/mining_stats.json
git commit -m "feat(training): AdaptLLM regex mining for French regulatory reading comprehension

6 task types: NLI, causal, conditional, reference, summarization, completion.
Connector-based mining from 28 FFE PDFs. Completion restricted to 2-sentence
radius around connectors (spec deviation documented).

Standard: Cheng et al. ICLR 2024 (arXiv:2309.09530)"
```

---

### Task 2: Kaggle Dataset Preparation

**Files:**
- Create: `scripts/training/prepare_kaggle_dataset.py`
- Create: `kaggle/dataset-generation/dataset-metadata.json`

**Context:**
- Kaggle dataset must contain: `corpus_paragraphs.jsonl` (for TAPT) + `reading_tasks.jsonl` (for SFT)
- `corpus_paragraphs.jsonl`: one paragraph per line, with source metadata
- Must NOT contain the raw PDFs or full markdown (just paragraphs)
- Spec §Pipeline: mining happens locally (Step 2), dataset uploaded (Step 3)

- [ ] **Step 1: Implement `prepare_kaggle_dataset.py`**

```python
"""Prepare Kaggle dataset for generation training.

Reads docling JSONs, splits into paragraphs for TAPT corpus.
Copies reading_tasks.jsonl for SFT. Writes to kaggle/dataset-generation/.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_paragraphs(input_dir: Path) -> list[dict]:
    """Extract paragraphs from docling JSONs with source metadata."""
    paragraphs = []
    for f in sorted(input_dir.glob("*.json")):
        data = json.load(open(f, encoding="utf-8"))
        md = data.get("markdown", "")
        source = data.get("source", f.stem)
        for para in md.split("\n\n"):
            text = para.strip()
            if len(text) > 20:  # skip tiny fragments
                paragraphs.append({"text": text, "source": source})
    logger.info("Extracted %d paragraphs from %d documents", len(paragraphs), len(list(input_dir.glob("*.json"))))
    return paragraphs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to docling_v2_fr/")
    parser.add_argument("--tasks", required=True, help="Path to reading_tasks.jsonl")
    parser.add_argument("--output", required=True, help="Output dir for Kaggle dataset")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Corpus paragraphs for TAPT
    paragraphs = extract_paragraphs(Path(args.corpus))
    para_path = output / "corpus_paragraphs.jsonl"
    with open(para_path, "w", encoding="utf-8") as f:
        for p in paragraphs:
            json.dump(p, f, ensure_ascii=False)
            f.write("\n")
    logger.info("Wrote %d paragraphs to %s", len(paragraphs), para_path)

    # Copy reading tasks for SFT
    tasks_src = Path(args.tasks)
    tasks_dst = output / "reading_tasks.jsonl"
    shutil.copy2(tasks_src, tasks_dst)
    task_count = sum(1 for _ in open(tasks_dst, encoding="utf-8"))
    logger.info("Copied %d reading tasks to %s", task_count, tasks_dst)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `kaggle/dataset-generation/dataset-metadata.json`**

```json
{
  "title": "Pocket Arbiter Generation Training Data",
  "id": "pguillemin/pocket-arbiter-gen-data",
  "licenses": [{"name": "CC0-1.0"}]
}
```

- [ ] **Step 3: Write tests `scripts/training/tests/test_prepare_kaggle.py`**

```python
"""Tests for Kaggle dataset preparation."""

import json
import tempfile
from pathlib import Path

from scripts.training.prepare_kaggle_dataset import extract_paragraphs


def test_extract_paragraphs_splits_on_double_newline(tmp_path):
    doc = {"markdown": "First paragraph.\n\nSecond paragraph.\n\nThird.", "source": "test.pdf"}
    (tmp_path / "test.json").write_text(json.dumps(doc), encoding="utf-8")
    result = extract_paragraphs(tmp_path)
    # "Third." is < 20 chars, should be filtered
    assert len(result) == 2
    assert result[0]["source"] == "test.pdf"


def test_extract_paragraphs_skips_tiny_fragments(tmp_path):
    doc = {"markdown": "Short.\n\nA long enough paragraph with real content.", "source": "t.pdf"}
    (tmp_path / "t.json").write_text(json.dumps(doc), encoding="utf-8")
    result = extract_paragraphs(tmp_path)
    assert len(result) == 1  # "Short." filtered out (< 20 chars)


def test_extract_paragraphs_preserves_source(tmp_path):
    doc = {"markdown": "A paragraph with enough text here.", "source": "R01.pdf"}
    (tmp_path / "r01.json").write_text(json.dumps(doc), encoding="utf-8")
    result = extract_paragraphs(tmp_path)
    assert result[0]["source"] == "R01.pdf"
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest scripts/training/tests/test_prepare_kaggle.py -v`
Expected: ALL PASS

- [ ] **Step 5: Test on real corpus**

Run: `python scripts/training/prepare_kaggle_dataset.py --corpus corpus/processed/docling_v2_fr --tasks data/training/reading_tasks.jsonl --output kaggle/dataset-generation/`
Expected: `corpus_paragraphs.jsonl` + `reading_tasks.jsonl` in output dir

Verify: `wc -l kaggle/dataset-generation/*.jsonl` — both files non-empty

- [ ] **Step 6: Commit**

```bash
git add scripts/training/prepare_kaggle_dataset.py scripts/training/tests/test_prepare_kaggle.py kaggle/dataset-generation/dataset-metadata.json
git commit -m "feat(training): Kaggle dataset prep for generation training

Extracts corpus paragraphs + copies mining tasks for upload."
```

---

### Task 3: Kaggle Training Kernel (TAPT + SFT)

**Files:**
- Create: `kaggle/kernel-generation/train_generation.py`
- Create: `kaggle/kernel-generation/kernel-metadata.json`

**Context:**
- Self-contained script: all config inline (Kaggle can't import local modules)
- Phase 1 (TAPT): CLM full FT on corpus_paragraphs.jsonl, 5 epochs, save checkpoint
- Phase 2 (SFT): SFT on reading_tasks.jsonl using Gemma IT chat template, 3 epochs, save checkpoint
- Both checkpoints saved to `/kaggle/working/` for download
- All validations from spec §Validations Phase 1 and Phase 2 must be included
- Dataset path: try `/kaggle/input/pocket-arbiter-gen-data/` with fallback
- Packages: pip install tiktoken (stats only)
- fp16 via TrainingArguments(fp16=True) — uses AMP internally
- NEFTune: neftune_noise_alpha=5
- max_grad_norm=1.0
- save_total_limit=2 per stage
- Check and inject dropout if Gemma 3 config lacks attention_dropout

- [ ] **Step 1: Create `kernel-metadata.json`**

```json
{
  "id": "pguillemin/pocket-arbiter-cpt-generation",
  "title": "Pocket Arbiter CPT+SFT Generation",
  "code_file": "train_generation.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": [
    "pguillemin/pocket-arbiter-gen-data"
  ],
  "keywords": [
    "pocket-arbiter", "gemma", "cpt", "tapt", "adaptllm",
    "sft", "generation", "chess", "full-finetuning"
  ]
}
```

- [ ] **Step 2: Implement `train_generation.py` — Phase 0-1 (environment + config)**

```python
"""TAPT + SFT generation fine-tuning for Gemma 3 270M IT.

Kaggle T4 script — self-contained, production-grade.

Input:  /kaggle/input/pocket-arbiter-gen-data/{corpus_paragraphs,reading_tasks}.jsonl
Output: /kaggle/working/{gemma-270m-cpt,gemma-270m-cpt-sft}/
        /kaggle/working/tapt_perplexity.json

Standards:
    TAPT: Gururangan et al. ACL 2020
    AdaptLLM: Cheng et al. ICLR 2024
    FFT > LoRA for CPT: Biderman et al. TMLR 2024
    NEFTune: Jain et al. arXiv:2310.05914
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

# ============================================================
# PHASE 0: Environment
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

logger.info("=== PHASE 0: Environment ===")

# GPU validation
assert torch.cuda.is_available(), "FATAL: No GPU detected"
GPU_NAME = torch.cuda.get_device_name(0)
GPU_PROPS = torch.cuda.get_device_properties(0)
GPU_VRAM_MB = GPU_PROPS.total_memory / 1024 / 1024
logger.info("GPU: %s (%.0f MB VRAM, compute %d.%d)", GPU_NAME, GPU_VRAM_MB, GPU_PROPS.major, GPU_PROPS.minor)
assert GPU_VRAM_MB >= 14000, f"FATAL: Need >= 14 GB VRAM, got {GPU_VRAM_MB:.0f} MB"
assert GPU_PROPS.major >= 7, f"FATAL: Need compute >= 7.0 (fp16 AMP), got {GPU_PROPS.major}.{GPU_PROPS.minor}"

# Install missing deps (pinned for reproducibility)
DEPS = ["tiktoken==0.9.0", "trl==0.16.0"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + DEPS)

# HuggingFace auth for gated Gemma 3 model (CRITICAL: secret must be set in Kaggle UI)
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    logger.info("HF_TOKEN loaded from Kaggle secrets")
except Exception:
    logger.warning("No Kaggle secrets available — using env HF_TOKEN or ~/.huggingface/token")

import tiktoken  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ============================================================
# PHASE 1: Config
# ============================================================

SEED = 42
MODEL_ID = "google/gemma-3-270m-it"
DRY_RUN = "--dry-run" in sys.argv

# Debug: list what Kaggle mounted (invaluable for dataset path debugging)
logger.info("=== /kaggle/input/ contents ===")
if os.path.isdir("/kaggle/input"):
    for entry in sorted(os.listdir("/kaggle/input")):
        entry_path = os.path.join("/kaggle/input", entry)
        if os.path.isdir(entry_path):
            files = os.listdir(entry_path)
            logger.info("  %s/ (%d files: %s)", entry, len(files), files[:5])
        else:
            logger.info("  %s", entry)
else:
    logger.info("  /kaggle/input does NOT exist (local mode)")

# Dataset path (Kaggle mount with fallback)
INPUT_CANDIDATES = [
    "/kaggle/input/pocket-arbiter-gen-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-gen-data",
    "kaggle/dataset-generation",  # local fallback for dry-run
]
INPUT_DIR = None
for candidate in INPUT_CANDIDATES:
    if os.path.isdir(candidate):
        INPUT_DIR = candidate
        break
assert INPUT_DIR is not None, f"FATAL: No input dir found. Tried: {INPUT_CANDIDATES}"

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "models"
TAPT_CHECKPOINT = os.path.join(OUTPUT_DIR, "gemma-270m-cpt")
SFT_CHECKPOINT = os.path.join(OUTPUT_DIR, "gemma-270m-cpt-sft")
PERPLEXITY_PATH = os.path.join(OUTPUT_DIR, "tapt_perplexity.json")

CORPUS_PATH = os.path.join(INPUT_DIR, "corpus_paragraphs.jsonl")
TASKS_PATH = os.path.join(INPUT_DIR, "reading_tasks.jsonl")

# Holdout docs for TAPT eval (by source, NOT by paragraph — prevent data leakage)
EVAL_HOLDOUT_SOURCES = {"H01_2025_26_Conduite_pour_joueur_handicapes.pdf",
                         "C04_2025_26_Coupe_de_la_parité.pdf",
                         "E02-Le_classement_rapide.pdf"}

TAPT_CONFIG = {
    "epochs": 1 if DRY_RUN else 5,
    "batch_size": 1 if DRY_RUN else 4,
    "grad_accum": 1 if DRY_RUN else 4,
    "lr": 5e-6,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "neftune_alpha": 5,
    "dropout": 0.1,
    "seq_length": 512 if DRY_RUN else 2048,
    "save_total_limit": 2,
}

SFT_CONFIG = {
    "epochs": 1 if DRY_RUN else 3,
    "batch_size": 1 if DRY_RUN else 4,
    "grad_accum": 1 if DRY_RUN else 4,
    "lr": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "neftune_alpha": 5,
    "save_total_limit": 2,
    "seq_length": 512 if DRY_RUN else 2048,
}

logger.info("Config: DRY_RUN=%s, INPUT_DIR=%s, OUTPUT_DIR=%s", DRY_RUN, INPUT_DIR, OUTPUT_DIR)

t_start = time.time()
```

- [ ] **Step 3: Implement Phase 2-3 (data + model loading with all validations)**

```python
# ============================================================
# PHASE 2: Data loading + validation
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(SEED)
logger.info("=== PHASE 2: Data loading ===")

# Load corpus paragraphs
assert os.path.exists(CORPUS_PATH), f"FATAL: Missing {CORPUS_PATH}"
with open(CORPUS_PATH, encoding="utf-8") as f:
    paragraphs = [json.loads(line) for line in f]
assert len(paragraphs) > 0, "FATAL: Empty corpus"

# Split train/eval by SOURCE document (NOT by paragraph — data leakage prevention)
train_paras = [p for p in paragraphs if p["source"] not in EVAL_HOLDOUT_SOURCES]
eval_paras = [p for p in paragraphs if p["source"] in EVAL_HOLDOUT_SOURCES]
logger.info("TAPT split: %d train / %d eval paragraphs (by doc, holdout: %s)",
            len(train_paras), len(eval_paras), EVAL_HOLDOUT_SOURCES)
assert len(eval_paras) > 0, f"FATAL: No eval paragraphs for sources {EVAL_HOLDOUT_SOURCES}"

# Load reading tasks
assert os.path.exists(TASKS_PATH), f"FATAL: Missing {TASKS_PATH}"
with open(TASKS_PATH, encoding="utf-8") as f:
    reading_tasks = [json.loads(line) for line in f]
assert len(reading_tasks) >= 500, f"FATAL: Gate G2 FAIL — only {len(reading_tasks)} tasks (need >= 500)"
logger.info("SFT tasks: %d (gate G2 PASS)", len(reading_tasks))

# Load tokenizer + validate
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
assert tokenizer.eos_token == "<eos>", f"FATAL: eos_token = '{tokenizer.eos_token}', expected '<eos>'"
assert tokenizer.eos_token_id == 1, f"FATAL: eos_token_id = {tokenizer.eos_token_id}, expected 1"
logger.info("Tokenizer: vocab=%d, eos='%s' (ID %d) — VERIFIED", tokenizer.vocab_size, tokenizer.eos_token, tokenizer.eos_token_id)

# Count tokens with GEMMA tokenizer (not tiktoken)
gemma_tokens = sum(len(tokenizer.encode(p["text"])) for p in paragraphs)
tik_enc = tiktoken.get_encoding("cl100k_base")
tiktoken_tokens = sum(len(tik_enc.encode(p["text"])) for p in paragraphs)
logger.info("Corpus tokens: %d (Gemma) / %d (tiktoken) — ratio %.2f", gemma_tokens, tiktoken_tokens, gemma_tokens / tiktoken_tokens)
assert gemma_tokens >= 300_000, f"FATAL: Gate G0 FAIL — {gemma_tokens} Gemma tokens < 300K"

# ============================================================
# PHASE 3: Model loading + validation
# ============================================================

logger.info("=== PHASE 3: Model loading ===")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

# Architecture validation
config = model.config
total_params = sum(p.numel() for p in model.parameters())
logger.info("Architecture: %d layers, hidden=%d, params=%d (%.0fM)",
            config.num_hidden_layers, config.hidden_size, total_params, total_params / 1e6)
assert 250_000_000 < total_params < 350_000_000, f"FATAL: Expected ~270M params, got {total_params}"

# Inject dropout if absent (spec: verify attention_dropout in config)
if not hasattr(config, "attention_dropout") or config.attention_dropout == 0.0:
    logger.warning("Gemma config has no attention_dropout — injecting 0.1")
    config.attention_dropout = TAPT_CONFIG["dropout"]
    # Reload model with updated config to apply dropout
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=config, torch_dtype=torch.float16, device_map="auto")
else:
    logger.info("attention_dropout = %.2f (native)", config.attention_dropout)

# Verify dropout layers actually exist in the model
dropout_layers = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout)]
logger.info("Dropout layers in model: %d", len(dropout_layers))
if len(dropout_layers) == 0:
    logger.warning("No Dropout layers found — config.attention_dropout may have no effect. "
                    "This is expected if Gemma 3 architecture hard-codes no dropout. "
                    "NEFTune + weight_decay remain as regularization.")

# VRAM after load
vram_used = torch.cuda.memory_allocated() / 1024 / 1024
logger.info("VRAM after model load: %.0f / %.0f MB (%.1f%%)", vram_used, GPU_VRAM_MB, 100 * vram_used / GPU_VRAM_MB)
assert vram_used < GPU_VRAM_MB * 0.8, f"FATAL: VRAM {vram_used:.0f} MB > 80% of {GPU_VRAM_MB:.0f} MB"
```

- [ ] **Step 4: Implement Phase 4 (TAPT training)**

```python
# ============================================================
# PHASE 4: TAPT training
# ============================================================

logger.info("=== PHASE 4: TAPT ===")

# Shuffle paragraphs (debias LA 63%)
random.shuffle(train_paras)

# Concatenate with eos_token, pack into sequences
def pack_sequences(paragraphs, tokenizer, seq_length):
    """Pack paragraphs into fixed-length sequences separated by eos_token."""
    all_ids = []
    for p in paragraphs:
        ids = tokenizer.encode(p["text"], add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id)
    # Pack into chunks of seq_length
    sequences = []
    for i in range(0, len(all_ids) - seq_length, seq_length):
        sequences.append(all_ids[i:i + seq_length])
    return sequences

train_seqs = pack_sequences(train_paras, tokenizer, TAPT_CONFIG["seq_length"])
eval_seqs = pack_sequences(eval_paras, tokenizer, TAPT_CONFIG["seq_length"])
logger.info("TAPT sequences: %d train, %d eval (seq_len=%d)", len(train_seqs), len(eval_seqs), TAPT_CONFIG["seq_length"])

# Do NOT set labels — DataCollatorForLanguageModeling handles label creation
# (clones input_ids, shifts for CLM). Setting both causes ambiguity.
train_dataset = Dataset.from_dict({"input_ids": train_seqs})
eval_dataset = Dataset.from_dict({"input_ids": eval_seqs})

# Baseline perplexity (BEFORE training)
model.eval()
with torch.no_grad():
    eval_losses = []
    for seq in eval_seqs[:min(len(eval_seqs), 20)]:
        input_ids = torch.tensor([seq], device=model.device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        eval_losses.append(outputs.loss.item())
    baseline_ppl = math.exp(sum(eval_losses) / len(eval_losses))
logger.info("Baseline perplexity: %.2f", baseline_ppl)

# Training
tapt_args = TrainingArguments(
    output_dir=TAPT_CHECKPOINT,
    num_train_epochs=TAPT_CONFIG["epochs"],
    per_device_train_batch_size=TAPT_CONFIG["batch_size"],
    per_device_eval_batch_size=TAPT_CONFIG["batch_size"],
    gradient_accumulation_steps=TAPT_CONFIG["grad_accum"],
    learning_rate=TAPT_CONFIG["lr"],
    warmup_ratio=TAPT_CONFIG["warmup_ratio"],
    weight_decay=TAPT_CONFIG["weight_decay"],
    max_grad_norm=TAPT_CONFIG["max_grad_norm"],
    neftune_noise_alpha=TAPT_CONFIG["neftune_alpha"],
    fp16=True,
    seed=SEED,
    logging_steps=1,
    logging_nan_inf_filter=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=TAPT_CONFIG["save_total_limit"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

t0 = time.time()
trainer = Trainer(
    model=model,
    args=tapt_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
train_result = trainer.train()
tapt_time = time.time() - t0
logger.info("TAPT training: %.1f min, final loss: %.4f", tapt_time / 60, train_result.training_loss)

# Post-TAPT perplexity
model.eval()
with torch.no_grad():
    post_losses = []
    for seq in eval_seqs[:min(len(eval_seqs), 20)]:
        input_ids = torch.tensor([seq], device=model.device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        post_losses.append(outputs.loss.item())
    tapt_ppl = math.exp(sum(post_losses) / len(post_losses))
logger.info("TAPT perplexity: %.2f (baseline: %.2f, delta: %.2f)", tapt_ppl, baseline_ppl, tapt_ppl - baseline_ppl)

# Gate G1: perplexity must decrease
g1_pass = tapt_ppl < baseline_ppl
logger.info("Gate G1: %s (%.2f < %.2f)", "PASS" if g1_pass else "FAIL", tapt_ppl, baseline_ppl)

# Save perplexity results as retrievable artefact
perplexity_data = {
    "baseline_perplexity": baseline_ppl,
    "tapt_perplexity": tapt_ppl,
    "delta": tapt_ppl - baseline_ppl,
    "gate_g1": "PASS" if g1_pass else "FAIL",
    "epochs": TAPT_CONFIG["epochs"],
    "train_sequences": len(train_seqs),
    "eval_sequences": len(eval_seqs),
    "training_time_min": tapt_time / 60,
}
with open(PERPLEXITY_PATH, "w") as f:
    json.dump(perplexity_data, f, indent=2)
logger.info("Perplexity saved: %s", PERPLEXITY_PATH)

# Save TAPT checkpoint
model.save_pretrained(TAPT_CHECKPOINT)
tokenizer.save_pretrained(TAPT_CHECKPOINT)
logger.info("TAPT checkpoint saved: %s", TAPT_CHECKPOINT)

# Post-training validation: generate 5 test responses
model.eval()
test_prompts = [
    "Que se passe-t-il si un joueur arrive en retard ?",
    "Quel est le role de l'arbitre principal ?",
    "Comment fonctionne le departage ?",
    "Quelles sont les cadences en parties rapides ?",
    "Un joueur peut-il utiliser son telephone ?",
]
logger.info("Post-TAPT generation test:")
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    logger.info("  Q: %s", prompt)
    logger.info("  A: %s", response[:100])

vram_after_tapt = torch.cuda.memory_allocated() / 1024 / 1024
logger.info("VRAM after TAPT: %.0f MB", vram_after_tapt)
```

- [ ] **Step 5: Implement Phase 5-6 (SFT + output validation)**

```python
# ============================================================
# PHASE 5: SFT training
# ============================================================

logger.info("=== PHASE 5: SFT ===")

# Use TRL SFTTrainer for proper label masking (user tokens masked, only assistant tokens trained)
# pip install trl was added to DEPS at Phase 0
from trl import SFTTrainer, SFTConfig as TRLSFTConfig  # noqa: E402

# Format reading tasks as chat messages
sft_messages = [task["messages"] for task in reading_tasks]
sft_dataset = Dataset.from_dict({"messages": sft_messages})

# Split SFT eval (10%, shuffle with seed for reproducibility)
sft_dataset = sft_dataset.shuffle(seed=SEED)
sft_split = sft_dataset.train_test_split(test_size=0.1, seed=SEED)
sft_train = sft_split["train"]
sft_eval = sft_split["test"]
logger.info("SFT split: %d train / %d eval", len(sft_train), len(sft_eval))

sft_args = TRLSFTConfig(
    output_dir=SFT_CHECKPOINT,
    num_train_epochs=SFT_CONFIG["epochs"],
    per_device_train_batch_size=SFT_CONFIG["batch_size"],
    per_device_eval_batch_size=SFT_CONFIG["batch_size"],
    gradient_accumulation_steps=SFT_CONFIG["grad_accum"],
    learning_rate=SFT_CONFIG["lr"],
    warmup_ratio=SFT_CONFIG["warmup_ratio"],
    weight_decay=SFT_CONFIG["weight_decay"],
    max_grad_norm=SFT_CONFIG["max_grad_norm"],
    neftune_noise_alpha=SFT_CONFIG["neftune_alpha"],
    fp16=True,
    seed=SEED,
    logging_steps=1,
    logging_nan_inf_filter=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=SFT_CONFIG["save_total_limit"],
    max_length=SFT_CONFIG.get("seq_length", 2048),
    # SFTTrainer handles chat template + label masking automatically
    # User tokens are masked (label=-100), only assistant tokens are trained
)

t0 = time.time()
sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_args,
    train_dataset=sft_train,
    eval_dataset=sft_eval,
)
sft_result = sft_trainer.train()
sft_time = time.time() - t0
logger.info("SFT training: %.1f min, final loss: %.4f", sft_time / 60, sft_result.training_loss)

# Save SFT checkpoint
model.save_pretrained(SFT_CHECKPOINT)
tokenizer.save_pretrained(SFT_CHECKPOINT)
logger.info("SFT checkpoint saved: %s", SFT_CHECKPOINT)

# Post-SFT generation test
logger.info("Post-SFT generation test:")
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    logger.info("  Q: %s", prompt)
    logger.info("  A: %s", response[:100])

# ============================================================
# PHASE 6: Output validation
# ============================================================

logger.info("=== PHASE 6: Output validation ===")
for ckpt_path, name in [(TAPT_CHECKPOINT, "TAPT"), (SFT_CHECKPOINT, "SFT")]:
    assert os.path.isdir(ckpt_path), f"FATAL: {name} checkpoint missing: {ckpt_path}"
    total_files = sum(len(files) for _, _, files in os.walk(ckpt_path))
    total_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(ckpt_path) for f in fns) / 1024 / 1024
    logger.info("%s: %d files, %.0f MB", name, total_files, total_size)
    assert total_size > 100, f"FATAL: {name} checkpoint too small ({total_size:.0f} MB)"

assert os.path.exists(PERPLEXITY_PATH), f"FATAL: Perplexity file missing: {PERPLEXITY_PATH}"

total_time = time.time() - t_start if 't_start' in dir() else 0
logger.info("=" * 60)
logger.info("TRAINING COMPLETE — %.1f min total", total_time / 60)
logger.info("TAPT: %s (%.1f min)", TAPT_CHECKPOINT, tapt_time / 60)
logger.info("SFT: %s (%.1f min)", SFT_CHECKPOINT, sft_time / 60)
logger.info("Perplexity: %s", PERPLEXITY_PATH)
logger.info("Gate G1: %s", perplexity_data["gate_g1"])
logger.info("GPU: %s", GPU_NAME)
logger.info("=" * 60)
```

Note: wrap the main block with `t_start = time.time()` at the top and `if __name__ == "__main__":` guard.

- [ ] **Step 6: Dry-run test locally (CPU, 1 step)**

The `--dry-run` flag (already in config) sets:
- epochs=1, batch_size=1, grad_accum=1, seq_length=512
- Uses local `kaggle/dataset-generation/` as input fallback

Run: `python kaggle/kernel-generation/train_generation.py --dry-run`
Expected output:
- "DRY_RUN=True"
- Model loads (downloads ~500MB first time)
- 1 training step for TAPT, 1 for SFT
- Both checkpoints created (small)
- No CUDA errors (runs on CPU if no GPU, but asserts will fail — for local test, temporarily comment GPU assert or run with `CUDA_VISIBLE_DEVICES=` to simulate)

If no local GPU: validate data flow only by running the data loading + validation phases up to Phase 3, then skip training.

- [ ] **Step 7: Commit**

```bash
git add kaggle/kernel-generation/
git commit -m "feat(training): Kaggle kernel for TAPT+SFT generation training

Gemma 3 270M IT, full fine-tuning, fp16 AMP. TAPT 5 epochs CLM + SFT 3 epochs
on AdaptLLM-mined reading comprehension. Self-contained for Kaggle T4.

Standards: TAPT (ACL 2020), AdaptLLM (ICLR 2024), NEFTune, Biderman TMLR 2024"
```

---

### Task 4: System Prompt + Evaluation Script (TDD)

**Files:**
- Create: `scripts/training/generation_prompt.py`
- Create: `scripts/training/eval_generation.py`
- Create: `scripts/training/tests/test_eval_generation.py`

**Context:**
- Runs 3 models: base, TAPT-only, TAPT+SFT (spec §Ordre d'eval obligatoire)
- Produces 3 separate output files (spec §Artefacts):
  - `generation_eval_base.json` — base model responses
  - `generation_eval_tapt.json` — TAPT-only responses (gate G1b diagnostic)
  - `generation_eval.json` — TAPT+SFT responses + human scores
- Auto citation check on 264 annales (gate G4b)
- System prompt for RAG must be defined explicitly

- [ ] **Step 1: Create `generation_prompt.py` — system prompt for RAG**

```python
"""RAG system prompt for Pocket Arbiter generation model.

Used at inference time AND for eval. Single source of truth.
"""

SYSTEM_PROMPT = """Tu es un assistant pour arbitres d'echecs. Tu reponds aux questions
en te basant UNIQUEMENT sur le contexte fourni (extraits des reglements FFE/FIDE).

REGLES:
- Cite TOUJOURS le document source et l'article/section concerne.
- Si le contexte ne contient pas la reponse, dis clairement "Information non trouvee
  dans les extraits fournis."
- Ne reponds JAMAIS avec des informations qui ne sont pas dans le contexte.
- Reponds en francais.
- Sois concis et actionnable (l'arbitre a besoin d'une decision rapide)."""


def build_rag_prompt(question: str, context: str) -> list[dict]:
    """Build chat messages for RAG inference.

    Returns list of messages for tokenizer.apply_chat_template().
    """
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nContexte:\n{context}\n\nQuestion: {question}"},
    ]
```

- [ ] **Step 2: Write tests `scripts/training/tests/test_eval_generation.py`**

```python
"""Tests for generation evaluation."""

import json
import re

import pytest

from scripts.training.eval_generation import (
    check_citation,
    load_human_questions,
    load_annales_questions,
)
from scripts.training.generation_prompt import build_rag_prompt, SYSTEM_PROMPT


class TestBuildRagPrompt:
    def test_includes_system_prompt(self):
        messages = build_rag_prompt("Quelle cadence ?", "Art 5.1 dit 90 min.")
        assert SYSTEM_PROMPT in messages[0]["content"]

    def test_includes_question_and_context(self):
        messages = build_rag_prompt("Ma question", "Mon contexte")
        assert "Ma question" in messages[0]["content"]
        assert "Mon contexte" in messages[0]["content"]

    def test_returns_valid_messages_format(self):
        messages = build_rag_prompt("Q", "C")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestCheckCitation:
    def test_detects_document_mention(self):
        response = "Selon le Livre de l'Arbitre, article 6.2, page 45..."
        assert check_citation(response, ["LA-octobre2025.pdf"], [45])

    def test_rejects_no_citation(self):
        response = "Le joueur doit se presenter a l'heure."
        assert not check_citation(response, ["LA-octobre2025.pdf"], [45])

    def test_detects_source_name_variants(self):
        response = "D'apres le reglement general (R01), section 3..."
        assert check_citation(response, ["R01_2025_26_Regles_generales.pdf"], [3])

    def test_detects_page_number(self):
        response = "Voir page 185 du Livre de l'Arbitre."
        assert check_citation(response, ["LA-octobre2025.pdf"], [185])


class TestLoadQuestions:
    def test_load_human_questions(self):
        qs = load_human_questions("tests/data/gold_standard_annales_fr_v8_adversarial.json")
        assert len(qs) == 34  # 40 ffe:human:* total, 6 are is_impossible=True
        assert all(q["id"].startswith("ffe:human:") for q in qs)
        assert all(not q.get("content", {}).get("is_impossible", False) for q in qs)

    def test_load_annales_questions(self):
        qs = load_annales_questions("tests/data/gold_standard_annales_fr_v8_adversarial.json")
        assert len(qs) == 264
        assert all(not q.get("content", {}).get("is_impossible", False) for q in qs)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest scripts/training/tests/test_eval_generation.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 4: Implement `eval_generation.py`**

Key functions:
- `load_human_questions(gs_path)` → 34 questions where `id.startswith("ffe:human:")` AND `not is_impossible`
- `load_annales_questions(gs_path)` → 264 answerable annales (with `annales_source`, not `is_impossible`)
- `load_chunk_context(db_path, source, page)` → chunk text by source+page (NOT chunk_id — format mismatch v1 vs v2)
- `check_citation(response, expected_docs, expected_pages)` → bool

**ATTENTION chunk_id mismatch** : le GS stocke des chunk_ids format v1
(`LA-octobre2025.pdf-p185-parent545-child00`) qui N'EXISTENT PAS dans la DB v2
(format `LA-octobre2025.pdf-c0430`). Le lookup DOIT utiliser `provenance.docs[0]` +
`provenance.pages[0]` contre les colonnes `source` et `page` de la table `children`.
Plusieurs children peuvent matcher la meme page — les concatener comme contexte.

```python
def load_chunk_context(db_path: str, source: str, page: int) -> str:
    """Load all children from source+page as context.

    GS chunk_ids are v1 format, incompatible with v2 DB.
    Fallback: lookup by source+page, concatenate matching children.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT text FROM children WHERE source = ? AND page = ? ORDER BY id",
        (source, page),
    ).fetchall()
    conn.close()
    return "\n\n".join(r[0] for r in rows) if rows else ""
```

Citation regex logic:
```python
def check_citation(response: str, expected_docs: list[str], expected_pages: list[int]) -> bool:
    """Check if response cites at least one expected source.

    Matches: document name variants (LA, Livre Arbitre, R01, Regles generales, etc.)
    AND/OR page numbers.
    """
    response_lower = response.lower()

    # Source name patterns (map PDF filenames to likely citation forms)
    SOURCE_PATTERNS = {
        "LA-octobre2025.pdf": r"(?:livre.{0,5}arbitre|l\.?a\.?\b|la\b.{0,10}octobre)",
        "R01": r"(?:r[eè]gles?\s+g[eé]n[eé]rales?|r\.?01)",
        "R02": r"(?:annexes?\s+aux\s+r[eè]gles|r\.?02)",
        "A01": r"(?:championnat\s+de\s+france\b|a\.?01)",
        "A02": r"(?:championnat\s+.{0,10}clubs?|a\.?02)",
    }
    doc_cited = False
    for doc in expected_docs:
        for pattern_key, pattern in SOURCE_PATTERNS.items():
            if pattern_key in doc:
                if re.search(pattern, response_lower):
                    doc_cited = True
                    break

    # Page match
    page_cited = False
    for page in expected_pages:
        if re.search(rf"\bpage\s*{page}\b", response_lower):
            page_cited = True
            break
        if re.search(rf"\bp\.?\s*{page}\b", response_lower):
            page_cited = True
            break

    return doc_cited or page_cited
```

Main workflow:
- For each model (base, TAPT, SFT), generate responses on 34 human questions
- Save each as separate file: `generation_eval_base.json`, `generation_eval_tapt.json`, `generation_eval.json`
- Run auto citation check on 264 annales for base and SFT models
- Output format per file:

```json
{
  "model": "path/or/id",
  "questions": [
    {
      "id": "ffe:human:...",
      "question": "...",
      "context": "...",
      "response": "...",
      "scores": {"useful": null, "faithful": null, "cited": null}
    }
  ],
  "auto_citation": {
    "total": 264,
    "cited_count": 42,
    "cited_pct": 15.9
  }
}
```

CLI args:
- `--model` : single model path or HuggingFace ID
- `--gs` : GS path
- `--db` : DB path
- `--output-dir` : directory for output files
- `--output-prefix` : prefix for output filename (e.g., `generation_eval_base` → `generation_eval_base.json`)
- `--max-questions` : limit for testing (default: all)

Note: the script is called 3 times (once per model), NOT once with multiple models.
Each invocation produces one output file.

**CPU inference** : ce script tourne en LOCAL sans GPU. Le modele 270M doit etre
charge en fp32 sur CPU (`device_map="cpu"`, `torch_dtype=torch.float32`).
Generation lente : ~5-10s/question, ~7 min pour 34Q, ~30 min pour 264Q. C'est voulu.

```python
# Model loading for CPU eval (NOT the Kaggle fp16 pattern)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32, device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest scripts/training/tests/test_eval_generation.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/training/generation_prompt.py scripts/training/eval_generation.py scripts/training/tests/test_eval_generation.py
git commit -m "feat(training): generation eval — 3 models, 3 output files, citation regex

System prompt defined in generation_prompt.py (single source of truth).
34Q human eval + 264Q auto citation check.
Gates G1b, G3, G4a, G4b from spec."
```

---

### Task 5: Integration — Run Full Pipeline

**Context:** All scripts exist. Now execute the pipeline end-to-end.

- [ ] **Step 1: DVC version current state**

```bash
dvc status
# If models/ has changes, dvc add + push
```

- [ ] **Step 2: Run mining (already done in Task 1 Step 5, verify)**

```bash
python -c "import json; d=json.load(open('data/training/mining_stats.json')); print(f'Total: {d[\"total\"]}, Gate G2: {\"PASS\" if d[\"total\"] >= 500 else \"FAIL\"}')"
```

- [ ] **Step 3: Prepare Kaggle dataset**

```bash
python scripts/training/prepare_kaggle_dataset.py \
  --corpus corpus/processed/docling_v2_fr \
  --tasks data/training/reading_tasks.jsonl \
  --output kaggle/dataset-generation/
```

- [ ] **Step 4: Upload dataset to Kaggle**

```bash
cd kaggle/dataset-generation && kaggle datasets create -r zip
# Or if already exists:
# kaggle datasets version -r zip -m "Update training data"
```

- [ ] **Step 5: Pre-flight Kaggle checks**

```bash
# Verify dataset uploaded
kaggle datasets files pguillemin/pocket-arbiter-gen-data
# Must show: corpus_paragraphs.jsonl, reading_tasks.jsonl

# Check kernel status — KILL if running
kaggle kernels status pguillemin/pocket-arbiter-cpt-generation 2>/dev/null || echo "New kernel"
# If "running" or "queued": KILL first (push on same slug QUEUES, not replaces)
# kaggle kernels cancel pguillemin/pocket-arbiter-cpt-generation
```

- [ ] **Step 6: Configure T4 + HF_TOKEN in Kaggle UI**

Open https://www.kaggle.com/code/pguillemin/pocket-arbiter-cpt-generation in browser.

**OBLIGATOIRE (API ne peut PAS faire ca) :**
1. Session options → Accelerator → **GPU T4** → Save
2. Add-ons → Secrets → Add secret: key=`HF_TOKEN`, value=[ton token HuggingFace avec read access]
3. Verifier visuellement que T4 est affiche dans Session options

**Pre-requis HuggingFace :**
- Accepter la licence Gemma 3 sur https://huggingface.co/google/gemma-3-270m-it
- Le token HF doit avoir les permissions `read` minimum

- [ ] **Step 7: Push kernel**

```bash
kaggle kernels push -p kaggle/kernel-generation/
```

- [ ] **Step 8: Wait for completion, check logs**

```bash
kaggle kernels status pguillemin/pocket-arbiter-cpt-generation
# Poll until COMPLETE or ERROR
# If ERROR: kaggle kernels output pguillemin/pocket-arbiter-cpt-generation -p /tmp/logs
# Read COMPLETE error log. DO NOT retry without diagnosis.
```

- [ ] **Step 9: Download outputs**

```bash
kaggle kernels output pguillemin/pocket-arbiter-cpt-generation -p models/
ls -la models/gemma-270m-cpt/ models/gemma-270m-cpt-sft/
# Both dirs must exist with model files
```

- [ ] **Step 10: Run eval on ALL 3 models (spec §Ordre d'eval obligatoire)**

```bash
# Base model (reference)
python scripts/training/eval_generation.py \
  --model google/gemma-3-270m-it \
  --gs tests/data/gold_standard_annales_fr_v8_adversarial.json \
  --db corpus/processed/corpus_v2_fr.db \
  --output-dir data/benchmarks/ \
  --output-prefix generation_eval_base

# TAPT only (gate G1b diagnostic)
python scripts/training/eval_generation.py \
  --model models/gemma-270m-cpt \
  --gs tests/data/gold_standard_annales_fr_v8_adversarial.json \
  --db corpus/processed/corpus_v2_fr.db \
  --output-dir data/benchmarks/ \
  --output-prefix generation_eval_tapt

# TAPT+SFT (gates G3/G4a/G4b)
python scripts/training/eval_generation.py \
  --model models/gemma-270m-cpt-sft \
  --gs tests/data/gold_standard_annales_fr_v8_adversarial.json \
  --db corpus/processed/corpus_v2_fr.db \
  --output-dir data/benchmarks/ \
  --output-prefix generation_eval
```

Verify: all 3 output files exist, each with 34 human questions + 264 auto citation results.

- [ ] **Step 11: DVC version checkpoints**

```bash
dvc add models/gemma-270m-cpt models/gemma-270m-cpt-sft
dvc push
git add models/gemma-270m-cpt.dvc models/gemma-270m-cpt-sft.dvc data/benchmarks/
git commit -m "feat(training): generation model checkpoints + eval results

TAPT + SFT on Gemma 3 270M IT. Perplexity and 34Q eval documented.
DVC tracked. Gates G1-G4 to be evaluated."
```

---

### Task 6: Human Evaluation + Gates

**Context:** Pierre reviews 3 eval files, fills scores, evaluates all gates.

- [ ] **Step 1: Compare base vs TAPT vs SFT side-by-side**

Open the 3 files and compare responses for the same questions:
- `data/benchmarks/generation_eval_base.json` — reference
- `data/benchmarks/generation_eval_tapt.json` — TAPT only (G1b diagnostic)
- `data/benchmarks/generation_eval.json` — TAPT+SFT (G3/G4a)

For each of the 34 human questions in `generation_eval.json`, fill `scores.useful`, `scores.faithful`, `scores.cited` (0 or 1) using the edge case rubric from the spec.

- [ ] **Step 2: Compute all gate scores**

```bash
python -c "
import json

# Gate G1b (diagnostic): TAPT-only vs base — auto citation comparison
base = json.load(open('data/benchmarks/generation_eval_base.json'))
tapt = json.load(open('data/benchmarks/generation_eval_tapt.json'))
sft = json.load(open('data/benchmarks/generation_eval.json'))

print('=== Gate G1b (TAPT diagnostic) ===')
print(f'  Base auto citation: {base[\"auto_citation\"][\"cited_count\"]}/{base[\"auto_citation\"][\"total\"]} ({base[\"auto_citation\"][\"cited_pct\"]:.1f}%)')
print(f'  TAPT auto citation: {tapt[\"auto_citation\"][\"cited_count\"]}/{tapt[\"auto_citation\"][\"total\"]} ({tapt[\"auto_citation\"][\"cited_pct\"]:.1f}%)')
print()

# Gate G3: no degradation vs base (human eval)
scored = [q for q in sft['questions'] if q['scores']['useful'] is not None]
passes = sum(1 for q in scored if all(v == 1 for v in q['scores'].values()))
total = len(scored)
pct = 100 * passes / total if total else 0
print('=== Gate G3 (no degradation) ===')
print(f'  SFT: {passes}/{total} = {pct:.0f}%')
print(f'  Compare with base responses qualitatively')
print()

# Gate G4a: quality >= 70%
print('=== Gate G4a (quality >= 70%) ===')
print(f'  Score: {pct:.0f}% — {\"PASS\" if pct >= 70 else \"FAIL\"}')
print()

# Gate G4b: auto citation >= 80%
print('=== Gate G4b (auto citation >= 80%) ===')
cited_pct = sft['auto_citation']['cited_pct']
print(f'  SFT auto citation: {sft[\"auto_citation\"][\"cited_count\"]}/{sft[\"auto_citation\"][\"total\"]} ({cited_pct:.1f}%)')
print(f'  Gate G4b: {\"PASS\" if cited_pct >= 80 else \"FAIL\"}')
"
```

- [ ] **Step 3: Document gate results and decision**

Update `models/model_card.json` section `generation_finetuning` with:
- Perplexity baseline vs TAPT (from `tapt_perplexity.json`)
- Gate G1b result (TAPT diagnostic)
- Gate G3 result (no degradation)
- Gate G4a score (human eval %)
- Gate G4b score (auto citation %)
- Decision: accept, or ADR-001 rollback to 1B

- [ ] **Step 4: Final commit**

```bash
git add data/benchmarks/generation_eval*.json models/model_card.json
git commit -m "docs(training): generation eval results — gates G1b/G3/G4a/G4b

Perplexity: [baseline] -> [tapt] (G1: [PASS/FAIL])
Human eval 34Q: [X]% (G4a: [PASS/FAIL])
Auto citation 264Q: [X]% (G4b: [PASS/FAIL])
Decision: [accept/rollback 1B per ADR-001]"
```

---

## Execution Notes

- Tasks 1-4 are independent code creation — can be parallelized
- Task 5 is sequential (depends on all scripts existing + Kaggle upload)
- Task 6 is human-in-the-loop (Pierre reviews)
- If Kaggle kernel fails: read COMPLETE log, fix root cause, increment slug, re-push
- If Gate G4a FAIL: review failure modes, decide per ADR-001
- Total estimated time: Tasks 1-4 (~2h dev), Task 5 (~1h including Kaggle wait), Task 6 (~30 min review)
