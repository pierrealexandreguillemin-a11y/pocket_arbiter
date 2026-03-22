# Eval Generation Kaggle T4 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run generation eval (3 models × 298 questions) on Kaggle T4 GPU instead of CPU local, producing 3 JSON files for gates G1b/G3/G4a/G4b.

**Architecture:** Two Kaggle datasets (SFT checkpoint + eval data with scripts/GS/DB), one kernel that imports eval functions via sys.path, loops over 3 models sequentially with GPU inference, saves 3 JSON outputs.

**Tech Stack:** Kaggle T4 GPU, HuggingFace Transformers, PyTorch fp16 inference, kaggle CLI

**Spec:** `docs/superpowers/specs/2026-03-23-eval-generation-kaggle-design.md`

**Skill:** `@kaggle-deployment` — MUST follow for all push/upload operations

---

## File Map

| File | Responsibility | Created in |
|------|---------------|------------|
| `kaggle/sft-checkpoint/` | Clean SFT checkpoint (model + tokenizer, no epoch dirs) | Task 1 |
| `kaggle/sft-checkpoint/dataset-metadata.json` | Kaggle dataset metadata | Task 1 |
| `kaggle/eval-data/` | Scripts + GS + DB for Kaggle import | Task 2 |
| `kaggle/eval-data/dataset-metadata.json` | Kaggle dataset metadata | Task 2 |
| `kaggle/kernel-eval/eval_generation_kaggle.py` | Kernel orchestrator (~100 lines) | Task 3 |
| `kaggle/kernel-eval/kernel-metadata.json` | Kaggle kernel metadata | Task 3 |
| `data/benchmarks/generation_eval_base.json` | Base model eval output | Task 5 |
| `data/benchmarks/generation_eval_tapt.json` | TAPT model eval output | Task 5 |
| `data/benchmarks/generation_eval.json` | SFT model eval output | Task 5 |

---

### Task 1: Prepare and Upload SFT Checkpoint Dataset

**Files:**
- Create: `kaggle/sft-checkpoint/dataset-metadata.json`
- Copy: `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja` from `models/kaggle-sft-output/gemma-270m-cpt-sft/`

**Context:**
- Source: `models/kaggle-sft-output/gemma-270m-cpt-sft/` (11 GB with epoch checkpoints)
- We copy ONLY the final checkpoint files (~1055 MB), NOT checkpoint-102/204/306, NOT training_args.bin
- Skill @kaggle-deployment: upload as dataset, not model_source

- [ ] **Step 1: Create clean checkpoint directory**

```bash
mkdir -p kaggle/sft-checkpoint
```

- [ ] **Step 2: Copy final checkpoint files only**

```bash
cp models/kaggle-sft-output/gemma-270m-cpt-sft/model.safetensors kaggle/sft-checkpoint/
cp models/kaggle-sft-output/gemma-270m-cpt-sft/config.json kaggle/sft-checkpoint/
cp models/kaggle-sft-output/gemma-270m-cpt-sft/tokenizer.json kaggle/sft-checkpoint/
cp models/kaggle-sft-output/gemma-270m-cpt-sft/tokenizer_config.json kaggle/sft-checkpoint/
cp models/kaggle-sft-output/gemma-270m-cpt-sft/generation_config.json kaggle/sft-checkpoint/
cp models/kaggle-sft-output/gemma-270m-cpt-sft/chat_template.jinja kaggle/sft-checkpoint/
```

- [ ] **Step 3: Verify files and sizes**

```bash
ls -la kaggle/sft-checkpoint/
# Expected: model.safetensors ~1023 MB, tokenizer.json ~32 MB, others < 10 KB
# NO checkpoint-102/, checkpoint-204/, checkpoint-306/, training_args.bin
```

- [ ] **Step 4: Create dataset-metadata.json**

```json
{
  "title": "Gemma 270M SFT Checkpoint (TAPT+SFT)",
  "id": "pguillemin/gemma-270m-sft-checkpoint",
  "licenses": [{"name": "other"}]
}
```

Write to `kaggle/sft-checkpoint/dataset-metadata.json`.

- [ ] **Step 5: Upload dataset to Kaggle**

```bash
cd kaggle/sft-checkpoint && kaggle datasets create -r zip
```

Expected: Upload ~1 GB, takes 2-3 min.

- [ ] **Step 6: Gate D2 — Verify SFT dataset upload**

```bash
kaggle datasets files pguillemin/gemma-270m-sft-checkpoint
```

Expected: `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja` listed.

---

### Task 2: Prepare and Upload Eval Data Dataset

**Files:**
- Create: `kaggle/eval-data/dataset-metadata.json`
- Copy: `eval_generation.py`, `generation_prompt.py`, GS JSON, DB

**Context:**
- Contains the Python scripts imported via sys.path on Kaggle
- Contains GS + DB for context lookup
- Spec §Imports: only pure functions imported (load_*, check_citation, build_rag_prompt)

- [ ] **Step 1: Create eval-data directory**

```bash
mkdir -p kaggle/eval-data
```

- [ ] **Step 2: Copy scripts, GS, and DB**

```bash
cp scripts/training/eval_generation.py kaggle/eval-data/
cp scripts/training/generation_prompt.py kaggle/eval-data/
cp tests/data/gold_standard_annales_fr_v8_adversarial.json kaggle/eval-data/
cp corpus/processed/corpus_v2_fr.db kaggle/eval-data/
```

- [ ] **Step 3: Verify files**

```bash
ls -la kaggle/eval-data/
# Expected: eval_generation.py (~8 KB), generation_prompt.py (~1 KB),
#           gold_standard_annales_fr_v8_adversarial.json (~1.2 MB),
#           corpus_v2_fr.db (~17 MB)
```

- [ ] **Step 4: Create dataset-metadata.json**

```json
{
  "title": "Pocket Arbiter Eval Data",
  "id": "pguillemin/pocket-arbiter-eval-data",
  "licenses": [{"name": "CC0-1.0"}]
}
```

Write to `kaggle/eval-data/dataset-metadata.json`.

- [ ] **Step 5: Upload dataset to Kaggle**

```bash
cd kaggle/eval-data && kaggle datasets create -r zip
```

Expected: Upload ~18 MB, takes < 1 min.

- [ ] **Step 6: Gate D1 — Verify eval data upload**

```bash
kaggle datasets files pguillemin/pocket-arbiter-eval-data
```

Expected: 4 files listed (2 .py + 1 .json + 1 .db).

- [ ] **Step 7: Gate D3 — Verify existing datasets**

```bash
kaggle datasets files pguillemin/gemma-3-270m-it
kaggle datasets files pguillemin/gemma-270m-tapt-checkpoint
```

Expected: Both return file listings. If either fails, STOP.

---

### Task 3: Write Kernel Script

**Files:**
- Create: `kaggle/kernel-eval/eval_generation_kaggle.py`
- Create: `kaggle/kernel-eval/kernel-metadata.json`

**Context:**
- Spec §Imports: import pure functions from eval_generation + generation_prompt via sys.path
- Spec §Device handling: reimplement generate_response_gpu with .to(model.device)
- Spec §SQLite: open DB once, not per-question
- Skill @kaggle-deployment: device_map={"": 0}, fp16 for inference, --accelerator flag
- Output: 3 JSON files in /kaggle/working/

- [ ] **Step 1: Create kernel-metadata.json**

Write `kaggle/kernel-eval/kernel-metadata.json`:

```json
{
  "id": "pguillemin/pocket-arbiter-eval-generation",
  "title": "Pocket Arbiter Eval Generation (3 models)",
  "code_file": "eval_generation_kaggle.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": [
    "pguillemin/gemma-3-270m-it",
    "pguillemin/gemma-270m-tapt-checkpoint",
    "pguillemin/gemma-270m-sft-checkpoint",
    "pguillemin/pocket-arbiter-eval-data"
  ]
}
```

- [ ] **Step 2: Write the kernel script**

Write `kaggle/kernel-eval/eval_generation_kaggle.py`:

```python
"""Eval generation on Kaggle T4 — 3 models x 298 questions.

Imports eval functions from pocket-arbiter-eval-data dataset.
Loops over base, TAPT, SFT models sequentially.
Outputs 3 JSON files to /kaggle/working/.

Spec: docs/superpowers/specs/2026-03-23-eval-generation-kaggle-design.md
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sqlite3
import sys
import time

import torch

# ============================================================
# PHASE 0: Environment + Imports
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("=== PHASE 0: Environment ===")

# Gate K1: GPU check
assert torch.cuda.is_available(), "FATAL: No GPU detected"
GPU_NAME = torch.cuda.get_device_name(0)
GPU_PROPS = torch.cuda.get_device_properties(0)
GPU_VRAM_MB = GPU_PROPS.total_memory / 1024 / 1024
logger.info(
    "GPU: %s (%.0f MB, compute %d.%d)",
    GPU_NAME, GPU_VRAM_MB, GPU_PROPS.major, GPU_PROPS.minor,
)
assert GPU_VRAM_MB >= 14000, f"FATAL K1: Need >= 14 GB VRAM, got {GPU_VRAM_MB:.0f}"
compute = GPU_PROPS.major + GPU_PROPS.minor / 10
assert compute >= 7.5, f"FATAL K1: Need compute >= 7.5 (T4), got {compute}"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Resolve dataset paths (two patterns with fallback)
EVAL_DATA_CANDIDATES = [
    "/kaggle/input/pocket-arbiter-eval-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-eval-data",
]
EVAL_DATA_DIR = None
for c in EVAL_DATA_CANDIDATES:
    if os.path.isdir(c):
        EVAL_DATA_DIR = c
        break
assert EVAL_DATA_DIR is not None, f"FATAL K2: Eval data not found. Tried: {EVAL_DATA_CANDIDATES}"
logger.info("Eval data: %s", EVAL_DATA_DIR)

# Import eval functions via sys.path
sys.path.insert(0, EVAL_DATA_DIR)
from eval_generation import (  # noqa: E402
    check_citation,
    load_annales_questions,
    load_human_questions,
)
from generation_prompt import build_rag_prompt  # noqa: E402

logger.info("Imports OK from %s", EVAL_DATA_DIR)

# Resolve model paths
def resolve_model_path(slug: str) -> str:
    """Find model path in Kaggle input, testing both mount patterns."""
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/pguillemin/{slug}",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Model {slug} not found. Tried: {candidates}")


MODEL_CONFIGS = [
    {
        "name": "base",
        "slug": "gemma-3-270m-it",
        "output": "generation_eval_base.json",
    },
    {
        "name": "tapt",
        "slug": "gemma-270m-tapt-checkpoint",
        "output": "generation_eval_tapt.json",
    },
    {
        "name": "sft",
        "slug": "gemma-270m-sft-checkpoint",
        "output": "generation_eval.json",
    },
]

# Gate K2: Verify all 4 datasets are mounted
for cfg in MODEL_CONFIGS:
    path = resolve_model_path(cfg["slug"])
    logger.info("Model %s: %s", cfg["name"], path)

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ============================================================
# PHASE 1: Load data
# ============================================================

logger.info("=== PHASE 1: Load data ===")

GS_PATH = os.path.join(EVAL_DATA_DIR, "gold_standard_annales_fr_v8_adversarial.json")
DB_PATH = os.path.join(EVAL_DATA_DIR, "corpus_v2_fr.db")

assert os.path.exists(GS_PATH), f"FATAL K2: GS not found: {GS_PATH}"
assert os.path.exists(DB_PATH), f"FATAL K2: DB not found: {DB_PATH}"

human_qs = load_human_questions(GS_PATH)
annales_qs = load_annales_questions(GS_PATH)

# Gate K4
assert len(human_qs) == 34, f"FATAL K4: Expected 34 human Qs, got {len(human_qs)}"
assert len(annales_qs) == 264, f"FATAL K4: Expected 264 annales Qs, got {len(annales_qs)}"
logger.info("GS loaded: %d human, %d annales", len(human_qs), len(annales_qs))

# Gate K3: Open DB once
conn = sqlite3.connect(DB_PATH)
tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()
table_names = [t[0] for t in tables]
assert "children" in table_names, f"FATAL K3: 'children' table missing. Tables: {table_names}"
logger.info("DB opened: %s (tables: %s)", DB_PATH, table_names)


# ============================================================
# PHASE 2: Inference helpers
# ============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def generate_response_gpu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
) -> str:
    """GPU inference with device transfer and chat template."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        output_ids[0][prompt_len:], skip_special_tokens=True,
    )


def load_context_from_conn(
    conn: sqlite3.Connection, source: str, page: int,
) -> str:
    """Load chunk context using shared DB connection."""
    rows = conn.execute(
        "SELECT text FROM children WHERE source = ? AND page = ?",
        (source, page),
    ).fetchall()
    return "\n\n".join(r[0] for r in rows)


def eval_model(
    model_path: str,
    model_name: str,
    output_path: str,
    human_qs: list[dict],
    annales_qs: list[dict],
    conn: sqlite3.Connection,
) -> None:
    """Evaluate one model on all questions and save JSON."""
    logger.info("--- Evaluating: %s (%s) ---", model_name, model_path)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map={"": 0},
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("Model loaded: %.0f MB VRAM", vram)

    # Gate K5: smoke test — 1 question
    test_msgs = build_rag_prompt("Test question?", "Test context.")
    test_resp = generate_response_gpu(model, tokenizer, test_msgs)
    assert len(test_resp) > 0, f"FATAL K5: Empty response from {model_name}"
    logger.info("Smoke test OK: %d tokens", len(test_resp.split()))

    # Human questions (34)
    human_results = []
    empty_count = 0
    for i, q in enumerate(human_qs):
        prov = q.get("provenance", {})
        source = (prov.get("docs") or [""])[0]
        page = (prov.get("pages") or [0])[0]
        context = load_context_from_conn(conn, source, page)
        messages = build_rag_prompt(q["content"]["question"], context)
        response = generate_response_gpu(model, tokenizer, messages)
        if not response.strip():
            empty_count += 1
        human_results.append({
            "id": q["id"],
            "question": q["content"]["question"],
            "context": context,
            "response": response,
            "scores": {"useful": None, "faithful": None, "cited": None},
        })
        if (i + 1) % 10 == 0:
            logger.info("  [human %d/%d]", i + 1, len(human_qs))

    # Annales questions (264) — auto citation only
    cited_count = 0
    for i, q in enumerate(annales_qs):
        prov = q.get("provenance", {})
        source = (prov.get("docs") or [""])[0]
        page = (prov.get("pages") or [0])[0]
        context = load_context_from_conn(conn, source, page)
        messages = build_rag_prompt(q["content"]["question"], context)
        response = generate_response_gpu(model, tokenizer, messages)
        if not response.strip():
            empty_count += 1
        if check_citation(response, prov.get("docs", []), prov.get("pages", [])):
            cited_count += 1
        if (i + 1) % 50 == 0:
            logger.info("  [annales %d/%d]", i + 1, len(annales_qs))

    total_annales = len(annales_qs)
    cited_pct = round(100 * cited_count / total_annales, 1)

    # Gate K6: warn if too many empty responses
    total_qs = len(human_qs) + total_annales
    if empty_count > total_qs * 0.1:
        logger.warning(
            "K6 WARNING: %d/%d empty responses (%.1f%%)",
            empty_count, total_qs, 100 * empty_count / total_qs,
        )

    output = {
        "model": model_path,
        "model_name": model_name,
        "questions": human_results,
        "auto_citation": {
            "total": total_annales,
            "cited_count": cited_count,
            "cited_pct": cited_pct,
        },
        "metadata": {
            "gpu": GPU_NAME,
            "empty_responses": empty_count,
            "inference_time_min": round((time.time() - t0) / 60, 1),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(
        "Saved: %s (citation: %d/%d = %.1f%%, empties: %d, time: %.1f min)",
        output_path, cited_count, total_annales, cited_pct,
        empty_count, (time.time() - t0) / 60,
    )

    # Free VRAM
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("VRAM freed: %.0f MB remaining", vram_after)


# ============================================================
# PHASE 3: Run eval on all 3 models
# ============================================================

logger.info("=== PHASE 3: Eval loop (3 models) ===")
t_total = time.time()

for cfg in MODEL_CONFIGS:
    model_path = resolve_model_path(cfg["slug"])
    output_path = os.path.join(OUTPUT_DIR, cfg["output"])
    eval_model(model_path, cfg["name"], output_path, human_qs, annales_qs, conn)

conn.close()

# ============================================================
# PHASE 4: Summary
# ============================================================

logger.info("=== PHASE 4: Summary ===")
total_min = (time.time() - t_total) / 60
logger.info("Total time: %.1f min", total_min)

for cfg in MODEL_CONFIGS:
    output_path = os.path.join(OUTPUT_DIR, cfg["output"])
    if os.path.exists(output_path):
        data = json.load(open(output_path))
        ac = data["auto_citation"]
        meta = data.get("metadata", {})
        logger.info(
            "  %s: citation %d/%d (%.1f%%), empties: %d, time: %.1f min",
            cfg["name"], ac["cited_count"], ac["total"], ac["cited_pct"],
            meta.get("empty_responses", -1), meta.get("inference_time_min", -1),
        )
    else:
        logger.error("  %s: OUTPUT MISSING — %s", cfg["name"], output_path)

logger.info("=== DONE ===")
```

- [ ] **Step 3: Verify script syntax**

```bash
python -c "import ast; ast.parse(open('kaggle/kernel-eval/eval_generation_kaggle.py').read()); print('Syntax OK')"
```

Expected: "Syntax OK"

---

### Task 4: Push Kernel and Monitor

**Context:**
- Skill @kaggle-deployment: `--accelerator NvidiaTeslaT4` OBLIGATOIRE
- Gate D4: check no previous kernel running
- Gate D5: accelerator flag in push command

- [ ] **Step 1: Gate D4 — Check no kernel running**

```bash
kaggle kernels status pguillemin/pocket-arbiter-eval-generation 2>/dev/null || echo "New kernel — OK"
```

Expected: "New kernel — OK" or "complete" or "error". If "running" or "queued": STOP, cancel first.

- [ ] **Step 2: Push kernel with T4**

```bash
kaggle kernels push -p kaggle/kernel-eval/ --accelerator NvidiaTeslaT4
```

Expected: "Kernel version N successfully pushed"

- [ ] **Step 3: Monitor status**

```bash
kaggle kernels status pguillemin/pocket-arbiter-eval-generation
```

Poll until COMPLETE or ERROR. Expected runtime: ~15-20 min.

If ERROR:
```bash
kaggle kernels output pguillemin/pocket-arbiter-eval-generation -p /tmp/eval-logs
cat /tmp/eval-logs/*.log
```
Read COMPLETE log. Diagnose root cause. Fix. Re-push. Do NOT retry blindly.

- [ ] **Step 4: Verify slug after first push**

Check if Kaggle renamed the slug. If so, update `kernel-metadata.json` `id` field.

```bash
kaggle kernels list --mine --search eval-generation
```

---

### Task 5: Download and Validate Outputs

**Files:**
- Create: `data/benchmarks/generation_eval_base.json`
- Create: `data/benchmarks/generation_eval_tapt.json`
- Create: `data/benchmarks/generation_eval.json`

- [ ] **Step 1: Download outputs**

```bash
kaggle kernels output pguillemin/pocket-arbiter-eval-generation -p data/benchmarks/
```

- [ ] **Step 2: Gate P1 — Verify 3 JSON files exist and parse**

```bash
python -c "
import json, os
files = ['generation_eval_base.json', 'generation_eval_tapt.json', 'generation_eval.json']
for f in files:
    path = os.path.join('data/benchmarks', f)
    assert os.path.exists(path), f'MISSING: {path}'
    data = json.load(open(path))
    print(f'{f}: OK ({len(data[\"questions\"])} questions)')
print('Gate P1: PASS')
"
```

Expected: 3 files, each with 34 questions. "Gate P1: PASS"

- [ ] **Step 3: Gate P2 — Verify responses are non-null**

```bash
python -c "
import json
files = ['generation_eval_base.json', 'generation_eval_tapt.json', 'generation_eval.json']
for f in files:
    data = json.load(open(f'data/benchmarks/{f}'))
    non_null = sum(1 for q in data['questions'] if q['response'])
    print(f'{f}: {non_null}/34 non-null responses')
    assert non_null == 34, f'FAIL P2: {34-non_null} null responses in {f}'
print('Gate P2: PASS')
"
```

- [ ] **Step 4: Gate P3 — Verify auto citation totals**

```bash
python -c "
import json
files = ['generation_eval_base.json', 'generation_eval_tapt.json', 'generation_eval.json']
for f in files:
    data = json.load(open(f'data/benchmarks/{f}'))
    ac = data['auto_citation']
    print(f'{f}: citation {ac[\"cited_count\"]}/{ac[\"total\"]} ({ac[\"cited_pct\"]}%)')
    assert ac['total'] == 264, f'FAIL P3: total={ac[\"total\"]} != 264 in {f}'
print('Gate P3: PASS')
"
```

- [ ] **Step 5: Gate P4 — Verify response lengths**

```bash
python -c "
import json
files = ['generation_eval_base.json', 'generation_eval_tapt.json', 'generation_eval.json']
for f in files:
    data = json.load(open(f'data/benchmarks/{f}'))
    lengths = [len(q['response'].split()) for q in data['questions']]
    avg = sum(lengths) / len(lengths)
    print(f'{f}: avg {avg:.0f} tokens, min {min(lengths)}, max {max(lengths)}')
    assert avg > 20, f'FAIL P4: avg {avg:.0f} tokens too short in {f}'
print('Gate P4: PASS')
"
```

- [ ] **Step 6: Quick comparison summary**

```bash
python -c "
import json
for f, name in [('generation_eval_base.json','BASE'), ('generation_eval_tapt.json','TAPT'), ('generation_eval.json','SFT')]:
    data = json.load(open(f'data/benchmarks/{f}'))
    ac = data['auto_citation']
    meta = data.get('metadata', {})
    print(f'{name}: citation {ac[\"cited_pct\"]}%, empties {meta.get(\"empty_responses\",\"?\")}, time {meta.get(\"inference_time_min\",\"?\")} min')
"
```

- [ ] **Step 7: Commit**

```bash
git add data/benchmarks/generation_eval_base.json data/benchmarks/generation_eval_tapt.json data/benchmarks/generation_eval.json kaggle/kernel-eval/ kaggle/eval-data/dataset-metadata.json kaggle/sft-checkpoint/dataset-metadata.json
git commit -m "feat(training): generation eval on Kaggle T4 — 3 models x 298Q

Base, TAPT, SFT evaluated on 34 human + 264 annales questions.
Auto citation check on annales. Gates P1-P4 PASS.
Replaces Task 5 Step 10 (CPU local -> Kaggle GPU)."
```

---

## Execution Notes

- Tasks 1 and 2 are independent — can be parallelized
- Task 3 depends on nothing (kernel code is self-contained)
- Task 4 depends on Tasks 1+2+3 (datasets + kernel must exist before push)
- Task 5 depends on Task 4 (kernel must complete before download)
- After Task 5: **out of scope for this plan**, handled by main plan Task 6:
  - Pierre does human eval, scores 34Q useful/faithful/cited
  - Gates G1b/G3/G4a/G4b evaluated
  - `models/model_card.json` updated with eval results
  - Final commit with scores
- Total estimated time: Tasks 1-3 (~15 min), Task 4 (~20 min Kaggle), Task 5 (~5 min)
