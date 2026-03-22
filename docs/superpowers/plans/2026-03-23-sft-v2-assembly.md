# SFT v2 Model Assembly — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run SFT on TAPT epoch 4 checkpoint with conservative hyperparameters (1 epoch, LR 1e-5, save_steps=20) to fix the 2-layer overfit from v1, then evaluate 3 models.

**Architecture:** Fork `train_sft.py` → `train_sft_v2.py` with 6 config changes (epochs, LR, save_strategy, save_steps, save_total_limit, output dir). Push to Kaggle T4. Download, analyze checkpoints locally, upload best to Kaggle. Re-run eval kernel unchanged.

**Tech Stack:** Python 3.10, PyTorch, TRL 0.16.0, HuggingFace Transformers, Kaggle CLI

**Spec:** `docs/superpowers/specs/2026-03-23-sft-v2-assembly-design.md`

---

### File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `kaggle/kernel-sft/train_sft_v2.py` | CREATE | SFT v2 training script (fork of train_sft.py) |
| `kaggle/kernel-sft/kernel-metadata.json` | MODIFY | Point code_file to train_sft_v2.py |

No test files — this is a Kaggle kernel (runs on remote GPU, not testable locally beyond dry-run).

---

### Task 1: Create train_sft_v2.py

**Files:**
- Create: `kaggle/kernel-sft/train_sft_v2.py`
- Reference: `kaggle/kernel-sft/train_sft.py`

- [ ] **Step 1: Copy train_sft.py to train_sft_v2.py**

```bash
cp kaggle/kernel-sft/train_sft.py kaggle/kernel-sft/train_sft_v2.py
```

- [ ] **Step 2: Update docstring (lines 1-19)**

Replace first line:
```python
"""SFT v2 generation fine-tuning for Gemma 3 270M IT (post-TAPT epoch 4).
```

Replace line 4:
```python
Loads TAPT checkpoint-88 (epoch 4) from dataset. 1 epoch, LR 1e-5, save_steps=20.
```

- [ ] **Step 3: Update SFT_CFG (line 110-122)**

Change these values in the `SFT_CFG` dict:

```python
SFT_CFG = dict(
    epochs=1 if _D else 1,           # was: 1 if _D else 3
    batch_size=1,
    grad_accum=1 if _D else 16,
    lr=1e-5,                          # was: 2e-5
    warmup_pct=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    neftune_alpha=5,
    save_total_limit=6,               # was: 3
    seq_length=512 if _D else 1024,
    eval_split=0.1,
    save_steps=20,                    # NEW
)
```

- [ ] **Step 4: Update output dir name (line 105-106)**

```python
SFT_CKPT = os.path.join(OUTPUT_DIR, "gemma-270m-cpt-sft-v2")
METRICS_PATH = os.path.join(OUTPUT_DIR, "sft_v2_metrics.json")
```

- [ ] **Step 5: Update SFTConfig to use save_steps (lines 399-422)**

Replace the save_strategy/save_total_limit section:

```python
sft_config = SFTConfig(
    output_dir=SFT_CKPT,
    num_train_epochs=S["epochs"],
    per_device_train_batch_size=S["batch_size"],
    gradient_accumulation_steps=S["grad_accum"],
    learning_rate=S["lr"],
    warmup_steps=warmup_steps,
    weight_decay=S["weight_decay"],
    max_grad_norm=S["max_grad_norm"],
    lr_scheduler_type="cosine",
    neftune_noise_alpha=S["neftune_alpha"],
    max_seq_length=S["seq_length"],
    fp16=True,
    gradient_checkpointing=True,
    seed=SEED,
    logging_steps=1,
    logging_nan_inf_filter=True,
    eval_strategy="no",
    save_strategy="steps",            # was: "epoch"
    save_steps=S["save_steps"],       # NEW: every 20 steps
    save_total_limit=S["save_total_limit"],
)
```

- [ ] **Step 6: Verify existing safeguards are present**

Confirm these lines exist unchanged in the script:
- Line 26: `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`
- Line 341: `torch_dtype=torch.float32, device_map={"": 0}`
- Line 402: `gradient_checkpointing=True`
- Line 411: `fp16=True`

- [ ] **Step 7: Run dry-run locally to verify no syntax errors**

```bash
cd C:/Dev/pocket_arbiter
python kaggle/kernel-sft/train_sft_v2.py --dry-run 2>&1 | head -30
```

Expected: script starts, loads tokenizer, creates dataset, hits GPU assert (no GPU locally) or runs with mock data.

Note: dry-run may fail at GPU assert on local machine — that's expected. We're checking for import errors and syntax errors only.

- [ ] **Step 8: Commit**

```bash
git add kaggle/kernel-sft/train_sft_v2.py
git commit -m "feat(training): SFT v2 script — 1 epoch, LR 1e-5, save_steps=20

Fork of train_sft.py with conservative hyperparameters to fix v1 overfit:
- Base: TAPT epoch 4 (checkpoint-88) instead of epoch 5
- Epochs: 1 (was 3), LR: 1e-5 (was 2e-5)
- save_strategy: steps (every 20), 5 intra-epoch checkpoints
- Output: gemma-270m-cpt-sft-v2/"
```

---

### Task 2: Update kernel-metadata.json

**Files:**
- Modify: `kaggle/kernel-sft/kernel-metadata.json`

- [ ] **Step 1: Change code_file to train_sft_v2.py**

Edit `kaggle/kernel-sft/kernel-metadata.json` line 3:

```json
"code_file": "train_sft_v2.py",
```

- [ ] **Step 2: Verify dataset_sources unchanged**

Confirm these 2 sources are present:
```json
"dataset_sources": [
    "pguillemin/pocket-arbiter-gen-data",
    "pguillemin/gemma-270m-tapt-checkpoint"
]
```

- [ ] **Step 3: Verify enable_internet is true**

Confirm: `"enable_internet": "true"` (needed for `pip install trl==0.16.0`)

- [ ] **Step 4: Commit**

```bash
git add kaggle/kernel-sft/kernel-metadata.json
git commit -m "fix(training): point kernel metadata to train_sft_v2.py"
```

---

### Task 3: Pre-push Kaggle checklist

**Files:** None (verification only)

- [ ] **Step 1: Verify TAPT dataset v2 is ready**

```bash
kaggle datasets files pguillemin/gemma-270m-tapt-checkpoint
```

Expected: 6 files listed, `model.safetensors` = 1072419256 bytes. Status = `ready`.

Note: both checkpoint-88 (epoch 4) and checkpoint-110 (epoch 5) have the same
`model.safetensors` size (1072419256 bytes). To confirm epoch 4 content, verify
the upload date is post-2026-03-23 (the date we ran `kaggle datasets version`).

- [ ] **Step 2: Verify previous SFT kernel is not running**

```bash
kaggle kernels status pguillemin/pocket-arbiter-sft-generation
```

Expected: `complete` or `error` (NOT `running` or `queued`).

- [ ] **Step 3: Verify gen-data dataset is available**

```bash
kaggle datasets files pguillemin/pocket-arbiter-gen-data
```

Expected: `reading_tasks.jsonl` present.

---

### Task 4: Push kernel SFT v2

**Files:** None (Kaggle CLI)

- [ ] **Step 1: Push with T4 accelerator**

```bash
kaggle kernels push -p kaggle/kernel-sft --accelerator NvidiaTeslaT4
```

CRITICAL: `--accelerator NvidiaTeslaT4` is MANDATORY. Without it, enable_gpu=true defaults to P100.
If `--accelerator` flag is rejected by CLI, verify GPU from kernel log after start and abort if P100.

- [ ] **Step 2: Verify kernel is queued/running**

```bash
kaggle kernels status pguillemin/pocket-arbiter-sft-generation
```

Expected: `running` or `queued`.

- [ ] **Step 3: BLOCKED — Wait for Kaggle completion (~20 min)**

Do NOT poll in a loop. Check status once, then wait for notification or check back in 5 min.
Poll:
```bash
kaggle kernels status pguillemin/pocket-arbiter-sft-generation
```

Expected: `complete`. If `error`, download log and diagnose.

- [ ] **Step 4: Download outputs**

```bash
kaggle kernels output pguillemin/pocket-arbiter-sft-generation -p models/kaggle-sft-v2-output/
```

Expected: `gemma-270m-cpt-sft-v2/` directory with 5 checkpoint subdirs + final model + `sft_v2_metrics.json`.

- [ ] **Step 5: Commit output reference**

```bash
git add -A models/kaggle-sft-v2-output/ 2>/dev/null  # gitignored, just verify
```

Note: models/ outputs are gitignored. DVC tracking happens after analysis.

---

### Task 5: Analyze checkpoints and select best

**Files:**
- Read: `models/kaggle-sft-v2-output/sft_v2_metrics.json`
- Read: `models/kaggle-sft-v2-output/gemma-270m-cpt-sft-v2/checkpoint-*/trainer_state.json`

- [ ] **Step 1: Extract loss and token_accuracy per checkpoint**

```python
import json, os

base = "models/kaggle-sft-v2-output/gemma-270m-cpt-sft-v2"
checkpoints = sorted([d for d in os.listdir(base) if d.startswith("checkpoint-")])

print("Checkpoint | Loss   | Token Acc | Delta Loss")
print("-----------|--------|-----------|----------")
prev_loss = None
for ckpt in checkpoints:
    state_path = os.path.join(base, ckpt, "trainer_state.json")
    with open(state_path) as f:
        state = json.load(f)
    last_log = [e for e in state["log_history"] if "loss" in e][-1]
    loss = last_log["loss"]
    acc = last_log.get("mean_token_accuracy", "N/A")
    delta = f"{loss - prev_loss:+.4f}" if prev_loss else "—"
    prev_loss = loss
    print(f"{ckpt:11s} | {loss:.4f} | {acc if isinstance(acc, str) else f'{acc:.4f}'} | {delta}")
```

- [ ] **Step 2: Verify gate G3 (loss decreased)**

Check that the last checkpoint loss < step 1 loss (from trainer_state.json log_history[0]).
If loss did NOT decrease → ABORT, the model did not learn.

- [ ] **Step 3: Identify best checkpoint**

Criteria (in order):
1. Min loss among checkpoints
2. If loss remounts between consecutive checkpoints → take checkpoint before remontee
3. Verify token_accuracy confirms the choice

Note: checkpoint-100 (step 100) and the final model (step 101, root dir) are 1 gradient
step apart — treat as the same candidate. If checkpoint-100 is best, use the final model's
files from the root directory (they have all standard output files in one place).

- [ ] **Step 4: Run 5-gram repetition check on best checkpoint (SKIP if no local GPU)**

```python
import json, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

best_ckpt = "models/kaggle-sft-v2-output/gemma-270m-cpt-sft-v2/checkpoint-XX"  # replace XX
model = AutoModelForCausalLM.from_pretrained(best_ckpt, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(best_ckpt)

TEST_PROMPTS = [
    "Qu'est-ce qu'un forfait en competition FFE ?",
    "Quelles sont les cadences pour le championnat de France ?",
    "Comment fonctionne le departage Buchholz ?",
    "Quel est le role de l'arbitre principal ?",
    "Quelles sont les conditions pour obtenir le titre de MI ?",
]

repetitive = 0
for p in TEST_PROMPTS:
    inputs = tokenizer(p, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    words = answer.split()
    if len(words) >= 15:
        ngrams = [" ".join(words[i:i+5]) for i in range(len(words)-4)]
        counts = Counter(ngrams)
        if counts.most_common(1)[0][1] >= 3:
            repetitive += 1
            print(f"REPETITIVE: {p[:50]} -> {answer[:100]}")

print(f"\nRepetition check: {repetitive}/5 ({100*repetitive/5:.0f}%)")
assert repetitive == 0, "FAIL: best checkpoint has repetitive outputs"
```

Note: this step requires a GPU or will be very slow on CPU (~30 min). If no local GPU,
skip — the eval kernel outputs (Task 7) will serve as the deferred degeneration check
on all 264 questions (stronger than 5 prompts anyway).

- [ ] **Step 5: Copy best checkpoint to sft-checkpoint dataset**

```bash
BEST=checkpoint-XX  # replace with actual best
SRC="models/kaggle-sft-v2-output/gemma-270m-cpt-sft-v2/$BEST"
DST="kaggle/sft-checkpoint"

cp "$SRC/model.safetensors" "$DST/"
cp "$SRC/config.json" "$DST/" 2>/dev/null || cp "models/kaggle-sft-v2-output/gemma-270m-cpt-sft-v2/config.json" "$DST/"
cp "$SRC/tokenizer.json" "$DST/" 2>/dev/null || cp "models/kaggle-sft-v2-output/gemma-270m-cpt-sft-v2/tokenizer.json" "$DST/"
cp "$SRC/tokenizer_config.json" "$DST/" 2>/dev/null || cp "models/kaggle-sft-v2-output/gemma-270m-cpt-sft-v2/tokenizer_config.json" "$DST/"
cp "models/kaggle-sft-v2-output/gemma-270m-cpt-sft-v2/generation_config.json" "$DST/"
cp "models/kaggle-sft-v2-output/gemma-270m-cpt-sft-v2/chat_template.jinja" "$DST/"
```

- [ ] **Step 6: Upload best checkpoint to Kaggle**

```bash
kaggle datasets version -p kaggle/sft-checkpoint/ -m "SFT v2 best checkpoint (step XX, loss X.XXXX)"
```

Wait for `ready` status:
```bash
kaggle datasets status pguillemin/gemma-270m-sft-checkpoint
```

- [ ] **Step 7: DVC track SFT v2 outputs**

```bash
python -m dvc add models/kaggle-sft-v2-output
python -m dvc push
```

- [ ] **Step 8: Commit analysis results**

```bash
git add models/kaggle-sft-v2-output.dvc models/.gitignore kaggle/sft-checkpoint/config.json kaggle/sft-checkpoint/tokenizer_config.json kaggle/sft-checkpoint/generation_config.json kaggle/sft-checkpoint/chat_template.jinja
git commit -m "data: SFT v2 best checkpoint (step XX) — loss X.XXXX, acc X.XX

Selected from 5 intra-epoch checkpoints. Base: TAPT epoch 4.
DVC tracked: models/kaggle-sft-v2-output/"
```

---

### Task 6: Push eval kernel

**Files:** None (Kaggle CLI, kernel unchanged)

- [ ] **Step 1: Verify SFT dataset v2 is ready**

```bash
kaggle datasets files pguillemin/gemma-270m-sft-checkpoint
```

Expected: `model.safetensors` with recent date.

- [ ] **Step 2: Verify all 4 eval datasets available**

```bash
kaggle datasets files pguillemin/gemma-3-270m-it
kaggle datasets files pguillemin/gemma-270m-tapt-checkpoint
kaggle datasets files pguillemin/gemma-270m-sft-checkpoint
kaggle datasets files pguillemin/pocket-arbiter-eval-data
```

All 4 must show files.

- [ ] **Step 3: Verify eval kernel not running**

```bash
kaggle kernels status pguillemin/pocket-arbiter-eval-generation-3-models
```

- [ ] **Step 4: Push eval kernel with T4**

```bash
kaggle kernels push -p kaggle/kernel-eval --accelerator NvidiaTeslaT4
```

- [ ] **Step 5: BLOCKED — Wait for Kaggle completion (~30 min)**

Do NOT poll in a loop. Check back in 10 min.
```bash
kaggle kernels status pguillemin/pocket-arbiter-eval-generation-3-models
```

- [ ] **Step 6: Download eval outputs**

```bash
kaggle kernels output pguillemin/pocket-arbiter-eval-generation-3-models -p data/benchmarks-v2/
```

Expected: `generation_eval_base.json`, `generation_eval_tapt.json`, `generation_eval.json`

---

### Task 7: Validate eval outputs and compare v1 vs v2

**Files:**
- Read: `data/benchmarks-v2/generation_eval_*.json`
- Read: `data/benchmarks/generation_eval_*.json` (v1 reference)

- [ ] **Step 1: Run automated analysis (same script as v1)**

```python
import json, re
from collections import Counter

def analyze(path, name):
    with open(path) as f:
        data = json.load(f)
    qs = data["questions"]
    responses = [q["response"] for q in qs]
    empty = sum(1 for r in responses if not r or not r.strip())
    lengths = [len(r.split()) for r in responses if r and r.strip()]
    repetitive = 0
    for r in responses:
        if not r or len(r.split()) < 15: continue
        words = r.split()
        ngrams = [" ".join(words[i:i+5]) for i in range(len(words)-4)]
        counts = Counter(ngrams)
        if counts and counts.most_common(1)[0][1] >= 3:
            repetitive += 1
    echo = 0
    for q in qs:
        if not q["response"]: continue
        if q["question"][:30].lower() in q["response"][:80].lower():
            echo += 1
    ac = data.get("auto_citation", {})
    print(f"{name}: empty={empty}, rep={repetitive}, echo={echo}, "
          f"med_len={sorted(lengths)[len(lengths)//2] if lengths else 0}, "
          f"cite={ac.get('cited_pct', '?')}%")

for f, n in [
    ("data/benchmarks-v2/generation_eval_base.json", "BASE v2"),
    ("data/benchmarks-v2/generation_eval_tapt.json", "TAPT v2"),
    ("data/benchmarks-v2/generation_eval.json", "SFT v2"),
    ("data/benchmarks/generation_eval.json", "SFT v1 (ref)"),
]:
    analyze(f, n)
```

- [ ] **Step 2: Verify gates P1-P4**

| Gate | Check |
|------|-------|
| P1 | 3 files, same question count |
| P2 | SFT v2 empty <= 0 |
| P3 | 264 annales in each file |
| P4 | SFT v2 avg length > 15 tokens |

- [ ] **Step 3: Compare v1 vs v2**

| Metrique | SFT v1 | SFT v2 | Pass? |
|----------|--------|--------|-------|
| Repetitions | 0% | ? | <= 0% |
| Empty | 0 | ? | <= 0 |
| Echo | 17.6% | ? | < 10% |
| Citations | 33.0% | ? | >= 30% |

- [ ] **Step 4: Update documentation**

Update `models/model_card.json`:
- Add `llmModel_sft_v2` section with new metrics
- Update `evaluation` section with v2 results

Update `docs/PROJECT_HISTORY.md`:
- Add SFT v2 results to Ere 9

Update `CLAUDE.md`:
- Update chantier 4 status

- [ ] **Step 5: Commit all results**

```bash
git add data/benchmarks-v2/ models/model_card.json docs/PROJECT_HISTORY.md CLAUDE.md
git commit -m "docs: SFT v2 eval results — compare v1 vs v2

SFT v2: 1 epoch LR 1e-5 on TAPT epoch 4 (vs v1: 3 epochs LR 2e-5 on TAPT epoch 5)
Results: [fill with actual metrics]"
```

---

### Task 8: Decision gate — eval humaine ou rollback

**Files:** None (decision)

- [ ] **Step 1: Present v2 results to user**

Show the comparison table v1 vs v2 with all metrics.

- [ ] **Step 2: Decision**

| If | Then |
|----|------|
| SFT v2 better on all metrics | Proceed to human eval (34 questions) |
| SFT v2 worse on repetitions | Try Option C (2 epochs, LR 5e-6) |
| SFT v2 worse on citations | Keep v1, model quality is fundamental 270M limit |
| All models too weak | ADR-001 gate: consider Gemma 1B rollback |
