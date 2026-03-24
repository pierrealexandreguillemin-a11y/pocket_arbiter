"""SFT v4 — Supervised Fine-Tuning on TAPT v2, all corrections applied.

Kaggle T4 script — self-contained, production-grade.

8 corrections from v3 (verified against Google FFT guide + literature + eval v4):
  1. Base model: TAPT v2 (corrected LR/dropout/scheduler) instead of v1
  2. attention_dropout: 0.0 (Gemma default, arXiv:2505.24788)
  3. lr_scheduler: constant_with_warmup (Google FFT, WSO arXiv:2603.16127)
  4. Loss masking: assistant-only via DataCollatorForCompletionOnlyLM
  5. Prompt: RAG v2 in training data (alignment train/inference)
  6. warmup: 5% (Secret Recipe arXiv:2412.13337)
  7. optimizer: adamw_torch_fused (Google FFT, ~5% speedup)
  8. Gen params: state-of-the-art (temp=0.2, rep_penalty=1.2)

Input:  /kaggle/input/pocket-arbiter-gen-data/reading_tasks_v2.jsonl
        /kaggle/input/gemma-270m-tapt-checkpoint/  (TAPT v2 model)
Output: /kaggle/working/gemma-270m-cpt-sft-v4/
        /kaggle/working/sft_v4_metrics.json

Standards: AdaptLLM (Cheng ICLR 2024), NEFTune (Jain ICLR 2024),
    Google Gemma FFT guide, WSO (arXiv:2603.16127).

Eval v4 reference (target to beat): Base 43.9% citations with 0 empty.
"""

from __future__ import annotations

import os

# Force single GPU — prevent DataParallel on T4x2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
t_start = time.time()

# ── PHASE 0: Environment ────────────────────────────────────────
logger.info("=== PHASE 0: Environment ===")
assert torch.cuda.is_available(), "FATAL: No GPU detected"
GPU_PROPS = torch.cuda.get_device_properties(0)
GPU_VRAM_MB = GPU_PROPS.total_memory / 1024 / 1024
logger.info(
    "GPU: %s (%.0f MB VRAM, compute %d.%d)",
    torch.cuda.get_device_name(0),
    GPU_VRAM_MB,
    GPU_PROPS.major,
    GPU_PROPS.minor,
)
assert GPU_VRAM_MB >= 14000, f"FATAL: Need >= 14 GB VRAM, got {GPU_VRAM_MB:.0f} MB"
assert (
    GPU_PROPS.major >= 7
), f"FATAL: Need compute >= 7.0, got {GPU_PROPS.major}.{GPU_PROPS.minor}"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "trl==0.16.0"])

from datasets import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer  # noqa: E402

# ── PHASE 1: Config ─────────────────────────────────────────────
SEED = 42
DRY_RUN = "--dry-run" in sys.argv

# TAPT v2 checkpoint
_TAPT_PATHS = [
    "/kaggle/input/gemma-270m-tapt-checkpoint",
    "/kaggle/input/datasets/pguillemin/gemma-270m-tapt-checkpoint",
    "models/kaggle-output/gemma-270m-cpt",  # local fallback
]
TAPT_MODEL_ID = None
for _tp in _TAPT_PATHS:
    if os.path.isdir(_tp) and os.path.exists(os.path.join(_tp, "model.safetensors")):
        TAPT_MODEL_ID = _tp
        logger.info("TAPT checkpoint found: %s", _tp)
        break
assert (
    TAPT_MODEL_ID is not None
), f"FATAL: TAPT checkpoint not found. Tried: {_TAPT_PATHS}"

# Reading tasks v2 (RAG prompt format)
INPUT_CANDIDATES = [
    "/kaggle/input/pocket-arbiter-gen-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-gen-data",
    "kaggle/dataset-generation",  # local fallback
]
INPUT_DIR = next((c for c in INPUT_CANDIDATES if os.path.isdir(c)), None)
assert INPUT_DIR is not None, f"FATAL: No input dir found. Tried: {INPUT_CANDIDATES}"
logger.info("INPUT_DIR: %s", INPUT_DIR)

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "models"
SFT_CKPT = os.path.join(OUTPUT_DIR, "gemma-270m-cpt-sft-v4")
METRICS_PATH = os.path.join(OUTPUT_DIR, "sft_v4_metrics.json")

# v4: use reading_tasks_v2.jsonl (RAG prompt format), fallback to v1
TASKS_V2 = os.path.join(INPUT_DIR, "reading_tasks_v2.jsonl")
TASKS_V1 = os.path.join(INPUT_DIR, "reading_tasks.jsonl")
if os.path.exists(TASKS_V2):
    TASKS_PATH = TASKS_V2
    logger.info("Using v2 tasks (RAG prompt format): %s", TASKS_V2)
else:
    TASKS_PATH = TASKS_V1
    logger.warning("WARNING: v2 tasks not found, falling back to v1: %s", TASKS_V1)

_D = DRY_RUN
SFT_CFG = dict(
    epochs=2,  # same as v3
    batch_size=1,
    grad_accum=1 if _D else 16,
    lr=1e-5,  # same as v3
    warmup_pct=0.05,  # v4 FIX: 10% → 5% (Secret Recipe arXiv:2412.13337)
    weight_decay=0.01,
    max_grad_norm=1.0,
    neftune_alpha=5,
    save_total_limit=12,  # 10 ckpts + final + margin
    save_steps=20,
    save_only_model=True,
    seq_length=512 if _D else 1024,
    eval_split=0.1,
)

# Gemma response marker for DataCollatorForCompletionOnlyLM
RESPONSE_TEMPLATE = "<start_of_turn>model\n"

TEST_PROMPTS = [
    "Qu'est-ce qu'un forfait en competition FFE ?",
    "Quelles sont les cadences pour le championnat de France ?",
    "Comment fonctionne le departage Buchholz ?",
    "Quel est le role de l'arbitre principal ?",
    "Quelles sont les conditions pour obtenir le titre de MI ?",
]


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def log_vram(label: str) -> None:
    """Log current GPU VRAM usage."""
    u = torch.cuda.memory_allocated() / 1024 / 1024
    t = GPU_PROPS.total_memory / 1024 / 1024
    logger.info("VRAM [%s]: %.0f / %.0f MB (%.1f%%)", label, u, t, 100 * u / t)


def generate_test(
    mdl: AutoModelForCausalLM,
    tok: AutoTokenizer,
    label: str,
) -> list:
    """Generate test responses with state-of-the-art gen params."""
    logger.info("%s generation test:", label)
    mdl.eval()
    results = []
    for p in TEST_PROMPTS:
        inputs = tok(p, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"].to(mdl.device)
        attention_mask = inputs["attention_mask"].to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.2,
                top_k=64,
                top_p=0.95,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=tok.eos_token_id,
            )
        answer = tok.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
        logger.info("  Q: %s\n  A: %s", p, answer[:200])
        results.append({"question": p, "answer": answer[:500]})
    return results


@torch.no_grad()
def compute_clm_loss(
    mdl: AutoModelForCausalLM,
    tok: AutoTokenizer,
    messages_list: list[list[dict]],
    max_len: int,
) -> float:
    """Compute CLM loss on chat-formatted examples, one at a time.

    Full-sequence loss for relative comparison (train vs eval).
    Runs post-training after del trainer: model ~1 GB only on GPU.
    """
    mdl.eval()
    total_loss, count = 0.0, 0
    with torch.amp.autocast("cuda", enabled=False):
        for msgs in messages_list:
            text = tok.apply_chat_template(msgs, tokenize=False)
            encoded = tok(
                text,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
                padding=False,
            )
            ids = encoded["input_ids"].to(mdl.device)
            if ids.shape[1] < 2:
                continue
            out = mdl(input_ids=ids, labels=ids)
            loss_val = float(out.loss)
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                total_loss += loss_val
                count += 1
    if count == 0:
        return float("inf")
    return total_loss / count


# ── PHASE 2: Diagnostics ────────────────────────────────────────
logger.info("=== PHASE 2: Diagnostics ===")
logger.info("=== /kaggle/input/ contents ===")
if os.path.isdir("/kaggle/input"):
    for e in sorted(os.listdir("/kaggle/input")):
        ep = os.path.join("/kaggle/input", e)
        if os.path.isdir(ep):
            logger.info("  %s/ (%d files)", e, len(os.listdir(ep)))
else:
    logger.info("  /kaggle/input does NOT exist (local mode)")

# Verify TAPT checkpoint integrity
tapt_files = os.listdir(TAPT_MODEL_ID)
logger.info("TAPT checkpoint files: %s", sorted(tapt_files))
required_files = {"model.safetensors", "config.json", "tokenizer.json"}
missing = required_files - set(tapt_files)
assert not missing, f"FATAL: TAPT checkpoint missing files: {missing}"

tapt_size_mb = (
    sum(
        os.path.getsize(os.path.join(TAPT_MODEL_ID, f))
        for f in tapt_files
        if os.path.isfile(os.path.join(TAPT_MODEL_ID, f))
    )
    / 1024
    / 1024
)
logger.info("TAPT checkpoint size: %.1f MB", tapt_size_mb)
assert tapt_size_mb > 500, f"FATAL: TAPT checkpoint too small ({tapt_size_mb:.1f} MB)"

# ── PHASE 3: Data loading ───────────────────────────────────────
logger.info("=== PHASE 3: Data loading ===")
set_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(TAPT_MODEL_ID)
EOS = "<eos>"
assert (
    tokenizer.eos_token == EOS
), f"FATAL: eos_token={tokenizer.eos_token!r}, expected {EOS!r}"
logger.info("Tokenizer: eos='%s' — VERIFIED", tokenizer.eos_token)

# Verify chat template produces expected response marker
test_msg = [
    {"role": "user", "content": "test"},
    {"role": "assistant", "content": "ok"},
]
ct = tokenizer.apply_chat_template(test_msg, tokenize=False)
assert (
    RESPONSE_TEMPLATE in ct
), f"FATAL: Response template '{RESPONSE_TEMPLATE}' not found in chat template: {ct}"
logger.info("Response template '%s' verified in chat template", RESPONSE_TEMPLATE)

# Verify DataCollatorForCompletionOnlyLM works with this tokenizer
response_token_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
logger.info("Response template token IDs: %s", response_token_ids)

with open(TASKS_PATH, encoding="utf-8") as f:
    reading_tasks = [json.loads(line) for line in f]
logger.info("Reading tasks loaded: %d from %s", len(reading_tasks), TASKS_PATH)
assert (
    len(reading_tasks) >= 500
), f"FATAL Gate GS1: Need >= 500 tasks, got {len(reading_tasks)}"
logger.info("Gate GS1 PASS: %d tasks >= 500", len(reading_tasks))

# Validate v2 format (RAG prompt in user content)
sample = reading_tasks[0]
assert (
    "REGLES:" in sample["messages"][0]["content"]
), "FATAL: v2 format expected (REGLES in user content). Got v1 format?"
logger.info("v2 format validation PASS (REGLES in user content)")

# Validate task structure
for i, task in enumerate(reading_tasks[:5]):
    assert "messages" in task, f"FATAL: Task {i} missing 'messages'"
    msgs = task["messages"]
    assert len(msgs) >= 2, f"FATAL: Task {i} has {len(msgs)} messages"
    assert msgs[0]["role"] == "user", f"FATAL: Task {i} first not user"
    assert msgs[-1]["role"] == "assistant", f"FATAL: Task {i} last not assistant"
logger.info("Task structure validation PASS")

# Distribution
type_counts: dict[str, int] = {}
for task in reading_tasks:
    t = task.get("task_type", "unknown")
    type_counts[t] = type_counts.get(t, 0) + 1
logger.info("Task distribution: %s", json.dumps(type_counts, indent=2))

# Train/eval split
random.shuffle(reading_tasks)
n_eval = max(1, int(len(reading_tasks) * SFT_CFG["eval_split"]))
eval_tasks = reading_tasks[:n_eval]
train_tasks = reading_tasks[n_eval:]
logger.info(
    "Train/eval split: %d train, %d eval (%.1f%%)",
    len(train_tasks),
    len(eval_tasks),
    100 * n_eval / len(reading_tasks),
)

eval_messages = [t["messages"] for t in eval_tasks]
train_messages_sample = [t["messages"] for t in train_tasks[:n_eval]]

# ── PHASE 4: Model loading ──────────────────────────────────────
logger.info("=== PHASE 4: Model loading ===")
model = AutoModelForCausalLM.from_pretrained(
    TAPT_MODEL_ID, dtype=torch.float32, device_map={"": 0}
)
cfg = model.config
logger.info(
    "Architecture: layers=%d, hidden=%d, heads=%d",
    cfg.num_hidden_layers,
    cfg.hidden_size,
    cfg.num_attention_heads,
)
total_params = sum(p.numel() for p in model.parameters())
logger.info("Total params: %d (%.1fM)", total_params, total_params / 1e6)
assert (
    200e6 < total_params < 400e6
), f"FATAL Gate GS0: Expected ~270M params, got {total_params / 1e6:.1f}M"

# v4 FIX: NO dropout injection. Gemma ships with attention_dropout=0.0.
current_dropout = getattr(cfg, "attention_dropout", 0.0)
logger.info(
    "attention_dropout=%.2f (Gemma default, NOT injected — v4 fix)",
    current_dropout,
)
log_vram("after model load")

# Pre-SFT generation test
pre_sft_results = generate_test(model, tokenizer, "Pre-SFT (TAPT v2)")

# ── PHASE 5: SFT training ───────────────────────────────────────
logger.info("=== PHASE 5: SFT training ===")

sft_train_messages = [t["messages"] for t in train_tasks]
sft_ds = Dataset.from_dict({"messages": sft_train_messages})
logger.info("SFT train dataset: %d examples (eval %d held out)", len(sft_ds), n_eval)

S = SFT_CFG
total_steps = (len(sft_ds) // S["batch_size"] // S["grad_accum"]) * S["epochs"]
warmup_steps = int(total_steps * S["warmup_pct"])
logger.info(
    "Training: %d examples, %d epochs, eff_batch=%d, steps=%d, warmup=%d",
    len(sft_ds),
    S["epochs"],
    S["batch_size"] * S["grad_accum"],
    total_steps,
    warmup_steps,
)
logger.info(
    "v4 corrections: scheduler=constant, dropout=0.0, warmup=5%%, "
    "loss=assistant-only (DataCollatorForCompletionOnlyLM), prompt=RAG v2",
)

# v4 FIX: DataCollatorForCompletionOnlyLM — masks prompt tokens
# Only trains on assistant response (after <start_of_turn>model\n)
# This fixes the echo behavior from v3 (full-sequence loss)
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_token_ids,
    tokenizer=tokenizer,
)
logger.info(
    "DataCollatorForCompletionOnlyLM configured with response_template=%s",
    response_token_ids,
)

sft_config = SFTConfig(
    output_dir=SFT_CKPT,
    num_train_epochs=S["epochs"],
    per_device_train_batch_size=S["batch_size"],
    gradient_accumulation_steps=S["grad_accum"],
    learning_rate=S["lr"],
    warmup_steps=warmup_steps,
    weight_decay=S["weight_decay"],
    max_grad_norm=S["max_grad_norm"],
    # v4 FIX: constant_with_warmup (Google FFT, WSO arXiv:2603.16127)
    lr_scheduler_type="constant_with_warmup",
    # v4 FIX: adamw_torch_fused (Google FFT, ~5% speedup)
    optim="adamw_torch_fused",
    neftune_noise_alpha=S["neftune_alpha"],
    max_seq_length=S["seq_length"],
    fp16=True,
    gradient_checkpointing=True,
    seed=SEED,
    logging_steps=1,
    logging_nan_inf_filter=True,
    eval_strategy="no",  # OOM with 262K vocab eval
    save_strategy="steps",
    save_steps=S["save_steps"],
    save_only_model=True,
    save_total_limit=S["save_total_limit"],
)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=sft_ds,
    processing_class=tokenizer,
    data_collator=collator,  # v4 FIX: assistant-only loss
)

logger.info("Starting SFT v4 training...")
log_vram("before SFT training")
sft_result = sft_trainer.train()
train_loss = sft_result.training_loss
logger.info("SFT v4 final train_loss: %.4f", train_loss)
assert not math.isnan(train_loss), "FATAL: SFT loss is NaN"
assert train_loss < 10, f"FATAL: SFT loss too high: {train_loss}"
log_vram("after SFT training")

# Save final model
sft_trainer.save_model(SFT_CKPT)
tokenizer.save_pretrained(SFT_CKPT)
logger.info("SFT v4 checkpoint saved: %s", SFT_CKPT)

# Free trainer + optimizer memory before manual eval
del sft_trainer
gc.collect()
torch.cuda.empty_cache()
log_vram("after trainer cleanup (before manual eval)")

# ── PHASE 6: Post-training eval ─────────────────────────────────
logger.info("=== PHASE 6: Post-training eval ===")

logger.info("Computing eval loss on %d held-out examples...", len(eval_messages))
eval_loss = compute_clm_loss(model, tokenizer, eval_messages, S["seq_length"])
logger.info("Eval loss (full-seq CLM): %.4f", eval_loss)

logger.info("Computing train loss on %d sample examples...", len(train_messages_sample))
train_loss_manual = compute_clm_loss(
    model, tokenizer, train_messages_sample, S["seq_length"]
)
logger.info("Train loss (full-seq CLM, sample): %.4f", train_loss_manual)

# Overfit detection
if train_loss_manual > 0:
    overfit_ratio = eval_loss / train_loss_manual
    logger.info(
        "Overfit ratio: eval/train = %.4f / %.4f = %.2f",
        eval_loss,
        train_loss_manual,
        overfit_ratio,
    )
    overfit_flag = overfit_ratio > 2.0
    if overfit_flag:
        logger.warning("WARNING: Overfit ratio %.2f > 2.0", overfit_ratio)
    else:
        logger.info("Overfit check PASS: ratio %.2f <= 2.0", overfit_ratio)
else:
    overfit_ratio = float("inf")
    overfit_flag = True

# Post-SFT generation test
post_sft_results = generate_test(model, tokenizer, "Post-SFT v4")

# ── PHASE 7: Checkpoint validation ──────────────────────────────
logger.info("=== PHASE 7: Checkpoint validation ===")
assert os.path.isdir(SFT_CKPT), f"FATAL: SFT checkpoint missing: {SFT_CKPT}"
ckpt_files = os.listdir(SFT_CKPT)
logger.info("SFT v4 checkpoint files: %s", sorted(ckpt_files))
assert "model.safetensors" in ckpt_files, "FATAL: model.safetensors missing"

ckpt_size_mb = (
    sum(
        os.path.getsize(os.path.join(SFT_CKPT, f))
        for f in ckpt_files
        if os.path.isfile(os.path.join(SFT_CKPT, f))
    )
    / 1024
    / 1024
)
logger.info("SFT v4 checkpoint size: %.1f MB", ckpt_size_mb)

epoch_ckpts = sorted([d for d in ckpt_files if d.startswith("checkpoint-")])
logger.info("Epoch checkpoints: %s", epoch_ckpts)

# ── PHASE 8: Save metrics ───────────────────────────────────────
logger.info("=== PHASE 8: Save metrics ===")
metrics = {
    "training_loss": train_loss,
    "eval_loss": eval_loss,
    "train_loss_manual": train_loss_manual,
    "overfit_ratio": round(overfit_ratio, 4),
    "overfit_flag": overfit_flag,
    "epochs": S["epochs"],
    "total_train": len(sft_ds),
    "total_eval": len(eval_messages),
    "task_distribution": type_counts,
    "tapt_checkpoint": TAPT_MODEL_ID,
    "tasks_file": TASKS_PATH,
    "config": {k: v for k, v in S.items()},
    "corrections_v3_to_v4": {
        "base_model": "TAPT v2 (corrected) instead of v1",
        "dropout": "0.0 (Gemma default, v3 injected 0.1)",
        "scheduler": "constant_with_warmup (v3 used cosine)",
        "loss_masking": "assistant-only via DataCollatorForCompletionOnlyLM (v3 full-seq)",
        "prompt": "RAG v2 in training data (v3 used AdaptLLM generic)",
        "warmup": "5% (v3 used 10%)",
        "optim": "adamw_torch_fused (v3 used adamw_torch)",
    },
    "eval_v4_reference": {
        "base_citations": "43.9% (target to beat)",
        "tapt_v1_citations": "36.4%",
        "sft_v3_citations": "28.8%",
    },
    "pre_sft_responses": pre_sft_results,
    "post_sft_responses": post_sft_results,
    "checkpoint_size_mb": round(ckpt_size_mb, 1),
    "epoch_checkpoints": epoch_ckpts,
    "train_metrics": sft_result.metrics if hasattr(sft_result, "metrics") else {},
    "gpu": torch.cuda.get_device_name(0),
    "vram_mb": round(GPU_VRAM_MB),
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
logger.info("Metrics saved: %s", METRICS_PATH)

# ── Summary ─────────────────────────────────────────────────────
elapsed = time.time() - t_start
logger.info("=" * 60)
logger.info("SFT v4 TRAINING COMPLETE — %.1f min total", elapsed / 60)
logger.info("Checkpoint: %s (%.1f MB)", SFT_CKPT, ckpt_size_mb)
logger.info(
    "train_loss=%.4f | eval_loss=%.4f | overfit_ratio=%.2f | flag=%s",
    train_loss,
    eval_loss,
    overfit_ratio,
    overfit_flag,
)
logger.info(
    "v4 fixes: TAPT v2 base, dropout=0.0, constant scheduler, "
    "assistant-only loss, RAG prompt v2, warmup=5%%, fused optim",
)
logger.info("Epoch checkpoints: %s", epoch_ckpts)
logger.info("=" * 60)
