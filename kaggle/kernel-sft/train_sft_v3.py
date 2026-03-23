"""SFT v3 generation fine-tuning for Gemma 3 270M IT (post-TAPT epoch 4).

Kaggle T4 script — self-contained, production-grade.
Loads TAPT checkpoint-88 (epoch 4) from dataset. 2 epochs, LR 1e-5, save_steps=20.
Sweet spot between v1 (over-learns: 3ep LR 2e-5) and v2 (under-learns: 1ep LR 1e-5).

Input:  /kaggle/input/pocket-arbiter-gen-data/reading_tasks.jsonl
        /kaggle/input/gemma-270m-tapt-checkpoint/  (TAPT model)
Output: /kaggle/working/gemma-270m-cpt-sft-v3/
        /kaggle/working/sft_v3_metrics.json

Standards: AdaptLLM (Cheng ICLR 2024), NEFTune (Jain ICLR 2024),
    Full FT > LoRA (Biderman TMLR 2024).

OOM context: Gemma vocab 262K → ForCausalLMLoss casts logits to fp32
    → 1024 × 262144 × 4 = 1 GB + cross_entropy internals = 5.2 GB.
    During training (model ~1 GB + optimizer ~7 GB + eval logits 5 GB = 13 GB > T4 15 GB).
    Fix: eval_strategy="no" during training, then manual eval AFTER del trainer
    (model ~1 GB + logits ~3 GB = 4 GB, fits T4 with 11 GB margin).
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
from trl import SFTConfig, SFTTrainer  # noqa: E402

# ── PHASE 1: Config ─────────────────────────────────────────────
SEED = 42
DRY_RUN = "--dry-run" in sys.argv

# TAPT checkpoint: prefer Kaggle dataset mount, fallback local
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

# Reading tasks input
INPUT_CANDIDATES = [
    "/kaggle/input/pocket-arbiter-gen-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-gen-data",
    "kaggle/dataset-generation",  # local fallback
]
INPUT_DIR = next((c for c in INPUT_CANDIDATES if os.path.isdir(c)), None)
assert INPUT_DIR is not None, f"FATAL: No input dir found. Tried: {INPUT_CANDIDATES}"
logger.info("INPUT_DIR: %s", INPUT_DIR)

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "models"
SFT_CKPT = os.path.join(OUTPUT_DIR, "gemma-270m-cpt-sft-v3")
METRICS_PATH = os.path.join(OUTPUT_DIR, "sft_v3_metrics.json")
TASKS_PATH = os.path.join(INPUT_DIR, "reading_tasks.jsonl")

_D = DRY_RUN
SFT_CFG = dict(
    epochs=2,  # v3: 2 epochs (v1=3 over-learns, v2=1 under-learns)
    batch_size=1,
    grad_accum=1 if _D else 16,
    lr=1e-5,  # v3: same as v2, budget increased via 2 epochs (v1=2e-5 over-learns)
    warmup_pct=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    neftune_alpha=5,
    save_total_limit=12,  # v3: 10 ckpts + final + margin
    save_steps=20,
    save_only_model=True,
    seq_length=512 if _D else 1024,
    eval_split=0.1,  # 10% holdout (spec: "10% stratifie par type")
)

TEST_PROMPTS = [
    "Qu'est-ce qu'un forfait en competition FFE ?",
    "Quelles sont les cadences pour le championnat de France ?",
    "Comment fonctionne le departage Buchholz ?",
    "Quel est le role de l'arbitre principal ?",
    "Quelles sont les conditions pour obtenir le titre de MI ?",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def log_vram(label: str) -> None:
    u = torch.cuda.memory_allocated() / 1024 / 1024
    t = GPU_PROPS.total_memory / 1024 / 1024
    logger.info("VRAM [%s]: %.0f / %.0f MB (%.1f%%)", label, u, t, 100 * u / t)


def generate_test(mdl: AutoModelForCausalLM, tok: AutoTokenizer, label: str) -> list:
    """Generate test responses for post-training validation."""
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
                do_sample=False,
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

    Full-sequence loss (user + assistant tokens). Use for relative
    comparison (train vs eval) — both computed the same way, so the
    ratio is a valid overfit signal even without label masking.

    Runs post-training after del trainer: model ~1 GB only on GPU,
    logits+CE ~3 GB per example → ~4 GB total, fits T4 with margin.
    """
    mdl.eval()
    total_loss, count = 0.0, 0

    with torch.amp.autocast("cuda", enabled=False):  # fp32 for stable loss
        for msgs in messages_list:
            # Two-step tokenization for portability
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
eos_id = tokenizer.convert_tokens_to_ids("<eos>")
assert eos_id == 1, f"FATAL: eos_token_id={eos_id}, expected 1"
logger.info("Tokenizer: eos='%s' (id=%d) — VERIFIED", tokenizer.eos_token, eos_id)

# Verify chat template works
test_msg = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "ok"}]
try:
    ct = tokenizer.apply_chat_template(test_msg, tokenize=False)
    logger.info("Chat template test: %s", ct[:100])
    assert "<start_of_turn>" in ct, f"FATAL: Chat template missing Gemma markers: {ct}"
except Exception as e:
    raise AssertionError(f"FATAL: Chat template failed: {e}") from e

with open(TASKS_PATH, encoding="utf-8") as f:
    reading_tasks = [json.loads(line) for line in f]
logger.info("Reading tasks loaded: %d", len(reading_tasks))
assert (
    len(reading_tasks) >= 500
), f"FATAL Gate G2: Need >= 500 tasks, got {len(reading_tasks)}"
logger.info("Gate G2 PASS: %d tasks >= 500", len(reading_tasks))

# Validate task format (sample 5)
for i, task in enumerate(reading_tasks[:5]):
    assert "messages" in task, f"FATAL: Task {i} missing 'messages' key"
    msgs = task["messages"]
    assert len(msgs) >= 2, f"FATAL: Task {i} has {len(msgs)} messages, need >= 2"
    assert msgs[0]["role"] == "user", f"FATAL: Task {i} first message not user"
    assert (
        msgs[-1]["role"] == "assistant"
    ), f"FATAL: Task {i} last message not assistant"
    assert len(msgs[0]["content"]) > 10, f"FATAL: Task {i} user content too short"
    assert len(msgs[-1]["content"]) > 5, f"FATAL: Task {i} assistant content too short"
logger.info("Task format validation PASS (sampled 5)")

# Distribution by task type (from prompt prefix)
type_counts: dict[str, int] = {}
for task in reading_tasks:
    content = task["messages"][0]["content"]
    if "consequence" in content.lower() or "contraste" in content.lower():
        t = "nli"
    elif "cause" in content.lower():
        t = "causal"
    elif "condition" in content.lower():
        t = "conditional"
    elif "reference" in content.lower() or "texte" in content.lower():
        t = "reference"
    elif content.lower().startswith("r\u00e9sumez") or content.lower().startswith(
        "resumez"
    ):
        t = "summarization"
    elif content.lower().startswith("compl\u00e9tez") or content.lower().startswith(
        "completez"
    ):
        t = "completion"
    else:
        t = "other"
    type_counts[t] = type_counts.get(t, 0) + 1
logger.info("Task distribution: %s", json.dumps(type_counts, indent=2))

# Train/eval split — 10% holdout (spec: "10% stratifie par type")
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

# Keep eval messages for post-training manual eval
eval_messages = [t["messages"] for t in eval_tasks]
train_messages_sample = [
    t["messages"] for t in train_tasks[:n_eval]
]  # same size for comparison

# Check token length distribution (sample 100 from train)
sample_lengths = []
for i in range(min(100, len(train_tasks))):
    toks = tokenizer.apply_chat_template(train_tasks[i]["messages"], tokenize=True)
    sample_lengths.append(len(toks))
logger.info(
    "Token length stats (n=%d): min=%d, median=%d, mean=%.0f, max=%d, >1024=%d",
    len(sample_lengths),
    min(sample_lengths),
    sorted(sample_lengths)[len(sample_lengths) // 2],
    sum(sample_lengths) / len(sample_lengths),
    max(sample_lengths),
    sum(1 for slen in sample_lengths if slen > 1024),
)

# ── PHASE 4: Model loading ──────────────────────────────────────
logger.info("=== PHASE 4: Model loading ===")
model = AutoModelForCausalLM.from_pretrained(
    TAPT_MODEL_ID, torch_dtype=torch.float32, device_map={"": 0}
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
), f"FATAL: Expected ~270M params, got {total_params / 1e6:.1f}M"

# Inject dropout if absent (spec: 0.1)
if not getattr(cfg, "attention_dropout", 0.0):
    cfg.attention_dropout = 0.1
    logger.info("Injected attention_dropout=0.1")
dropout_layers = [
    n for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout)
]
logger.info("Dropout layers: %d", len(dropout_layers))
log_vram("after model load")

# Pre-SFT generation test (verify TAPT model is coherent)
pre_sft_results = generate_test(model, tokenizer, "Pre-SFT (TAPT checkpoint)")

# ── PHASE 5: SFT training ───────────────────────────────────────
logger.info("=== PHASE 5: SFT training ===")

# Build training dataset (eval held out for manual post-training eval)
sft_train_messages = [t["messages"] for t in train_tasks]
sft_ds = Dataset.from_dict({"messages": sft_train_messages})
logger.info("SFT train dataset: %d examples (eval %d held out)", len(sft_ds), n_eval)

S = SFT_CFG
total_steps = (len(sft_ds) // S["batch_size"] // S["grad_accum"]) * S["epochs"]
warmup_steps = int(total_steps * S["warmup_pct"])
logger.info(
    "Training config: %d examples, %d epochs, eff_batch=%d, steps=%d, warmup=%d",
    len(sft_ds),
    S["epochs"],
    S["batch_size"] * S["grad_accum"],
    total_steps,
    warmup_steps,
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
    lr_scheduler_type="cosine",
    neftune_noise_alpha=S["neftune_alpha"],
    max_seq_length=S["seq_length"],
    fp16=True,
    gradient_checkpointing=True,
    seed=SEED,
    logging_steps=1,
    logging_nan_inf_filter=True,
    # eval_strategy="no" during training — OOM with Trainer eval pipeline:
    # model(~1GB) + optimizer(~7GB) + eval logits fp32(~5GB) = 13GB > T4 15GB.
    # Manual eval runs AFTER del trainer: model(~1GB) + logits(~3GB) = 4GB OK.
    eval_strategy="no",
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
)

logger.info("Starting SFT training...")
log_vram("before SFT training")
sft_result = sft_trainer.train()
train_loss = sft_result.training_loss
logger.info("SFT final train_loss: %.4f", train_loss)
assert not math.isnan(train_loss), "FATAL: SFT loss is NaN"
assert train_loss < 10, f"FATAL: SFT loss suspiciously high: {train_loss}"
log_vram("after SFT training")

# Save final model
sft_trainer.save_model(SFT_CKPT)
tokenizer.save_pretrained(SFT_CKPT)
logger.info("SFT checkpoint saved: %s", SFT_CKPT)

# Free trainer + optimizer memory (~7 GB) before manual eval
del sft_trainer
gc.collect()
torch.cuda.empty_cache()
log_vram("after trainer cleanup (before manual eval)")

# ── PHASE 6: Post-training eval ─────────────────────────────────
logger.info("=== PHASE 6: Post-training eval ===")

# Manual eval loss — runs with model only (~1 GB), no optimizer
# Full-sequence CLM loss (not label-masked). Both train_sample and eval
# computed the same way → ratio is valid overfit signal.
logger.info("Computing eval loss on %d held-out examples...", len(eval_messages))
eval_loss = compute_clm_loss(model, tokenizer, eval_messages, S["seq_length"])
logger.info("Eval loss (full-seq CLM): %.4f", eval_loss)
log_vram("after eval loss")

logger.info("Computing train loss on %d sample examples...", len(train_messages_sample))
train_loss_manual = compute_clm_loss(
    model, tokenizer, train_messages_sample, S["seq_length"]
)
logger.info("Train loss (full-seq CLM, sample): %.4f", train_loss_manual)

# Overfit detection — compare losses computed with identical method
if train_loss_manual > 0:
    overfit_ratio = eval_loss / train_loss_manual
    logger.info(
        "Overfit ratio: eval/train = %.4f / %.4f = %.2f",
        eval_loss,
        train_loss_manual,
        overfit_ratio,
    )
    # Heuristic: ratio > 2.0 = likely overfitting on 1802 examples
    overfit_flag = overfit_ratio > 2.0
    if overfit_flag:
        logger.warning(
            "WARNING: Overfit ratio %.2f > 2.0 — model may be overfitting. "
            "Consider using an earlier epoch checkpoint.",
            overfit_ratio,
        )
    else:
        logger.info("Overfit check PASS: ratio %.2f <= 2.0", overfit_ratio)
else:
    overfit_ratio = float("inf")
    overfit_flag = True
    logger.warning("WARNING: train_loss_manual = 0, cannot compute overfit ratio")

# Generate test responses with fine-tuned model
post_sft_results = generate_test(model, tokenizer, "Post-SFT")

# ── PHASE 7: Checkpoint validation ──────────────────────────────
logger.info("=== PHASE 7: Checkpoint validation ===")

assert os.path.isdir(SFT_CKPT), f"FATAL: SFT checkpoint missing: {SFT_CKPT}"
ckpt_files = os.listdir(SFT_CKPT)
logger.info("SFT checkpoint files: %s", sorted(ckpt_files))
assert "model.safetensors" in ckpt_files, "FATAL: model.safetensors missing"
assert (
    "tokenizer.json" in ckpt_files or "tokenizer_config.json" in ckpt_files
), "FATAL: tokenizer missing from checkpoint"

ckpt_size_mb = (
    sum(
        os.path.getsize(os.path.join(SFT_CKPT, f))
        for f in ckpt_files
        if os.path.isfile(os.path.join(SFT_CKPT, f))
    )
    / 1024
    / 1024
)
logger.info("SFT checkpoint size: %.1f MB", ckpt_size_mb)
assert ckpt_size_mb > 100, f"FATAL: SFT checkpoint too small ({ckpt_size_mb:.1f} MB)"

epoch_ckpts = sorted([d for d in os.listdir(SFT_CKPT) if d.startswith("checkpoint-")])
logger.info("Epoch checkpoints: %s", epoch_ckpts)

# ── PHASE 8: Save metrics ───────────────────────────────────────
logger.info("=== PHASE 8: Save metrics ===")

train_log = sft_result.metrics if hasattr(sft_result, "metrics") else {}
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
    "config": {k: v for k, v in S.items()},
    "pre_sft_responses": pre_sft_results,
    "post_sft_responses": post_sft_results,
    "checkpoint_size_mb": round(ckpt_size_mb, 1),
    "epoch_checkpoints": epoch_ckpts,
    "train_metrics": train_log,
    "gpu": torch.cuda.get_device_name(0),
    "vram_mb": round(GPU_VRAM_MB),
    "note": (
        "eval_loss and train_loss_manual are full-sequence CLM losses "
        "(not label-masked). Comparable to each other for overfit detection, "
        "but NOT directly comparable to training_loss (which is assistant-only)."
    ),
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
logger.info("Metrics saved: %s", METRICS_PATH)

# ── Summary ─────────────────────────────────────────────────────
elapsed = time.time() - t_start
logger.info("=" * 60)
logger.info("SFT TRAINING COMPLETE — %.1f min total", elapsed / 60)
logger.info("Checkpoint: %s (%.1f MB)", SFT_CKPT, ckpt_size_mb)
logger.info(
    "train_loss=%.4f | eval_loss=%.4f | overfit_ratio=%.2f | flag=%s",
    train_loss,
    eval_loss,
    overfit_ratio,
    overfit_flag,
)
logger.info(
    "Tasks: %d train + %d eval | Epochs: %d | GPU: %s",
    len(sft_ds),
    len(eval_messages),
    S["epochs"],
    torch.cuda.get_device_name(0),
)
logger.info("Epoch checkpoints: %s", epoch_ckpts)
logger.info("=" * 60)
