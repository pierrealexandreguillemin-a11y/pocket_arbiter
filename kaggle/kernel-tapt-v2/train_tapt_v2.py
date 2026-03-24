"""TAPT v2 — Task-Adaptive Pre-Training for Gemma 3 270M IT.

Kaggle T4 script — TAPT-only, no SFT. Corrected parameters from v1 audit.

4 bugs fixed from v1 (verified against Google FFT guide + literature):
  1. LR 5e-6 → 5e-5 (Google FFT guide, v1 was 10x too low)
  2. attention_dropout 0.1 → 0.0 (Google ships 0.0, arXiv:2505.24788)
  3. cosine → constant_with_warmup (Google FFT guide, WSO arXiv:2603.16127)
  4. warmup 10% → 5% (Secret Recipe arXiv:2412.13337)

Input:  /kaggle/input/pocket-arbiter-gen-data/corpus_paragraphs.jsonl
        /kaggle/input/gemma-3-270m-it/ (base model)
Output: /kaggle/working/gemma-270m-cpt-v2/
        /kaggle/working/tapt_v2_metrics.json

Standards: TAPT (Gururangan ACL 2020), Google Gemma FFT guide,
    NEFTune (Jain ICLR 2024), WSO (arXiv:2603.16127).
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

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tiktoken==0.9.0"])

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

# ── PHASE 1: Config ─────────────────────────────────────────────
SEED = 42
DRY_RUN = "--dry-run" in sys.argv

# Model: prefer Kaggle dataset mount, fallback to Hub
_KAGGLE_MODEL_PATHS = [
    "/kaggle/input/gemma-3-270m-it",
    "/kaggle/input/datasets/pguillemin/gemma-3-270m-it",
]
MODEL_ID = "google/gemma-3-270m-it"
for _mp in _KAGGLE_MODEL_PATHS:
    if os.path.isdir(_mp) and os.path.exists(os.path.join(_mp, "config.json")):
        MODEL_ID = _mp
        logger.info("Using Kaggle-mounted model: %s", _mp)
        break
else:
    logger.info("No Kaggle model mount found, using Hub: %s", MODEL_ID)

# Input data
INPUT_CANDIDATES = [
    "/kaggle/input/pocket-arbiter-gen-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-gen-data",
    "kaggle/dataset-generation",  # local fallback
]
INPUT_DIR = next((c for c in INPUT_CANDIDATES if os.path.isdir(c)), None)
assert INPUT_DIR is not None, f"FATAL: No input dir found. Tried: {INPUT_CANDIDATES}"
logger.info("INPUT_DIR: %s", INPUT_DIR)

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "models"
TAPT_CKPT = os.path.join(OUTPUT_DIR, "gemma-270m-cpt-v2")
METRICS_PATH = os.path.join(OUTPUT_DIR, "tapt_v2_metrics.json")
CORPUS_PATH = os.path.join(INPUT_DIR, "corpus_paragraphs.jsonl")

# Eval holdout: same 3 docs as v1 for comparability
EVAL_HOLDOUT = {
    "H01_2025_26_Conduite_pour_joueur_handicapes.pdf",
    "C04_2025_26_Coupe_de_la_parit\u00e9.pdf",
    "E02-Le_classement_rapide.pdf",
}

_D = DRY_RUN
TAPT_CFG = dict(
    epochs=1 if _D else 5,
    batch_size=1,
    grad_accum=1 if _D else 16,
    lr=5e-5,  # v2 FIX: Google FFT guide. v1 was 5e-6 (10x too low)
    warmup_pct=0.05,  # v2 FIX: 10% → 5% (Secret Recipe arXiv:2412.13337)
    weight_decay=0.01,
    max_grad_norm=1.0,
    neftune_alpha=5,
    seq_length=512 if _D else 1024,
    save_total_limit=5,  # keep ALL 5 epoch checkpoints for best selection
)

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
) -> list[dict]:
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

# ── PHASE 3: Data loading ───────────────────────────────────────
logger.info("=== PHASE 3: Data loading ===")
set_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
EOS = "<eos>"
assert (
    tokenizer.eos_token == EOS
), f"FATAL: eos_token={tokenizer.eos_token!r}, expected {EOS!r}"
eos_id = tokenizer.convert_tokens_to_ids("<eos>")
assert eos_id == 1, f"FATAL: eos_token_id={eos_id}, expected 1"
logger.info("Tokenizer: eos='%s' (id=%d) — VERIFIED", tokenizer.eos_token, eos_id)

with open(CORPUS_PATH, encoding="utf-8") as f:
    paragraphs = [json.loads(line) for line in f]
logger.info("Corpus: %d paragraphs", len(paragraphs))

# Token count verification
gemma_tok = sum(len(tokenizer.encode(p["text"])) for p in paragraphs)
tik_enc = tiktoken.get_encoding("cl100k_base")
tik_tok = sum(len(tik_enc.encode(p["text"])) for p in paragraphs)
logger.info(
    "Tokens: %d (Gemma), %d (tiktoken), ratio=%.2f",
    gemma_tok,
    tik_tok,
    gemma_tok / tik_tok,
)
assert gemma_tok >= 300_000, f"FATAL Gate GT1: {gemma_tok} Gemma tokens < 300K"
logger.info("Gate GT1 PASS: %d Gemma tokens >= 300K", gemma_tok)

# Train/eval split — same holdout as v1 for comparability
train_paras = [p for p in paragraphs if p["source"] not in EVAL_HOLDOUT]
eval_paras = [p for p in paragraphs if p["source"] in EVAL_HOLDOUT]
logger.info(
    "TAPT split: %d train, %d eval (holdout %d docs)",
    len(train_paras),
    len(eval_paras),
    len(EVAL_HOLDOUT),
)
assert len(eval_paras) > 0, "FATAL: No eval paragraphs from holdout sources"

# ── PHASE 4: Model loading ──────────────────────────────────────
logger.info("=== PHASE 4: Model loading ===")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map={"": 0}
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

# v2 FIX: NO dropout injection. Gemma ships with attention_dropout=0.0.
# v1 injected 0.1, which hurts learning in 1-5 epochs (arXiv:2505.24788).
current_dropout = getattr(cfg, "attention_dropout", 0.0)
logger.info(
    "attention_dropout=%.2f (Gemma default, NOT injected — v2 fix)",
    current_dropout,
)

log_vram("after model load")

# Pre-TAPT generation test
pre_tapt_results = generate_test(model, tokenizer, "Pre-TAPT (base)")

# ── PHASE 5: Sequence packing ───────────────────────────────────
logger.info("=== PHASE 5: Sequence packing ===")
random.shuffle(train_paras)


def pack_sequences(paras: list[dict], seq_len: int) -> list[list[int]]:
    """Pack paragraphs into fixed-length sequences for CLM training."""
    all_ids: list[int] = []
    for p in paras:
        all_ids.extend(tokenizer.encode(p["text"], add_special_tokens=False))
        all_ids.append(eos_id)
    seqs = [
        all_ids[i : i + seq_len] for i in range(0, len(all_ids) - seq_len + 1, seq_len)
    ]
    logger.info("Packed %d tokens -> %d seqs of %d", len(all_ids), len(seqs), seq_len)
    return seqs


train_seqs = pack_sequences(train_paras, int(TAPT_CFG["seq_length"]))
eval_seqs = pack_sequences(eval_paras, int(TAPT_CFG["seq_length"]))
assert len(train_seqs) > 0, "FATAL: No training sequences after packing"
assert len(eval_seqs) > 0, "FATAL: No eval sequences after packing"

train_ds = Dataset.from_dict(
    {"input_ids": train_seqs, "attention_mask": [[1] * len(s) for s in train_seqs]}
)
eval_ds = Dataset.from_dict(
    {"input_ids": eval_seqs, "attention_mask": [[1] * len(s) for s in eval_seqs]}
)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


@torch.no_grad()
def compute_perplexity(mdl: AutoModelForCausalLM, ds: Dataset) -> float:
    """Compute perplexity on eval set. fp32, no AMP (stable loss)."""
    mdl.eval()
    total_loss, total_tok = 0.0, 0
    with torch.amp.autocast("cuda", enabled=False):
        for i in range(len(ds)):
            ids = torch.tensor([ds[i]["input_ids"]], device=mdl.device)
            out = mdl(input_ids=ids.long(), labels=ids.long())
            loss_val = float(out.loss)
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                total_loss += loss_val * ids.shape[1]
                total_tok += ids.shape[1]
    if total_tok == 0:
        return float("inf")
    avg_loss = total_loss / total_tok
    return math.exp(min(avg_loss, 20.0))


# ── PHASE 6: Baseline perplexity ────────────────────────────────
logger.info("=== PHASE 6: Baseline perplexity ===")
ppl_before = compute_perplexity(model, eval_ds)
logger.info("Baseline perplexity: %.2f", ppl_before)

# ── PHASE 7: TAPT training ──────────────────────────────────────
logger.info("=== PHASE 7: TAPT training ===")
C = TAPT_CFG
total_steps = (len(train_ds) // C["batch_size"] // C["grad_accum"]) * C["epochs"]
warmup_steps = int(total_steps * C["warmup_pct"])
logger.info(
    "Training: %d seqs, %d epochs, eff_batch=%d, ~%d steps, warmup=%d",
    len(train_ds),
    C["epochs"],
    C["batch_size"] * C["grad_accum"],
    total_steps,
    warmup_steps,
)
logger.info(
    "v2 corrections: lr=%.0e (v1=5e-6), scheduler=constant, dropout=0.0, warmup=5%%",
    C["lr"],
)

tapt_args = TrainingArguments(
    output_dir=TAPT_CKPT,
    num_train_epochs=C["epochs"],
    per_device_train_batch_size=C["batch_size"],
    per_device_eval_batch_size=C["batch_size"],
    gradient_accumulation_steps=C["grad_accum"],
    learning_rate=C["lr"],
    warmup_steps=warmup_steps,
    weight_decay=C["weight_decay"],
    max_grad_norm=C["max_grad_norm"],
    # v2 FIX: constant_with_warmup (Google FFT guide, WSO arXiv:2603.16127)
    # v1 used cosine which decays LR prematurely on ~110 steps
    lr_scheduler_type="constant_with_warmup",
    # v2 FIX: adamw_torch_fused (Google FFT guide, ~5% speedup)
    optim="adamw_torch_fused",
    neftune_noise_alpha=C["neftune_alpha"],
    fp16=True,
    gradient_checkpointing=True,
    seed=SEED,
    logging_steps=1,
    logging_nan_inf_filter=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=C["save_total_limit"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=tapt_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

logger.info("Starting TAPT v2 training...")
log_vram("before TAPT training")
result = trainer.train()
train_loss = result.training_loss
logger.info("TAPT v2 final loss: %.4f", train_loss)
assert not math.isnan(train_loss), "FATAL: TAPT loss is NaN"
assert train_loss < 50, f"FATAL: TAPT loss too high: {train_loss}"
log_vram("after TAPT training")

# ── PHASE 8: Post-training eval ─────────────────────────────────
logger.info("=== PHASE 8: Post-training eval ===")
ppl_after = compute_perplexity(model, eval_ds)
logger.info(
    "Perplexity: %.2f → %.2f (delta=%.2f)",
    ppl_before,
    ppl_after,
    ppl_before - ppl_after,
)

# Gate GT2: perplexity must decrease
gate_gt2 = ppl_after < ppl_before
if gate_gt2:
    logger.info(
        "Gate GT2 PASS: perplexity decreased (%.2f < %.2f)", ppl_after, ppl_before
    )
else:
    logger.warning(
        "Gate GT2 FAIL: perplexity increased (%.2f >= %.2f)",
        ppl_after,
        ppl_before,
    )

# Save model
trainer.save_model(TAPT_CKPT)
tokenizer.save_pretrained(TAPT_CKPT)
logger.info("TAPT v2 checkpoint saved: %s", TAPT_CKPT)

# Post-TAPT generation test
post_tapt_results = generate_test(model, tokenizer, "Post-TAPT v2")

# Free trainer memory
del trainer
gc.collect()
torch.cuda.empty_cache()
log_vram("after trainer cleanup")

# ── PHASE 9: Checkpoint validation ──────────────────────────────
logger.info("=== PHASE 9: Checkpoint validation ===")
assert os.path.isdir(TAPT_CKPT), f"FATAL: TAPT checkpoint missing: {TAPT_CKPT}"
ckpt_files = os.listdir(TAPT_CKPT)
logger.info("TAPT v2 checkpoint files: %s", sorted(ckpt_files))
assert "model.safetensors" in ckpt_files, "FATAL: model.safetensors missing"

ckpt_size_mb = (
    sum(
        os.path.getsize(os.path.join(TAPT_CKPT, f))
        for f in ckpt_files
        if os.path.isfile(os.path.join(TAPT_CKPT, f))
    )
    / 1024
    / 1024
)
logger.info("TAPT v2 checkpoint size: %.1f MB", ckpt_size_mb)
assert ckpt_size_mb > 100, f"FATAL: checkpoint too small ({ckpt_size_mb:.1f} MB)"

epoch_ckpts = sorted([d for d in ckpt_files if d.startswith("checkpoint-")])
logger.info("Epoch checkpoints: %s", epoch_ckpts)

# ── PHASE 10: Save metrics ──────────────────────────────────────
logger.info("=== PHASE 10: Save metrics ===")
train_log = result.metrics if hasattr(result, "metrics") else {}
metrics = {
    "perplexity_baseline": round(ppl_before, 4),
    "perplexity_post": round(ppl_after, 4),
    "perplexity_delta": round(ppl_before - ppl_after, 4),
    "gate_GT2": gate_gt2,
    "training_loss": round(train_loss, 4),
    "config": {k: v for k, v in C.items()},
    "corrections_v1_to_v2": {
        "lr": "5e-6 → 5e-5 (Google FFT guide, 10x increase)",
        "dropout": "0.1 → 0.0 (Gemma default, arXiv:2505.24788)",
        "scheduler": "cosine → constant_with_warmup (Google FFT, WSO arXiv:2603.16127)",
        "warmup": "10% → 5% (Secret Recipe arXiv:2412.13337)",
        "optim": "adamw_torch → adamw_torch_fused (Google FFT, ~5% speedup)",
    },
    "v1_reference": {
        "perplexity_baseline": 37.74,
        "perplexity_post": 7.98,
        "lr": "5e-6 (10x too low)",
        "scheduler": "cosine",
        "dropout": "0.1 (injected)",
    },
    "pre_tapt_responses": pre_tapt_results,
    "post_tapt_responses": post_tapt_results,
    "epoch_checkpoints": epoch_ckpts,
    "checkpoint_size_mb": round(ckpt_size_mb, 1),
    "train_metrics": train_log,
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
logger.info("TAPT v2 TRAINING COMPLETE — %.1f min total", elapsed / 60)
logger.info("Checkpoint: %s (%.1f MB)", TAPT_CKPT, ckpt_size_mb)
logger.info(
    "Perplexity: %.2f → %.2f (v1 ref: 37.74 → 7.98)",
    ppl_before,
    ppl_after,
)
logger.info(
    "train_loss=%.4f | Gate GT2=%s | GPU: %s",
    train_loss,
    gate_gt2,
    torch.cuda.get_device_name(0),
)
logger.info("Epoch checkpoints: %s", epoch_ckpts)
logger.info(
    "v2 fixes: lr=5e-5 (was 5e-6), dropout=0.0 (was 0.1), "
    "scheduler=constant (was cosine), warmup=5%% (was 10%%)",
)
logger.info("=" * 60)
