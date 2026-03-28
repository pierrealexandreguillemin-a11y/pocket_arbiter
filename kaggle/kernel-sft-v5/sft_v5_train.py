"""SFT v5: Fine-tune Gemma 3 270M IT on RAFT training data (Gemma 27B teacher).

Based on train_sft_v3.py (PROVEN on Kaggle T4). Only 2 changes vs v3:
1. BETTER DATA: 2142 RAFT samples (27B teacher, 95.1% citations valid)
   instead of 1802 regex garbage (AdaptLLM, 16 tok median, no citations)
2. ASSISTANT-ONLY LOSS: DataCollatorForCompletionOnlyLM (fixes bug #3)
   instead of full-sequence loss (which caused echo behavior in v1-v3)

Everything else IDENTICAL to v3 (proven hyperparams):
- LR 1e-5 (NOT 2e-5 — overcorrection destroyed faithfulness in TAPT v2)
- Cosine scheduler (NOT constant — proven in v3 sweep)
- Dropout: model default (NOT forced to 0.0 — v3 sweep showed 0.1 is fine)
- No NEFTune (not validated for sub-1B RAG)
- eval_strategy="no" (OOM on 262K vocab, manual eval post-training)

Memory refs: feedback_tapt_v2_overcorrection, feedback_tapt_v3_sweep,
  feedback_sft_data_quality_disaster, project_rag_faithfulness_findings

Base: TAPT ep1 (checkpoint-22, 46.2% citations — best from v3 sweep)
Data: 2142 samples (1729 oracle cited, 307 abstain, 106 memorize)
Target: citations > 46.2% (beat TAPT ep1 alone)

Input:  pguillemin/pa-tapt-ep1-checkpoint (TAPT ep1 model)
        pguillemin/pa-sft-v5-data (sft_train_v5.jsonl)
Output: /kaggle/working/gemma-270m-sft-v5/
"""

from __future__ import annotations

import os

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

# -- PHASE 0: Environment --------------------------------------------------
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

# trl==0.16.0 EXACT — proven on Kaggle. >=0.16.0 installs latest which REMOVED
# DataCollatorForCompletionOnlyLM (deprecated after v0.20.0).
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "trl==0.16.0"])

from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer  # noqa: E402

# -- Config (v3 proven hyperparams, only data + loss changed) ---------------
SEED = 42

_MODEL_PATHS = [
    "/kaggle/input/pa-tapt-ep1-checkpoint",
    "/kaggle/input/datasets/pguillemin/pa-tapt-ep1-checkpoint",
]
_DATA_PATHS = [
    "/kaggle/input/pa-sft-v5-data",
    "/kaggle/input/datasets/pguillemin/pa-sft-v5-data",
]

MODEL_PATH = next((p for p in _MODEL_PATHS if os.path.isdir(p)), None)
assert MODEL_PATH is not None, f"FATAL: Model not found: {_MODEL_PATHS}"

DATA_DIR = next((p for p in _DATA_PATHS if os.path.isdir(p)), None)
assert DATA_DIR is not None, f"FATAL: Data not found: {_DATA_PATHS}"
DATA_PATH = os.path.join(DATA_DIR, "sft_train_v5.jsonl")
assert os.path.exists(DATA_PATH), f"FATAL: {DATA_PATH} not found"

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "models"
SFT_OUTPUT = os.path.join(OUTPUT_DIR, "gemma-270m-sft-v5")

# Hyperparams: v3 proven values. DO NOT overcorrect (memory: feedback_tapt_v2_overcorrection)
SFT_CFG = {
    "epochs": 2,  # v3: 2 epochs (v1=3 over-learns, v2=1 under-learns)
    "batch_size": 1,
    "grad_accum": 16,  # effective batch = 16
    "lr": 1e-5,  # v3 value. NOT 2e-5 (v1 over-learns). NOT 5e-5 (TAPT v2 disaster)
    "warmup_pct": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_steps": 20,  # intra-epoch checkpoints for MA analysis
    "save_only_model": True,  # 1.1 GB/ckpt, not 3.1 GB
    "save_total_limit": 12,
    "seq_length": 1024,  # NOT 2048 (OOM on T4)
    # NO neftune_alpha — not validated for sub-1B RAG (memory: feedback_sft_data_quality_disaster)
    # Cosine scheduler — proven in v3 sweep (memory: feedback_tapt_v3_sweep)
    # Dropout: model default from config.json (do NOT override)
}

TEST_PROMPTS = [
    "Qu'est-ce qu'un forfait en competition FFE ?",
    "Quelles sont les cadences pour le championnat de France ?",
    "Comment fonctionne le departage Buchholz ?",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_vram(label: str) -> None:
    u = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("VRAM [%s]: %.0f / %.0f MB", label, u, GPU_VRAM_MB)


def generate_test(mdl, tok, label: str) -> list:
    """Generate test responses for post-training validation."""
    logger.info("%s generation test:", label)
    mdl.eval()
    results = []
    for p in TEST_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(
            text, return_tensors="pt", truncation=True, max_length=SFT_CFG["seq_length"]
        )
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=150,
                min_new_tokens=10,
                do_sample=True,
                temperature=0.2,
                top_k=64,
                top_p=0.95,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=tok.eos_token_id,
            )
        answer = tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        logger.info("  Q: %s\n  A: %s", p, answer[:200])
        results.append({"question": p, "answer": answer[:500]})
    return results


@torch.no_grad()
def compute_clm_loss(mdl, tok, messages_list: list, max_len: int) -> float:
    """Compute CLM loss on chat-formatted examples (full-seq, for overfit detection)."""
    mdl.eval()
    total_loss, count = 0.0, 0
    with torch.amp.autocast("cuda", enabled=False):
        for msgs in messages_list:
            text = tok.apply_chat_template(msgs, tokenize=False)
            enc = tok(text, return_tensors="pt", max_length=max_len, truncation=True)
            ids = enc["input_ids"].to(mdl.device)
            if ids.shape[1] < 2:
                continue
            out = mdl(input_ids=ids, labels=ids)
            loss_val = float(out.loss)
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                total_loss += loss_val
                count += 1
    return total_loss / count if count > 0 else float("inf")


# -- PHASE 1: Diagnostics --------------------------------------------------
logger.info("=== PHASE 1: Diagnostics ===")
set_seed(SEED)

logger.info("Model path: %s", MODEL_PATH)
logger.info("Data path: %s", DATA_PATH)

tapt_files = os.listdir(MODEL_PATH)
logger.info("TAPT files: %s", sorted(tapt_files))
required = {"model.safetensors", "config.json", "tokenizer.json"}
missing = required - set(tapt_files)
assert not missing, f"FATAL: Missing files: {missing}"

# -- PHASE 2: Load data ----------------------------------------------------
logger.info("=== PHASE 2: Load data ===")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
logger.info("Tokenizer: eos='%s' (id=%d)", tokenizer.eos_token, tokenizer.eos_token_id)

# Verify chat template
test_msg = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "ok"}]
ct = tokenizer.apply_chat_template(test_msg, tokenize=False)
assert "<start_of_turn>" in ct, f"FATAL: Bad chat template: {ct}"
logger.info("Chat template OK: %s", ct[:100])

with open(DATA_PATH, encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]
logger.info("Loaded %d training samples", len(raw_data))

# Validate 5 samples
for i, d in enumerate(raw_data[:5]):
    msgs = d["messages"]
    assert len(msgs) == 2, f"Task {i}: need 2 messages, got {len(msgs)}"
    assert msgs[0]["role"] == "user", f"Task {i}: first not user"
    assert msgs[-1]["role"] == "assistant", f"Task {i}: last not assistant"
    assert len(msgs[-1]["content"]) > 5, f"Task {i}: response too short"
logger.info("Data format validation PASS (5 samples)")

# Train/eval split (10% holdout, same as v3)
random.shuffle(raw_data)
n_eval = max(1, int(len(raw_data) * 0.1))
eval_data = raw_data[:n_eval]
train_data = raw_data[n_eval:]
eval_messages = [d["messages"] for d in eval_data]
train_messages_sample = [d["messages"] for d in train_data[:n_eval]]
logger.info("Split: %d train, %d eval", len(train_data), len(eval_data))


# Format dataset: apply chat template
def format_chat(example: dict) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


sft_ds = Dataset.from_list(train_data).map(format_chat, remove_columns=["messages"])
logger.info("SFT dataset: %d examples", len(sft_ds))

# Token length stats
sample_lengths = []
for i in range(min(100, len(train_data))):
    toks = tokenizer.apply_chat_template(train_data[i]["messages"], tokenize=True)
    sample_lengths.append(len(toks))
logger.info(
    "Token lengths (n=%d): min=%d, median=%d, max=%d, >1024=%d",
    len(sample_lengths),
    min(sample_lengths),
    sorted(sample_lengths)[len(sample_lengths) // 2],
    max(sample_lengths),
    sum(1 for s in sample_lengths if s > 1024),
)

# -- PHASE 3: Load model ---------------------------------------------------
logger.info("=== PHASE 3: Load model ===")

# fp32 model + AMP (fp16=True) — proven for Gemma 3 270M on T4 (v3 kernel)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float32, device_map={"": 0}
)
cfg = model.config
logger.info(
    "Model: layers=%d, hidden=%d, heads=%d, params=%.1fM, dropout=%.1f",
    cfg.num_hidden_layers,
    cfg.hidden_size,
    cfg.num_attention_heads,
    sum(p.numel() for p in model.parameters()) / 1e6,
    getattr(cfg, "attention_dropout", 0.0),
)
log_vram("after model load")

# Pre-SFT generation test
pre_sft_results = generate_test(model, tokenizer, "Pre-SFT")

# -- PHASE 4: SFT training -------------------------------------------------
logger.info("=== PHASE 4: SFT training ===")

S = SFT_CFG
total_steps = (len(sft_ds) // S["batch_size"] // S["grad_accum"]) * S["epochs"]
warmup_steps = int(total_steps * S["warmup_pct"])
logger.info(
    "Steps: %d total, %d warmup, save every %d",
    total_steps,
    warmup_steps,
    S["save_steps"],
)

# Assistant-only loss: DataCollatorForCompletionOnlyLM (THE fix for bug #3)
# Masks user tokens with labels=-100, loss only on assistant response.
# response_template = Gemma 3 marker for start of model turn.
response_template = "<start_of_turn>model\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

sft_config = SFTConfig(
    output_dir=SFT_OUTPUT,
    num_train_epochs=S["epochs"],
    per_device_train_batch_size=S["batch_size"],
    gradient_accumulation_steps=S["grad_accum"],
    learning_rate=S["lr"],
    warmup_steps=warmup_steps,
    weight_decay=S["weight_decay"],
    max_grad_norm=S["max_grad_norm"],
    lr_scheduler_type="cosine",  # proven in v3 (NOT constant — overcorrection risk)
    max_seq_length=S["seq_length"],
    fp16=True,
    gradient_checkpointing=True,
    seed=SEED,
    logging_steps=1,
    logging_nan_inf_filter=True,
    eval_strategy="no",  # OOM with 262K vocab eval. Manual eval post-training.
    save_strategy="steps",
    save_steps=S["save_steps"],
    save_only_model=True,
    save_total_limit=S["save_total_limit"],
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=sft_ds,
    data_collator=collator,
    processing_class=tokenizer,
)

logger.info("Starting SFT training (assistant-only loss)...")
log_vram("before training")
result = trainer.train()
train_loss = result.training_loss
logger.info("SFT train_loss: %.4f", train_loss)
assert not math.isnan(train_loss), "FATAL: loss NaN"

trainer.save_model(SFT_OUTPUT)
tokenizer.save_pretrained(SFT_OUTPUT)
logger.info("Model saved: %s", SFT_OUTPUT)

# Free trainer memory for eval
del trainer
gc.collect()
torch.cuda.empty_cache()
log_vram("after trainer cleanup")

# -- PHASE 5: Post-training eval -------------------------------------------
logger.info("=== PHASE 5: Post-training eval ===")

eval_loss = compute_clm_loss(model, tokenizer, eval_messages, S["seq_length"])
train_loss_manual = compute_clm_loss(
    model, tokenizer, train_messages_sample, S["seq_length"]
)
overfit_ratio = eval_loss / train_loss_manual if train_loss_manual > 0 else float("inf")
overfit_flag = overfit_ratio > 2.0
logger.info(
    "Overfit: eval=%.4f / train=%.4f = %.2f (flag=%s)",
    eval_loss,
    train_loss_manual,
    overfit_ratio,
    overfit_flag,
)

post_sft_results = generate_test(model, tokenizer, "Post-SFT")

# Checkpoint listing
epoch_ckpts = sorted([d for d in os.listdir(SFT_OUTPUT) if d.startswith("checkpoint-")])
logger.info("Checkpoints: %s", epoch_ckpts)

# -- PHASE 6: Metrics ------------------------------------------------------
elapsed = (time.time() - t_start) / 60
metrics = {
    "model": "gemma-3-270m-it-tapt-ep1-sft-v5",
    "base": "TAPT ep1 (checkpoint-22)",
    "data": f"{len(sft_ds)} train + {n_eval} eval (from 2142 RAFT, 90 invalid filtered)",
    "training_loss": train_loss,
    "eval_loss": eval_loss,
    "train_loss_manual": train_loss_manual,
    "overfit_ratio": round(overfit_ratio, 4),
    "overfit_flag": overfit_flag,
    "loss_type": "assistant-only (DataCollatorForCompletionOnlyLM, bug #3 fix)",
    "config": S,
    "scheduler": "cosine",
    "neftune": "disabled",
    "checkpoints": epoch_ckpts,
    "pre_sft_responses": pre_sft_results,
    "post_sft_responses": post_sft_results,
    "elapsed_min": round(elapsed, 1),
    "gpu": torch.cuda.get_device_name(0),
    "standards": [
        "RAFT arXiv:2403.10131 (training data: oracle + distractors)",
        "v3 proven hyperparams (memory: feedback_tapt_v3_sweep)",
        "1 change at a time (memory: feedback_tapt_v2_overcorrection)",
    ],
}
metrics_path = os.path.join(OUTPUT_DIR, "sft_v5_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

logger.info("=" * 60)
logger.info("SFT v5 COMPLETE — %.1f min", elapsed)
logger.info(
    "train_loss=%.4f | eval_loss=%.4f | overfit=%.2f | flag=%s",
    train_loss,
    eval_loss,
    overfit_ratio,
    overfit_flag,
)
logger.info("Checkpoints: %s", epoch_ckpts)
logger.info("=" * 60)
