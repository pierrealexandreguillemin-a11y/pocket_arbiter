"""SFT v5 on 1B: LoRA fine-tune Gemma 3 1B IT on RAFT data (Unsloth).

Adapts the proven SFT v5 pipeline (270M) for Gemma 3 1B IT:
- Unsloth + LoRA NF4 (1B too large for FFT on T4 16 GB)
- Gemma 3 1B IT base (no TAPT — 1B at 56.7% pipeline already)
- Same RAFT data: 2142 entries (95.1% citations valid, Gemma 3 4B teacher)
- train_on_responses_only via Unsloth (bug #3 fix: assistant-only loss)

Memory refs: feedback_unsloth_mandatory, feedback_kaggle_t4_mandatory,
  feedback_tapt_v3_sweep, feedback_sft_data_quality_disaster

Base: Gemma 3 1B IT (pipeline 56.7% citations, oracle 46.2%)
Data: 2142 samples (1729 oracle cited, 307 abstain, 106 memorize)
Gate: pipeline citations > 56.7% = success. Else 1B base = final model.

Input:  pguillemin/gemma-3-1b-it (base model, 2 GB)
        pguillemin/pa-sft-v5-data (sft_train_v5.jsonl, 8.6 MB)
Output: /kaggle/working/gemma-1b-sft-v5/ (merged model for eval)

VRAM budget (T4 15 GB):
  Model NF4 ~600 MB + LoRA ~50 MB + optimizer ~200 MB + activations ~2 GB
  Total ~3 GB => comfortable on T4 (12 GB headroom)

Runtime estimate: 2142 samples x 2 epochs x ~0.3s/step = ~20 min training
  + load/save/eval ~15 min = ~35 min total

ISO tracking: seed=42, torch.deterministic, dataset pguillemin/pa-sft-v5-data,
  model pguillemin/gemma-3-1b-it, Kaggle T4 (sm_75)
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

# ============================================================
# PHASE 0: Environment
# ============================================================
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
), f"FATAL: Need compute >= 7.0 (T4), got {GPU_PROPS.major}.{GPU_PROPS.minor}"

logger.info("torch=%s, cuda=%s", torch.__version__, torch.version.cuda)

# Install Unsloth — MANDATORY for Gemma 3 on T4 (feedback_unsloth_mandatory)
# CRITICAL: Kaggle image has transformers==5.0.0, trl==1.0.0, datasets==4.8.3
# Unsloth pyproject.toml excludes !=5.0.0 and requires trl<=0.24.0.
# --no-deps avoids downgrading transformers. Hard deps installed explicitly.
# v1 CRASH: bitsandbytes missing → ModuleNotFoundError at unsloth import.
logger.info("Installing Unsloth (--no-deps) + pinned hard deps...")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "--upgrade",
        "--no-cache-dir",
        "unsloth",
        "unsloth_zoo",
    ],
    timeout=600,
)
# Hard deps: bitsandbytes (NF4 quant), trl==0.24.0 (SFTTrainer compat),
# peft (LoRA), cut_cross_entropy/msgspec/tyro/hf_transfer (unsloth_zoo reqs)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "trl==0.24.0",
        "peft",
        "bitsandbytes",
        "cut_cross_entropy",
        "msgspec",
        "tyro",
        "hf_transfer",
    ],
    timeout=300,
)

# Unsloth MUST be imported BEFORE transformers (warning from v1 log line 25)
import unsloth  # noqa: E402, I001
from unsloth import FastModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

import transformers  # noqa: E402
import trl  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import GenerationConfig  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

logger.info(
    "unsloth=%s, transformers=%s, trl=%s",
    getattr(unsloth, "__version__", "?"),
    transformers.__version__,
    trl.__version__,
)

# ============================================================
# Config
# ============================================================
SEED = 42

# Resolve dataset paths (two Kaggle mount patterns)
_MODEL_PATHS = [
    "/kaggle/input/gemma-3-1b-it",
    "/kaggle/input/datasets/pguillemin/gemma-3-1b-it",
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

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SFT_OUTPUT = os.path.join(OUTPUT_DIR, "gemma-1b-sft-v5")

# LoRA config
LORA_CFG = {
    "r": 16,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "lora_alpha": 32,  # 2x rank = standard (Unsloth docs, Medium GRPO example)
    "lora_dropout": 0,
    "use_gradient_checkpointing": "unsloth",
    "random_state": SEED,
}

# Training config — adapted from 270M SFT v5 proven hyperparams
SFT_CFG = {
    "epochs": 2,
    "batch_size": 2,  # NF4 + LoRA fits batch_size=2 on T4
    "grad_accum": 8,  # effective batch = 16 (same as 270M)
    "lr": 2e-5,  # slightly higher for LoRA (adapters, not full weights)
    "warmup_pct": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_steps": 20,
    "save_only_model": True,
    "save_total_limit": 12,
    "seq_length": 1024,
}

TEST_PROMPTS = [
    "Qu'est-ce qu'un forfait en competition FFE ?",
    "Quelles sont les cadences pour le championnat de France ?",
    "Comment fonctionne le departage Buchholz ?",
]

GEN_CONFIG = GenerationConfig(
    do_sample=True,
    temperature=0.2,
    top_k=64,
    top_p=0.95,
    max_new_tokens=150,
    min_new_tokens=10,
    repetition_penalty=1.2,
    no_repeat_ngram_size=4,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_vram(label: str) -> None:
    u = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("VRAM [%s]: %.0f / %.0f MB", label, u, GPU_VRAM_MB)


def generate_test(mdl, tok, label: str) -> list:
    """Generate test responses for pre/post-training comparison."""
    logger.info("%s generation test:", label)
    mdl.eval()
    results = []
    for p in TEST_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=SFT_CFG["seq_length"],
        )
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                generation_config=GEN_CONFIG,
                pad_token_id=tok.eos_token_id,
            )
        answer = tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        logger.info("  Q: %s\n  A: %s", p, answer[:200])
        results.append({"question": p, "answer": answer[:500]})
    return results


# ============================================================
# PHASE 1: Diagnostics
# ============================================================
logger.info("=== PHASE 1: Diagnostics ===")
set_seed(SEED)

logger.info("Model: %s", MODEL_PATH)
logger.info("Data:  %s", DATA_PATH)
logger.info("Output: %s", SFT_OUTPUT)

model_files = os.listdir(MODEL_PATH)
logger.info(
    "Model files: %s", sorted([f for f in model_files if not f.startswith(".")])
)
required = {"model.safetensors", "config.json", "tokenizer.json"}
missing = required - set(model_files)
assert not missing, f"FATAL: Missing model files: {missing}"

# ============================================================
# PHASE 2: Load data
# ============================================================
logger.info("=== PHASE 2: Load data ===")

with open(DATA_PATH, encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]
logger.info("Loaded %d training samples", len(raw_data))

# Validate 5 samples — MANDATORY (feedback_sft_data_quality_disaster)
for i, d in enumerate(raw_data[:5]):
    msgs = d["messages"]
    assert len(msgs) == 2, f"Sample {i}: need 2 messages, got {len(msgs)}"
    assert msgs[0]["role"] == "user", f"Sample {i}: first not user"
    assert msgs[-1]["role"] == "assistant", f"Sample {i}: last not assistant"
    assert (
        len(msgs[-1]["content"]) > 10
    ), f"Sample {i}: response too short ({len(msgs[-1]['content'])} chars)"
    logger.info(
        "  Sample %d: Q=%s... A=%s...",
        i,
        msgs[0]["content"][:60],
        msgs[-1]["content"][:60],
    )
logger.info("Data validation PASS (5 samples inspected)")

# Train/eval split (10% holdout)
random.shuffle(raw_data)
n_eval = max(1, int(len(raw_data) * 0.1))
eval_data = raw_data[:n_eval]
train_data = raw_data[n_eval:]
eval_messages = [d["messages"] for d in eval_data]
train_messages_sample = [d["messages"] for d in train_data[:n_eval]]
logger.info("Split: %d train, %d eval", len(train_data), len(eval_data))

# ============================================================
# PHASE 3: Load model (Unsloth + LoRA)
# ============================================================
logger.info("=== PHASE 3: Load model (Unsloth NF4 + LoRA) ===")

model, tokenizer = FastModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=SFT_CFG["seq_length"],
    load_in_4bit=True,
)
log_vram("after NF4 load")

cfg = model.config
logger.info(
    "Model: layers=%d, hidden=%d, heads=%d, vocab=%d",
    cfg.num_hidden_layers,
    cfg.hidden_size,
    cfg.num_attention_heads,
    cfg.vocab_size,
)

# Verify chat template
test_msg = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "ok"}]
ct = tokenizer.apply_chat_template(test_msg, tokenize=False)
assert "<start_of_turn>" in ct, f"FATAL: Bad chat template: {ct}"
logger.info("Chat template OK: %s", ct[:100])

# Apply LoRA
model = FastModel.get_peft_model(
    model,
    r=LORA_CFG["r"],
    target_modules=LORA_CFG["target_modules"],
    lora_alpha=LORA_CFG["lora_alpha"],
    lora_dropout=LORA_CFG["lora_dropout"],
    use_gradient_checkpointing=LORA_CFG["use_gradient_checkpointing"],
    random_state=LORA_CFG["random_state"],
)
log_vram("after LoRA")

# Count trainable params
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(
    "Params: %.1fM total, %.1fM trainable (%.1f%%)",
    total_params / 1e6,
    trainable_params / 1e6,
    100 * trainable_params / total_params,
)

# Pre-SFT generation test
pre_sft_results = generate_test(model, tokenizer, "Pre-SFT")

# ============================================================
# PHASE 4: Format dataset + train
# ============================================================
logger.info("=== PHASE 4: SFT training ===")


def format_chat(example: dict) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
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
    "Token lengths (n=%d): min=%d, median=%d, max=%d, >%d=%d",
    len(sample_lengths),
    min(sample_lengths),
    sorted(sample_lengths)[len(sample_lengths) // 2],
    max(sample_lengths),
    SFT_CFG["seq_length"],
    sum(1 for s in sample_lengths if s > SFT_CFG["seq_length"]),
)

S = SFT_CFG
total_steps = (len(sft_ds) // S["batch_size"] // S["grad_accum"]) * S["epochs"]
warmup_steps = int(total_steps * S["warmup_pct"])
logger.info(
    "Steps: %d total, %d warmup, save every %d",
    total_steps,
    warmup_steps,
    S["save_steps"],
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
    optim="adamw_8bit",  # Unsloth recommended for memory efficiency
    lr_scheduler_type="cosine",
    max_seq_length=S["seq_length"],
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    seed=SEED,
    logging_steps=1,
    logging_nan_inf_filter=True,
    eval_strategy="no",
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
    processing_class=tokenizer,
)

# Apply Unsloth train_on_responses_only (bug #3 fix: assistant-only loss)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)
logger.info("train_on_responses_only applied (assistant-only loss)")

logger.info("Starting SFT training...")
log_vram("before training")
result = trainer.train()
train_loss = result.training_loss
logger.info("SFT train_loss: %.4f", train_loss)
assert not math.isnan(train_loss), "FATAL: loss NaN"

# Save LoRA adapters
trainer.save_model(SFT_OUTPUT)
tokenizer.save_pretrained(SFT_OUTPUT)
logger.info("LoRA adapters saved: %s", SFT_OUTPUT)

# Free trainer memory
del trainer
gc.collect()
torch.cuda.empty_cache()
log_vram("after trainer cleanup")

# ============================================================
# PHASE 5: Merge and save full model
# ============================================================
logger.info("=== PHASE 5: Merge LoRA + save full model ===")

merged_dir = os.path.join(OUTPUT_DIR, "gemma-1b-sft-v5-merged")
# maximum_memory_usage=0.5 prevents OOM during merge (issue #1667)
model.save_pretrained_merged(
    merged_dir,
    tokenizer,
    save_method="merged_16bit",
    maximum_memory_usage=0.5,
)
logger.info("Merged model saved: %s", merged_dir)
log_vram("after merge")

# ============================================================
# PHASE 6: Post-training eval
# ============================================================
logger.info("=== PHASE 6: Post-training eval ===")

post_sft_results = generate_test(model, tokenizer, "Post-SFT")


# Manual CLM loss for overfit detection
@torch.no_grad()
def compute_clm_loss(mdl, tok, messages_list: list, max_len: int) -> float:
    mdl.eval()
    total_loss, count = 0.0, 0
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


eval_loss = compute_clm_loss(model, tokenizer, eval_messages, S["seq_length"])
train_loss_manual = compute_clm_loss(
    model, tokenizer, train_messages_sample, S["seq_length"]
)
overfit_ratio = eval_loss / train_loss_manual if train_loss_manual > 0 else float("inf")
overfit_flag = overfit_ratio > 2.0
logger.info(
    "Overfit: eval=%.4f / train=%.4f = %.3f (flag=%s)",
    eval_loss,
    train_loss_manual,
    overfit_ratio,
    overfit_flag,
)

# List checkpoints
ckpts = sorted([d for d in os.listdir(SFT_OUTPUT) if d.startswith("checkpoint-")])
logger.info("Checkpoints: %s", ckpts)

# ============================================================
# PHASE 7: Metrics
# ============================================================
elapsed = (time.time() - t_start) / 60
metrics = {
    "model": "gemma-3-1b-it-sft-v5-lora",
    "base": "Gemma 3 1B IT (no TAPT)",
    "method": "Unsloth LoRA NF4 + train_on_responses_only",
    "data": f"{len(sft_ds)} train + {n_eval} eval (from 2142 RAFT)",
    "training_loss": train_loss,
    "eval_loss": eval_loss,
    "train_loss_manual": train_loss_manual,
    "overfit_ratio": round(overfit_ratio, 4),
    "overfit_flag": overfit_flag,
    "lora_config": LORA_CFG,
    "training_config": S,
    "trainable_params_M": round(trainable_params / 1e6, 1),
    "total_params_M": round(total_params / 1e6, 1),
    "trainable_pct": round(100 * trainable_params / total_params, 2),
    "checkpoints": ckpts,
    "pre_sft_responses": pre_sft_results,
    "post_sft_responses": post_sft_results,
    "elapsed_min": round(elapsed, 1),
    "gpu": torch.cuda.get_device_name(0),
    "seed": SEED,
    "standards": [
        "RAFT arXiv:2403.10131 (training data: oracle + distractors)",
        "Unsloth mandatory for Gemma 3 on T4 (memory: feedback_unsloth_mandatory)",
        "1 change at a time (memory: feedback_tapt_v2_overcorrection)",
    ],
    "baselines": {
        "1b_base_pipeline": 56.7,
        "1b_base_oracle": 46.2,
        "270m_sft80_pipeline": 48.7,
    },
}
metrics_path = os.path.join(OUTPUT_DIR, "sft_v5_1b_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
logger.info("Metrics saved: %s", metrics_path)

logger.info("=" * 60)
logger.info("SFT v5 1B COMPLETE — %.1f min", elapsed)
logger.info(
    "train_loss=%.4f | eval_loss=%.4f | overfit=%.3f | flag=%s",
    train_loss,
    eval_loss,
    overfit_ratio,
    overfit_flag,
)
logger.info(
    "Trainable: %.1fM / %.1fM (%.1f%%)",
    trainable_params / 1e6,
    total_params / 1e6,
    100 * trainable_params / total_params,
)
logger.info("Checkpoints: %s", ckpts)
logger.info("=" * 60)
