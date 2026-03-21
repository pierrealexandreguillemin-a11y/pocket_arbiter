"""TAPT + SFT generation fine-tuning for Gemma 3 270M IT.

Kaggle T4 script — self-contained, production-grade.

Input:  /kaggle/input/pocket-arbiter-gen-data/{corpus_paragraphs,reading_tasks}.jsonl
Output: /kaggle/working/{gemma-270m-cpt,gemma-270m-cpt-sft}/
        /kaggle/working/tapt_perplexity.json

Standards: TAPT (Gururangan ACL 2020), AdaptLLM (Cheng ICLR 2024),
    NEFTune (Jain ICLR 2024), Full FT > LoRA (Biderman TMLR 2024).
"""

from __future__ import annotations

import json  # noqa: E401
import logging
import math
import os
import random
import subprocess
import sys
import time

import numpy as np  # noqa: E401
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
t_start = time.time()

# === PHASE 0: Environment ===
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

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "tiktoken==0.9.0", "trl==0.16.0"]
)

try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    logger.info("HF_TOKEN loaded from Kaggle secrets")
except Exception:
    logger.warning("No Kaggle secrets — using env HF_TOKEN or cached token")

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
from trl import SFTConfig, SFTTrainer  # noqa: E402

# === PHASE 1: Config ===
SEED, MODEL_ID, DRY_RUN = 42, "google/gemma-3-270m-it", "--dry-run" in sys.argv

logger.info("=== /kaggle/input/ contents ===")
if os.path.isdir("/kaggle/input"):
    for e in sorted(os.listdir("/kaggle/input")):
        ep = os.path.join("/kaggle/input", e)
        if os.path.isdir(ep):
            logger.info("  %s/ (%d files)", e, len(os.listdir(ep)))
else:
    logger.info("  /kaggle/input does NOT exist (local mode)")

INPUT_CANDIDATES = [
    "/kaggle/input/pocket-arbiter-gen-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-gen-data",
    "kaggle/dataset-generation",
]
INPUT_DIR = next((c for c in INPUT_CANDIDATES if os.path.isdir(c)), None)
assert INPUT_DIR is not None, f"FATAL: No input dir found. Tried: {INPUT_CANDIDATES}"
logger.info("INPUT_DIR: %s", INPUT_DIR)

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "models"
TAPT_CKPT = os.path.join(OUTPUT_DIR, "gemma-270m-cpt")
SFT_CKPT = os.path.join(OUTPUT_DIR, "gemma-270m-cpt-sft")
PPL_PATH = os.path.join(OUTPUT_DIR, "tapt_perplexity.json")
CORPUS_PATH = os.path.join(INPUT_DIR, "corpus_paragraphs.jsonl")
TASKS_PATH = os.path.join(INPUT_DIR, "reading_tasks.jsonl")

EVAL_HOLDOUT = {
    "H01_2025_26_Conduite_pour_joueur_handicapes.pdf",
    "C04_2025_26_Coupe_de_la_parit\u00e9.pdf",
    "E02-Le_classement_rapide.pdf",
}

_D = DRY_RUN
TAPT_CFG = dict(
    epochs=1 if _D else 5,
    batch_size=1 if _D else 4,
    grad_accum=1 if _D else 4,
    lr=5e-6,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    neftune_alpha=5,
    dropout=0.1,
    seq_length=512 if _D else 2048,
    save_total_limit=2,
)
SFT_CFG = dict(
    epochs=1 if _D else 3,
    batch_size=1 if _D else 4,
    grad_accum=1 if _D else 4,
    lr=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    neftune_alpha=5,
    save_total_limit=2,
    seq_length=512 if _D else 2048,
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
    torch.manual_seed(seed)  # noqa: E702
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # noqa: E702
    os.environ["PYTHONHASHSEED"] = str(seed)


def log_vram(label: str) -> None:
    u = torch.cuda.memory_allocated() / 1024 / 1024
    t = GPU_PROPS.total_memory / 1024 / 1024
    logger.info("VRAM [%s]: %.0f / %.0f MB (%.1f%%)", label, u, t, 100 * u / t)


def generate_test(mdl: AutoModelForCausalLM, tok: AutoTokenizer, label: str) -> None:
    """Generate 5 test responses for post-training validation."""
    logger.info("%s generation test:", label)
    mdl.eval()
    for p in TEST_PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(ids, max_new_tokens=50, do_sample=False)
        logger.info(
            "  Q: %s\n  A: %s",
            p,
            tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)[:120],
        )


# === PHASE 2: Data loading ===
logger.info("=== PHASE 2: Data loading ===")
set_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
EOS = "<eos>"  # noqa: S105 — not a password, it's a special token
assert (
    tokenizer.eos_token == EOS
), f"FATAL: eos_token={tokenizer.eos_token!r}, expected {EOS!r}"
eos_id = tokenizer.convert_tokens_to_ids("<eos>")
assert eos_id == 1, f"FATAL: eos_token_id={eos_id}, expected 1"
logger.info("Tokenizer: eos='%s' (id=%d) — VERIFIED", tokenizer.eos_token, eos_id)

with open(CORPUS_PATH, encoding="utf-8") as f:
    paragraphs = [json.loads(line) for line in f]
logger.info("Corpus: %d paragraphs", len(paragraphs))

gemma_tok = sum(len(tokenizer.encode(p["text"])) for p in paragraphs)
tik_tok = sum(
    len(tiktoken.get_encoding("cl100k_base").encode(p["text"])) for p in paragraphs
)
logger.info(
    "Tokens: %d (Gemma), %d (tiktoken), ratio=%.2f",
    gemma_tok,
    tik_tok,
    gemma_tok / tik_tok,
)

train_paras = [p for p in paragraphs if p["source"] not in EVAL_HOLDOUT]
eval_paras = [p for p in paragraphs if p["source"] in EVAL_HOLDOUT]
logger.info(
    "TAPT split: %d train, %d eval (holdout %d docs)",
    len(train_paras),
    len(eval_paras),
    len(EVAL_HOLDOUT),
)
assert len(eval_paras) > 0, "FATAL: No eval paragraphs from holdout sources"

with open(TASKS_PATH, encoding="utf-8") as f:
    reading_tasks = [json.loads(line) for line in f]
logger.info("Reading tasks: %d", len(reading_tasks))
assert (
    len(reading_tasks) >= 500
), f"FATAL Gate G2: Need >= 500 tasks, got {len(reading_tasks)}"

# === PHASE 3: Model loading ===
logger.info("=== PHASE 3: Model loading ===")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
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

if not getattr(cfg, "attention_dropout", 0.0):
    cfg.attention_dropout = TAPT_CFG["dropout"]
    logger.info("Injected attention_dropout=%.1f", TAPT_CFG["dropout"])
dropout_layers = [
    n for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout)
]
logger.info("Dropout layers: %d", len(dropout_layers))
log_vram("after model load")

# === PHASE 4: TAPT training ===
logger.info("=== PHASE 4: TAPT training ===")
random.shuffle(train_paras)


def pack_sequences(paras: list[dict], seq_len: int) -> list[list[int]]:
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
    mdl.eval()
    total_loss, total_tok = 0.0, 0
    for i in range(len(ds)):
        ids = torch.tensor([ds[i]["input_ids"]], device=mdl.device)
        out = mdl(input_ids=ids, labels=ids)
        total_loss += out.loss.item() * ids.shape[1]
        total_tok += ids.shape[1]
    return math.exp(total_loss / total_tok)


ppl_before = compute_perplexity(model, eval_ds)
logger.info("Baseline perplexity: %.2f", ppl_before)

C = TAPT_CFG
tapt_args = TrainingArguments(
    output_dir=TAPT_CKPT,
    num_train_epochs=C["epochs"],
    per_device_train_batch_size=C["batch_size"],
    per_device_eval_batch_size=C["batch_size"],
    gradient_accumulation_steps=C["grad_accum"],
    learning_rate=C["lr"],
    warmup_ratio=C["warmup_ratio"],
    weight_decay=C["weight_decay"],
    max_grad_norm=C["max_grad_norm"],
    neftune_noise_alpha=C["neftune_alpha"],
    fp16=True,
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
result = trainer.train()
logger.info("TAPT final loss: %.4f", result.training_loss)
assert not math.isnan(result.training_loss), "FATAL: TAPT loss is NaN"
assert result.training_loss < 50, f"FATAL: TAPT loss too high: {result.training_loss}"

ppl_after = compute_perplexity(model, eval_ds)
logger.info(
    "Perplexity after TAPT: %.2f (before: %.2f, delta: %.2f)",
    ppl_after,
    ppl_before,
    ppl_after - ppl_before,
)

ppl_result = {
    "baseline": ppl_before,
    "post_tapt": ppl_after,
    "delta": ppl_before - ppl_after,
    "gate_G1": ppl_after < ppl_before,
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(PPL_PATH, "w", encoding="utf-8") as f:
    json.dump(ppl_result, f, indent=2)
logger.info("Perplexity saved: %s (G1=%s)", PPL_PATH, ppl_result["gate_G1"])

trainer.save_model(TAPT_CKPT)
tokenizer.save_pretrained(TAPT_CKPT)
logger.info("TAPT checkpoint: %s", TAPT_CKPT)
generate_test(model, tokenizer, "Post-TAPT")
log_vram("after TAPT")

# === PHASE 5: SFT training ===
logger.info("=== PHASE 5: SFT training ===")

formatted = [
    tokenizer.apply_chat_template(
        t["messages"], tokenize=False, add_generation_prompt=False
    )
    for t in reading_tasks
]
sft_ds = Dataset.from_dict({"text": formatted})
logger.info(
    "SFT dataset: %d examples, sample len: %d chars", len(sft_ds), len(formatted[0])
)

S = SFT_CFG
sft_config = SFTConfig(
    output_dir=SFT_CKPT,
    num_train_epochs=S["epochs"],
    per_device_train_batch_size=S["batch_size"],
    gradient_accumulation_steps=S["grad_accum"],
    learning_rate=S["lr"],
    warmup_ratio=S["warmup_ratio"],
    weight_decay=S["weight_decay"],
    max_grad_norm=S["max_grad_norm"],
    neftune_noise_alpha=S["neftune_alpha"],
    max_seq_length=S["seq_length"],
    fp16=True,
    seed=SEED,
    logging_steps=1,
    logging_nan_inf_filter=True,
    save_strategy="epoch",
    save_total_limit=S["save_total_limit"],
    dataset_text_field="text",
)

sft_trainer = SFTTrainer(
    model=model, args=sft_config, train_dataset=sft_ds, processing_class=tokenizer
)
sft_result = sft_trainer.train()
logger.info("SFT final loss: %.4f", sft_result.training_loss)
assert not math.isnan(sft_result.training_loss), "FATAL: SFT loss is NaN"

sft_trainer.save_model(SFT_CKPT)
tokenizer.save_pretrained(SFT_CKPT)
logger.info("SFT checkpoint: %s", SFT_CKPT)
generate_test(model, tokenizer, "Post-SFT")
log_vram("after SFT")

# === PHASE 6: Output validation ===
logger.info("=== PHASE 6: Output validation ===")
for ckpt, name in [(TAPT_CKPT, "TAPT"), (SFT_CKPT, "SFT")]:
    assert os.path.isdir(ckpt), f"FATAL: {name} checkpoint missing: {ckpt}"
    sz = (
        sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(ckpt)
            for f in fns
        )
        / 1024
        / 1024
    )
    logger.info("%s checkpoint: %.1f MB at %s", name, sz, ckpt)
    assert sz > 100, f"FATAL: {name} checkpoint too small ({sz:.1f} MB)"
assert os.path.exists(PPL_PATH), f"FATAL: Perplexity file missing: {PPL_PATH}"

elapsed = time.time() - t_start
logger.info("=" * 60)
logger.info("TRAINING COMPLETE — %.1f min total", elapsed / 60)
logger.info("TAPT: %s | SFT: %s | PPL: %s", TAPT_CKPT, SFT_CKPT, PPL_PATH)
logger.info(
    "Gate G1 (ppl decreased): %s | GPU: %s",
    ppl_result["gate_G1"],
    torch.cuda.get_device_name(0),
)
logger.info("=" * 60)
