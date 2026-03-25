"""TAPT v3 + Sweep -- v1 exact params, training + eval in one kernel.

TAPT v3 reproduces v1 params exactly (LR 5e-6, dropout 0.1, cosine).
Then sweeps base + 5 epoch checkpoints through RAG pipeline (298 questions).

v2 overcorrected: 3 "fixes" all pushed toward more aggressive training,
destroying faithfulness (45.1% base -> 1-6% TAPT v2). Lesson: one change
at a time, and Google FFT guide != RAG faithfulness guide.

v3 = v1 params reproduced, with per-epoch eval we never had.

Input:
  pguillemin/gemma-3-270m-it (base model)
  pguillemin/pocket-arbiter-gen-data (corpus_paragraphs.jsonl)
  pguillemin/pocket-arbiter-eval-data (DB, GS, eval scripts)
Output:
  /kaggle/working/tapt_v3_metrics.json
  /kaggle/working/tapt_v3_sweep_results.json
  /kaggle/working/generation_eval_base.json
  /kaggle/working/generation_eval_tapt_ep{1-5}.json
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sqlite3  # noqa: E402
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

# -- PHASE 0: Environment ----------------------------------------
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

from datasets import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# -- PHASE 1: Config ---------------------------------------------
logger.info("=== PHASE 1: Config ===")
SEED = 42
OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# Resolve paths
_MODEL_PATHS = [
    "/kaggle/input/gemma-3-270m-it",
    "/kaggle/input/datasets/pguillemin/gemma-3-270m-it",
]
MODEL_ID = next((p for p in _MODEL_PATHS if os.path.isdir(p)), "google/gemma-3-270m-it")
logger.info("Base model: %s", MODEL_ID)

_INPUT_PATHS = [
    "/kaggle/input/pocket-arbiter-gen-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-gen-data",
]
INPUT_DIR = next((p for p in _INPUT_PATHS if os.path.isdir(p)), None)
assert INPUT_DIR is not None, f"FATAL: Gen data not found: {_INPUT_PATHS}"
logger.info("Gen data: %s", INPUT_DIR)

_EVAL_PATHS = [
    "/kaggle/input/pocket-arbiter-eval-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-eval-data",
]
EVAL_DATA_DIR = next((p for p in _EVAL_PATHS if os.path.isdir(p)), None)
assert EVAL_DATA_DIR is not None, f"FATAL: Eval data not found: {_EVAL_PATHS}"
logger.info("Eval data: %s", EVAL_DATA_DIR)

TAPT_CKPT = os.path.join(OUTPUT_DIR, "gemma-270m-cpt-v3")
CORPUS_PATH = os.path.join(INPUT_DIR, "corpus_paragraphs.jsonl")

# v1 EXACT params -- no corrections, no "improvements"
EVAL_HOLDOUT = {
    "H01_2025_26_Conduite_pour_joueur_handicapes.pdf",
    "C04_2025_26_Coupe_de_la_parit\u00e9.pdf",
    "E02-Le_classement_rapide.pdf",
}
TAPT_CFG = dict(
    epochs=5,
    batch_size=1,
    grad_accum=16,
    lr=5e-6,  # v1 actual (NOT 5e-5)
    warmup_pct=0.10,  # v1 actual (NOT 5%)
    weight_decay=0.01,
    max_grad_norm=1.0,
    neftune_alpha=5,
    seq_length=1024,
    dropout=0.1,  # v1 injected (NOT 0.0)
    scheduler="cosine",  # v1 actual (NOT constant)
)
logger.info("TAPT v3 config (= v1 exact): %s", json.dumps(TAPT_CFG, indent=2))


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


# -- PHASE 2: Data loading ---------------------------------------
logger.info("=== PHASE 2: Data loading ===")
set_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
assert tokenizer.eos_token == "<eos>", f"eos={tokenizer.eos_token!r}"
eos_id = tokenizer.convert_tokens_to_ids("<eos>")
logger.info("Tokenizer OK: eos='<eos>' (id=%d)", eos_id)

with open(CORPUS_PATH, encoding="utf-8") as f:
    paragraphs = [json.loads(line) for line in f]
logger.info("Corpus: %d paragraphs", len(paragraphs))

gemma_tok = sum(len(tokenizer.encode(p["text"])) for p in paragraphs)
logger.info("Tokens: %d (Gemma tokenizer)", gemma_tok)
assert gemma_tok >= 300_000, f"Gate GT1 FAIL: {gemma_tok} < 300K"
logger.info("Gate GT1 PASS: %d tokens", gemma_tok)

train_paras = [p for p in paragraphs if p["source"] not in EVAL_HOLDOUT]
eval_paras = [p for p in paragraphs if p["source"] in EVAL_HOLDOUT]
logger.info("Split: %d train, %d eval", len(train_paras), len(eval_paras))

# -- PHASE 3: Model loading --------------------------------------
logger.info("=== PHASE 3: Model loading ===")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map={"": 0}
)
total_params = sum(p.numel() for p in model.parameters())
logger.info("Params: %.1fM", total_params / 1e6)
assert 200e6 < total_params < 400e6, f"Expected ~270M, got {total_params / 1e6:.1f}M"

# v1 INJECT dropout 0.1 -- this is what v1 did
model.config.attention_dropout = TAPT_CFG["dropout"]
logger.info(
    "attention_dropout=%.2f (v1 injected -- v2 removed it, v3 restores it)",
    model.config.attention_dropout,
)
log_vram("after model load")

# -- PHASE 4: Sequence packing -----------------------------------
logger.info("=== PHASE 4: Sequence packing ===")
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


train_seqs = pack_sequences(train_paras, TAPT_CFG["seq_length"])
eval_seqs = pack_sequences(eval_paras, TAPT_CFG["seq_length"])
assert len(train_seqs) > 0 and len(eval_seqs) > 0

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
    return math.exp(min(total_loss / total_tok, 20.0))


# -- PHASE 5: Baseline perplexity --------------------------------
logger.info("=== PHASE 5: Baseline perplexity ===")
ppl_before = compute_perplexity(model, eval_ds)
logger.info("Baseline perplexity: %.2f", ppl_before)

# -- PHASE 6: TAPT training --------------------------------------
logger.info("=== PHASE 6: TAPT v3 training (v1 exact params) ===")
C = TAPT_CFG
total_steps = (len(train_ds) // C["batch_size"] // C["grad_accum"]) * C["epochs"]
warmup_steps = int(total_steps * C["warmup_pct"])
steps_per_epoch = len(train_ds) // C["batch_size"] // C["grad_accum"]
logger.info(
    "Training: %d seqs, %d epochs, eff_batch=%d, ~%d steps (%d/epoch), warmup=%d",
    len(train_ds),
    C["epochs"],
    C["batch_size"] * C["grad_accum"],
    total_steps,
    steps_per_epoch,
    warmup_steps,
)
logger.info(
    "v3 = v1 exact: lr=%.0e, scheduler=%s, dropout=%.1f, warmup=%.0f%%",
    C["lr"],
    C["scheduler"],
    C["dropout"],
    C["warmup_pct"] * 100,
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
    lr_scheduler_type=C["scheduler"],  # cosine (v1)
    optim="adamw_torch",  # v1 (NOT fused)
    neftune_noise_alpha=C["neftune_alpha"],
    fp16=True,
    gradient_checkpointing=True,
    seed=SEED,
    logging_steps=1,
    logging_nan_inf_filter=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=6,  # keep ALL 5 epoch checkpoints + final
    save_only_model=True,
)

trainer = Trainer(
    model=model,
    args=tapt_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
)

logger.info("Starting TAPT v3 training...")
log_vram("before training")
result = trainer.train()
train_loss = result.training_loss
logger.info("TAPT v3 final loss: %.4f", train_loss)
assert not math.isnan(train_loss), "FATAL: loss is NaN"
log_vram("after training")

# Post-training perplexity
ppl_after = compute_perplexity(model, eval_ds)
logger.info("Perplexity: %.2f -> %.2f", ppl_before, ppl_after)

# Save final model
trainer.save_model(TAPT_CKPT)
tokenizer.save_pretrained(TAPT_CKPT)

# List epoch checkpoints
ckpt_dirs = sorted(
    [d for d in os.listdir(TAPT_CKPT) if d.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[1]),
)
logger.info("Epoch checkpoints: %s", ckpt_dirs)

# Save TAPT metrics
tapt_metrics = {
    "perplexity_baseline": round(ppl_before, 4),
    "perplexity_post": round(ppl_after, 4),
    "training_loss": round(train_loss, 4),
    "config": {
        k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in C.items()
    },
    "epoch_checkpoints": ckpt_dirs,
    "v1_reference": {"perplexity_baseline": 37.74, "perplexity_post": 7.98},
    "v2_reference": {
        "citations_base": 45.1,
        "citations_ep1": 4.2,
        "note": "overcorrected",
    },
}
with open(os.path.join(OUTPUT_DIR, "tapt_v3_metrics.json"), "w") as f:
    json.dump(tapt_metrics, f, indent=2)
logger.info("TAPT v3 metrics saved")

# Free trainer
del trainer
gc.collect()
torch.cuda.empty_cache()

# Free model too -- we'll reload per-checkpoint in sweep
del model
gc.collect()
torch.cuda.empty_cache()
log_vram("after full cleanup")

tapt_elapsed = (time.time() - t_start) / 60
logger.info("TAPT v3 training done in %.1f min", tapt_elapsed)

# ==================================================================
# PART 2: SWEEP EVAL -- base + 5 epoch checkpoints
# ==================================================================
logger.info("=" * 60)
logger.info("=== SWEEP EVAL: base + %d TAPT epochs ===", len(ckpt_dirs))
logger.info("=" * 60)

# Import eval functions
sys.path.insert(0, EVAL_DATA_DIR)
from eval_generation import (  # noqa: E402
    check_citation,
    load_annales_questions,
    load_human_questions,
)
from generation_prompt import build_rag_prompt  # noqa: E402

GS_PATH = os.path.join(EVAL_DATA_DIR, "gold_standard_annales_fr_v8_adversarial.json")
DB_PATH = os.path.join(EVAL_DATA_DIR, "corpus_v2_fr.db")
assert os.path.exists(GS_PATH) and os.path.exists(DB_PATH)

human_qs = load_human_questions(GS_PATH)
annales_qs = load_annales_questions(GS_PATH)
assert len(human_qs) == 34 and len(annales_qs) == 264
logger.info("GS loaded: %d human, %d annales", len(human_qs), len(annales_qs))

conn = sqlite3.connect(DB_PATH)

# Build model configs: base + epoch checkpoints
MODEL_CONFIGS = [
    {"name": "base", "path": MODEL_ID, "output": "generation_eval_base.json"},
]
for ckpt in ckpt_dirs:
    step = int(ckpt.split("-")[1])
    epoch = step // steps_per_epoch
    MODEL_CONFIGS.append(
        {
            "name": f"tapt_ep{epoch}",
            "path": os.path.join(TAPT_CKPT, ckpt),
            "output": f"generation_eval_tapt_ep{epoch}.json",
            "step": step,
            "epoch": epoch,
        }
    )
logger.info("Models to evaluate: %s", [c["name"] for c in MODEL_CONFIGS])


def generate_response(mdl, tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=512,
            min_new_tokens=10,
            do_sample=True,
            temperature=0.2,
            top_k=64,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def load_context(conn, source, page):
    rows = conn.execute(
        "SELECT text FROM children WHERE source = ? AND page = ?",
        (source, page),
    ).fetchall()
    return "\n\n".join(r[0] for r in rows)


def eval_model(model_path, model_name, output_path):
    logger.info("--- Evaluating: %s ---", model_name)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map={"": 0}
    )
    mdl.eval()
    logger.info("Loaded: %.0f MB VRAM", torch.cuda.memory_allocated() / 1024 / 1024)

    # Human questions (34)
    human_results = []
    empty_count = 0
    for i, q in enumerate(human_qs):
        prov = q.get("provenance", {})
        src = (prov.get("docs") or [""])[0]
        pg = (prov.get("pages") or [0])[0]
        ctx = load_context(conn, src, pg)
        msgs = build_rag_prompt(q["content"]["question"], ctx)
        resp = generate_response(mdl, tok, msgs)
        if not resp.strip():
            empty_count += 1
        human_results.append(
            {
                "id": q["id"],
                "question": q["content"]["question"],
                "response": resp,
            }
        )
        if (i + 1) % 10 == 0:
            logger.info("  [human %d/%d]", i + 1, len(human_qs))

    # Annales questions (264)
    cited_count = 0
    for i, q in enumerate(annales_qs):
        prov = q.get("provenance", {})
        src = (prov.get("docs") or [""])[0]
        pg = (prov.get("pages") or [0])[0]
        ctx = load_context(conn, src, pg)
        msgs = build_rag_prompt(q["content"]["question"], ctx)
        resp = generate_response(mdl, tok, msgs)
        if not resp.strip():
            empty_count += 1
        if check_citation(resp, prov.get("docs", []), prov.get("pages", [])):
            cited_count += 1
        if (i + 1) % 50 == 0:
            logger.info("  [annales %d/%d]", i + 1, len(annales_qs))

    cited_pct = round(100 * cited_count / len(annales_qs), 1)
    lengths = [len(r["response"].split()) for r in human_results]
    median_len = sorted(lengths)[len(lengths) // 2] if lengths else 0

    output = {
        "model_name": model_name,
        "model_path": model_path,
        "questions": human_results,
        "auto_citation": {
            "total": len(annales_qs),
            "cited_count": cited_count,
            "cited_pct": cited_pct,
        },
        "metadata": {
            "empty_responses": empty_count,
            "median_response_words": median_len,
            "inference_time_min": round((time.time() - t0) / 60, 1),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    elapsed = (time.time() - t0) / 60
    logger.info(
        "%s: citations=%d/%d (%.1f%%), empty=%d, median=%d words, %.1f min",
        model_name,
        cited_count,
        len(annales_qs),
        cited_pct,
        empty_count,
        median_len,
        elapsed,
    )

    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": model_name,
        "cited_pct": cited_pct,
        "empty": empty_count,
        "median_words": median_len,
        "time_min": round(elapsed, 1),
    }


# Run sweep
sweep_results = []
for cfg in MODEL_CONFIGS:
    r = eval_model(cfg["path"], cfg["name"], os.path.join(OUTPUT_DIR, cfg["output"]))
    r["epoch"] = cfg.get("epoch", 0)
    r["step"] = cfg.get("step", 0)
    sweep_results.append(r)

conn.close()

# -- SUMMARY ------------------------------------------------------
logger.info("=" * 60)
logger.info("=== FINAL SUMMARY ===")
total_min = (time.time() - t_start) / 60
logger.info(
    "Total time: %.1f min (TAPT: %.1f + sweep: %.1f)",
    total_min,
    tapt_elapsed,
    total_min - tapt_elapsed,
)
logger.info("")
logger.info(
    "%-12s | %8s | %5s | %8s | %8s", "Model", "Citations", "Empty", "Median", "Time"
)
logger.info("-" * 60)
for r in sweep_results:
    logger.info(
        "%-12s | %7.1f%% | %5d | %8d | %7.1fm",
        r["name"],
        r["cited_pct"],
        r["empty"],
        r["median_words"],
        r["time_min"],
    )

best = max(sweep_results, key=lambda x: x["cited_pct"])
logger.info("")
logger.info("BEST: %s with %.1f%% citations", best["name"], best["cited_pct"])

base_result = next(r for r in sweep_results if r["name"] == "base")
tapt_results = [r for r in sweep_results if r["name"] != "base"]
any_beats_base = any(r["cited_pct"] > base_result["cited_pct"] for r in tapt_results)
logger.info("Any TAPT epoch beats base? %s", any_beats_base)
if not any_beats_base:
    logger.info(
        "CONCLUSION: TAPT hurts faithfulness at v1 params too. SFT on base recommended."
    )

sweep_output = {
    "sweep_results": sweep_results,
    "best_model": best["name"],
    "best_citations_pct": best["cited_pct"],
    "any_tapt_beats_base": any_beats_base,
    "tapt_config": TAPT_CFG,
    "tapt_perplexity": {"before": round(ppl_before, 2), "after": round(ppl_after, 2)},
    "total_time_min": round(total_min, 1),
    "references": {
        "v1": {"tapt_ep5_citations": 36.4, "note": "LR 5e-6, dropout 0.1, cosine"},
        "v2": {
            "best_citations": 6.4,
            "note": "overcorrected (LR 5e-5, dropout 0.0, constant)",
        },
        "eval_v4": {"base": 43.9, "tapt_v1": 36.4, "sft_v3": 28.8},
    },
}
with open(os.path.join(OUTPUT_DIR, "tapt_v3_sweep_results.json"), "w") as f:
    json.dump(sweep_output, f, indent=2, ensure_ascii=False)

logger.info("Results saved: tapt_v3_sweep_results.json")
logger.info("=== DONE ===")
