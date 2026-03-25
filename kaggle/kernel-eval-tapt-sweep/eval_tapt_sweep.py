"""Eval TAPT v2 sweep — base + 5 epoch checkpoints through RAG pipeline.

Answers the question: which TAPT epoch gives optimal RAG faithfulness?
Literature (17 papers) + eval v4 show more fine-tuning = less faithfulness.
This sweep finds the perplexity sweet spot for RAG grounding.

Uses kernel_sources to chain TAPT v2 output (no upload needed).
Paths: /kaggle/input/notebooks/pguillemin/pocket-arbiter-tapt-v2/

Input:
  kernel_sources: pguillemin/pocket-arbiter-tapt-v2 (5 epoch checkpoints)
  dataset_sources: pguillemin/gemma-3-270m-it (base model)
                   pguillemin/pocket-arbiter-eval-data (DB, GS, eval scripts)
Output: /kaggle/working/tapt_sweep_results.json
        /kaggle/working/generation_eval_tapt_ep{1-5}.json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── PHASE 0: Environment ────────────────────────────────────────
logger.info("=== PHASE 0: Environment ===")
assert torch.cuda.is_available(), "FATAL: No GPU detected"
GPU_NAME = torch.cuda.get_device_name(0)
GPU_PROPS = torch.cuda.get_device_properties(0)
GPU_VRAM_MB = GPU_PROPS.total_memory / 1024 / 1024
logger.info(
    "GPU: %s (%.0f MB, compute %d.%d)",
    GPU_NAME,
    GPU_VRAM_MB,
    GPU_PROPS.major,
    GPU_PROPS.minor,
)
assert GPU_VRAM_MB >= 14000, f"FATAL: Need >= 14 GB VRAM, got {GPU_VRAM_MB:.0f}"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Resolve paths ────────────────────────────────────────────────
# Eval data (dataset_sources)
EVAL_DATA_CANDIDATES = [
    "/kaggle/input/pocket-arbiter-eval-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-eval-data",
]
EVAL_DATA_DIR = next((c for c in EVAL_DATA_CANDIDATES if os.path.isdir(c)), None)
assert (
    EVAL_DATA_DIR is not None
), f"FATAL: Eval data not found. Tried: {EVAL_DATA_CANDIDATES}"
logger.info("Eval data: %s", EVAL_DATA_DIR)

# Base model (dataset_sources)
BASE_MODEL_CANDIDATES = [
    "/kaggle/input/gemma-3-270m-it",
    "/kaggle/input/datasets/pguillemin/gemma-3-270m-it",
]
BASE_MODEL_DIR = next((c for c in BASE_MODEL_CANDIDATES if os.path.isdir(c)), None)
assert (
    BASE_MODEL_DIR is not None
), f"FATAL: Base model not found. Tried: {BASE_MODEL_CANDIDATES}"
logger.info("Base model: %s", BASE_MODEL_DIR)

# TAPT v2 output (kernel_sources — mounts under /kaggle/input/notebooks/)
TAPT_OUTPUT_CANDIDATES = [
    "/kaggle/input/notebooks/pguillemin/pocket-arbiter-tapt-v2",
    "/kaggle/input/pocket-arbiter-tapt-v2",
]
TAPT_OUTPUT_DIR = None
for c in TAPT_OUTPUT_CANDIDATES:
    if os.path.isdir(c):
        TAPT_OUTPUT_DIR = c
        break
assert (
    TAPT_OUTPUT_DIR is not None
), f"FATAL: TAPT v2 output not found. Tried: {TAPT_OUTPUT_CANDIDATES}"
logger.info("TAPT v2 output: %s", TAPT_OUTPUT_DIR)

# List TAPT output contents for diagnostics
logger.info("TAPT v2 output contents:")
for item in sorted(os.listdir(TAPT_OUTPUT_DIR)):
    item_path = os.path.join(TAPT_OUTPUT_DIR, item)
    if os.path.isdir(item_path):
        logger.info("  %s/ (%d files)", item, len(os.listdir(item_path)))
    else:
        size_kb = os.path.getsize(item_path) / 1024
        logger.info("  %s (%.0f KB)", item, size_kb)

# Find TAPT checkpoint directory
TAPT_CKPT_DIR = os.path.join(TAPT_OUTPUT_DIR, "gemma-270m-cpt-v2")
if not os.path.isdir(TAPT_CKPT_DIR):
    # Fallback: files might be directly in output dir
    TAPT_CKPT_DIR = TAPT_OUTPUT_DIR
logger.info("TAPT checkpoint dir: %s", TAPT_CKPT_DIR)

# Import eval functions
sys.path.insert(0, EVAL_DATA_DIR)
from eval_generation import (  # noqa: E402
    check_citation,
    load_annales_questions,
    load_human_questions,
)
from generation_prompt import build_rag_prompt  # noqa: E402

logger.info("Imports OK from %s", EVAL_DATA_DIR)

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ── PHASE 1: Load data ──────────────────────────────────────────
logger.info("=== PHASE 1: Load data ===")

GS_PATH = os.path.join(EVAL_DATA_DIR, "gold_standard_annales_fr_v8_adversarial.json")
DB_PATH = os.path.join(EVAL_DATA_DIR, "corpus_v2_fr.db")
assert os.path.exists(GS_PATH), f"FATAL: GS not found: {GS_PATH}"
assert os.path.exists(DB_PATH), f"FATAL: DB not found: {DB_PATH}"

human_qs = load_human_questions(GS_PATH)
annales_qs = load_annales_questions(GS_PATH)
assert len(human_qs) == 34, f"FATAL: Expected 34 human Qs, got {len(human_qs)}"
assert len(annales_qs) == 264, f"FATAL: Expected 264 annales Qs, got {len(annales_qs)}"
logger.info("GS loaded: %d human, %d annales", len(human_qs), len(annales_qs))

conn = sqlite3.connect(DB_PATH)
logger.info("DB opened: %s", DB_PATH)

# ── PHASE 2: Build model configs ────────────────────────────────
logger.info("=== PHASE 2: Build model configs ===")

# Discover TAPT epoch checkpoints
epoch_ckpts = sorted(
    [d for d in os.listdir(TAPT_CKPT_DIR) if d.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[1]),
)
logger.info("TAPT v2 epoch checkpoints: %s", epoch_ckpts)

MODEL_CONFIGS = [
    {"name": "base", "path": BASE_MODEL_DIR, "output": "generation_eval_base.json"},
]
for ckpt in epoch_ckpts:
    ckpt_path = os.path.join(TAPT_CKPT_DIR, ckpt)
    step = int(ckpt.split("-")[1])
    epoch = step // 22  # 22 steps per epoch (346 seqs / 16 eff_batch)
    MODEL_CONFIGS.append(
        {
            "name": f"tapt_ep{epoch}",
            "path": ckpt_path,
            "output": f"generation_eval_tapt_ep{epoch}.json",
            "step": str(step),
            "epoch": str(epoch),
        }
    )

for cfg in MODEL_CONFIGS:
    safetensors = os.path.join(cfg["path"], "model.safetensors")
    assert os.path.exists(
        safetensors
    ), f"FATAL: {cfg['name']} missing model.safetensors at {cfg['path']}"
    size_mb = os.path.getsize(safetensors) / 1024 / 1024
    logger.info("  %s: %s (%.0f MB)", cfg["name"], cfg["path"], size_mb)

logger.info("Total models to evaluate: %d", len(MODEL_CONFIGS))

# ── PHASE 3: Inference ──────────────────────────────────────────
logger.info("=== PHASE 3: Inference ===")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def generate_response_gpu(model, tokenizer, messages):
    """GPU inference with chat template + state-of-the-art gen params."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            min_new_tokens=10,
            do_sample=True,
            temperature=0.2,
            top_k=64,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def load_context_from_conn(conn, source, page):
    """Load chunk context using shared DB connection."""
    rows = conn.execute(
        "SELECT text FROM children WHERE source = ? AND page = ?",
        (source, page),
    ).fetchall()
    return "\n\n".join(r[0] for r in rows)


def eval_model(model_path, model_name, output_path, human_qs, annales_qs, conn):
    """Evaluate one model on all questions and save JSON."""
    logger.info("--- Evaluating: %s (%s) ---", model_name, model_path)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map={"": 0},
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("Model loaded: %.0f MB VRAM (fp32)", vram)

    # Smoke test
    q0 = human_qs[0]
    prov0 = q0.get("provenance", {})
    src0 = (prov0.get("docs") or [""])[0]
    pg0 = (prov0.get("pages") or [0])[0]
    ctx0 = load_context_from_conn(conn, src0, pg0)
    test_msgs = build_rag_prompt(q0["content"]["question"], ctx0)
    test_resp = generate_response_gpu(model, tokenizer, test_msgs)
    if not test_resp.strip():
        logger.warning("WARNING: Empty smoke test from %s", model_name)
    else:
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
        human_results.append(
            {
                "id": q["id"],
                "question": q["content"]["question"],
                "response": response,
                "scores": {"useful": None, "faithful": None, "cited": None},
            }
        )
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

    # Response length stats
    lengths = [len(r["response"].split()) for r in human_results]
    median_len = sorted(lengths)[len(lengths) // 2] if lengths else 0
    mean_len = sum(lengths) / len(lengths) if lengths else 0

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
            "median_response_words": median_len,
            "mean_response_words": round(mean_len, 1),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(
        "Saved: %s (citation: %d/%d = %.1f%%, empties: %d, median: %d words, time: %.1f min)",
        output_path,
        cited_count,
        total_annales,
        cited_pct,
        empty_count,
        median_len,
        (time.time() - t0) / 60,
    )

    # Free VRAM
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": model_name,
        "cited_pct": cited_pct,
        "empty": empty_count,
        "median_words": median_len,
        "mean_words": round(mean_len, 1),
        "time_min": round((time.time() - t0) / 60, 1),
    }


# ── Run eval loop ────────────────────────────────────────────────
t_total = time.time()
sweep_results = []

for cfg in MODEL_CONFIGS:
    output_path = os.path.join(OUTPUT_DIR, cfg["output"])
    result = eval_model(
        cfg["path"],
        cfg["name"],
        output_path,
        human_qs,
        annales_qs,
        conn,
    )
    result["epoch"] = int(cfg.get("epoch", 0))
    result["step"] = int(cfg.get("step", 0))
    sweep_results.append(result)

conn.close()

# ── PHASE 4: Summary ────────────────────────────────────────────
logger.info("=== PHASE 4: Summary ===")
total_min = (time.time() - t_total) / 60
logger.info("Total time: %.1f min", total_min)

logger.info("")
logger.info(
    "%-12s | %8s | %5s | %8s | %8s",
    "Model",
    "Citations",
    "Empty",
    "Median w",
    "Time min",
)
logger.info("-" * 60)
for r in sweep_results:
    logger.info(
        "%-12s | %7.1f%% | %5d | %8d | %8.1f",
        r["name"],
        r["cited_pct"],
        r["empty"],
        r["median_words"],
        r["time_min"],
    )

# Find optimal epoch (max citations)
best = max(sweep_results, key=lambda x: x["cited_pct"])
logger.info("")
logger.info(
    "BEST FAITHFULNESS: %s with %.1f%% citations", best["name"], best["cited_pct"]
)

# Save sweep summary
sweep_output = {
    "sweep_results": sweep_results,
    "best_model": best["name"],
    "best_citations_pct": best["cited_pct"],
    "total_time_min": round(total_min, 1),
    "eval_v4_reference": {"base": 43.9, "tapt_v1": 36.4, "sft_v3": 28.8},
}
sweep_path = os.path.join(OUTPUT_DIR, "tapt_sweep_results.json")
with open(sweep_path, "w", encoding="utf-8") as f:
    json.dump(sweep_output, f, indent=2, ensure_ascii=False)
logger.info("Sweep results saved: %s", sweep_path)

logger.info("=== DONE ===")
