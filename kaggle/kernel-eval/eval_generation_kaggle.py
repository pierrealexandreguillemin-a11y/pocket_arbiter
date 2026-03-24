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
    GPU_NAME,
    GPU_VRAM_MB,
    GPU_PROPS.major,
    GPU_PROPS.minor,
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
assert (
    EVAL_DATA_DIR is not None
), f"FATAL K2: Eval data not found. Tried: {EVAL_DATA_CANDIDATES}"
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
assert (
    len(annales_qs) == 264
), f"FATAL K4: Expected 264 annales Qs, got {len(annales_qs)}"
logger.info("GS loaded: %d human, %d annales", len(human_qs), len(annales_qs))

# Gate K3: Open DB once
conn = sqlite3.connect(DB_PATH)
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
table_names = [t[0] for t in tables]
assert (
    "children" in table_names
), f"FATAL K3: 'children' table missing. Tables: {table_names}"
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
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        # Gen params state-of-the-art for RAG faithfulness (2026-03-24):
        # - do_sample=True + temp=0.2: Google Gemma 3 default, avoids greedy loops
        # - repetition_penalty=1.2: critical for 270M (Li et al. 2022)
        # - no_repeat_ngram_size=4: prevents loops, safe for domain terms
        # - min_new_tokens=10: prevents empty/1-word responses
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
    return tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
    )


def load_context_from_conn(
    conn: sqlite3.Connection,
    source: str,
    page: int,
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
    # Gemma 3 was NEVER designed for fp16 (HF team confirmation, issue #36822).
    # fp16 produces NaN/empty outputs. T4 lacks native bf16.
    # float32 is correct: 270M = ~1 GB VRAM, fits T4 with margin.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map={"": 0},
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("Model loaded: %.0f MB VRAM (fp32)", vram)

    # Gate K5: smoke test — first real human question (not a dummy prompt)
    q0 = human_qs[0]
    prov0 = q0.get("provenance", {})
    src0 = (prov0.get("docs") or [""])[0]
    pg0 = (prov0.get("pages") or [0])[0]
    ctx0 = load_context_from_conn(conn, src0, pg0)
    test_msgs = build_rag_prompt(q0["content"]["question"], ctx0)
    test_resp = generate_response_gpu(model, tokenizer, test_msgs)
    if not test_resp.strip():
        logger.warning("K5 WARNING: Empty smoke test response from %s", model_name)
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
                "context": context,
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

    # Gate K6: warn if too many empty responses
    total_qs = len(human_qs) + total_annales
    if empty_count > total_qs * 0.1:
        logger.warning(
            "K6 WARNING: %d/%d empty responses (%.1f%%)",
            empty_count,
            total_qs,
            100 * empty_count / total_qs,
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
        output_path,
        cited_count,
        total_annales,
        cited_pct,
        empty_count,
        (time.time() - t0) / 60,
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
        with open(output_path) as fh:
            data = json.load(fh)
        ac = data["auto_citation"]
        meta = data.get("metadata", {})
        logger.info(
            "  %s: citation %d/%d (%.1f%%), empties: %d, time: %.1f min",
            cfg["name"],
            ac["cited_count"],
            ac["total"],
            ac["cited_pct"],
            meta.get("empty_responses", -1),
            meta.get("inference_time_min", -1),
        )
    else:
        logger.error("  %s: OUTPUT MISSING — %s", cfg["name"], output_path)

logger.info("=== DONE ===")
