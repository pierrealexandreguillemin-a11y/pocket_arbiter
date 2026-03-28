"""Eval v5 generation on Kaggle T4 — 5 models x 298 questions.

Compares base, TAPT ep1, and 3 SFT v5 checkpoints (80, 120, 242).
Imports eval functions from pocket-arbiter-eval-data dataset.
Loops over models sequentially, freeing VRAM between each.
Outputs 5 JSON files + 1 summary to /kaggle/working/.

Ref: SFT v5 = RAFT-style (Gemma 27B teacher, 2232 entries, 95.1% citations valid)
     TAPT ep1 = best baseline (46.2% citations, sweep v3)

Reproducibility: seed=42 set before each model eval (torch + cuda).
  Results reproducible on same hardware/driver. PyTorch docs recommend
  cudnn.benchmark=False for deterministic behavior.

Metric: cited_pct = regex match (doc name OR page number) on 264 annales.
  This is a PROXY for faithfulness, not semantic entailment (ICTIR 2025).
  Industry standard (FACTS Grounding, Google 2025) uses LLM-as-judge,
  not feasible offline on T4. Relative comparison across models is valid.
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

# Reproducibility (PyTorch docs, HF issue #3154)
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
logger.info("Seed: %d, cudnn.benchmark=False, cudnn.deterministic=True", SEED)

# Version log (traceability — reproducible from git commit + these versions)
import transformers  # noqa: E402

logger.info(
    "torch=%s, transformers=%s, cuda=%s",
    torch.__version__,
    transformers.__version__,
    torch.version.cuda,
)

# Resolve dataset paths (two mount patterns with fallback)
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


def resolve_model_path(slug: str, subdir: str | None = None) -> str:
    """Find model path in Kaggle input, testing both mount patterns.

    Args:
        slug: Kaggle dataset slug.
        subdir: Optional subdirectory within the dataset.

    Returns:
        Absolute path to the model directory.

    Raises:
        FileNotFoundError: If neither mount pattern contains the model.
    """
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/pguillemin/{slug}",
    ]
    for c in candidates:
        if os.path.isdir(c):
            if subdir is None:
                return c
            path = os.path.join(c, subdir)
            if os.path.isdir(path):
                return path
            raise FileNotFoundError(f"Subdir '{subdir}' not found in {c}")
    raise FileNotFoundError(f"Model {slug} not found. Tried: {candidates}")


# 5 models: base, tapt_ep1, 3 SFT v5 checkpoints
_ModelCfg = dict[str, str | None]
MODEL_CONFIGS: list[_ModelCfg] = [
    {
        "name": "base",
        "slug": "gemma-3-270m-it",
        "subdir": None,
        "output": "generation_eval_v5_base.json",
    },
    {
        "name": "tapt_ep1",
        "slug": "pa-tapt-ep1-checkpoint",
        "subdir": None,
        "output": "generation_eval_v5_tapt_ep1.json",
    },
    {
        "name": "sft_v5_ckpt80",
        "slug": "pa-sft-v5-checkpoints",
        "subdir": "checkpoint-80",
        "output": "generation_eval_v5_sft80.json",
    },
    {
        "name": "sft_v5_ckpt120",
        "slug": "pa-sft-v5-checkpoints",
        "subdir": "checkpoint-120",
        "output": "generation_eval_v5_sft120.json",
    },
    {
        "name": "sft_v5_ckpt242",
        "slug": "pa-sft-v5-checkpoints",
        "subdir": "checkpoint-242",
        "output": "generation_eval_v5_sft242.json",
    },
]

# Gate K2: Verify all datasets are mounted
for cfg in MODEL_CONFIGS:
    slug = cfg["slug"]
    assert slug is not None
    path = resolve_model_path(slug, cfg["subdir"])
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
        # - do_sample=True + temp=0.2: Google Gemma 3 default
        # - repetition_penalty=1.2: critical for 270M (Li et al. 2022)
        # - no_repeat_ngram_size=4: prevents loops
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
) -> dict:
    """Evaluate one model on all questions and save JSON. Returns summary."""
    logger.info("--- Evaluating: %s (%s) ---", model_name, model_path)
    t0 = time.time()

    # Reset seed per model for reproducible comparison (HF #3154)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Gemma 3 fp16 = NaN on T4 (issue #36822). float32 = ~1 GB VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map={"": 0},
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("Model loaded: %.0f MB VRAM (fp32)", vram)

    # Gate K5: smoke test — first human question
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

    # Response length stats
    lengths = [len(r["response"].split()) for r in human_results]
    median_len = sorted(lengths)[len(lengths) // 2] if lengths else 0
    mean_len = sum(lengths) / len(lengths) if lengths else 0

    # Gate K6: warn if too many empty responses
    total_qs = len(human_qs) + total_annales
    if empty_count > total_qs * 0.1:
        logger.warning(
            "K6 WARNING: %d/%d empty responses (%.1f%%)",
            empty_count,
            total_qs,
            100 * empty_count / total_qs,
        )

    elapsed_min = round((time.time() - t0) / 60, 1)

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
            "seed": SEED,
            "empty_responses": empty_count,
            "inference_time_min": elapsed_min,
            "median_response_words": median_len,
            "mean_response_words": round(mean_len, 1),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(
        "Saved: %s (citation: %d/%d = %.1f%%, empties: %d, "
        "median: %d words, time: %.1f min)",
        output_path,
        cited_count,
        total_annales,
        cited_pct,
        empty_count,
        median_len,
        elapsed_min,
    )

    # Free VRAM
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("VRAM freed: %.0f MB remaining", vram_after)

    return {
        "name": model_name,
        "cited_pct": cited_pct,
        "empty": empty_count,
        "median_words": median_len,
        "mean_words": round(mean_len, 1),
        "time_min": elapsed_min,
    }


# ============================================================
# PHASE 3: Run eval on all 5 models
# ============================================================

logger.info("=== PHASE 3: Eval loop (5 models) ===")
t_total = time.time()
eval_results = []

for cfg in MODEL_CONFIGS:
    slug = cfg["slug"]
    assert slug is not None
    model_path = resolve_model_path(slug, cfg["subdir"])
    name = cfg["name"] or "unknown"
    out = cfg["output"] or "output.json"
    output_path = os.path.join(OUTPUT_DIR, out)
    result = eval_model(model_path, name, output_path, human_qs, annales_qs, conn)
    eval_results.append(result)

conn.close()

# ============================================================
# PHASE 4: Comparison summary
# ============================================================

logger.info("=== PHASE 4: Summary ===")
total_min = (time.time() - t_total) / 60
logger.info("Total time: %.1f min", total_min)

logger.info("")
logger.info(
    "%-16s | %8s | %5s | %8s | %8s",
    "Model",
    "Citations",
    "Empty",
    "Median w",
    "Time min",
)
logger.info("-" * 65)
for r in eval_results:
    logger.info(
        "%-16s | %7.1f%% | %5d | %8d | %8.1f",
        r["name"],
        r["cited_pct"],
        r["empty"],
        r["median_words"],
        r["time_min"],
    )

# Best model
best = max(eval_results, key=lambda x: x["cited_pct"])
logger.info("")
logger.info(
    "BEST FAITHFULNESS: %s with %.1f%% citations", best["name"], best["cited_pct"]
)

# Reference baselines from previous evals
BASELINES = {"base_ref": 43.9, "tapt_ep1_ref": 46.2}

# Save comparison summary
summary = {
    "eval_results": eval_results,
    "best_model": best["name"],
    "best_citations_pct": best["cited_pct"],
    "total_time_min": round(total_min, 1),
    "baselines": BASELINES,
    "decision": (
        f"SFT v5 WINS ({best['name']} {best['cited_pct']}% > 46.2%)"
        if best["cited_pct"] > 46.2
        else "TAPT ep1 remains best (46.2%)"
    ),
}
summary_path = os.path.join(OUTPUT_DIR, "eval_v5_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
logger.info("Summary saved: %s", summary_path)

logger.info("=== DONE ===")
