"""Eval v5b: end-to-end RAG pipeline (real retrieval, not oracle).

Uses pre-computed retrieval contexts from retrieval_contexts_v5b.jsonl.
Splits questions into HIT (retrieval found correct page) vs MISS.
Key metric: abstention rate on MISS (UAEval4RAG, Meta CRAG).

Standards: RAGChecker NeurIPS 2024, RAGAS 2023, eRAG 2024, ICTIR 2025.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import sys
import time

import torch

# ============================================================
# PHASE 0: Environment
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("=== PHASE 0: Environment ===")

assert torch.cuda.is_available(), "FATAL: No GPU"
GPU_NAME = torch.cuda.get_device_name(0)
logger.info("GPU: %s", GPU_NAME)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import transformers  # noqa: E402

logger.info(
    "torch=%s, transformers=%s, cuda=%s",
    torch.__version__,
    transformers.__version__,
    torch.version.cuda,
)

# Resolve eval data
EVAL_DATA_CANDIDATES = [
    "/kaggle/input/pocket-arbiter-eval-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-eval-data",
]
EVAL_DATA_DIR = next((c for c in EVAL_DATA_CANDIDATES if os.path.isdir(c)), None)
assert EVAL_DATA_DIR is not None, f"FATAL: eval data not found: {EVAL_DATA_CANDIDATES}"

RETRIEVAL_PATH = os.path.join(EVAL_DATA_DIR, "retrieval_contexts_v5b.jsonl")
assert os.path.exists(RETRIEVAL_PATH), f"FATAL: {RETRIEVAL_PATH} not found"

sys.path.insert(0, EVAL_DATA_DIR)
from generation_prompt import build_rag_prompt  # noqa: E402

logger.info("Eval data: %s", EVAL_DATA_DIR)

# Model path resolver (same as v5)
_ModelCfg = dict[str, str | None]


def resolve_model_path(slug: str, subdir: str | None = None) -> str:
    """Find model in Kaggle input (both mount patterns)."""
    for prefix in ["/kaggle/input", "/kaggle/input/datasets/pguillemin"]:
        c = f"{prefix}/{slug}"
        if os.path.isdir(c):
            if subdir is None:
                return c
            path = os.path.join(c, subdir)
            if os.path.isdir(path):
                return path
    raise FileNotFoundError(f"Model {slug} not found")


MODEL_CONFIGS: list[_ModelCfg] = [
    {
        "name": "base",
        "slug": "gemma-3-270m-it",
        "subdir": None,
        "output": "eval_v5b_base.json",
    },
    {
        "name": "tapt_ep1",
        "slug": "pa-tapt-ep1-checkpoint",
        "subdir": None,
        "output": "eval_v5b_tapt_ep1.json",
    },
    {
        "name": "sft_v5_ckpt80",
        "slug": "pa-sft-v5-checkpoints",
        "subdir": "checkpoint-80",
        "output": "eval_v5b_sft80.json",
    },
]

for cfg in MODEL_CONFIGS:
    slug = cfg["slug"]
    assert slug is not None
    p = resolve_model_path(slug, cfg["subdir"])
    logger.info("Model %s: %s", cfg["name"], p)

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ============================================================
# PHASE 1: Load retrieval contexts
# ============================================================

logger.info("=== PHASE 1: Load data ===")

with open(RETRIEVAL_PATH, encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

hit_entries = [e for e in entries if e["retrieval_hit"]]
miss_entries = [e for e in entries if not e["retrieval_hit"]]
logger.info(
    "Loaded %d entries: %d hits (%.1f%%), %d misses (%.1f%%)",
    len(entries),
    len(hit_entries),
    100 * len(hit_entries) / len(entries),
    len(miss_entries),
    100 * len(miss_entries) / len(entries),
)

# Log context length stats (truncation awareness)
ctx_lens = [len(e["retrieval_context"]) for e in entries]
logger.info(
    "Retrieval context chars: min=%d, median=%d, max=%d (will truncate to 2048 tokens)",
    min(ctx_lens),
    sorted(ctx_lens)[len(ctx_lens) // 2],
    max(ctx_lens),
)

# Citation patterns (same as eval v5)
SOURCE_PATTERNS: dict[str, str] = {
    "LA": r"(?:livre.{0,10}arbitre|l\.?a\.?\b)",
    "R01": r"(?:r[eè]gles?\s+g[eé]n[eé]rales?|r\.?01)",
    "R02": r"(?:annexes?\s+aux\s+r[eè]gles|r\.?02)",
    "A01": r"(?:championnat\s+de\s+france\b|a\.?01)",
    "A02": r"(?:championnat\s+.{0,10}clubs?|a\.?02)",
}

ABSTAIN_PATTERNS = [
    r"non trouv[eé]",
    r"pas trouv[eé]",
    r"pas dans (le|les) (contexte|extrait)",
    r"information.*non.*disponible",
    r"ne (se |)trouve pas",
    r"ne figure pas",
    r"pas mentionn[eé]",
    r"aucune information",
]


def check_citation(
    response: str, expected_docs: list[str], expected_pages: list[int]
) -> bool:
    """Regex citation check (same as eval v5)."""
    rl = response.lower()
    doc = any(
        re.search(pat, rl)
        for doc in expected_docs
        for key, pat in SOURCE_PATTERNS.items()
        if key in doc.upper()
    )
    page = any(re.search(rf"\bpage\s*{p}\b|\bp\.?\s*{p}\b", rl) for p in expected_pages)
    return doc or page


def check_abstain(response: str) -> bool:
    """Check if model says 'not found in context'."""
    rl = response.lower()
    return any(re.search(p, rl) for p in ABSTAIN_PATTERNS)


# ============================================================
# PHASE 2: Inference
# ============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def generate_response_gpu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
) -> str:
    """GPU inference (same params as eval v5)."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # No truncation: Gemma 3 supports 32K tokens (max_position_embeddings=32768).
    # Retrieval contexts avg 23K chars (~6K tokens) — fits within model capacity.
    # Same as eval v5 oracle kernel (no truncation).
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
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
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def eval_subset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    subset: list[dict],
    label: str,
) -> dict:
    """Eval a subset (hit or miss) and return metrics."""
    cited = 0
    abstained = 0
    empty = 0
    lengths: list[int] = []

    for i, entry in enumerate(subset):
        msgs = build_rag_prompt(entry["question"], entry["retrieval_context"])
        resp = generate_response_gpu(model, tokenizer, msgs)

        if check_citation(resp, entry["expected_docs"], entry["expected_pages"]):
            cited += 1
        if check_abstain(resp):
            abstained += 1
        if not resp.strip():
            empty += 1
        lengths.append(len(resp.split()))

        if (i + 1) % 50 == 0:
            logger.info("  [%s %d/%d]", label, i + 1, len(subset))

    n = len(subset)
    return {
        "count": n,
        "cited": cited,
        "cited_pct": round(100 * cited / n, 1) if n else 0,
        "abstained": abstained,
        "abstain_pct": round(100 * abstained / n, 1) if n else 0,
        "empty": empty,
        "median_words": sorted(lengths)[len(lengths) // 2] if lengths else 0,
    }


def eval_model_pipeline(
    model_path: str,
    model_name: str,
) -> dict:
    """Evaluate one model on hit + miss subsets."""
    logger.info("--- Evaluating: %s (%s) ---", model_name, model_path)
    t0 = time.time()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map={"": 0}
    )
    model.eval()
    logger.info("Model loaded: %.0f MB VRAM", torch.cuda.memory_allocated() / 1e6)

    hit_results = eval_subset(model, tokenizer, hit_entries, "hit")
    miss_results = eval_subset(model, tokenizer, miss_entries, "miss")

    total = len(entries)
    all_cited = hit_results["cited"] + miss_results["cited"]
    all_abstain = hit_results["abstained"] + miss_results["abstained"]
    elapsed = round((time.time() - t0) / 60, 1)

    logger.info(
        "%s DONE: hit_cite=%.1f%% miss_cite=%.1f%% miss_abstain=%.1f%% (%.1f min)",
        model_name,
        hit_results["cited_pct"],
        miss_results["cited_pct"],
        miss_results["abstain_pct"],
        elapsed,
    )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "hit": hit_results,
        "miss": miss_results,
        "overall_cited_pct": round(100 * all_cited / total, 1),
        "overall_abstain_pct": round(100 * all_abstain / total, 1),
        "elapsed_min": elapsed,
    }


# ============================================================
# PHASE 3: Run eval
# ============================================================

logger.info("=== PHASE 3: Eval loop (3 models) ===")
t_total = time.time()
all_results = []

for cfg in MODEL_CONFIGS:
    slug = cfg["slug"]
    assert slug is not None
    path = resolve_model_path(slug, cfg["subdir"])
    name = cfg["name"] or "unknown"
    result = eval_model_pipeline(path, name)
    all_results.append(result)

    # Save per-model
    out = cfg["output"] or "output.json"
    per_model_path = os.path.join(OUTPUT_DIR, out)
    with open(per_model_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

# ============================================================
# PHASE 4: Summary
# ============================================================

logger.info("=== PHASE 4: Summary ===")
logger.info("Total: %.1f min", (time.time() - t_total) / 60)
logger.info("")
logger.info(
    "%-16s | %8s | %8s | %8s | %9s | %9s",
    "Model",
    "Overall%",
    "Hit%",
    "Miss%",
    "Abstain%",
    "MissAbst%",
)
logger.info("-" * 78)
for r in all_results:
    logger.info(
        "%-16s | %7.1f%% | %7.1f%% | %7.1f%% | %8.1f%% | %8.1f%%",
        r["model"],
        r["overall_cited_pct"],
        r["hit"]["cited_pct"],
        r["miss"]["cited_pct"],
        r["overall_abstain_pct"],
        r["miss"]["abstain_pct"],
    )

# Decision
best_miss_abstain = max(all_results, key=lambda x: x["miss"]["abstain_pct"])
best_hit_cite = max(all_results, key=lambda x: x["hit"]["cited_pct"])
logger.info("")
logger.info(
    "Best on HIT (faithful reader): %s (%.1f%%)",
    best_hit_cite["model"],
    best_hit_cite["hit"]["cited_pct"],
)
logger.info(
    "Best on MISS (safe abstainer): %s (%.1f%%)",
    best_miss_abstain["model"],
    best_miss_abstain["miss"]["abstain_pct"],
)

summary = {
    "results": all_results,
    "best_hit_cite": best_hit_cite["model"],
    "best_miss_abstain": best_miss_abstain["model"],
    "retrieval_recall": round(100 * len(hit_entries) / len(entries), 1),
}
summary_path = os.path.join(OUTPUT_DIR, "eval_v5b_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
logger.info("Saved: %s", summary_path)
logger.info("=== DONE ===")
