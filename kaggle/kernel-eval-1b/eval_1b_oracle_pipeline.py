"""Eval Gemma 3 1B IT: oracle + pipeline on 298 questions.

ADR-001 gate: 270M failed (postmortem), 1B is the fallback.
Runs BOTH oracle (same as v5) and pipeline (same as v5b) contexts
on the same model in one kernel to enable direct comparison.

Standards: RAGChecker NeurIPS 2024, ICTIR 2025, RAGAS 2023.
Gemma 3 fp16=NaN on T4: use float32 (~4 GB VRAM for 1B).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import sqlite3
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
GPU_VRAM_MB = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
logger.info("GPU: %s (%.0f MB)", GPU_NAME, GPU_VRAM_MB)

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

# Resolve paths
EVAL_CANDIDATES = [
    "/kaggle/input/pocket-arbiter-eval-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-eval-data",
]
EVAL_DIR = next((c for c in EVAL_CANDIDATES if os.path.isdir(c)), None)
assert EVAL_DIR is not None, f"FATAL: eval data not found: {EVAL_CANDIDATES}"

MODEL_CANDIDATES = [
    "/kaggle/input/gemma-3-1b-it",
    "/kaggle/input/datasets/pguillemin/gemma-3-1b-it",
]
MODEL_DIR = next((c for c in MODEL_CANDIDATES if os.path.isdir(c)), None)
assert MODEL_DIR is not None, f"FATAL: model not found: {MODEL_CANDIDATES}"

logger.info("Eval data: %s", EVAL_DIR)
logger.info("Model: %s", MODEL_DIR)

sys.path.insert(0, EVAL_DIR)
from eval_generation import (  # noqa: E402
    check_citation,
    load_annales_questions,
    load_human_questions,
)
from generation_prompt import build_rag_prompt  # noqa: E402

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ============================================================
# PHASE 1: Load data
# ============================================================

logger.info("=== PHASE 1: Load data ===")

GS_PATH = os.path.join(EVAL_DIR, "gold_standard_annales_fr_v8_adversarial.json")
DB_PATH = os.path.join(EVAL_DIR, "corpus_v2_fr.db")
RETRIEVAL_PATH = os.path.join(EVAL_DIR, "retrieval_contexts_v5b.jsonl")

assert os.path.exists(GS_PATH), f"FATAL: {GS_PATH}"
assert os.path.exists(DB_PATH), f"FATAL: {DB_PATH}"
assert os.path.exists(RETRIEVAL_PATH), f"FATAL: {RETRIEVAL_PATH}"

human_qs = load_human_questions(GS_PATH)
annales_qs = load_annales_questions(GS_PATH)
assert len(human_qs) == 34, f"Expected 34, got {len(human_qs)}"
assert len(annales_qs) == 264, f"Expected 264, got {len(annales_qs)}"
logger.info("GS: %d human, %d annales", len(human_qs), len(annales_qs))

with open(RETRIEVAL_PATH, encoding="utf-8") as f:
    retrieval_entries = [json.loads(line) for line in f]
hit_entries = [e for e in retrieval_entries if e["retrieval_hit"]]
miss_entries = [e for e in retrieval_entries if not e["retrieval_hit"]]
logger.info(
    "Retrieval: %d entries, %d hits (%.1f%%), %d misses",
    len(retrieval_entries),
    len(hit_entries),
    100 * len(hit_entries) / len(retrieval_entries),
    len(miss_entries),
)

conn = sqlite3.connect(DB_PATH)
logger.info("DB opened")

# ============================================================
# PHASE 2: Load model
# ============================================================

logger.info("=== PHASE 2: Load model ===")

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# Gemma 3 fp16=NaN on T4 (HF #36822). bf16 unsupported on T4 (sm_75 < sm_80).
# 1B fp32 = ~4 GB VRAM on T4 15 GB = safe.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32, device_map={"": 0}
)
model.eval()
vram = torch.cuda.memory_allocated() / 1024 / 1024
logger.info("Model loaded: %.0f MB VRAM (fp32)", vram)

# GenerationConfig avoids transformers 5.0 "flags not valid" warning
# (top_p/top_k rejected when passed as kwargs, must use GenerationConfig).
RAG_GEN_CONFIG = GenerationConfig(
    max_new_tokens=512,
    min_new_tokens=10,
    do_sample=True,
    temperature=0.2,
    top_k=64,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=4,
)

# Smoke test
test_msgs = build_rag_prompt("Test question?", "Test context.")
test_text = tokenizer.apply_chat_template(
    test_msgs, tokenize=False, add_generation_prompt=True
)
test_inputs = tokenizer(test_text, return_tensors="pt")
test_inputs = {k: v.to(model.device) for k, v in test_inputs.items()}
with torch.no_grad():
    test_out = model.generate(
        **test_inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
test_resp = tokenizer.decode(
    test_out[0][test_inputs["input_ids"].shape[1] :], skip_special_tokens=True
)
logger.info("Smoke test: '%s'", test_resp[:100])
assert len(test_resp.strip()) > 0, "FATAL: empty smoke test"


def generate_gpu(
    mdl: AutoModelForCausalLM,
    tok: AutoTokenizer,
    messages: list[dict],
) -> str:
    """Generate with standard RAG params via GenerationConfig."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            generation_config=RAG_GEN_CONFIG,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def load_oracle_context(source: str, page: int) -> str:
    """Load oracle context from DB."""
    rows = conn.execute(
        "SELECT text FROM children WHERE source = ? AND page = ?",
        (source, page),
    ).fetchall()
    return "\n\n".join(r[0] for r in rows)


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


def check_abstain(response: str) -> bool:
    """Check if model says 'not found'."""
    rl = response.lower()
    return any(re.search(p, rl) for p in ABSTAIN_PATTERNS)


# ============================================================
# PHASE 3A: Oracle eval (same as v5)
# ============================================================

logger.info("=== PHASE 3A: Oracle eval (298Q) ===")
t_oracle = time.time()

# Human questions (34) — save full responses
human_results = []
empty_oracle = 0
for i, q in enumerate(human_qs):
    prov = q.get("provenance", {})
    source = (prov.get("docs") or [""])[0]
    page = (prov.get("pages") or [0])[0]
    context = load_oracle_context(source, page)
    messages = build_rag_prompt(q["content"]["question"], context)
    response = generate_gpu(model, tokenizer, messages)
    if not response.strip():
        empty_oracle += 1
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
        logger.info("  [oracle human %d/%d]", i + 1, len(human_qs))

# Annales questions (264) — auto citation
oracle_cited = 0
for i, q in enumerate(annales_qs):
    prov = q.get("provenance", {})
    source = (prov.get("docs") or [""])[0]
    page = (prov.get("pages") or [0])[0]
    context = load_oracle_context(source, page)
    messages = build_rag_prompt(q["content"]["question"], context)
    response = generate_gpu(model, tokenizer, messages)
    if not response.strip():
        empty_oracle += 1
    if check_citation(response, prov.get("docs", []), prov.get("pages", [])):
        oracle_cited += 1
    if (i + 1) % 50 == 0:
        logger.info("  [oracle annales %d/%d]", i + 1, len(annales_qs))

oracle_pct = round(100 * oracle_cited / len(annales_qs), 1)
oracle_time = round((time.time() - t_oracle) / 60, 1)

# Response length stats
lengths = [len(r["response"].split()) for r in human_results]
median_words = sorted(lengths)[len(lengths) // 2] if lengths else 0

oracle_output = {
    "model": MODEL_DIR,
    "model_name": "gemma_3_1b_it",
    "questions": human_results,
    "auto_citation": {
        "total": len(annales_qs),
        "cited_count": oracle_cited,
        "cited_pct": oracle_pct,
    },
    "metadata": {
        "gpu": GPU_NAME,
        "seed": SEED,
        "empty_responses": empty_oracle,
        "inference_time_min": oracle_time,
        "median_response_words": median_words,
    },
}

oracle_path = os.path.join(OUTPUT_DIR, "eval_1b_oracle.json")
with open(oracle_path, "w", encoding="utf-8") as f:
    json.dump(oracle_output, f, ensure_ascii=False, indent=2)

logger.info(
    "Oracle DONE: cited=%d/%d (%.1f%%), empties=%d, median=%dw, %.1f min",
    oracle_cited,
    len(annales_qs),
    oracle_pct,
    empty_oracle,
    median_words,
    oracle_time,
)

# ============================================================
# PHASE 3B: Pipeline eval (same as v5b)
# ============================================================

logger.info("=== PHASE 3B: Pipeline eval (298Q) ===")
t_pipeline = time.time()

# Reset seed for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

pipeline_results: dict[str, dict] = {}
for label, subset in [("hit", hit_entries), ("miss", miss_entries)]:
    cited = 0
    abstained = 0
    empty = 0
    plengths: list[int] = []

    for i, entry in enumerate(subset):
        messages = build_rag_prompt(entry["question"], entry["retrieval_context"])
        response = generate_gpu(model, tokenizer, messages)

        if check_citation(response, entry["expected_docs"], entry["expected_pages"]):
            cited += 1
        if check_abstain(response):
            abstained += 1
        if not response.strip():
            empty += 1
        plengths.append(len(response.split()))

        if (i + 1) % 50 == 0:
            logger.info("  [pipeline %s %d/%d]", label, i + 1, len(subset))

    n = len(subset)
    pipeline_results[label] = {
        "count": n,
        "cited": cited,
        "cited_pct": round(100 * cited / n, 1) if n else 0,
        "abstained": abstained,
        "abstain_pct": round(100 * abstained / n, 1) if n else 0,
        "empty": empty,
        "median_words": sorted(plengths)[len(plengths) // 2] if plengths else 0,
    }
    logger.info(
        "  %s: cited=%.1f%% abstain=%.1f%% empty=%d",
        label,
        pipeline_results[label]["cited_pct"],
        pipeline_results[label]["abstain_pct"],
        empty,
    )

total = len(retrieval_entries)
all_cited = pipeline_results["hit"]["cited"] + pipeline_results["miss"]["cited"]
all_abstain = (
    pipeline_results["hit"]["abstained"] + pipeline_results["miss"]["abstained"]
)
pipeline_time = round((time.time() - t_pipeline) / 60, 1)

pipeline_output = {
    "model": "gemma_3_1b_it",
    "hit": pipeline_results["hit"],
    "miss": pipeline_results["miss"],
    "overall_cited_pct": round(100 * all_cited / total, 1),
    "overall_abstain_pct": round(100 * all_abstain / total, 1),
    "elapsed_min": pipeline_time,
}

pipeline_path = os.path.join(OUTPUT_DIR, "eval_1b_pipeline.json")
with open(pipeline_path, "w", encoding="utf-8") as f:
    json.dump(pipeline_output, f, ensure_ascii=False, indent=2)

logger.info(
    "Pipeline DONE: overall_cite=%.1f%% hit=%.1f%% miss=%.1f%% miss_abstain=%.1f%% %.1f min",
    pipeline_output["overall_cited_pct"],
    pipeline_results["hit"]["cited_pct"],
    pipeline_results["miss"]["cited_pct"],
    pipeline_results["miss"]["abstain_pct"],
    pipeline_time,
)

# ============================================================
# PHASE 4: Summary comparison
# ============================================================

conn.close()
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

logger.info("=== PHASE 4: Summary ===")
total_time = oracle_time + pipeline_time
logger.info(
    "Total: %.1f min (oracle %.1f + pipeline %.1f)",
    total_time,
    oracle_time,
    pipeline_time,
)

# Comparison with 270M baselines (eval v5 + v5b)
logger.info("")
logger.info("=== COMPARISON: 1B vs 270M baselines ===")
logger.info("")
logger.info("ORACLE (auto-citation 264 annales):")
logger.info("  270M base:     40.5%%")
logger.info("  270M tapt_ep1: 46.6%%")
logger.info("  270M sft80:    59.1%%")
logger.info("  1B base:       %.1f%%", oracle_pct)
logger.info("")
logger.info("PIPELINE (overall citation 298Q):")
logger.info("  270M base:     24.8%%")
logger.info("  270M tapt_ep1: 40.3%%")
logger.info("  270M sft80:    48.7%%")
logger.info("  1B base:       %.1f%%", pipeline_output["overall_cited_pct"])
logger.info("")
logger.info("PIPELINE MISS abstention:")
logger.info("  270M base:     3.1%%")
logger.info("  270M tapt_ep1: 3.1%%")
logger.info("  270M sft80:    0.0%%")
logger.info("  1B base:       %.1f%%", pipeline_results["miss"]["abstain_pct"])

summary = {
    "model": "gemma_3_1b_it",
    "oracle_cited_pct": oracle_pct,
    "oracle_empty": empty_oracle,
    "oracle_median_words": median_words,
    "oracle_time_min": oracle_time,
    "pipeline": pipeline_output,
    "pipeline_time_min": pipeline_time,
    "total_time_min": total_time,
    "baselines_270m": {
        "oracle": {"base": 40.5, "tapt_ep1": 46.6, "sft80": 59.1},
        "pipeline": {"base": 24.8, "tapt_ep1": 40.3, "sft80": 48.7},
        "miss_abstain": {"base": 3.1, "tapt_ep1": 3.1, "sft80": 0.0},
    },
}
summary_path = os.path.join(OUTPUT_DIR, "eval_1b_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
logger.info("Saved: %s", summary_path)

logger.info("=== DONE ===")
