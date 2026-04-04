"""Eval 1B Base vs SFT v5: pipeline comparison on 298 questions.

Loads two models SEQUENTIALLY (del + gc between each):
1. Gemma 3 1B IT base (baseline 56.7% pipeline citations)
2. Gemma 3 1B IT SFT v5 (LoRA merged, RAFT-trained)

Both run the SAME pipeline eval (retrieval contexts from v5b).
Gate: SFT v5 > 56.7% = success. Else 1B base = final model.

Based on proven eval_1b_oracle_pipeline.py (v4, 56.7% baseline).
Gemma 3 fp32 mandatory on T4 (HF #36822, bf16 unsupported sm_75).

Runtime estimate: 298Q x 2 models x ~20s/Q = ~200 min (~3.3h).
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


# Resolve paths (two Kaggle mount patterns)
def _resolve(slug: str) -> str:
    for c in [f"/kaggle/input/{slug}", f"/kaggle/input/datasets/pguillemin/{slug}"]:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Not found: {slug}")


EVAL_DIR = _resolve("pocket-arbiter-eval-data")
BASE_DIR = _resolve("gemma-3-1b-it")
SFT_DIR = _resolve("gemma-1b-sft-v5-merged")

logger.info("Eval data: %s", EVAL_DIR)
logger.info("Base model: %s", BASE_DIR)
logger.info("SFT model: %s", SFT_DIR)

sys.path.insert(0, EVAL_DIR)
from eval_generation import (  # noqa: E402
    check_citation,
    load_annales_questions,
    load_human_questions,
)
from generation_prompt import build_rag_prompt  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ============================================================
# PHASE 1: Load data
# ============================================================

logger.info("=== PHASE 1: Load data ===")

GS_PATH = os.path.join(EVAL_DIR, "gold_standard_annales_fr_v8_adversarial.json")
RETRIEVAL_PATH = os.path.join(EVAL_DIR, "retrieval_contexts_v5b.jsonl")

assert os.path.exists(GS_PATH), f"FATAL: {GS_PATH}"
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

# GenerationConfig avoids transformers 5.0 "flags not valid" warning
# Gemma 3 EOS: token 1 (<eos>) + token 106 (<end_of_turn>)
# CRITICAL: Unsloth save_pretrained_merged drops generation_config.json,
# so merged models only have eos_token_id=1 from config.json.
# Without token 106, model generates past <end_of_turn> → garbage + hang.
# Also set use_cache=True explicitly (Unsloth saves use_cache=false from training).
GEMMA3_EOS_TOKEN_IDS = [1, 106]

RAG_GEN_CONFIG = GenerationConfig(
    max_new_tokens=512,
    min_new_tokens=10,
    do_sample=True,
    temperature=0.2,
    top_k=64,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=4,
    eos_token_id=GEMMA3_EOS_TOKEN_IDS,
)

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
    rl = response.lower()
    return any(re.search(p, rl) for p in ABSTAIN_PATTERNS)


def generate_gpu(
    mdl: AutoModelForCausalLM,
    tok: AutoTokenizer,
    messages: list[dict],
) -> str:
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


def eval_pipeline(
    model_dir: str,
    model_name: str,
) -> dict:
    """Load model, run pipeline eval on 298Q, save results, free VRAM."""
    logger.info("--- Eval: %s (%s) ---", model_name, model_dir)
    t0 = time.time()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, device_map={"": 0}
    )
    model.eval()
    # Force use_cache=True for inference (Unsloth merged models save use_cache=False)
    if hasattr(model.config, "use_cache") and not model.config.use_cache:
        logger.info("WARN: use_cache was False (training artifact), forcing True")
        model.config.use_cache = True
    vram = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("Model loaded: %.0f MB VRAM (fp32)", vram)

    # Smoke test with timing guard
    t_smoke = time.time()
    test_msgs = build_rag_prompt("Test question?", "Test context.")
    test_resp = generate_gpu(model, tokenizer, test_msgs)
    smoke_secs = time.time() - t_smoke
    assert len(test_resp.strip()) > 0, f"FATAL: empty smoke test for {model_name}"
    logger.info("Smoke test OK (%.1fs): '%s'", smoke_secs, test_resp[:80])
    # Guard: if smoke test takes >30s, model is broken (EOS issue, use_cache=False, etc.)
    # Base model: ~2s. Broken merged: ~128s.
    if smoke_secs > 30:
        logger.error(
            "FATAL: Smoke test took %.1fs (>30s threshold). Model likely broken "
            "(missing EOS token 106 or use_cache=False). Skipping %s.",
            smoke_secs,
            model_name,
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return {"model": model_name, "error": f"smoke_test_too_slow_{smoke_secs:.0f}s"}

    # Pipeline eval (hit + miss)
    pipeline_results: dict[str, dict] = {}
    for label, subset in [("hit", hit_entries), ("miss", miss_entries)]:
        cited = 0
        abstained = 0
        empty = 0
        plengths: list[int] = []

        for i, entry in enumerate(subset):
            messages = build_rag_prompt(entry["question"], entry["retrieval_context"])
            response = generate_gpu(model, tokenizer, messages)

            if check_citation(
                response, entry["expected_docs"], entry["expected_pages"]
            ):
                cited += 1
            if check_abstain(response):
                abstained += 1
            if not response.strip():
                empty += 1
            plengths.append(len(response.split()))

            if (i + 1) % 50 == 0:
                logger.info("  [%s %s %d/%d]", model_name, label, i + 1, len(subset))

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
            "  %s %s: cited=%.1f%% abstain=%.1f%% empty=%d",
            model_name,
            label,
            pipeline_results[label]["cited_pct"],
            pipeline_results[label]["abstain_pct"],
            empty,
        )

    total_n = len(retrieval_entries)
    all_cited = pipeline_results["hit"]["cited"] + pipeline_results["miss"]["cited"]
    all_abstain = (
        pipeline_results["hit"]["abstained"] + pipeline_results["miss"]["abstained"]
    )
    elapsed = round((time.time() - t0) / 60, 1)

    output = {
        "model": model_name,
        "hit": pipeline_results["hit"],
        "miss": pipeline_results["miss"],
        "overall_cited_pct": round(100 * all_cited / total_n, 1),
        "overall_abstain_pct": round(100 * all_abstain / total_n, 1),
        "elapsed_min": elapsed,
    }

    # Save IMMEDIATELY (kaggle-deployment: checkpoint after each model)
    out_path = os.path.join(OUTPUT_DIR, f"eval_1b_{model_name}_pipeline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", out_path)

    logger.info(
        "%s DONE: overall_cite=%.1f%% hit=%.1f%% miss=%.1f%% %.1f min",
        model_name,
        output["overall_cited_pct"],
        pipeline_results["hit"]["cited_pct"],
        pipeline_results["miss"]["cited_pct"],
        elapsed,
    )

    # Free VRAM before next model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("VRAM freed: %.0f MB remaining", vram_after)

    return output


# ============================================================
# PHASE 2: Eval both models sequentially
# ============================================================

logger.info("=== PHASE 2: Sequential eval ===")
t_total = time.time()

MODELS = [
    {"dir": BASE_DIR, "name": "base"},
    {"dir": SFT_DIR, "name": "sft_v5"},
]

results = {}
for cfg in MODELS:
    results[cfg["name"]] = eval_pipeline(cfg["dir"], cfg["name"])

# ============================================================
# PHASE 3: Summary comparison
# ============================================================

logger.info("=== PHASE 3: Summary ===")
total_min = round((time.time() - t_total) / 60, 1)

logger.info("")
logger.info("=== PIPELINE CITATIONS (298Q) ===")
for name, r in results.items():
    logger.info(
        "  %s: overall=%.1f%% hit=%.1f%% miss=%.1f%% abstain=%.1f%% (%.1f min)",
        name,
        r["overall_cited_pct"],
        r["hit"]["cited_pct"],
        r["miss"]["cited_pct"],
        r["overall_abstain_pct"],
        r["elapsed_min"],
    )

logger.info("")
logger.info("BASELINES:")
logger.info("  270m base:     24.8%%")
logger.info("  270m tapt_ep1: 40.3%%")
logger.info("  270m sft80:    48.7%%")
logger.info("  1B base (v4):  56.7%%")
logger.info("")

gate = results["sft_v5"]["overall_cited_pct"] > 56.7
logger.info(
    "GATE: SFT v5 (%.1f%%) > 56.7%% = %s",
    results["sft_v5"]["overall_cited_pct"],
    "PASS" if gate else "FAIL",
)

summary = {
    "models": results,
    "total_time_min": total_min,
    "gate_sft_v5_gt_56_7": gate,
    "baselines": {
        "270m_base": 24.8,
        "270m_tapt_ep1": 40.3,
        "270m_sft80": 48.7,
        "1b_base_v4": 56.7,
    },
}
summary_path = os.path.join(OUTPUT_DIR, "eval_1b_comparison.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
logger.info("Saved: %s", summary_path)

logger.info("Total: %.1f min", total_min)
logger.info("=== DONE ===")
