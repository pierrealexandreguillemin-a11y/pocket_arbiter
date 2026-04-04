"""Eval 1B SFT v5 only: pipeline eval on 298 questions.

SFT-v5-only kernel — base already evaluated at 56.4% (run 2026-04-02).
Saves ~90 min GPU by skipping the base model.

Fixes applied vs previous hang (pocket-arbiter-eval-1b-base-vs-sft-v5):
  1. eos_token_id=[1, 106]: Unsloth save_pretrained_merged drops
     generation_config.json. Token 106 (<end_of_turn>) was missing
     from EOS list -> model generated past response -> Tamil garbage
     + input echo -> 128s smoke test (71x slower than base 1.8s).
  2. use_cache=True override: Unsloth saves training-time config
     (use_cache=false) which disables KV cache -> full recompute
     per generated token.
  3. Smoke test timing guard: abort if >30s (base ~2s, broken ~128s).

Gate: SFT v5 > 56.7% pipeline citations = success.
Runtime estimate: 298Q x ~18s/Q = ~90 min.

Gemma 3 fp32 mandatory on T4 (HF #36822, bf16 unsupported sm_75).
GenerationConfig object required (transformers 5.0 deprecation).

Input:  pguillemin/gemma-1b-sft-v5-merged (2 GB, with fixed config)
        pguillemin/pocket-arbiter-eval-data (DB + GS + eval scripts)
Output: eval_1b_sft_v5_pipeline.json
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
SFT_DIR = _resolve("gemma-1b-sft-v5-merged")

logger.info("Eval data: %s", EVAL_DIR)
logger.info("SFT model: %s", SFT_DIR)

# Verify dataset version: detect if Kaggle mounted old vs fixed dataset.
# Even if old dataset is mounted, the kernel still works because:
#   - RAG_GEN_CONFIG explicitly sets eos_token_id=[1,106] (overrides model config)
#   - use_cache is forced True at runtime after from_pretrained()
#   - Smoke test timing guard catches any remaining issue
sft_files = sorted(os.listdir(SFT_DIR))
logger.info("SFT model files: %s", sft_files)

gen_cfg_path = os.path.join(SFT_DIR, "generation_config.json")
if os.path.exists(gen_cfg_path):
    with open(gen_cfg_path) as f:
        gc_data = json.load(f)
    logger.info(
        "generation_config.json FOUND: eos_token_id=%s (FIXED dataset)",
        gc_data.get("eos_token_id"),
    )
else:
    logger.warning(
        "generation_config.json MISSING — Kaggle may have mounted OLD dataset. "
        "Kernel will still work (eos_token_id set in RAG_GEN_CONFIG)."
    )

cfg_path = os.path.join(SFT_DIR, "config.json")
with open(cfg_path) as f:
    model_cfg = json.load(f)
cfg_eos = model_cfg.get("eos_token_id")
cfg_cache = model_cfg.get("use_cache")
logger.info("config.json: eos_token_id=%s, use_cache=%s", cfg_eos, cfg_cache)
if cfg_eos == 1 or cfg_cache is False:
    logger.warning(
        "OLD config.json detected (eos=1, use_cache=False). "
        "Kernel overrides both at runtime — proceeding."
    )

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

# Gemma 3 EOS: token 1 (<eos>) + token 106 (<end_of_turn>)
# CRITICAL fix for Unsloth merged models (see docstring).
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


# ============================================================
# PHASE 2: Load SFT v5 model
# ============================================================

logger.info("=== PHASE 2: Load SFT v5 model ===")
t0 = time.time()

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

tokenizer = AutoTokenizer.from_pretrained(SFT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    SFT_DIR, torch_dtype=torch.float32, device_map={"": 0}
)
model.eval()

# Force use_cache=True (Unsloth merged models save use_cache=False from training)
if hasattr(model.config, "use_cache") and not model.config.use_cache:
    logger.info("WARN: use_cache was False (training artifact), forcing True")
    model.config.use_cache = True

vram = torch.cuda.memory_allocated() / 1024 / 1024
logger.info("Model loaded: %.0f MB VRAM (fp32)", vram)

# Log EOS config for verification
logger.info(
    "Model config eos_token_id=%s, tokenizer eos_token_id=%s",
    model.config.eos_token_id,
    tokenizer.eos_token_id,
)

# ============================================================
# PHASE 3: Smoke test with timing guard
# ============================================================

logger.info("=== PHASE 3: Smoke test ===")
t_smoke = time.time()
test_msgs = build_rag_prompt("Test question?", "Test context.")
test_resp = generate_gpu(model, tokenizer, test_msgs)
smoke_secs = time.time() - t_smoke

assert len(test_resp.strip()) > 0, "FATAL: empty smoke test"
logger.info("Smoke test (%.1fs): '%s'", smoke_secs, test_resp[:120])

# Guard: base ~2s, broken merged ~128s. 30s threshold is generous.
if smoke_secs > 30:
    logger.error(
        "FATAL: Smoke test took %.1fs (>30s threshold). "
        "EOS token 106 likely still missing or use_cache still False. "
        "Aborting to save GPU quota.",
        smoke_secs,
    )
    # Save error output for diagnosis
    error_output = {
        "model": "sft_v5",
        "error": f"smoke_test_too_slow_{smoke_secs:.0f}s",
        "smoke_response": test_resp[:500],
        "model_eos_token_id": str(model.config.eos_token_id),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "use_cache": model.config.use_cache,
    }
    err_path = os.path.join(OUTPUT_DIR, "eval_1b_sft_v5_ERROR.json")
    with open(err_path, "w", encoding="utf-8") as f:
        json.dump(error_output, f, indent=2, ensure_ascii=False)
    logger.info("Error output saved: %s", err_path)
    sys.exit(1)

logger.info("Smoke test PASS (%.1fs < 30s threshold)", smoke_secs)

# ============================================================
# PHASE 4: Pipeline eval (298Q)
# ============================================================

logger.info("=== PHASE 4: Pipeline eval (298Q) ===")

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
            elapsed_so_far = (time.time() - t0) / 60
            logger.info(
                "  [sft_v5 %s %d/%d] %.1f min elapsed",
                label,
                i + 1,
                len(subset),
                elapsed_so_far,
            )

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
        "  sft_v5 %s: cited=%.1f%% abstain=%.1f%% empty=%d median_words=%d",
        label,
        pipeline_results[label]["cited_pct"],
        pipeline_results[label]["abstain_pct"],
        empty,
        pipeline_results[label]["median_words"],
    )

total_n = len(retrieval_entries)
all_cited = pipeline_results["hit"]["cited"] + pipeline_results["miss"]["cited"]
all_abstain = (
    pipeline_results["hit"]["abstained"] + pipeline_results["miss"]["abstained"]
)
elapsed = round((time.time() - t0) / 60, 1)

output = {
    "model": "sft_v5",
    "hit": pipeline_results["hit"],
    "miss": pipeline_results["miss"],
    "overall_cited_pct": round(100 * all_cited / total_n, 1),
    "overall_abstain_pct": round(100 * all_abstain / total_n, 1),
    "elapsed_min": elapsed,
    "smoke_test_secs": round(smoke_secs, 1),
    "fixes_applied": [
        "eos_token_id=[1,106] in GenerationConfig",
        "use_cache=True override",
        "generation_config.json added to merged model",
    ],
}

# Save IMMEDIATELY
out_path = os.path.join(OUTPUT_DIR, "eval_1b_sft_v5_pipeline.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
logger.info("Saved: %s", out_path)

# ============================================================
# PHASE 5: Gate check + summary
# ============================================================

logger.info("=== PHASE 5: Gate check ===")

# Base results from previous run (2026-04-02, confirmed 56.4%)
BASE_RESULTS = {
    "overall_cited_pct": 56.4,
    "hit_cited_pct": 48.1,
    "miss_cited_pct": 69.2,
    "elapsed_min": 89.0,
    "source": "eval_1b_base_pipeline.json (2026-04-02)",
}

sft_cited = output["overall_cited_pct"]
gate = sft_cited > 56.7

logger.info("")
logger.info("=== PIPELINE CITATIONS (298Q) ===")
logger.info(
    "  base (cached): overall=%.1f%% hit=%.1f%% miss=%.1f%%",
    BASE_RESULTS["overall_cited_pct"],
    BASE_RESULTS["hit_cited_pct"],
    BASE_RESULTS["miss_cited_pct"],
)
logger.info(
    "  sft_v5:        overall=%.1f%% hit=%.1f%% miss=%.1f%% (%.1f min)",
    sft_cited,
    pipeline_results["hit"]["cited_pct"],
    pipeline_results["miss"]["cited_pct"],
    elapsed,
)
logger.info("")
logger.info("BASELINES:")
logger.info("  270m base:     24.8%%")
logger.info("  270m tapt_ep1: 40.3%%")
logger.info("  270m sft80:    48.7%%")
logger.info("  1B base (v4):  56.7%%")
logger.info("  1B base (v5):  56.4%% (rerun)")
logger.info("")
logger.info(
    "GATE: SFT v5 (%.1f%%) > 56.7%% = %s",
    sft_cited,
    "PASS" if gate else "FAIL",
)

summary = {
    "sft_v5": output,
    "base_cached": BASE_RESULTS,
    "gate_sft_v5_gt_56_7": gate,
    "delta_vs_base": round(sft_cited - BASE_RESULTS["overall_cited_pct"], 1),
    "baselines": {
        "270m_base": 24.8,
        "270m_tapt_ep1": 40.3,
        "270m_sft80": 48.7,
        "1b_base_v4": 56.7,
        "1b_base_v5_rerun": 56.4,
    },
}
summary_path = os.path.join(OUTPUT_DIR, "eval_1b_sft_v5_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
logger.info("Saved: %s", summary_path)

# Cleanup
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

logger.info("Total: %.1f min", elapsed)
logger.info("=== DONE ===")
