"""Eval 1B BASE: pipeline eval on 298 questions with per-question responses.

Runs Gemma 3 1B IT (base, no SFT) on the same retrieval contexts as
SFT v5 v3 to compare response quality. Diagnostic: are the garbage
responses from SFT v5 caused by the model or the retrieval contexts?

Gemma 3 fp32 mandatory on T4 (HF #36822, bf16 unsupported sm_75).
GenerationConfig object required (transformers 5.0 deprecation).
eos_token_id=[1, 106] for all Gemma 3 models.

Input:  pguillemin/gemma-3-1b-it (base model)
        pguillemin/pocket-arbiter-eval-data (DB + GS + retrieval contexts)
Output: eval_1b_base_pipeline.json + eval_1b_base_responses.jsonl
Runtime estimate: 298Q x ~18s/Q = ~90 min.
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

logger.info("Eval data: %s", EVAL_DIR)
logger.info("Base model: %s", BASE_DIR)

base_files = sorted(os.listdir(BASE_DIR))
logger.info("Base model files: %s", base_files[:10])

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

logger.info("=== PHASE 2: Load BASE model ===")
t0 = time.time()

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_DIR, torch_dtype=torch.float32, device_map={"": 0}
)
model.eval()

vram = torch.cuda.memory_allocated() / 1024 / 1024
logger.info("Model loaded: %.0f MB VRAM (fp32)", vram)

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
        "model": "base",
        "error": f"smoke_test_too_slow_{smoke_secs:.0f}s",
        "smoke_response": test_resp[:500],
        "model_eos_token_id": str(model.config.eos_token_id),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "use_cache": model.config.use_cache,
    }
    err_path = os.path.join(OUTPUT_DIR, "eval_1b_base_ERROR.json")
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
per_question_results: list[dict] = []

for label, subset in [("hit", hit_entries), ("miss", miss_entries)]:
    cited = 0
    abstained = 0
    empty = 0
    plengths: list[int] = []

    for i, entry in enumerate(subset):
        messages = build_rag_prompt(entry["question"], entry["retrieval_context"])
        response = generate_gpu(model, tokenizer, messages)

        is_cited = check_citation(
            response, entry["expected_docs"], entry["expected_pages"]
        )
        is_abstain = check_abstain(response)
        is_empty = not response.strip()

        if is_cited:
            cited += 1
        if is_abstain:
            abstained += 1
        if is_empty:
            empty += 1
        plengths.append(len(response.split()))

        per_question_results.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "response": response,
                "retrieval_context": entry["retrieval_context"],
                "retrieval_hit": entry["retrieval_hit"],
                "cited": is_cited,
                "abstained": is_abstain,
                "empty": is_empty,
                "words": len(response.split()),
            }
        )

        if (i + 1) % 50 == 0:
            elapsed_so_far = (time.time() - t0) / 60
            logger.info(
                "  [base %s %d/%d] %.1f min elapsed",
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
        "  base %s: cited=%.1f%% abstain=%.1f%% empty=%d median_words=%d",
        label,
        pipeline_results[label]["cited_pct"],
        pipeline_results[label]["abstain_pct"],
        empty,
        pipeline_results[label]["median_words"],
    )

# Save per-question responses (for HHEM faithfulness eval)
responses_path = os.path.join(OUTPUT_DIR, "eval_1b_base_responses.jsonl")
with open(responses_path, "w", encoding="utf-8") as f:
    for r in per_question_results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
logger.info("Saved %d responses: %s", len(per_question_results), responses_path)

total_n = len(retrieval_entries)
all_cited = pipeline_results["hit"]["cited"] + pipeline_results["miss"]["cited"]
all_abstain = (
    pipeline_results["hit"]["abstained"] + pipeline_results["miss"]["abstained"]
)
elapsed = round((time.time() - t0) / 60, 1)

output = {
    "model": "base",
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
out_path = os.path.join(OUTPUT_DIR, "eval_1b_base_pipeline.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
logger.info("Saved: %s", out_path)

# ============================================================
# PHASE 5: Gate check + summary
# ============================================================

logger.info("=== PHASE 5: Gate check ===")

base_cited = output["overall_cited_pct"]

logger.info("")
logger.info("=== PIPELINE CITATIONS (298Q) ===")
logger.info(
    "  base:  overall=%.1f%% hit=%.1f%% miss=%.1f%% (%.1f min)",
    base_cited,
    pipeline_results["hit"]["cited_pct"],
    pipeline_results["miss"]["cited_pct"],
    elapsed,
)

summary = {
    "base": output,
}
summary_path = os.path.join(OUTPUT_DIR, "eval_1b_base_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
logger.info("Saved: %s", summary_path)

# Cleanup
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

logger.info("Total: %.1f min", elapsed)
logger.info("=== DONE ===")
