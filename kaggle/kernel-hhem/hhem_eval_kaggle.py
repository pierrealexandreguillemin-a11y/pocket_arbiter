"""HHEM-2.1-Open faithfulness evaluation — Kaggle CPU kernel.

Scores 298 SFT v5 responses against retrieval contexts using Vectara's
hallucination evaluation model. Option C (chained): max(score per chunk).

ISO compliance:
  - ISO 25010 FA-04: hallucination detection (score < 0.5 = hallucinated)
  - ISO 25010 TB-04: RAGAS faithfulness proxy (HHEM offline alternative)
  - ISO 42001 A.6.2.2: model documented (version, source, method)
  - ISO 42001 A.7.2: sanity check 5 known FR pairs before full run
  - ISO 29119: reproducible (deterministic inference)

Input:  pguillemin/pocket-arbiter-eval-data (eval_1b_sft_v5_responses.jsonl)
Output: hhem_faithfulness.json (per-question scores + aggregates + gate)

Standards:
  - Vectara HHEM-2.1-Open (2024): T5-base hallucination classifier, FR native
  - RAGAS (arXiv:2309.15217): HHEM as offline alternative to LLM judge
  - Wallat et al. ICTIR 2025: citation != faithfulness

Runtime estimate: 298Q × ~5 chunks × ~0.1s/pair = ~2.5 min on Kaggle CPU.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# PHASE 0: Environment
# ============================================================

logger.info("=== PHASE 0: Environment ===")


def _resolve(slug: str) -> str:
    for c in [f"/kaggle/input/{slug}", f"/kaggle/input/datasets/pguillemin/{slug}"]:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Not found: {slug}")


EVAL_DIR = _resolve("pocket-arbiter-eval-data")
logger.info("Eval data: %s", EVAL_DIR)

RESPONSES_PATH = os.path.join(EVAL_DIR, "eval_1b_sft_v5_responses.jsonl")
assert os.path.exists(RESPONSES_PATH), f"FATAL: {RESPONSES_PATH}"

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ============================================================
# PHASE 1: Load HHEM model
# ============================================================

logger.info("=== PHASE 1: Load HHEM model ===")

MODEL_ID = "vectara/hallucination_evaluation_model"
t0 = time.time()

from transformers import AutoModelForSequenceClassification  # noqa: E402

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, trust_remote_code=True
)
logger.info("Model loaded in %.1fs: %s", time.time() - t0, MODEL_ID)

# ============================================================
# Configuration
# ============================================================

CHUNK_SEPARATOR = "\n\n"
MIN_CHUNK_WORDS = 20
MAX_CONTEXT_WORDS = 150
MAX_RESPONSE_WORDS = 100

# Gate TB-04 (QUALITY_REQUIREMENTS.md). HHEM FR scores compressed vs EN.
# We report raw + document the FR compression factor from sanity check.
GATE_FAITHFULNESS = 0.85

# Sanity checks: known FR pairs, thresholds calibrated for FR range.
# Consistent pairs: 0.3-0.65 FR (vs 0.7-0.95 EN). Contradictions: < 0.1 both.
SANITY_CHECKS = [
    {
        "context": "L'arbitre doit etre present au moins 30 minutes avant le debut de la ronde.",
        "response": "L'arbitre doit arriver 30 minutes avant la ronde.",
        "expected_min": 0.3,
        "label": "consistent_paraphrase",
    },
    {
        "context": "L'arbitre doit etre present au moins 30 minutes avant le debut de la ronde.",
        "response": "L'arbitre n'a pas besoin d'etre present avant la ronde.",
        "expected_max": 0.4,
        "label": "contradiction",
    },
    {
        "context": "Le forfait est prononce si le joueur ne se presente pas dans les 30 minutes.",
        "response": "Le joueur est declare forfait apres 30 minutes d'absence.",
        "expected_min": 0.2,
        "label": "consistent_inference",
    },
    {
        "context": "Les parties rapides se jouent avec un temps de reflexion de 15 minutes.",
        "response": "La cadence rapide est de 60 minutes par joueur.",
        "expected_max": 0.4,
        "label": "factual_error",
    },
    {
        "context": "Le classement Elo est calcule selon la formule de la FIDE.",
        "response": "Le classement Elo utilise la formule FIDE pour calculer les performances.",
        "expected_min": 0.4,
        "label": "faithful_restatement",
    },
]


# ============================================================
# Helpers
# ============================================================


def truncate(text: str, max_words: int) -> str:
    """Truncate to max_words. FR tokenizes ~1.3x denser than EN."""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def split_context(context: str) -> list[str]:
    """Split concatenated retrieval context into individual chunks."""
    chunks = context.split(CHUNK_SEPARATOR)
    return [c.strip() for c in chunks if len(c.split()) >= MIN_CHUNK_WORDS]


def score_pair(context: str, response: str) -> float:
    """Score (context, response). Returns P(consistent) in [0,1]."""
    ctx = truncate(context, MAX_CONTEXT_WORDS)
    resp = truncate(response, MAX_RESPONSE_WORDS)
    raw = model.predict([[ctx, resp]])
    return float(raw[0]) if hasattr(raw, "__len__") else float(raw)


# ============================================================
# PHASE 2: Sanity check (ISO 42001 A.7.2)
# ============================================================

logger.info("=== PHASE 2: Sanity check (FR) ===")

sanity_pass = True
for sc in SANITY_CHECKS:
    score = score_pair(str(sc["context"]), str(sc["response"]))
    if "expected_min" in sc:
        ok = score >= float(sc["expected_min"])  # type: ignore[arg-type]
        logger.info(
            "  [%s] %s: score=%.3f >= %.1f",
            sc["label"],
            "PASS" if ok else "FAIL",
            score,
            float(sc["expected_min"]),  # type: ignore[arg-type]
        )
    else:
        ok = score <= float(sc["expected_max"])  # type: ignore[arg-type]
        logger.info(
            "  [%s] %s: score=%.3f <= %.1f",
            sc["label"],
            "PASS" if ok else "FAIL",
            score,
            float(sc["expected_max"]),  # type: ignore[arg-type]
        )
    if not ok:
        sanity_pass = False

if not sanity_pass:
    logger.error("FATAL: Sanity check FAILED. HHEM unreliable on FR. Aborting.")
    sys.exit(1)
logger.info("Sanity check PASS — HHEM functional on FR")

# ============================================================
# PHASE 3: Load responses
# ============================================================

logger.info("=== PHASE 3: Load responses ===")

with open(RESPONSES_PATH, encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

hits = [e for e in entries if e["retrieval_hit"]]
misses = [e for e in entries if not e["retrieval_hit"]]
logger.info(
    "Loaded %d entries (%d hits, %d misses)", len(entries), len(hits), len(misses)
)

# ============================================================
# PHASE 4: Faithfulness scoring (298Q, Option C chained)
# ============================================================

logger.info("=== PHASE 4: Faithfulness scoring (298Q) ===")

results: list[dict] = []
t_start = time.time()

for i, entry in enumerate(entries):
    response = entry["response"].strip()
    if not response or entry["empty"]:
        results.append(
            {
                "id": entry["id"],
                "retrieval_hit": entry["retrieval_hit"],
                "hhem_max": 0.0,
                "hhem_scores": [],
                "n_chunks": 0,
                "cited": entry["cited"],
                "abstained": entry["abstained"],
            }
        )
        continue

    chunks = split_context(entry["retrieval_context"])
    if not chunks:
        results.append(
            {
                "id": entry["id"],
                "retrieval_hit": entry["retrieval_hit"],
                "hhem_max": 0.0,
                "hhem_scores": [],
                "n_chunks": 0,
                "cited": entry["cited"],
                "abstained": entry["abstained"],
            }
        )
        continue

    chunk_scores = [score_pair(chunk, response) for chunk in chunks]

    results.append(
        {
            "id": entry["id"],
            "retrieval_hit": entry["retrieval_hit"],
            "hhem_max": round(float(max(chunk_scores)), 4),
            "hhem_scores": [round(float(s), 4) for s in chunk_scores],
            "n_chunks": len(chunks),
            "cited": entry["cited"],
            "abstained": entry["abstained"],
        }
    )

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_start
        logger.info(
            "[%d/%d] %.1fs elapsed, avg %.2fs/Q",
            i + 1,
            len(entries),
            elapsed,
            elapsed / (i + 1),
        )

elapsed = time.time() - t_start
logger.info(
    "Scoring done: %d entries, %.1fs (%.2fs/Q)",
    len(results),
    elapsed,
    elapsed / len(results),
)

# ============================================================
# PHASE 5: Aggregate + gate check
# ============================================================

logger.info("=== PHASE 5: Gate check ===")

all_scores = [r["hhem_max"] for r in results if r["n_chunks"] > 0]
hit_scores = [
    r["hhem_max"] for r in results if r["retrieval_hit"] and r["n_chunks"] > 0
]
miss_scores = [
    r["hhem_max"] for r in results if not r["retrieval_hit"] and r["n_chunks"] > 0
]


def _stats(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0, "mean": 0, "median": 0}
    s = sorted(scores)
    n = len(s)
    return {
        "count": n,
        "mean": round(sum(s) / n, 4),
        "median": round(s[n // 2], 4),
        "green": sum(1 for x in s if x >= 0.8),
        "amber": sum(1 for x in s if 0.5 <= x < 0.8),
        "red": sum(1 for x in s if x < 0.5),
        "green_pct": round(100 * sum(1 for x in s if x >= 0.8) / n, 1),
        "red_pct": round(100 * sum(1 for x in s if x < 0.5) / n, 1),
    }


global_stats = _stats(all_scores)
gate_pass = global_stats["mean"] >= GATE_FAITHFULNESS

output = {
    "model": MODEL_ID,
    "model_type": "T5-base (~250 MB), HHEM-2.1-Open, trust_remote_code=True",
    "method": "Option C chained: max(score per chunk), ctx 150w + resp 100w truncation",
    "standards": [
        "ISO 25010 FA-04 (hallucination detection)",
        "ISO 25010 TB-04 (RAGAS faithfulness proxy, target >= 0.85)",
        "Vectara HHEM-2.1-Open (2024), FR support since v2.0",
        "Wallat et al. ICTIR 2025 (citation != faithfulness)",
    ],
    "fr_calibration_note": (
        "HHEM FR scores are compressed vs EN. "
        "Consistent pairs: 0.3-0.65 FR vs 0.7-0.95 EN. "
        "Contradictions: <0.1 both. Discrimination is preserved."
    ),
    "gate": {
        "target": f"mean >= {GATE_FAITHFULNESS}",
        "actual": global_stats["mean"],
        "pass": gate_pass,
    },
    "sanity_check": "PASS (5/5 FR pairs)",
    "global": global_stats,
    "hit": _stats(hit_scores),
    "miss": _stats(miss_scores),
    "elapsed_sec": round(elapsed, 1),
    "per_question": results,
}

out_path = os.path.join(OUTPUT_DIR, "hhem_faithfulness.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
logger.info("Saved: %s", out_path)

# Summary
logger.info("")
logger.info("=== HHEM Faithfulness Results ===")
for label, stats in [
    ("global", global_stats),
    ("hit", _stats(hit_scores)),
    ("miss", _stats(miss_scores)),
]:
    logger.info(
        "  %s: mean=%.3f median=%.3f green=%d (%.0f%%) amber=%d red=%d (%.0f%%) [n=%d]",
        label,
        stats["mean"],
        stats["median"],
        stats["green"],
        stats["green_pct"],
        stats["amber"],
        stats["red"],
        stats["red_pct"],
        stats["count"],
    )
logger.info("")
logger.info(
    "GATE TB-04: mean %.3f >= %.2f = %s",
    global_stats["mean"],
    GATE_FAITHFULNESS,
    "PASS" if gate_pass else "FAIL",
)
logger.info("=== DONE ===")
