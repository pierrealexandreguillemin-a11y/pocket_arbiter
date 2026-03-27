"""Phase B: Generate CoT answers for SFT v5 training data.

RAFT (Berkeley 2024, arXiv:2403.10131) Phase B:
- For each question from Phase A, build context with oracle + distractors
- 80% oracle present, 15% abstain (no oracle), 5% memorize (no context)
- Teacher (Gemma 3 4B IT via Unsloth) generates cited answers
- Validation: ##begin_quote## markers (RAFT Section 3) for reliable extraction
- Student (270M) gets CONCISE answer only, markers stripped before save

TWO-SESSION DESIGN (total ~14.7h, 49% weekly T4 quota):
- Session 1: QUESTION_RANGE = (0, 1116)    -> 904 oracle calls -> ~7.3h
- Session 2: QUESTION_RANGE = (1116, 2232)  -> 915 oracle calls -> ~7.4h
- Merge locally after both complete
- Per-question RNG (seed=SEED+i) -> deterministic regardless of session order

Design decisions backed by web research (2026-03-26):
- P=80%: RAFT Section 4.5, secondary sources cite 80% as strong default
- 4 distractors: RAFT Section 5.1 default
- 20% abstain: DTA (ACL 2025, arXiv:2505.20871) for high-stakes domains
- No CoT for 270M: Pleias-RAG-350M needed 3.1M examples, we have 2232
- ##begin_quote## markers: RAFT Section 3, +5-15pp (Table 2), stripped for clean SFT
- temp=0.3: low vs industry 0.6-1.0, justified for factual RAG (less hallucination)
- 15x teacher/student ratio: within range (DeepSeek 450x, Phi 500x, Gemma 3.5x)
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sqlite3  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
t_start = time.time()

# =====================================================================
# SESSION CONFIG — CHANGE THESE BETWEEN SESSIONS
# Session 1: (0, 1116)   -> ~904 oracle calls -> ~7.3h
# Session 2: (1116, 2232) -> ~915 oracle calls -> ~7.4h
# =====================================================================
QUESTION_RANGE = (0, 1116)  # <-- CHANGE FOR SESSION 2: (1116, 2232)
SESSION_TAG = f"s{QUESTION_RANGE[0]}_{QUESTION_RANGE[1]}"

# -- PHASE 0: Environment --------------------------------------------------
logger.info("=== PHASE 0: Environment ===")
logger.info("SESSION: questions %d to %d (tag=%s)", *QUESTION_RANGE, SESSION_TAG)
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

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth>=2025.3"])

from unsloth import FastModel  # noqa: E402

# -- Config -----------------------------------------------------------------
SEED = 42

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# Checkpoints: every 100 questions + every 30 min (whichever first)
CHECKPOINT_EVERY = 100
CHECKPOINT_TIME_MIN = 30
MAX_RUNTIME_MIN = 470  # 7h50 safety cutoff (Kaggle limit 9h, margin for Phase 5)

MAX_NEW_TOKENS_A = 250
MAX_WORDS_ORACLE = 400  # asymmetric: oracle gets more space
MAX_WORDS_DISTRACTOR = 150
N_DISTRACTORS = 4  # RAFT Section 5.1 default

# RAFT distribution (Section 4.5, P=80%)
ORACLE_RATIO = 0.80
ABSTAIN_RATIO = 0.15  # DTA (ACL 2025): abstain for high-stakes
MEMORIZE_RATIO = 0.05

ABSTAIN_ANSWER = "Information non trouvee dans les extraits fournis."

# Paths (dual fallback)
_EVAL_PATHS = [
    "/kaggle/input/pocket-arbiter-eval-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-eval-data",
]
_Q_PATHS = [
    "/kaggle/input/pa-sft-v5-questions",
    "/kaggle/input/datasets/pguillemin/pa-sft-v5-questions",
]

EVAL_DATA_DIR = next((p for p in _EVAL_PATHS if os.path.isdir(p)), None)
assert EVAL_DATA_DIR is not None, f"FATAL: Eval data not found: {_EVAL_PATHS}"
DB_PATH = os.path.join(EVAL_DATA_DIR, "corpus_v2_fr.db")
assert os.path.exists(DB_PATH), f"FATAL: DB not found: {DB_PATH}"

Q_DATA_DIR = next((p for p in _Q_PATHS if os.path.isdir(p)), None)
assert Q_DATA_DIR is not None, f"FATAL: Questions not found: {_Q_PATHS}"
Q_PATH = os.path.join(Q_DATA_DIR, "questions_v5.jsonl")
assert os.path.exists(Q_PATH), f"FATAL: Questions file not found: {Q_PATH}"

logger.info("DB: %s", DB_PATH)
logger.info("Questions: %s", Q_PATH)

# -- Prompt (RAFT ##begin_quote## markers for validation, stripped before save) --
ANSWER_PROMPT = """Tu es un assistant expert en reglements d'echecs FFE/FIDE.
Reponds a la question en citant UNIQUEMENT les extraits fournis.

EXTRAITS:
{context}

QUESTION: {question}

FORMAT DE REPONSE OBLIGATOIRE:
D'apres [NOM DU DOCUMENT] (p.XX) : ##begin_quote##"[citation exacte mot pour mot]"##end_quote##. [Explication concise en 1-2 phrases].

REGLES:
1. La citation entre ##begin_quote## et ##end_quote## doit etre EXACTEMENT copiee d'un extrait.
2. Indique le nom du document source et le numero de page.
3. Si AUCUN extrait ne contient la reponse, reponds exactement : "Information non trouvee dans les extraits fournis."
4. Sois concis (3 phrases max au total).
5. Reponds en francais."""


# -- Helpers ----------------------------------------------------------------


def truncate_words(text: str, max_words: int) -> str:
    """Truncate text to max_words, preserving word boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


def generate(model, tokenizer, prompt: str, max_tokens: int, temperature: float = 0.3):
    """Generate text. temp=0.3: factual RAG (industry 0.6-1.0 for diversity)."""
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def format_context(passages: list[dict]) -> str:
    """Format context passages for the prompt."""
    parts = []
    for i, p in enumerate(passages, 1):
        source_name = p["source"].replace(".pdf", "")
        max_w = MAX_WORDS_ORACLE if p.get("is_oracle") else MAX_WORDS_DISTRACTOR
        text = truncate_words(p["text"], max_w)
        parts.append(f"[{i}] {source_name} (p.{p['page']}):\n{text}")
    return "\n\n".join(parts)


def extract_quote(answer: str) -> tuple[str | None, str]:
    """Extract quote: prefer ##begin_quote## markers (RAFT), fallback regex."""
    # RAFT markers (preferred)
    match = re.search(r"##begin_quote##(.+?)##end_quote##", answer, re.DOTALL)
    if match:
        q = match.group(1).strip().strip("\"'")
        q = q.strip("\u201c\u201d\u00ab\u00bb")
        if len(q) >= 10:
            return q, "markers"
    # Regex fallback
    for pat in [
        r'"([^"]{10,})"',
        r"\u201c([^\u201d]{10,})\u201d",
        r"\u00ab([^\u00bb]{10,})\u00bb",
        r"'([^']{10,})'",
    ]:
        m = re.search(pat, answer)
        if m:
            return m.group(1).strip(), "regex"
    return None, "none"


def strip_markers(answer: str) -> str:
    """Remove RAFT markers for clean training data (student never sees them)."""
    return answer.replace("##begin_quote##", "").replace("##end_quote##", "")


def validate_quote(quote: str | None, oracle_text: str) -> bool:
    """Check quote is substring of oracle (80% prefix for minor truncation)."""
    if not quote or len(quote) < 10:
        return False
    nq = " ".join(quote.lower().split())
    no = " ".join(oracle_text.lower().split())
    check_len = max(10, int(len(nq) * 0.8))
    return nq[:check_len] in no


def assign_mode(index: int, seed: int = 42) -> str:
    """Assign RAFT mode for question at index. Deterministic per question."""
    rng = random.Random(seed + index)  # noqa: S311 — ML shuffling, not crypto
    r = rng.random()
    if r < ORACLE_RATIO:
        return "oracle"
    elif r < ORACLE_RATIO + ABSTAIN_RATIO:
        return "abstain"
    return "memorize"


def pick_distractors(
    chunk_id: str, index: int, all_ids: list[str], n: int, seed: int = 42
) -> list[str]:
    """Pick n random distractors. Per-question RNG for cross-session determinism."""
    rng = random.Random(seed + index + 10000)  # noqa: S311 — ML shuffling
    candidates = [cid for cid in all_ids if cid != chunk_id]
    return rng.sample(candidates, min(n, len(candidates)))


def shuffle_passages(passages: list[dict], index: int, seed: int = 42) -> list[dict]:
    """Shuffle passages with per-question RNG."""
    rng = random.Random(seed + index + 20000)  # noqa: S311 — ML shuffling
    shuffled = list(passages)
    rng.shuffle(shuffled)
    return shuffled


def save_checkpoint(
    output_path: str, output_file, stats: dict, i: int, total: int, trigger: str
) -> None:
    """Save data checkpoint + stats JSON."""
    output_file.flush()
    ckpt_path = os.path.join(OUTPUT_DIR, f"sft_data_v5_{SESSION_TAG}_ckpt_{i+1}.jsonl")
    shutil.copy2(output_path, ckpt_path)
    stats_path = os.path.join(
        OUTPUT_DIR, f"sft_data_v5_{SESSION_TAG}_ckpt_{i+1}_stats.json"
    )
    elapsed = (time.time() - t_start) / 60
    with open(stats_path, "w") as f:
        json.dump(
            {
                **stats,
                "elapsed_min": round(elapsed, 1),
                "progress": f"{i+1}/{total}",
                "trigger": trigger,
            },
            f,
            indent=2,
        )
    logger.info(
        "  CHECKPOINT [%s]: %s (%d entries, %.1f min)",
        trigger,
        ckpt_path,
        stats["total"],
        elapsed,
    )


# -- PHASE 1: Load teacher model -------------------------------------------
logger.info("=== PHASE 1: Load teacher model ===")

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-3-4b-it",
    max_seq_length=2048,
    load_in_4bit=True,
)
model.eval()

vram = torch.cuda.memory_allocated() / 1024 / 1024
logger.info("Teacher loaded: %.0f MB VRAM", vram)

test_resp = generate(model, tokenizer, "Qu'est-ce qu'un forfait aux echecs ?", 50)
logger.info("Smoke test: %s", test_resp[:100])
assert len(test_resp.strip()) > 0, "FATAL: Empty smoke test"

# -- PHASE 2: Load data ----------------------------------------------------
logger.info("=== PHASE 2: Load data ===")

questions = []
with open(Q_PATH, encoding="utf-8") as f:
    for line in f:
        questions.append(json.loads(line))
logger.info("Loaded %d questions total", len(questions))

conn = sqlite3.connect(DB_PATH)
all_chunks = {}
for row in conn.execute("SELECT id, source, page, text FROM children").fetchall():
    all_chunks[row[0]] = {
        "id": row[0],
        "source": row[1],
        "page": row[2],
        "text": row[3],
    }
conn.close()
all_chunk_ids = list(all_chunks.keys())
logger.info("Loaded %d chunks from DB", len(all_chunk_ids))

# Slice to this session's range
start_idx, end_idx = QUESTION_RANGE
session_questions = list(enumerate(questions))[start_idx:end_idx]
logger.info(
    "This session: questions %d-%d (%d questions)",
    start_idx,
    end_idx - 1,
    len(session_questions),
)

# Pre-compute modes for this session
session_modes = {i: assign_mode(i) for i, _ in session_questions}
mode_counts = {"oracle": 0, "abstain": 0, "memorize": 0}
for m in session_modes.values():
    mode_counts[m] += 1
logger.info("Mode distribution: %s", mode_counts)
logger.info("LLM calls needed: %d oracle", mode_counts["oracle"])

# -- PHASE 4: Generate answers ---------------------------------------------
logger.info("=== PHASE 4: Generate answers ===")

output_path = os.path.join(OUTPUT_DIR, f"sft_data_v5_{SESSION_TAG}.jsonl")
output_file = open(output_path, "w", encoding="utf-8")  # noqa: SIM115 — streaming writes across loop

stats = {
    "total": 0,
    "oracle_generated": 0,
    "abstain_written": 0,
    "memorize_written": 0,
    "quote_valid": 0,
    "quote_invalid": 0,
    "quote_missing": 0,
    "quote_markers": 0,
    "quote_regex": 0,
    "errors": 0,
    "empty_answers": 0,
}

last_ckpt_time = time.time()
processed = 0

for global_idx, q in session_questions:
    # Safety cutoff
    elapsed_min = (time.time() - t_start) / 60
    if elapsed_min > MAX_RUNTIME_MIN:
        logger.warning(
            "!! SAFETY CUTOFF at %.1f min (limit %d). Processed %d/%d.",
            elapsed_min,
            MAX_RUNTIME_MIN,
            processed,
            len(session_questions),
        )
        # Final checkpoint before exit
        save_checkpoint(
            output_path,
            output_file,
            stats,
            global_idx,
            len(session_questions),
            "cutoff",
        )
        break

    mode = session_modes[global_idx]
    chunk_id = q["chunk_id"]
    question = q["question"]

    # Defensive defaults
    context_ids = []
    quote = None
    quote_method = "none"
    quote_ok = False
    answer = ""

    try:
        t_q = time.time()

        if mode == "oracle":
            oracle_chunk = all_chunks.get(chunk_id)
            if not oracle_chunk:
                logger.warning("  SKIP %s: chunk not in DB", chunk_id)
                stats["errors"] += 1
                continue

            d_ids = pick_distractors(chunk_id, global_idx, all_chunk_ids, N_DISTRACTORS)
            distractors = [all_chunks[cid] for cid in d_ids]

            passages = [{**oracle_chunk, "is_oracle": True}]
            for d in distractors:
                passages.append({**d, "is_oracle": False})
            passages = shuffle_passages(passages, global_idx)

            context_str = format_context(passages)
            prompt = ANSWER_PROMPT.format(context=context_str, question=question)
            raw_answer = generate(model, tokenizer, prompt, MAX_NEW_TOKENS_A)

            quote, quote_method = extract_quote(raw_answer)
            quote_ok = validate_quote(quote, oracle_chunk["text"])

            if quote_ok:
                stats["quote_valid"] += 1
            elif quote is not None:
                stats["quote_invalid"] += 1
            else:
                stats["quote_missing"] += 1
            if quote_method == "markers":
                stats["quote_markers"] += 1
            elif quote_method == "regex":
                stats["quote_regex"] += 1

            answer = strip_markers(raw_answer)
            context_ids = [p["id"] for p in passages]
            stats["oracle_generated"] += 1

        elif mode == "abstain":
            d_ids = pick_distractors(chunk_id, global_idx, all_chunk_ids, N_DISTRACTORS)
            context_ids = d_ids
            answer = ABSTAIN_ANSWER
            quote_ok = True
            stats["abstain_written"] += 1

        elif mode == "memorize":
            answer = ABSTAIN_ANSWER
            quote_ok = True
            stats["memorize_written"] += 1

        if not answer or len(answer.strip()) < 5:
            stats["empty_answers"] += 1
            logger.warning("  EMPTY %s (mode=%s)", chunk_id, mode)

        entry = {
            "global_idx": global_idx,
            "chunk_id": chunk_id,
            "question": question,
            "oracle_source": q["source"],
            "oracle_page": q["page"],
            "mode": mode,
            "context_ids": context_ids,
            "oracle_in_context": mode == "oracle",
            "answer": answer.strip(),
            "quote": quote,
            "quote_valid": quote_ok,
            "quote_method": quote_method,
            "generation_time_s": round(time.time() - t_q, 1),
        }
        output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        stats["total"] += 1
        processed += 1

    except Exception as e:
        logger.warning("  ERROR %s: %s", chunk_id, str(e)[:100])
        stats["errors"] += 1

    # Progress every 25
    if processed % 25 == 0 and processed > 0:
        output_file.flush()
        elapsed = (time.time() - t_start) / 60
        rate = processed / elapsed if elapsed > 0 else 1
        eta = (len(session_questions) - processed) / rate if rate > 0 else 0
        logger.info(
            "  [%d/%d] oracle=%d valid=%d inv=%d miss=%d abs=%d err=%d | %.0f min (ETA %.0f)",
            processed,
            len(session_questions),
            stats["oracle_generated"],
            stats["quote_valid"],
            stats["quote_invalid"],
            stats["quote_missing"],
            stats["abstain_written"],
            stats["errors"],
            elapsed,
            eta,
        )

    # Checkpoint: every N questions OR every M minutes
    now = time.time()
    is_count = processed % CHECKPOINT_EVERY == 0 and processed > 0
    is_time = (now - last_ckpt_time) / 60 >= CHECKPOINT_TIME_MIN

    if is_count or is_time:
        save_checkpoint(
            output_path,
            output_file,
            stats,
            global_idx,
            len(session_questions),
            "count" if is_count else "time",
        )
        last_ckpt_time = now

# Final checkpoint (captures everything since last periodic checkpoint)
output_file.flush()
if processed > 0:
    save_checkpoint(
        output_path,
        output_file,
        stats,
        global_idx,
        len(session_questions),
        "final",
    )
output_file.close()

# -- PHASE 5: Summary + validation -----------------------------------------
elapsed = (time.time() - t_start) / 60
logger.info("=" * 60)
logger.info(
    "SESSION %s COMPLETE -- %.1f min (%d/%d processed)",
    SESSION_TAG,
    elapsed,
    processed,
    len(session_questions),
)
logger.info("Stats: %s", json.dumps(stats, indent=2))

oracle_total = stats["oracle_generated"]
quote_rate = stats["quote_valid"] / max(oracle_total, 1)
marker_rate = stats["quote_markers"] / max(oracle_total, 1)

logger.info("=== QUALITY GATES ===")
logger.info(
    "G1: Oracle generated: %d / %d expected", oracle_total, mode_counts["oracle"]
)
logger.info(
    "G2: Quote validation rate: %.1f%% (%d/%d)",
    100 * quote_rate,
    stats["quote_valid"],
    oracle_total,
)
logger.info("G2b: Marker extraction: %.1f%% (vs regex fallback)", 100 * marker_rate)
logger.info("G3: Empty: %d | G4: Errors: %d", stats["empty_answers"], stats["errors"])
logger.info("G5: Processed: %d/%d", processed, len(session_questions))

if quote_rate < 0.30:
    logger.warning("!! G2 FAIL: Quote rate < 30%%")
if stats["empty_answers"] > 10:
    logger.warning("!! G3 WARN: %d empty answers", stats["empty_answers"])
if processed < len(session_questions):
    logger.warning(
        "!! G5 INCOMPLETE: %d/%d (cutoff or errors)", processed, len(session_questions)
    )

# Visual inspection: 10 oracle + 3 abstain (MANDATORY)
with open(output_path, encoding="utf-8") as f:
    all_entries = [json.loads(line) for line in f]

oracle_entries = [entry for entry in all_entries if entry["mode"] == "oracle"]
logger.info("\n=== 10 ORACLE SAMPLES (MANDATORY VISUAL AUDIT) ===")
rng_inspect = random.Random(SEED)  # noqa: S311 — ML sampling
for idx in rng_inspect.sample(range(len(oracle_entries)), min(10, len(oracle_entries))):
    sample = oracle_entries[idx]
    logger.info(
        "--- [%s] valid=%s method=%s ---",
        sample["chunk_id"][:35],
        sample["quote_valid"],
        sample.get("quote_method"),
    )
    logger.info("  Q: %s", sample["question"][:100])
    logger.info("  A: %s", sample["answer"][:250])
    logger.info("")

abstain_entries = [entry for entry in all_entries if entry["mode"] == "abstain"]
if abstain_entries:
    logger.info("=== 3 ABSTAIN SAMPLES ===")
    for sample in abstain_entries[:3]:
        logger.info("  Q: %s  |  A: %s", sample["question"][:60], sample["answer"][:60])

# Answer length distribution
median_words = 0
answer_words = [len(entry["answer"].split()) for entry in oracle_entries]
if answer_words:
    answer_words.sort()
    n = len(answer_words)
    median_words = answer_words[n // 2]
    logger.info("=== ORACLE ANSWER LENGTH (words) ===")
    logger.info(
        "  Min=%d P10=%d Median=%d Mean=%.1f P90=%d Max=%d",
        answer_words[0],
        answer_words[n // 10],
        median_words,
        sum(answer_words) / n,
        answer_words[9 * n // 10],
        answer_words[-1],
    )
    if median_words < 20:
        logger.warning(
            "!! FAIL: Median %d < 20 words (SFT v1-v4 garbage was 16)", median_words
        )

# Save metrics
metrics = {
    "session": SESSION_TAG,
    "question_range": list(QUESTION_RANGE),
    "total_in_range": len(session_questions),
    "processed": processed,
    "complete": processed >= len(session_questions),
    **{k: v for k, v in stats.items()},
    "quote_validation_rate": round(quote_rate, 3),
    "quote_marker_rate": round(marker_rate, 3),
    "median_oracle_answer_words": median_words,
    "elapsed_min": round(elapsed, 1),
    "gpu": torch.cuda.get_device_name(0),
    "config": {
        "oracle_ratio": ORACLE_RATIO,
        "abstain_ratio": ABSTAIN_RATIO,
        "memorize_ratio": MEMORIZE_RATIO,
        "n_distractors": N_DISTRACTORS,
        "max_new_tokens": MAX_NEW_TOKENS_A,
        "max_words_oracle": MAX_WORDS_ORACLE,
        "max_words_distractor": MAX_WORDS_DISTRACTOR,
        "temperature": 0.3,
        "seed": SEED,
        "checkpoint_every": CHECKPOINT_EVERY,
    },
    "standards": [
        "RAFT arXiv:2403.10131 (P=80%, 4 distractors, ##begin_quote##)",
        "DTA arXiv:2505.20871 (abstain for high-stakes)",
        "Pleias-RAG arXiv:2504.18225 (350M learns citations)",
        "SQuAD 2.0 arXiv:1806.03822 (33% unanswerable benchmark)",
    ],
}
with open(
    os.path.join(OUTPUT_DIR, f"gen_answers_metrics_{SESSION_TAG}.json"), "w"
) as f:
    json.dump(metrics, f, indent=2)

logger.info("=== DONE (session %s) ===", SESSION_TAG)
