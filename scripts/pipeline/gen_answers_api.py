"""Phase B (API): Generate RAFT answers via Google Gemini API (Gemma 3 27B IT).

Replaces Kaggle T4 inference for remaining questions after Session 1 cutoff.
Same RAFT logic as gen_answers_cot.py but 13x faster via API (3.5s vs 46s/call).

Advantages over Kaggle T4:
- Gemma 3 27B full precision (vs 4B NF4 quantized)
- 3.5s/call (vs 46s/call)
- Zero T4 quota consumed
- Runs on any machine (HTTP calls, no GPU)

Design: identical RAFT distribution, quote validation, output format.
Standards: RAFT arXiv:2403.10131, DTA arXiv:2505.20871

Usage:
    export GEMINI_API_KEY="your-key"
    python scripts/pipeline/gen_answers_api.py --start 743 --end 2232
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sqlite3
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# -- Config -----------------------------------------------------------------
SEED = 42
ORACLE_RATIO = 0.80
ABSTAIN_RATIO = 0.15
MEMORIZE_RATIO = 0.05
N_DISTRACTORS = 4
MAX_WORDS_ORACLE = 400
MAX_WORDS_DISTRACTOR = 150

ABSTAIN_ANSWER = "Information non trouvee dans les extraits fournis."
MODEL = "gemma-3-27b-it"

# Retry config for 429 rate limits
MAX_RETRIES = 5
BASE_BACKOFF_S = 10

# Checkpoint
CHECKPOINT_EVERY = 50

# Paths
DB_PATH = "corpus/processed/corpus_v2_fr.db"
Q_PATH = "models/kaggle-gen-questions-output/questions_v5.jsonl"

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
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


def format_context(passages: list[dict]) -> str:
    parts = []
    for i, p in enumerate(passages, 1):
        source_name = p["source"].replace(".pdf", "")
        max_w = MAX_WORDS_ORACLE if p.get("is_oracle") else MAX_WORDS_DISTRACTOR
        text = truncate_words(p["text"], max_w)
        parts.append(f"[{i}] {source_name} (p.{p['page']}):\n{text}")
    return "\n\n".join(parts)


def generate_api(client, prompt: str) -> str:  # noqa: S311
    """Call Gemini API with retry/backoff for 429 rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            return response.text or ""
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = BASE_BACKOFF_S * (2**attempt)
                logger.warning(
                    "  429 rate limit, backoff %ds (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                logger.error("  API error: %s", err[:120])
                raise
    logger.error("  Max retries exceeded")
    return ""


def extract_quote(answer: str) -> tuple[str | None, str]:
    match = re.search(r"##begin_quote##(.+?)##end_quote##", answer, re.DOTALL)
    if match:
        q = match.group(1).strip().strip("\"'")
        q = q.strip("\u201c\u201d\u00ab\u00bb")
        if len(q) >= 10:
            return q, "markers"
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
    return answer.replace("##begin_quote##", "").replace("##end_quote##", "")


def validate_quote(quote: str | None, oracle_text: str) -> bool:
    if not quote or len(quote) < 10:
        return False
    nq = " ".join(quote.lower().split())
    no = " ".join(oracle_text.lower().split())
    check_len = max(10, int(len(nq) * 0.8))
    return nq[:check_len] in no


def assign_mode(index: int, seed: int = 42) -> str:
    rng = random.Random(seed + index)  # noqa: S311
    r = rng.random()
    if r < ORACLE_RATIO:
        return "oracle"
    elif r < ORACLE_RATIO + ABSTAIN_RATIO:
        return "abstain"
    return "memorize"


def pick_distractors(
    chunk_id: str, index: int, all_ids: list[str], n: int, seed: int = 42
) -> list[str]:
    rng = random.Random(seed + index + 10000)  # noqa: S311
    candidates = [cid for cid in all_ids if cid != chunk_id]
    return rng.sample(candidates, min(n, len(candidates)))


def shuffle_passages(passages: list[dict], index: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed + index + 20000)  # noqa: S311
    shuffled = list(passages)
    rng.shuffle(shuffled)
    return shuffled


def _process_question(
    client,
    q: dict,
    global_idx: int,
    mode: str,
    all_chunks: dict,
    all_chunk_ids: list[str],
    stats: dict,
) -> dict | None:
    """Process a single question and return an entry dict, or None on skip."""
    chunk_id = q["chunk_id"]
    question = q["question"]
    context_ids: list[str] = []
    quote = None
    quote_method = "none"
    quote_ok = False
    answer = ""
    t_q = time.time()

    if mode == "oracle":
        oracle_chunk = all_chunks.get(chunk_id)
        if not oracle_chunk:
            logger.warning("  SKIP %s: chunk not in DB", chunk_id)
            stats["errors"] += 1
            return None

        d_ids = pick_distractors(chunk_id, global_idx, all_chunk_ids, N_DISTRACTORS)
        distractors = [all_chunks[cid] for cid in d_ids]
        passages = [{**oracle_chunk, "is_oracle": True}]
        for d in distractors:
            passages.append({**d, "is_oracle": False})
        passages = shuffle_passages(passages, global_idx)

        context_str = format_context(passages)
        prompt = ANSWER_PROMPT.format(context=context_str, question=question)
        raw_answer = generate_api(client, prompt)

        if not raw_answer:
            stats["errors"] += 1
            stats["rate_limited"] += 1
            return None

        quote, quote_method = extract_quote(raw_answer)
        quote_ok = validate_quote(quote, oracle_chunk["text"])
        _update_quote_stats(stats, quote_ok, quote, quote_method)
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

    return {
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


def _update_quote_stats(
    stats: dict, quote_ok: bool, quote: str | None, method: str
) -> None:
    """Update quote validation counters."""
    if quote_ok:
        stats["quote_valid"] += 1
    elif quote is not None:
        stats["quote_invalid"] += 1
    else:
        stats["quote_missing"] += 1
    if method == "markers":
        stats["quote_markers"] += 1
    elif method == "regex":
        stats["quote_regex"] += 1


def _log_progress(
    output_file, stats: dict, processed: int, total: int, t_start: float
) -> None:
    """Log progress and flush output."""
    output_file.flush()
    elapsed = (time.time() - t_start) / 60
    rate = processed / elapsed if elapsed > 0 else 1
    eta = (total - processed) / rate if rate > 0 else 0
    logger.info(
        "  [%d/%d] oracle=%d valid=%d inv=%d miss=%d abs=%d err=%d rl=%d | %.0f min (ETA %.0f)",
        processed,
        total,
        stats["oracle_generated"],
        stats["quote_valid"],
        stats["quote_invalid"],
        stats["quote_missing"],
        stats["abstain_written"],
        stats["errors"],
        stats["rate_limited"],
        elapsed,
        eta,
    )


# -- Main -------------------------------------------------------------------


def _load_data() -> tuple[list[dict], dict, list[str]]:
    """Load questions and corpus chunks."""
    assert os.path.exists(DB_PATH), f"DB not found: {DB_PATH}"
    assert os.path.exists(Q_PATH), f"Questions not found: {Q_PATH}"

    questions = []
    with open(Q_PATH, encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    logger.info("Loaded %d questions", len(questions))

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
    logger.info("Loaded %d chunks", len(all_chunks))
    return questions, all_chunks, list(all_chunks.keys())


def _make_stats(skip_n: int = 0) -> dict:
    """Create fresh stats dict."""
    return {
        "total": skip_n,
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
        "rate_limited": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAFT answers via Gemini API")
    parser.add_argument(
        "--start", type=int, required=True, help="Start question index (inclusive)"
    )
    parser.add_argument(
        "--end", type=int, required=True, help="End question index (exclusive)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key, "FATAL: Set GEMINI_API_KEY environment variable"

    from google import genai

    client = genai.Client(api_key=api_key)
    test = client.models.generate_content(model=MODEL, contents="Test: 1+1=?")
    assert test.text, "FATAL: API smoke test returned empty"
    logger.info("API OK (model=%s)", MODEL)

    questions, all_chunks, all_chunk_ids = _load_data()

    tag = f"s{args.start}_{args.end}"
    output_path = (
        args.output or f"models/kaggle-gen-answers-output/sft_data_v5_{tag}.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    skip_n = 0
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            skip_n = sum(1 for _ in f)
        logger.info("RESUME: %d entries exist, skipping", skip_n)

    session_questions = list(enumerate(questions))[args.start : args.end]
    logger.info(
        "Processing questions %d-%d (%d total, skip %d)",
        args.start,
        args.end - 1,
        len(session_questions),
        skip_n,
    )

    session_modes = {i: assign_mode(i) for i, _ in session_questions}
    mode_counts = {"oracle": 0, "abstain": 0, "memorize": 0}
    for m in session_modes.values():
        mode_counts[m] += 1
    logger.info("Modes: %s", mode_counts)

    # Generate
    output_file = open(output_path, "a" if skip_n > 0 else "w", encoding="utf-8")  # noqa: SIM115
    t_start = time.time()
    stats = _make_stats(skip_n)

    processed = 0
    for global_idx, q in session_questions:
        if processed < skip_n:
            processed += 1
            continue

        mode = session_modes[global_idx]
        try:
            entry = _process_question(
                client, q, global_idx, mode, all_chunks, all_chunk_ids, stats
            )
            if entry:
                output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stats["total"] += 1
        except Exception as e:
            logger.warning("  ERROR %s: %s", q["chunk_id"], str(e)[:100])
            stats["errors"] += 1

        processed += 1
        if processed % 25 == 0:
            _log_progress(
                output_file, stats, processed, len(session_questions), t_start
            )
        if processed % CHECKPOINT_EVERY == 0:
            output_file.flush()
            logger.info("  CHECKPOINT: %d entries saved", stats["total"])

    output_file.flush()
    output_file.close()
    _summarize(output_path, stats, t_start, args.start, args.end)


def _summarize(
    output_path: str, stats: dict, t_start: float, start: int, end: int
) -> None:
    """Log summary, visual inspection, and save metrics."""
    elapsed = (time.time() - t_start) / 60
    oracle_total = stats["oracle_generated"]
    quote_rate = stats["quote_valid"] / max(oracle_total, 1)

    logger.info("=" * 60)
    logger.info("COMPLETE — %.1f min, %d entries", elapsed, stats["total"])
    logger.info("Stats: %s", json.dumps(stats, indent=2))
    logger.info(
        "G2: Quote validation: %.1f%% (%d/%d)",
        100 * quote_rate,
        stats["quote_valid"],
        oracle_total,
    )

    with open(output_path, encoding="utf-8") as f:
        all_entries = [json.loads(line) for line in f]
    oracle_entries = [entry for entry in all_entries if entry["mode"] == "oracle"]

    logger.info("\n=== 10 ORACLE SAMPLES ===")
    rng_inspect = random.Random(SEED)  # noqa: S311
    for idx in rng_inspect.sample(
        range(len(oracle_entries)), min(10, len(oracle_entries))
    ):
        sample = oracle_entries[idx]
        logger.info(
            "--- [%s] valid=%s method=%s ---",
            sample["chunk_id"][:35],
            sample["quote_valid"],
            sample.get("quote_method"),
        )
        logger.info("  Q: %s", sample["question"][:100])
        logger.info("  A: %s", sample["answer"][:250])

    median_words = 0
    answer_words = [len(entry["answer"].split()) for entry in oracle_entries]
    if answer_words:
        answer_words.sort()
        median_words = answer_words[len(answer_words) // 2]
        logger.info(
            "Answer words: Min=%d Median=%d Mean=%.1f Max=%d",
            answer_words[0],
            median_words,
            sum(answer_words) / len(answer_words),
            answer_words[-1],
        )

    metrics = {
        "range": [start, end],
        "model": MODEL,
        "method": "google_gemini_api",
        **stats,
        "quote_validation_rate": round(quote_rate, 3),
        "median_oracle_answer_words": median_words,
        "elapsed_min": round(elapsed, 1),
    }
    metrics_path = output_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Output: %s", output_path)


if __name__ == "__main__":
    main()
