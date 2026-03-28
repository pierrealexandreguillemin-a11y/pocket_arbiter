"""Pre-compute retrieval contexts for end-to-end RAG evaluation.

Runs the REAL hybrid search pipeline (cosine + BM25 + structured cells)
on all 298 testable GS questions and saves retrieved contexts as JSONL.

This separates retrieval (slow, needs embedding model) from generation
(needs LLM, can run on Kaggle T4). Standard approach: eRAG (2024).

Output: retrieval_contexts_v5b.jsonl — one line per question with:
  - question text
  - retrieved context (concatenated top-k chunks)
  - oracle context (from GS provenance, for comparison)
  - retrieval metadata (hit/miss, scores, sources)

Usage:
    python -m scripts.pipeline.precompute_retrieval
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

from scripts.pipeline.recall import load_gs, page_match
from scripts.pipeline.search import search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH = Path("corpus/processed/corpus_v2_fr.db")
GS_PATH = Path("tests/data/gold_standard_annales_fr_v8_adversarial.json")
OUTPUT_PATH = Path("data/benchmarks/eval_v5/retrieval_contexts_v5b.jsonl")


def load_oracle_context(db_path: Path, source: str, page: int) -> str:
    """Load oracle context (same as eval v5 kernel)."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT text FROM children WHERE source = ? AND page = ?",
            (source, page),
        ).fetchall()
        return "\n\n".join(r[0] for r in rows)
    finally:
        conn.close()


def main() -> None:
    """Pre-compute retrieval for all 298 testable questions."""
    logger.info("Loading GS from %s", GS_PATH)
    questions = load_gs(GS_PATH)
    logger.info("Loaded %d testable questions", len(questions))

    # Load embedding model once
    from scripts.pipeline.indexer import load_model

    model = load_model()
    logger.info("Embedding model loaded")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    hits = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions):
            # Real pipeline retrieval
            result = search(DB_PATH, q["question"], model=model, max_k=5)

            # Build retrieval context (concatenated)
            retrieval_text = "\n\n".join(ctx.text for ctx in result.contexts)
            retrieval_sources = [
                {"source": ctx.source, "page": ctx.page, "score": ctx.score}
                for ctx in result.contexts
            ]

            # Check if retrieval hit the right page
            # Parents have page=None, so also check matched children pages
            conn = sqlite3.connect(str(DB_PATH))
            retrieved_pages: list[tuple[str, int | None]] = []
            for ctx in result.contexts:
                if ctx.page is not None:
                    retrieved_pages.append((ctx.source, ctx.page))
                # Also check children_matched (they have real pages)
                for cid in ctx.children_matched:
                    row = conn.execute(
                        "SELECT source, page FROM children WHERE id = ?",
                        (cid,),
                    ).fetchone()
                    if row:
                        retrieved_pages.append((row[0], row[1]))
            conn.close()
            is_hit = page_match(q["expected_pairs"], retrieved_pages)
            if is_hit:
                hits += 1

            # Oracle context (for comparison)
            oracle_source = q["expected_docs"][0] if q["expected_docs"] else ""
            oracle_page = q["expected_pages"][0] if q["expected_pages"] else 0
            oracle_text = load_oracle_context(DB_PATH, oracle_source, oracle_page)

            entry = {
                "id": q["id"],
                "question": q["question"],
                "retrieval_context": retrieval_text,
                "oracle_context": oracle_text,
                "retrieval_hit": is_hit,
                "retrieval_sources": retrieval_sources,
                "expected_docs": q["expected_docs"],
                "expected_pages": q["expected_pages"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "[%d/%d] hits=%d (%.1f%%) elapsed=%.1fs",
                    i + 1,
                    len(questions),
                    hits,
                    100 * hits / (i + 1),
                    elapsed,
                )

    elapsed = time.time() - t0
    recall = 100 * hits / len(questions)
    logger.info(
        "DONE: %d questions, %d hits (recall@5=%.1f%%), %.1fs",
        len(questions),
        hits,
        recall,
        elapsed,
    )
    logger.info("Saved: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
