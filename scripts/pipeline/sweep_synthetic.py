"""Sweep synthetic query channel weight + measure inter-channel redundancy.

Usage: python -m scripts.pipeline.sweep_synthetic
Requires: corpus_v2_fr.db with synthetic_queries table populated.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from scripts.pipeline.search import (
    _has_table_triggers,
    adaptive_k,
    bm25_search,
    cosine_search,
    expand_query,
    format_query,
    load_model,
    reciprocal_rank_fusion,
    structured_cell_search,
    synthetic_query_cosine_search,
    table_row_cosine_search,
)

DB_PATH = "corpus/processed/corpus_v2_fr.db"
GS_PATH = "tests/data/gold_standard_annales_fr_v8_adversarial.json"
TABLE_PAGES = {2, 5, 150, 184, 186, 189, 196, 197}


def load_testable() -> list[dict]:
    """Load testable GS questions (non-adversarial, with pages)."""
    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)
    return [
        q
        for q in gs["questions"]
        if q.get("classification", {}).get("question_type") != "adversarial"
        and q.get("provenance", {}).get("pages")
    ]


def measure_redundancy(testable: list[dict], model: SentenceTransformer) -> None:
    """Measure overlap between cosine prose (ch1) and synthetic (ch5)."""
    import sqlite3

    conn = sqlite3.connect(DB_PATH)

    overlap_counts = []
    unique_syn_counts = []
    unique_cos_counts = []

    for q in testable:
        question = q["content"]["question"]
        q_emb = model.encode([format_query(question)], normalize_embeddings=True)[
            0
        ].astype(np.float32)

        cos_top10 = {
            did for did, _ in cosine_search(conn, q_emb, max_k=10, db_path=DB_PATH)
        }
        syn_top10 = {
            did
            for did, _ in synthetic_query_cosine_search(
                conn, q_emb, max_k=10, db_path=DB_PATH
            )
        }

        overlap = cos_top10 & syn_top10
        unique_syn = syn_top10 - cos_top10
        unique_cos = cos_top10 - syn_top10

        overlap_counts.append(len(overlap))
        unique_syn_counts.append(len(unique_syn))
        unique_cos_counts.append(len(unique_cos))

    conn.close()

    n = len(testable)
    avg_overlap = sum(overlap_counts) / n
    avg_unique_syn = sum(unique_syn_counts) / n
    avg_unique_cos = sum(unique_cos_counts) / n

    print("=== Inter-Channel Redundancy (cosine vs synthetic, top-10) ===")
    print(
        f"  Avg overlap:       {avg_overlap:.1f} / 10  ({avg_overlap/10:.0%} redundant)"
    )
    print(f"  Avg unique synth:  {avg_unique_syn:.1f} / 10  (new discoveries)")
    print(f"  Avg unique cosine: {avg_unique_cos:.1f} / 10")
    print(f"  Questions with 0 overlap: {sum(1 for o in overlap_counts if o == 0)}/{n}")
    print(
        f"  Questions with 100% overlap: {sum(1 for o in overlap_counts if o >= 10)}/{n}"
    )


def sweep_weights(testable: list[dict], model: SentenceTransformer) -> None:
    """Sweep synthetic_weight and measure recall."""
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    weights = [0.0, 0.2, 0.3, 0.5, 0.8, 1.0]

    print("\n=== Weight Sweep (synthetic channel) ===")
    print(
        f"{'Weight':<8} {'Global':>8} {'Tab':>8} {'Prose':>8} {'Tab hits':>10} {'Prose hits':>12}"
    )

    for w in weights:
        hits_all = hits_tab = hits_prose = 0
        n_tab = n_prose = 0

        for q in testable:
            question = q["content"]["question"]
            expected = set(q["provenance"]["pages"])
            is_tabular = bool(expected & TABLE_PAGES)
            if is_tabular:
                n_tab += 1
            else:
                n_prose += 1

            stemmed = expand_query(question)
            q_emb = model.encode([format_query(question)], normalize_embeddings=True)[
                0
            ].astype(np.float32)

            cos_r = cosine_search(conn, q_emb, max_k=20, db_path=DB_PATH)
            bm25_r = bm25_search(conn, stemmed, max_k=20)
            trow_r = table_row_cosine_search(conn, q_emb, max_k=10, db_path=DB_PATH)
            struct_r = (
                structured_cell_search(conn, question, max_k=10)
                if _has_table_triggers(question)
                else []
            )
            syn_r = (
                synthetic_query_cosine_search(conn, q_emb, max_k=10, db_path=DB_PATH)
                if w > 0
                else []
            )

            merged = reciprocal_rank_fusion(
                cos_r, bm25_r, struct_r, trow_r, syn_r, synthetic_weight=w
            )
            final = adaptive_k(merged, min_score=0.005, max_k=10)

            result_pages = set()
            for did, _ in final:
                _PAGE_QUERIES = {
                    "children": "SELECT page FROM children WHERE id = ?",
                    "table_summaries": "SELECT page FROM table_summaries WHERE id = ?",
                    "table_rows": "SELECT page FROM table_rows WHERE id = ?",
                }
                for sql in _PAGE_QUERIES.values():
                    row = conn.execute(sql, (did,)).fetchone()
                    if row and row[0]:
                        result_pages.add(row[0])
                        break

            hit = bool(result_pages & expected)
            hits_all += hit
            if is_tabular:
                hits_tab += hit
            else:
                hits_prose += hit

        n_all = len(testable)
        print(
            f"w={w:<5.1f}  {hits_all/n_all*100:>6.1f}%  "
            f"{hits_tab/n_tab*100:>6.1f}%  {hits_prose/n_prose*100:>6.1f}%  "
            f"{hits_tab:>4}/{n_tab:<4}  {hits_prose:>5}/{n_prose}"
        )

    conn.close()


def main() -> None:
    """Run redundancy analysis + weight sweep."""
    testable = load_testable()
    print(f"Loaded {len(testable)} testable questions")

    model = load_model()

    t0 = time.time()
    measure_redundancy(testable, model)
    print(f"\nRedundancy analysis: {time.time() - t0:.1f}s")

    t0 = time.time()
    sweep_weights(testable, model)
    print(f"\nWeight sweep: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
