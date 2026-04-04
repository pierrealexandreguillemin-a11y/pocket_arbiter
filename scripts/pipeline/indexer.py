"""Indexer: orchestrate chunking, embedding, SQLite DB build, integrity gates.

Re-exports from indexer_db and indexer_embed for backward compatibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tiktoken

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Re-exports for backward compatibility (search.py, tests, recall.py import from here)
from scripts.pipeline.indexer_db import (  # noqa: F401
    create_db,
    insert_children,
    insert_parents,
    insert_structured_cells,
    insert_synthetic_queries,
    insert_table_rows,
    insert_table_summaries,
    insert_targeted_rows,
    populate_fts,
)
from scripts.pipeline.indexer_embed import (  # noqa: F401
    BATCH_SIZE,
    DEFAULT_MODEL_ID,
    EMBEDDING_DIM,
    blob_to_embedding,
    embed_documents,
    embed_queries,
    embedding_to_blob,
    format_document,
    format_query,
    load_model,
    make_cch_title,
)
from scripts.pipeline.integrity import run_integrity_gates

logger = logging.getLogger(__name__)
_enc = tiktoken.get_encoding("cl100k_base")

# === Source display titles (28 PDFs) ===

SOURCE_TITLES: dict[str, str] = {
    "2018_Reglement_Disciplinaire20180422.pdf": "Reglement Disciplinaire FFE",
    "2022_Reglement_medical_19082022.pdf": "Reglement Medical FFE",
    "2023_Reglement_Financier20230610.pdf": "Reglement Financier FFE",
    "2024_Statuts20240420.pdf": "Statuts FFE",
    "2025_Reglement_Interieur_20250503.pdf": "Reglement Interieur FFE",
    "A01_2025_26_Championnat_de_France.pdf": "Championnat de France Individuel",
    "A02_2025_26_Championnat_de_France_des_Clubs.pdf": (
        "Championnat de France des Clubs"
    ),
    "A03_2025_26_Championnat_de_France_des_Clubs_rapides.pdf": (
        "Championnat de France des Clubs Rapides"
    ),
    "C01_2025_26_Coupe_de_France.pdf": "Coupe de France",
    "C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf": "Coupe Jean-Claude Loubatiere",
    "C04_2025_26_Coupe_de_la_parité.pdf": "Coupe de la Parite",
    "Contrat_de_delegation_15032022.pdf": "Contrat de Delegation FFE",
    "E02-Le_classement_rapide.pdf": "Classement Rapide",
    "F01_2025_26_Championnat_de_France_des_clubs_Feminin.pdf": (
        "Championnat France Clubs Feminin"
    ),
    "F02_2025_26_Championnat_individuel_Feminin_parties_rapides.pdf": (
        "Championnat Individuel Feminin Rapide"
    ),
    "H01_2025_26_Conduite_pour_joueur_handicapes.pdf": "Conduite Joueurs Handicapes",
    "H02_2025_26_Joueurs_a_mobilite_reduite.pdf": "Joueurs Mobilite Reduite",
    "InterclubsJeunes_PACABdr.pdf": "Interclubs Jeunes PACA",
    "Interclubs_DepartementalBdr.pdf": "Interclubs Departemental BdR",
    "J01_2025_26_Championnat_de_France_Jeunes.pdf": "Championnat France Jeunes",
    "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf": (
        "Championnat France Interclubs Jeunes"
    ),
    "J03_2025_26_Championnat_de_France_scolaire.pdf": "Championnat France Scolaire",
    "LA-octobre2025.pdf": "Livre de l'Arbitre FFE",
    "R01_2025_26_Regles_generales.pdf": "Regles Generales FFE",
    "R02_2025_26_Regles_generales_Annexes.pdf": "Regles Generales Annexes",
    "R03_2025_26_Competitions_homologuees.pdf": "Competitions Homologuees",
    "règlement_n4_2024_2025__1_.pdf": "Reglement N4",
    "règlement_régionale_2024_2025.pdf": "Reglement Regional",
}


def _table_section_from_summary(summary_text: str, max_words: int = 8) -> str:
    """Extract short descriptor from summary text (fallback CCH)."""
    text = summary_text.split(":")[0] if ":" in summary_text else summary_text
    words = text.split()[:max_words]
    return " ".join(words) if words else "Table"


def load_table_summaries(
    summaries_path: Path,
    docling_dir: Path,
) -> list[dict]:
    """Load table summaries and cross-reference with raw tables."""
    with open(summaries_path, encoding="utf-8") as f:
        data = json.load(f)
    summaries_map: dict[str, str] = data["summaries"]

    raw_tables: dict[str, dict] = {}
    for json_path in sorted(docling_dir.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            extraction = json.load(f)
        for table in extraction.get("tables", []):
            raw_tables[table["id"]] = table

    result: list[dict] = []
    for table_id, summary_text in summaries_map.items():
        raw = raw_tables.get(table_id)
        if raw is None:
            logger.warning("No raw table for summary %s", table_id)
            continue
        result.append(
            {
                "id": table_id,
                "summary_text": summary_text,
                "raw_table_text": raw["text"],
                "source": raw["source"],
                "page": raw.get("page"),
                "tokens": len(_enc.encode(summary_text)),
            }
        )
    logger.info("Loaded %d table summaries", len(result))
    return result


def _embed_table_summaries(
    table_sums: list[dict],
    chunker_tables: list[dict],
    model: object,
) -> np.ndarray:
    """Embed table summaries with CCH from chunker heading hierarchy."""
    if not table_sums:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    section_lookup: dict[tuple[str, str], str] = {}
    for ct in chunker_tables:
        key = (ct.get("source", ""), ct.get("raw_text", "")[:80])
        if ct.get("section"):
            section_lookup[key] = ct["section"]

    ts_titles = []
    for s in table_sums:
        key = (s["source"], s.get("raw_table_text", "")[:80])
        section = section_lookup.get(
            key, _table_section_from_summary(s["summary_text"])
        )
        ts_titles.append(make_cch_title(s["source"], section, SOURCE_TITLES))
    ts_texts = [s["summary_text"] for s in table_sums]
    return embed_documents(ts_texts, ts_titles, model)


def _build_table_section_lookup(
    table_sums: list[dict],
    chunker_tables: list[dict],
) -> dict[str, str]:
    """Map table_summary IDs to their section from chunker heading hierarchy."""
    lookup: dict[str, str] = {s["id"]: "" for s in table_sums}
    for ct in chunker_tables:
        if not ct.get("section"):
            continue
        for s in table_sums:
            if (
                ct.get("source") == s["source"]
                and ct.get("raw_text", "")[:80] == s.get("raw_table_text", "")[:80]
            ):
                lookup[s["id"]] = ct["section"]
    return lookup


def _embed_table_rows(
    table_sums: list[dict],
    chunker_tables: list[dict],
    model: object,
) -> tuple[list[dict], np.ndarray]:
    """Parse and embed table rows as narrative prose (Phase 2 chantier 5).

    Uses narrate_table_rows() instead of parse_table_rows() to produce
    self-contained sentences that embed with richer semantics.
    """
    from scripts.pipeline.enrichment import narrate_table_rows

    row_chunks = narrate_table_rows(table_sums) if table_sums else []
    if not row_chunks:
        return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    section_lookup = _build_table_section_lookup(table_sums, chunker_tables)
    logger.info("=== Step 7: Embedding %d narrative table rows ===", len(row_chunks))
    tr_titles = [
        make_cch_title(
            r["source"], section_lookup.get(r["table_id"], ""), SOURCE_TITLES
        )
        for r in row_chunks
    ]
    tr_texts = [r["text"] for r in row_chunks]
    return row_chunks, embed_documents(tr_texts, tr_titles, model)


SYNTHETIC_QUERIES_PATH = Path("models/kaggle-gen-questions-output/questions_v5.jsonl")


def _load_synthetic_queries(
    model: SentenceTransformer,
) -> tuple[list[dict], np.ndarray]:
    """Load questions_v5.jsonl, embed as queries (Doc2Query, 5th channel).

    Each question is embedded with format_query (query prompt, not document
    prompt) because at search time we compare query↔query, not query↔doc.
    """
    if not SYNTHETIC_QUERIES_PATH.exists():
        logger.warning("No synthetic queries at %s — skipping", SYNTHETIC_QUERIES_PATH)
        return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    with open(SYNTHETIC_QUERIES_PATH, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    queries = [
        {
            "id": f"{r['chunk_id']}-sq{i % 2}",
            "question": r["question"],
            "child_id": r["chunk_id"],
            "source": r["source"],
            "page": r.get("page"),
        }
        for i, r in enumerate(raw)
    ]
    logger.info("=== Step 7b: Embedding %d synthetic queries ===", len(queries))

    # Embed as QUERIES (not documents) — format_query adds the query prompt
    texts = [format_query(q["question"]) for q in queries]
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE)
    return queries, embeddings.astype(np.float32)


def build_index(
    docling_dir: Path,
    table_summaries_path: Path,
    output_db: Path,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict:
    """Build corpus_v2_fr.db: chunk, embed, store, validate."""
    from scripts.pipeline.chunker import chunk_document

    # 1. Chunk documents
    logger.info("=== Step 1: Chunking documents ===")
    all_children: list[dict] = []
    all_parents: list[dict] = []
    all_chunker_tables: list[dict] = []

    json_files = sorted(docling_dir.glob("*.json"))
    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        result = chunk_document(
            data["markdown"],
            data["source"],
            data.get("heading_pages"),
        )
        all_children.extend(result["children"])
        all_parents.extend(result["parents"])
        all_chunker_tables.extend(result.get("tables", []))
        logger.info(
            "  %s: %d children, %d parents",
            data["source"],
            len(result["children"]),
            len(result["parents"]),
        )
    logger.info(
        "Total: %d children, %d parents from %d documents",
        len(all_children),
        len(all_parents),
        len(json_files),
    )

    # 2. Load table summaries
    logger.info("=== Step 2: Loading table summaries ===")
    table_sums = load_table_summaries(table_summaries_path, docling_dir)

    # 3. Enrich chunks (OPT 1-2-4)
    logger.info("=== Step 3: Enriching chunks ===")
    contexts_path = docling_dir.parent / "chunk_contexts.json"
    if contexts_path.exists():
        from scripts.pipeline.enrichment import (
            apply_chapter_override,
            enrich_chunks,
            enrich_table_summaries,
            load_contexts,
        )

        contexts = load_contexts(contexts_path)
        enrich_chunks(all_children, contexts)
        enrich_table_summaries(table_sums)
        logger.info(
            "Enriched %d children, %d table summaries" " (contexts + abbreviations)",
            len(all_children),
            len(table_sums),
        )
    else:
        apply_chapter_override = None  # type: ignore[assignment]
        logger.warning("No chunk_contexts.json found — skipping enrichment")

    # 4. Load embedding model
    logger.info("=== Step 4: Loading embedding model ===")
    model = load_model(model_id)

    # 5. Embed children with CCH (+ chapter overrides OPT-4)
    logger.info("=== Step 5: Embedding %d children ===", len(all_children))
    child_titles = []
    for c in all_children:
        title = make_cch_title(c["source"], c.get("section", ""), SOURCE_TITLES)
        if apply_chapter_override is not None:
            title = apply_chapter_override(c["source"], c.get("page"), title)
        child_titles.append(title)
    child_texts = [c["text"] for c in all_children]
    child_embeddings = embed_documents(child_texts, child_titles, model)
    logger.info("Children embeddings shape: %s", child_embeddings.shape)

    # 6. Embed table summaries with CCH from chunker heading hierarchy
    logger.info("=== Step 6: Embedding %d table summaries ===", len(table_sums))
    ts_embeddings = _embed_table_summaries(table_sums, all_chunker_tables, model)

    # 7. Embed table rows (row-as-chunk, level 2)
    table_row_chunks, tr_embeddings = _embed_table_rows(
        table_sums, all_chunker_tables, model
    )

    # 7b. Load and embed synthetic queries (Doc2Query, 5th channel)
    syn_queries, syn_embeddings = _load_synthetic_queries(model)

    # 8. Build SQLite DB
    logger.info("=== Step 8: Building SQLite DB ===")
    conn = create_db(output_db)
    insert_parents(conn, all_parents)
    logger.info("Inserted %d parents", len(all_parents))
    insert_children(conn, all_children, child_embeddings)
    logger.info("Inserted %d children", len(all_children))
    if table_sums:
        insert_table_summaries(conn, table_sums, ts_embeddings)
        logger.info("Inserted %d table summaries", len(table_sums))
    if table_row_chunks:
        insert_table_rows(conn, table_row_chunks, tr_embeddings)
        logger.info("Inserted %d table rows", len(table_row_chunks))
    if syn_queries:
        insert_synthetic_queries(conn, syn_queries, syn_embeddings)
        logger.info("Inserted %d synthetic queries", len(syn_queries))

    # 9. Structured cells (level 3 — deterministic lookup, no embedding)
    from scripts.pipeline.enrichment import parse_structured_cells

    struct_cells = parse_structured_cells(table_sums) if table_sums else []
    if struct_cells:
        insert_structured_cells(conn, struct_cells)
        logger.info("Inserted %d structured cells", len(struct_cells))

    # 9b. Targeted row-chunks (C.10 — 6 priority tables, ~45 rows)
    from scripts.pipeline.enrichment import format_targeted_rows

    targeted_rows = format_targeted_rows(table_sums) if table_sums else []
    if targeted_rows:
        targeted_titles = [
            make_cch_title(r["source"], "", SOURCE_TITLES) for r in targeted_rows
        ]
        targeted_texts = [r["text"] for r in targeted_rows]
        targeted_embs = embed_documents(targeted_texts, targeted_titles, model)
        insert_targeted_rows(conn, targeted_rows, targeted_embs)
        logger.info("Inserted %d targeted rows (C.10)", len(targeted_rows))

    # 10. Build FTS5 index
    logger.info("=== Step 10: Populating FTS5 index ===")
    populate_fts(conn)
    fts_c = conn.execute("SELECT COUNT(*) FROM children_fts").fetchone()[0]
    fts_t = conn.execute("SELECT COUNT(*) FROM table_summaries_fts").fetchone()[0]
    fts_r = conn.execute("SELECT COUNT(*) FROM table_rows_fts").fetchone()[0]
    logger.info(
        "FTS5: %d children + %d summaries + %d rows indexed", fts_c, fts_t, fts_r
    )

    # 11. Integrity gates
    logger.info("=== Step 11: Relational integrity gates ===")
    run_integrity_gates(conn)

    conn.close()

    stats = {
        "documents": len(json_files),
        "children": len(all_children),
        "parents": len(all_parents),
        "table_summaries": len(table_sums),
        "embedding_dim": EMBEDDING_DIM,
        "model_id": model_id,
        "output_db": str(output_db),
    }
    logger.info("=== Build complete: %s ===", stats)
    return stats


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    docling_dir = Path("corpus/processed/docling_v2_fr")
    summaries_path = Path("corpus/processed/table_summaries_claude.json")
    output_path = Path("corpus/processed/corpus_v2_fr.db")

    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])

    build_index(docling_dir, summaries_path, output_path)
