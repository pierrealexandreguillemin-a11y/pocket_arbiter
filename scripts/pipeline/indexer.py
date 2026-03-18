"""Indexer: orchestrate chunking, embedding, SQLite DB build, integrity gates.

Re-exports from indexer_db and indexer_embed for backward compatibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import tiktoken

# Re-exports for backward compatibility (search.py, tests, recall.py import from here)
from scripts.pipeline.indexer_db import (  # noqa: F401
    create_db,
    insert_children,
    insert_parents,
    insert_table_summaries,
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
    "A02_2025_26_Championnat_de_France_des_Clubs.pdf": "Championnat de France des Clubs",
    "A03_2025_26_Championnat_de_France_des_Clubs_rapides.pdf": "Championnat de France des Clubs Rapides",
    "C01_2025_26_Coupe_de_France.pdf": "Coupe de France",
    "C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf": "Coupe Jean-Claude Loubatiere",
    "C04_2025_26_Coupe_de_la_parité.pdf": "Coupe de la Parite",
    "Contrat_de_delegation_15032022.pdf": "Contrat de Delegation FFE",
    "E02-Le_classement_rapide.pdf": "Classement Rapide",
    "F01_2025_26_Championnat_de_France_des_clubs_Feminin.pdf": "Championnat France Clubs Feminin",
    "F02_2025_26_Championnat_individuel_Feminin_parties_rapides.pdf": "Championnat Individuel Feminin Rapide",
    "H01_2025_26_Conduite_pour_joueur_handicapes.pdf": "Conduite Joueurs Handicapes",
    "H02_2025_26_Joueurs_a_mobilite_reduite.pdf": "Joueurs Mobilite Reduite",
    "InterclubsJeunes_PACABdr.pdf": "Interclubs Jeunes PACA",
    "Interclubs_DepartementalBdr.pdf": "Interclubs Departemental BdR",
    "J01_2025_26_Championnat_de_France_Jeunes.pdf": "Championnat France Jeunes",
    "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf": "Championnat France Interclubs Jeunes",
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

    # 3. Load embedding model
    logger.info("=== Step 3: Loading embedding model ===")
    model = load_model(model_id)

    # 4. Embed children with CCH
    logger.info("=== Step 4: Embedding %d children ===", len(all_children))
    child_titles = [
        make_cch_title(c["source"], c.get("section", ""), SOURCE_TITLES)
        for c in all_children
    ]
    child_texts = [c["text"] for c in all_children]
    child_embeddings = embed_documents(child_texts, child_titles, model)
    logger.info("Children embeddings shape: %s", child_embeddings.shape)

    # 5. Embed table summaries with CCH from chunker heading hierarchy
    logger.info("=== Step 5: Embedding %d table summaries ===", len(table_sums))
    _ts_section_lookup: dict[tuple[str, str], str] = {}
    for ct in all_chunker_tables:
        key = (ct.get("source", ""), ct.get("raw_text", "")[:80])
        if ct.get("section"):
            _ts_section_lookup[key] = ct["section"]

    if table_sums:
        ts_titles = []
        for s in table_sums:
            key = (s["source"], s.get("raw_table_text", "")[:80])
            section = _ts_section_lookup.get(
                key,
                _table_section_from_summary(s["summary_text"]),
            )
            ts_titles.append(make_cch_title(s["source"], section, SOURCE_TITLES))
        ts_texts = [s["summary_text"] for s in table_sums]
        ts_embeddings = embed_documents(ts_texts, ts_titles, model)
    else:
        ts_embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    # 6. Build SQLite DB
    logger.info("=== Step 6: Building SQLite DB ===")
    conn = create_db(output_db)
    insert_parents(conn, all_parents)
    logger.info("Inserted %d parents", len(all_parents))
    insert_children(conn, all_children, child_embeddings)
    logger.info("Inserted %d children", len(all_children))
    if table_sums:
        insert_table_summaries(conn, table_sums, ts_embeddings)
        logger.info("Inserted %d table summaries", len(table_sums))

    # 7. Build FTS5 index
    logger.info("=== Step 7: Populating FTS5 index ===")
    populate_fts(conn)
    fts_c = conn.execute("SELECT COUNT(*) FROM children_fts").fetchone()[0]
    fts_t = conn.execute("SELECT COUNT(*) FROM table_summaries_fts").fetchone()[0]
    logger.info("FTS5: %d children + %d summaries indexed", fts_c, fts_t)

    # 8. Integrity gates
    logger.info("=== Step 8: Relational integrity gates ===")
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
