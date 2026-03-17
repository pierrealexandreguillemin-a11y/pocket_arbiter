"""Indexer: embed children + table summaries, store in SQLite.

Applies Contextual Chunk Headers (CCH) at build-time:
- Documents: "title: {cch_title} | text: {chunk_text}"
- Queries: "task: search result | query: {question}"

Model: EmbeddingGemma-300M QAT (768D, L2 normalized).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tiktoken

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

# === Model config ===

DEFAULT_MODEL_ID = "google/embeddinggemma-300m-qat-q4_0-unquantized"
EMBEDDING_DIM = 768
BATCH_SIZE = 128

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
    "E02-Le_classement_rapide.pdf": "Le Classement Rapide",
    "F01_2025_26_Championnat_de_France_des_clubs_Feminin.pdf": "Championnat de France des Clubs Feminin",
    "F02_2025_26_Championnat_individuel_Feminin_parties_rapides.pdf": "Championnat Individuel Feminin Rapides",
    "H01_2025_26_Conduite_pour_joueur_handicapes.pdf": "Conduite pour Joueurs Handicapes",
    "H02_2025_26_Joueurs_a_mobilite_reduite.pdf": "Joueurs a Mobilite Reduite",
    "Interclubs_DepartementalBdr.pdf": "Interclubs Departemental Bouches-du-Rhone",
    "InterclubsJeunes_PACABdr.pdf": "Interclubs Jeunes PACA Bouches-du-Rhone",
    "J01_2025_26_Championnat_de_France_Jeunes.pdf": "Championnat de France Jeunes",
    "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf": "Championnat de France Interclubs Jeunes",
    "J03_2025_26_Championnat_de_France_scolaire.pdf": "Championnat de France Scolaire",
    "LA-octobre2025.pdf": "Lois des Echecs FFE",
    "R01_2025_26_Regles_generales.pdf": "Regles Generales Competitions FFE",
    "R02_2025_26_Regles_generales_Annexes.pdf": "Annexes Regles Generales FFE",
    "R03_2025_26_Competitions_homologuees.pdf": "Competitions Homologuees FFE",
    "règlement_n4_2024_2025__1_.pdf": "Reglement Nationale 4",
    "règlement_régionale_2024_2025.pdf": "Reglement Regionale",
}

# === SQLite schema ===

SCHEMA = """
CREATE TABLE IF NOT EXISTS parents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    section TEXT,
    tokens INTEGER,
    page INTEGER
);

CREATE TABLE IF NOT EXISTS children (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    parent_id TEXT NOT NULL REFERENCES parents(id),
    source TEXT NOT NULL,
    page INTEGER,
    article_num TEXT,
    section TEXT,
    tokens INTEGER
);

CREATE TABLE IF NOT EXISTS table_summaries (
    id TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL,
    raw_table_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    source TEXT NOT NULL,
    page INTEGER,
    tokens INTEGER
);

CREATE VIRTUAL TABLE IF NOT EXISTS children_fts USING fts5(
    id UNINDEXED,
    text_stemmed,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS table_summaries_fts USING fts5(
    id UNINDEXED,
    text_stemmed,
    tokenize='unicode61 remove_diacritics 2'
);
"""


# === CCH + prompt formatting ===


def make_cch_title(
    source: str,
    section: str,
    source_titles: dict[str, str],
) -> str:
    """Build CCH title for embedding context.

    Args:
        source: PDF filename (e.g., "R01_2025_26_Regles_generales.pdf").
        section: Section heading from the chunk.
        source_titles: Mapping of PDF filenames to display titles.

    Returns:
        Display title string: "{doc_title} | {section}".
    """
    display = source_titles.get(source, source.replace(".pdf", "").replace("_", " "))
    if section:
        return f"{display} | {section}"
    return display


def _table_section_from_summary(summary_text: str, max_words: int = 8) -> str:
    """Extract a short descriptive section from a table summary.

    Takes the first ``max_words`` words of the summary text, truncated at the
    first colon if one appears within that span.

    Args:
        summary_text: Full summary text of the table.
        max_words: Maximum words to include.

    Returns:
        Short descriptor string, e.g. "Bareme frais deplacement FFE".
    """
    # Truncate at first colon if present (summaries use "Title: details" pattern)
    text = summary_text.split(":")[0] if ":" in summary_text else summary_text
    words = text.split()[:max_words]
    return " ".join(words) if words else "Table"


def format_document(text: str, cch_title: str) -> str:
    """Format document text with Google EmbeddingGemma prompt.

    Args:
        text: Raw chunk text.
        cch_title: CCH title from make_cch_title().

    Returns:
        Formatted string: "title: {cch_title} | text: {text}".
    """
    return f"title: {cch_title} | text: {text}"


def format_query(query: str) -> str:
    """Format query with Google EmbeddingGemma prompt.

    Args:
        query: User question.

    Returns:
        Formatted string: "task: search result | query: {query}".
    """
    return f"task: search result | query: {query}"


# === Embedding blob serialization ===


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Convert float32 numpy array to bytes for SQLite BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Convert SQLite BLOB back to float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


# === Embedding functions ===


def load_model(model_id: str = DEFAULT_MODEL_ID) -> SentenceTransformer:
    """Load sentence-transformers model.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        SentenceTransformer model instance.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading model %s", model_id)
    model = SentenceTransformer(model_id)
    return model


def embed_documents(
    texts: list[str],
    titles: list[str],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed documents with CCH titles using Google prompt format.

    Args:
        texts: Raw chunk texts.
        titles: CCH titles (one per text).
        model: SentenceTransformer model.
        batch_size: Batch size for encoding.

    Returns:
        np.ndarray of shape (N, 768), L2 normalized.
    """
    formatted = [
        format_document(t, title) for t, title in zip(texts, titles, strict=False)
    ]
    embeddings = model.encode(
        formatted,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_queries(
    queries: list[str],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed queries using Google prompt format.

    Args:
        queries: User questions.
        model: SentenceTransformer model.
        batch_size: Batch size for encoding.

    Returns:
        np.ndarray of shape (N, 768), L2 normalized.
    """
    formatted = [format_query(q) for q in queries]
    embeddings = model.encode(
        formatted,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


# === SQLite DB ===


def create_db(path: Path) -> sqlite3.Connection:
    """Create SQLite database with schema.

    Args:
        path: Path for the database file.

    Returns:
        Open sqlite3 connection.
    """
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def insert_parents(conn: sqlite3.Connection, parents: list[dict]) -> None:
    """Insert parent records.

    Args:
        conn: SQLite connection.
        parents: List of parent dicts with id, text, source, section, tokens, page.
    """
    conn.executemany(
        "INSERT OR REPLACE INTO parents (id, text, source, section, tokens, page) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                p["id"],
                p["text"],
                p["source"],
                p.get("section"),
                p.get("tokens"),
                p.get("page"),
            )
            for p in parents
        ],
    )
    conn.commit()


def insert_children(
    conn: sqlite3.Connection,
    children: list[dict],
    embeddings: np.ndarray,
) -> None:
    """Insert children with embedding blobs.

    Args:
        conn: SQLite connection.
        children: List of child dicts.
        embeddings: np.ndarray of shape (N, 768).
    """
    conn.executemany(
        "INSERT OR REPLACE INTO children "
        "(id, text, embedding, parent_id, source, page, article_num, section, tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                c["id"],
                c["text"],
                embedding_to_blob(embeddings[i]),
                c["parent_id"],
                c["source"],
                c.get("page"),
                c.get("article_num"),
                c.get("section"),
                c.get("tokens"),
            )
            for i, c in enumerate(children)
        ],
    )
    conn.commit()


def insert_table_summaries(
    conn: sqlite3.Connection,
    summaries: list[dict],
    embeddings: np.ndarray,
) -> None:
    """Insert table summaries with embedding blobs.

    Args:
        conn: SQLite connection.
        summaries: List of summary dicts with id, summary_text, raw_table_text, etc.
        embeddings: np.ndarray of shape (N, 768).
    """
    conn.executemany(
        "INSERT OR REPLACE INTO table_summaries "
        "(id, summary_text, raw_table_text, embedding, source, page, tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                s["id"],
                s["summary_text"],
                s["raw_table_text"],
                embedding_to_blob(embeddings[i]),
                s["source"],
                s.get("page"),
                s.get("tokens"),
            )
            for i, s in enumerate(summaries)
        ],
    )
    conn.commit()


# === FTS5 population ===


def populate_fts(conn: sqlite3.Connection) -> None:
    """Populate FTS5 tables with stemmed text from children and table_summaries.

    Must be called after insert_children and insert_table_summaries.

    Args:
        conn: SQLite connection with children and table_summaries populated.
    """
    from scripts.pipeline.synonyms import stem_text

    # Clear existing FTS data (idempotent rebuild)
    conn.execute("DELETE FROM children_fts")
    conn.execute("DELETE FROM table_summaries_fts")

    # Populate children_fts
    rows = conn.execute("SELECT id, text FROM children").fetchall()
    conn.executemany(
        "INSERT INTO children_fts (id, text_stemmed) VALUES (?, ?)",
        [(row[0], stem_text(row[1])) for row in rows],
    )

    # Populate table_summaries_fts
    rows = conn.execute("SELECT id, summary_text FROM table_summaries").fetchall()
    conn.executemany(
        "INSERT INTO table_summaries_fts (id, text_stemmed) VALUES (?, ?)",
        [(row[0], stem_text(row[1])) for row in rows],
    )

    conn.commit()


# === Table summaries loading ===


def load_table_summaries(
    summaries_path: Path,
    docling_dir: Path,
) -> list[dict]:
    """Load table summaries and cross-reference with raw tables.

    Args:
        summaries_path: Path to table_summaries_claude.json.
        docling_dir: Path to docling_v2_fr/ with extracted JSONs.

    Returns:
        List of dicts with id, summary_text, raw_table_text, source, page, tokens.
    """
    with open(summaries_path, encoding="utf-8") as f:
        data = json.load(f)
    summaries_map: dict[str, str] = data["summaries"]

    # Build raw table lookup from all extraction files
    raw_tables: dict[str, dict] = {}
    for json_path in sorted(docling_dir.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            extraction = json.load(f)
        for table in extraction.get("tables", []):
            raw_tables[table["id"]] = table

    # Cross-reference
    result: list[dict] = []
    missing = 0
    for table_id, summary_text in summaries_map.items():
        raw = raw_tables.get(table_id)
        if raw is None:
            logger.warning("No raw table found for summary %s", table_id)
            missing += 1
            continue
        tokens = len(_enc.encode(summary_text))
        result.append(
            {
                "id": table_id,
                "summary_text": summary_text,
                "raw_table_text": raw["text"],
                "source": raw["source"],
                "page": raw.get("page"),
                "tokens": tokens,
            }
        )

    if missing:
        logger.warning("Skipped %d summaries with no matching raw table", missing)
    logger.info("Loaded %d table summaries", len(result))
    return result


# === Orchestrator ===


def build_index(
    docling_dir: Path,
    table_summaries_path: Path,
    output_db: Path,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict:
    """Build the complete SQLite index from extracted data.

    Args:
        docling_dir: Directory with docling_v2_fr/*.json files.
        table_summaries_path: Path to table_summaries_claude.json.
        output_db: Output path for corpus_v2_fr.db.
        model_id: Embedding model identifier.

    Returns:
        Dict with build statistics.
    """
    from scripts.pipeline.chunker import chunk_document

    # 1. Load extracted JSONs and chunk each document
    logger.info("=== Step 1: Chunking documents ===")
    all_children: list[dict] = []
    all_parents: list[dict] = []

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
    logger.info("Loaded %d table summaries", len(table_sums))

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

    # 5. Embed table summaries with CCH (descriptive section from summary text)
    logger.info("=== Step 5: Embedding %d table summaries ===", len(table_sums))
    if table_sums:
        ts_titles = [
            make_cch_title(
                s["source"],
                _table_section_from_summary(s["summary_text"]),
                SOURCE_TITLES,
            )
            for s in table_sums
        ]
        ts_texts = [s["summary_text"] for s in table_sums]
        ts_embeddings = embed_documents(ts_texts, ts_titles, model)
    else:
        ts_embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    # 6. Build SQLite DB
    logger.info("=== Step 6: Building SQLite DB ===")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
    conn = create_db(output_db)

    # Insert parents first (FK target)
    insert_parents(conn, all_parents)
    logger.info("Inserted %d parents", len(all_parents))

    # Insert children with embeddings
    insert_children(conn, all_children, child_embeddings)
    logger.info("Inserted %d children", len(all_children))

    # Insert table summaries with embeddings
    if table_sums:
        insert_table_summaries(conn, table_sums, ts_embeddings)
        logger.info("Inserted %d table summaries", len(table_sums))

    # 7. Build FTS5 index for BM25
    logger.info("=== Step 7: Populating FTS5 index ===")
    populate_fts(conn)
    fts_children = conn.execute("SELECT COUNT(*) FROM children_fts").fetchone()[0]
    fts_summaries = conn.execute("SELECT COUNT(*) FROM table_summaries_fts").fetchone()[
        0
    ]
    logger.info("FTS5: %d children + %d summaries indexed", fts_children, fts_summaries)

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

    stats = build_index(docling_dir, summaries_path, output_path)
    print(json.dumps(stats, indent=2))
