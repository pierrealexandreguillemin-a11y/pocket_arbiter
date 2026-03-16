"""Index children and table summaries into SQLite with embeddings.

Embeds text with EmbeddingGemma-300M (768D) via sentence-transformers.
Applies Contextual Chunk Headers (CCH) before embedding for +35% recall.
Stores in SQLite for brute-force cosine retrieval.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_ID = "google/embeddinggemma-300m"
EMBEDDING_DIM = 768

SCHEMA = """
CREATE TABLE IF NOT EXISTS children (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    parent_id TEXT NOT NULL,
    source TEXT NOT NULL,
    page INTEGER,
    article_num TEXT,
    section TEXT,
    tokens INTEGER
);

CREATE TABLE IF NOT EXISTS parents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    section TEXT,
    tokens INTEGER,
    page INTEGER
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
"""

# Human-readable titles for CCH (source filename → display title)
# Updated when corpus changes (once per year)
SOURCE_TITLES: dict[str, str] = {
    "LA-octobre2025.pdf": "Lois des Echecs FFE octobre 2025",
    "R01_2025_26_Regles_generales.pdf": "Regles generales competitions FFE 2025-26",
    "R02_2025_26_Regles_generales_Annexes.pdf": "Annexes regles generales FFE 2025-26",
    "R03_2025_26_Competitions_homologuees.pdf": "Competitions homologuees FFE 2025-26",
    "A01_2025_26_Championnat_de_France.pdf": "Championnat de France individuel 2025-26",
    "A02_2025_26_Championnat_de_France_des_Clubs.pdf": "Championnat de France des Clubs 2025-26",
    "A03_2025_26_Championnat_de_France_des_Clubs_rapides.pdf": "Ch. France Clubs rapide 2025-26",
    "C01_2025_26_Coupe_de_France.pdf": "Coupe de France 2025-26",
    "C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf": "Coupe Loubatiere 2025-26",
    "C04_2025_26_Coupe_de_la_parite.pdf": "Coupe de la parite 2025-26",
    "J01_2025_26_Championnat_de_France_Jeunes.pdf": "Ch. France Jeunes 2025-26",
    "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf": "Ch. France Interclubs Jeunes 2025-26",
    "J03_2025_26_Championnat_de_France_scolaire.pdf": "Ch. France scolaire 2025-26",
    "F01_2025_26_Championnat_de_France_des_clubs_Feminin.pdf": "Ch. France Clubs Feminin 2025-26",
    "F02_2025_26_Championnat_individuel_Feminin_parties_rapides.pdf": "Ch. individuel Feminin rapide 2025-26",
    "H01_2025_26_Conduite_pour_joueur_handicapes.pdf": "Conduite joueurs handicapes FFE",
    "H02_2025_26_Joueurs_a_mobilite_reduite.pdf": "Joueurs mobilite reduite FFE",
    "E02-Le_classement_rapide.pdf": "Classement rapide FFE",
    "2024_Statuts20240420.pdf": "Statuts FFE 2024",
    "2025_Reglement_Interieur_20250503.pdf": "Reglement interieur FFE 2025",
    "2023_Reglement_Financier20230610.pdf": "Reglement financier FFE 2023",
    "2022_Reglement_medical_19082022.pdf": "Reglement medical FFE 2022",
    "2018_Reglement_Disciplinaire20180422.pdf": "Reglement disciplinaire FFE 2018",
    "Contrat_de_delegation_15032022.pdf": "Contrat delegation FFE 2022",
    "Interclubs_DepartementalBdr.pdf": "Interclubs Departemental BdR",
    "InterclubsJeunes_PACABdr.pdf": "Interclubs Jeunes PACA BdR",
}


def contextualize_text(
    text: str,
    source: str,
    section: str | None,
    source_titles: dict[str, str] | None = None,
) -> str:
    """Prepend Contextual Chunk Header (CCH) to text for embedding.

    The CCH disambiguates chunks from different documents/sections
    that may have similar content (e.g., forfait rules across competitions).

    Args:
        text: Raw chunk text.
        source: Source PDF filename.
        section: Section/heading name.
        source_titles: Optional mapping source → human-readable title.

    Returns:
        Contextualized text with header prepended.
    """
    titles = source_titles or SOURCE_TITLES
    doc_title = titles.get(source, source)
    if section:
        header = f"[Document: {doc_title} | Section: {section}]"
    else:
        header = f"[Document: {doc_title}]"
    return f"{header}\n{text}"


def create_db(db_path: Path) -> None:
    """Create SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.close()


def _to_blob(arr: np.ndarray) -> bytes:
    """Convert numpy float32 array to bytes."""
    return arr.astype(np.float32).tobytes()


def embed_texts(texts: list[str], model_id: str = EMBEDDING_MODEL_ID) -> np.ndarray:
    """Embed texts using sentence-transformers.

    Args:
        texts: List of strings to embed.
        model_id: HuggingFace model ID.

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def insert_parents(db_path: Path, parents: list[dict]) -> None:
    """Insert parent records."""
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT OR REPLACE INTO parents (id, text, source, section, tokens, page) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(p["id"], p["text"], p["source"], p.get("section"),
          p.get("tokens"), p.get("page")) for p in parents],
    )
    conn.commit()
    conn.close()


def insert_children(
    db_path: Path,
    children: list[dict],
    embeddings: dict[str, np.ndarray],
) -> None:
    """Insert children with embeddings."""
    conn = sqlite3.connect(db_path)
    rows = []
    for c in children:
        emb = embeddings[c["id"]]
        rows.append((
            c["id"], c["text"], _to_blob(emb), c["parent_id"],
            c["source"], c.get("page"),
            c.get("article_num"), c.get("section"), c.get("tokens"),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO children "
        "(id, text, embedding, parent_id, source, page, article_num, section, tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def insert_table_summaries(
    db_path: Path,
    summaries: list[dict],
    embeddings: dict[str, np.ndarray],
) -> None:
    """Insert table summaries with embeddings."""
    conn = sqlite3.connect(db_path)
    rows = []
    for s in summaries:
        emb = embeddings[s["id"]]
        rows.append((
            s["id"], s["summary_text"], s["raw_table_text"],
            _to_blob(emb), s["source"], s.get("page"), s.get("tokens"),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO table_summaries "
        "(id, summary_text, raw_table_text, embedding, source, page, tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def load_table_summaries(path: Path) -> list[dict]:
    """Load table summaries from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summaries = []
    for key, summary_text in data["summaries"].items():
        source = key.rsplit("-table", 1)[0] + ".pdf"
        summaries.append({
            "id": key,
            "summary_text": summary_text,
            "raw_table_text": summary_text,
            "source": source,
        })
    return summaries
