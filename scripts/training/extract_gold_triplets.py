"""
Extraction triplets depuis Gold Standard - Pocket Arbiter

Convertit le Gold Standard (questions avec expected_pages) en triplets
pour fine-tuning, en utilisant le retrieval existant pour hard negatives.

Supporte Mode B (JSON chunks) et legacy Mode A (SQLite).

ISO Reference: ISO/IEC 42001 A.6.2.2, ISO/IEC 25010 S4.2

Usage Mode B (recommandé pour QLoRA):
    python -m scripts.training.extract_gold_triplets \
        --gs-fr tests/data/gold_standard_fr.json \
        --gs-intl tests/data/gold_standard_intl.json \
        --chunks-fr corpus/processed/chunks_mode_b_fr.json \
        --chunks-intl corpus/processed/chunks_mode_b_intl.json \
        --output data/training/gold_triplets_mode_b.jsonl

Usage legacy (Mode A SQLite):
    python -m scripts.training.extract_gold_triplets \
        --gs tests/data/gold_standard_fr.json \
        --db corpus/processed/corpus_fr.db \
        --output data/training/gold_triplets.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from scripts.pipeline.utils import get_timestamp, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MIN_SCORE = 0.3
DEFAULT_MAX_SCORE = 0.85  # Slightly lower than synthetic to avoid near-duplicates


def load_gold_standard(gs_path: Path) -> list[dict]:
    """Charge le Gold Standard."""
    with open(gs_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def load_chunks_mode_b(chunks_path: Path) -> dict[int, list[dict]]:
    """Charge chunks Mode B depuis JSON, indexés par page.

    Args:
        chunks_path: Path to chunks_mode_b_*.json file

    Returns:
        Dict mapping page number to list of chunks on that page
    """
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)

    # Index chunks by page for fast lookup
    page_index: dict[int, list[dict]] = {}
    for chunk in data["chunks"]:
        page = chunk["page"]
        if page not in page_index:
            page_index[page] = []
        page_index[page].append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"],
                "section": chunk.get("section"),
                "tokens": chunk.get("tokens", 0),
            }
        )

    logger.info(f"Loaded {len(data['chunks'])} chunks from {chunks_path.name}")
    return page_index


def get_chunks_for_pages_mode_b(
    page_index: dict[int, list[dict]],
    pages: list[int],
    source: str | None = None,
) -> list[dict]:
    """Récupère les chunks Mode B pour les pages données.

    Args:
        page_index: Dict mapping page number to chunks
        pages: List of page numbers to retrieve
        source: Optional source file filter

    Returns:
        List of matching chunks
    """
    chunks = []
    for page in pages:
        if page in page_index:
            for chunk in page_index[page]:
                if source is None or source in chunk["source"]:
                    chunks.append(chunk)
    return chunks


def get_chunks_for_pages_sqlite(
    db_path: Path, pages: list[int], source: str | None = None
) -> list[dict]:
    """Récupère les chunks depuis SQLite (legacy Mode A)."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    placeholders = ",".join("?" * len(pages))
    query = f"SELECT id, text, page, source FROM chunks WHERE page IN ({placeholders})"
    params = list(pages)

    if source:
        query += " AND source LIKE ?"
        params.append(f"%{source}%")

    cursor.execute(query, params)
    chunks = [
        {"id": r[0], "text": r[1], "page": r[2], "source": r[3]}
        for r in cursor.fetchall()
    ]
    conn.close()

    return chunks


def select_best_positive(chunks: list[dict], expected_pages: list[int]) -> dict | None:
    """Sélectionne le meilleur chunk positif (page la plus prioritaire)."""
    if not chunks:
        return None

    # Priorité aux pages dans l'ordre de expected_pages
    for page in expected_pages:
        for chunk in chunks:
            if chunk["page"] == page:
                return chunk

    # Fallback: premier chunk disponible
    return chunks[0]


def mine_hard_negative(
    db_path: Path,
    query_embedding,
    positive_page: int,
    min_score: float = DEFAULT_MIN_SCORE,
    max_score: float = DEFAULT_MAX_SCORE,
    top_k: int = 10,
) -> dict | None:
    """Mine un hard negative via retrieval (exclut la page positive)."""
    from scripts.pipeline.export_search import retrieve_similar

    candidates = retrieve_similar(db_path, query_embedding, top_k=top_k * 2)

    # Filtrer: exclure pages proches du positive (±2)
    filtered = [c for c in candidates if abs(c["page"] - positive_page) > 2]

    # Sélectionner dans la plage de score
    eligible = [c for c in filtered if min_score <= c.get("score", 0) <= max_score]

    if eligible:
        return max(eligible, key=lambda x: x.get("score", 0))

    # Fallback: meilleur sous max_score
    below_max = [c for c in filtered if c.get("score", 0) <= max_score]
    return max(below_max, key=lambda x: x.get("score", 0)) if below_max else None


def extract_gold_triplets_mode_b(
    questions: list[dict],
    page_index: dict[int, list[dict]],
    all_chunks: list[dict],
    corpus: str,
) -> list[dict]:
    """Extrait triplets depuis Gold Standard en mode B (sans embeddings).

    Pour Mode B, on utilise random sampling pour hard negatives car:
    1. Les chunks Mode B sont plus longs (685 chars avg)
    2. On évite le coût des embeddings pour la génération initiale
    3. Le fine-tuning QLoRA fera l'alignement sémantique

    Args:
        questions: Liste des questions GS (answerable uniquement)
        page_index: Dict page -> chunks
        all_chunks: Liste de tous les chunks pour sampling
        corpus: "fr" ou "intl"

    Returns:
        Liste de triplets {anchor, positive, negative, ...}
    """
    import random

    triplets = []
    skipped = {"no_pages": 0, "no_chunk": 0, "unanswerable": 0}

    for q in tqdm(questions, desc=f"Extracting {corpus.upper()} triplets"):
        # Skip unanswerable questions
        hard_type = q.get("metadata", {}).get("hard_type", "ANSWERABLE")
        if hard_type != "ANSWERABLE":
            skipped["unanswerable"] += 1
            continue

        expected_pages = q.get("expected_pages", [])
        expected_docs = q.get("expected_docs", [])
        source_hint = expected_docs[0] if expected_docs else None

        if not expected_pages:
            skipped["no_pages"] += 1
            continue

        # 1. Trouver le positive (chunk à expected_page)
        chunks = get_chunks_for_pages_mode_b(page_index, expected_pages, source_hint)
        positive = select_best_positive(chunks, expected_pages)

        if not positive:
            logger.warning(f"No chunk for {q['id']} pages {expected_pages}")
            skipped["no_chunk"] += 1
            continue

        # 2. Random hard negative (différente page, même corpus)
        negative_candidates = [
            c
            for c in all_chunks
            if abs(c["page"] - positive["page"]) > 2 and c["id"] != positive["id"]
        ]

        if not negative_candidates:
            logger.warning(f"No negative candidates for {q['id']}")
            skipped["no_chunk"] += 1
            continue

        negative = random.choice(negative_candidates)

        # 3. Créer triplet
        triplets.append(
            {
                "anchor": q["question"],
                "positive": positive["text"],
                "negative": negative["text"],
                "metadata": {
                    "source": "gold_standard",
                    "gs_id": q["id"],
                    "corpus": corpus,
                    "positive_chunk_id": positive["id"],
                    "positive_page": positive["page"],
                    "negative_chunk_id": negative["id"],
                    "negative_page": negative["page"],
                    "answer_type": q.get("metadata", {}).get("answer_type", "FACTUAL"),
                    "cognitive_level": q.get("metadata", {}).get(
                        "cognitive_level", "UNDERSTAND"
                    ),
                    "reasoning_type": q.get("metadata", {}).get(
                        "reasoning_type", "SINGLE_SENTENCE"
                    ),
                },
            }
        )

    logger.info(
        f"Extracted {len(triplets)} triplets from {corpus.upper()} "
        f"(skipped: {skipped})"
    )
    return triplets


def extract_gold_triplets_sqlite(
    gs_path: Path,
    db_path: Path,
    model,
    min_score: float = DEFAULT_MIN_SCORE,
    max_score: float = DEFAULT_MAX_SCORE,
) -> list[dict]:
    """Extrait les triplets depuis le Gold Standard (legacy SQLite mode)."""
    from scripts.pipeline.embeddings import embed_query

    questions = load_gold_standard(gs_path)
    triplets = []
    skipped = 0

    for q in tqdm(questions, desc="Extracting GS triplets"):
        expected_pages = q.get("expected_pages", [])
        expected_docs = q.get("expected_docs", [])
        source_hint = expected_docs[0] if expected_docs else None

        if not expected_pages:
            skipped += 1
            continue

        # 1. Trouver le positive (chunk à expected_page)
        chunks = get_chunks_for_pages_sqlite(db_path, expected_pages, source_hint)
        positive = select_best_positive(chunks, expected_pages)

        if not positive:
            logger.warning(f"No chunk found for {q['id']} pages {expected_pages}")
            skipped += 1
            continue

        # 2. Embed query et mine hard negative
        query_emb = embed_query(q["question"], model)
        negative = mine_hard_negative(
            db_path, query_emb, positive["page"], min_score, max_score
        )

        if not negative:
            logger.warning(f"No hard negative for {q['id']}")
            skipped += 1
            continue

        # 3. Créer triplet
        triplets.append(
            {
                "anchor": q["question"],
                "positive": positive["text"],
                "negative": negative["text"],
                "gs_id": q["id"],
                "positive_page": positive["page"],
                "negative_page": negative["page"],
                "negative_score": negative.get("score", 0),
            }
        )

    logger.info(f"Extracted {len(triplets)} triplets ({skipped} skipped)")
    return triplets


def save_triplets_jsonl(
    triplets: list[dict], output_path: Path, training_format: bool = True
) -> None:
    """Sauvegarde les triplets au format JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for t in triplets:
            if training_format:
                # Format sentence-transformers (anchor, positive, negative only)
                out = {
                    "anchor": t["anchor"],
                    "positive": t["positive"],
                    "negative": t["negative"],
                }
            else:
                out = t
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(triplets)} triplets to {output_path}")


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Extract Gold Standard triplets (Mode B or legacy SQLite)"
    )

    # Mode B arguments (recommended for QLoRA)
    parser.add_argument("--gs-fr", type=Path, help="Gold Standard FR JSON (Mode B)")
    parser.add_argument("--gs-intl", type=Path, help="Gold Standard INTL JSON (Mode B)")
    parser.add_argument("--chunks-fr", type=Path, help="Chunks Mode B FR JSON")
    parser.add_argument("--chunks-intl", type=Path, help="Chunks Mode B INTL JSON")

    # Legacy arguments (Mode A SQLite)
    parser.add_argument("--gs", "-g", type=Path, help="Gold Standard JSON (legacy)")
    parser.add_argument("--db", "-d", type=Path, help="Corpus SQLite DB (legacy)")

    # Common arguments
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--max-score", type=float, default=DEFAULT_MAX_SCORE)
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model ID (legacy)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    import random

    random.seed(args.seed)

    # Detect mode
    mode_b = args.chunks_fr or args.chunks_intl
    legacy = args.gs and args.db

    if mode_b:
        # Mode B: JSON chunks (recommended for QLoRA)
        logger.info("Mode B: Extracting from JSON chunks")
        all_triplets = []

        if args.gs_fr and args.chunks_fr:
            logger.info("Processing FR corpus...")
            with open(args.chunks_fr, encoding="utf-8") as f:
                chunks_data = json.load(f)
            all_chunks_fr = [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "page": c["page"],
                    "source": c["source"],
                }
                for c in chunks_data["chunks"]
            ]
            page_index_fr = load_chunks_mode_b(args.chunks_fr)
            questions_fr = load_gold_standard(args.gs_fr)
            triplets_fr = extract_gold_triplets_mode_b(
                questions_fr, page_index_fr, all_chunks_fr, "fr"
            )
            all_triplets.extend(triplets_fr)

        if args.gs_intl and args.chunks_intl:
            logger.info("Processing INTL corpus...")
            with open(args.chunks_intl, encoding="utf-8") as f:
                chunks_data = json.load(f)
            all_chunks_intl = [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "page": c["page"],
                    "source": c["source"],
                }
                for c in chunks_data["chunks"]
            ]
            page_index_intl = load_chunks_mode_b(args.chunks_intl)
            questions_intl = load_gold_standard(args.gs_intl)
            triplets_intl = extract_gold_triplets_mode_b(
                questions_intl, page_index_intl, all_chunks_intl, "intl"
            )
            all_triplets.extend(triplets_intl)

        triplets = all_triplets

        # Save training format (anchor, positive, negative only)
        save_triplets_jsonl(triplets, args.output, training_format=True)

        # Save full format (with metadata)
        full_output = args.output.with_stem(args.output.stem + "_full")
        save_triplets_jsonl(triplets, full_output, training_format=False)

        # Report
        fr_count = len(
            [t for t in triplets if t.get("metadata", {}).get("corpus") == "fr"]
        )
        intl_count = len(
            [t for t in triplets if t.get("metadata", {}).get("corpus") == "intl"]
        )

        report = {
            "mode": "mode_b",
            "gs_fr": str(args.gs_fr) if args.gs_fr else None,
            "gs_intl": str(args.gs_intl) if args.gs_intl else None,
            "chunks_fr": str(args.chunks_fr) if args.chunks_fr else None,
            "chunks_intl": str(args.chunks_intl) if args.chunks_intl else None,
            "triplets_fr": fr_count,
            "triplets_intl": intl_count,
            "total_triplets": len(triplets),
            "seed": args.seed,
            "timestamp": get_timestamp(),
        }

    elif legacy:
        # Legacy mode: SQLite with embeddings
        logger.info("Legacy mode: Extracting from SQLite with embeddings")

        from scripts.pipeline.embeddings import MODEL_ID, load_embedding_model

        model_id = args.model or MODEL_ID
        logger.info(f"Loading model: {model_id}")
        model = load_embedding_model(model_id)

        triplets = extract_gold_triplets_sqlite(
            args.gs, args.db, model, args.min_score, args.max_score
        )

        save_triplets_jsonl(triplets, args.output, training_format=True)
        full_output = args.output.with_stem(args.output.stem + "_full")
        save_triplets_jsonl(triplets, full_output, training_format=False)

        report = {
            "mode": "legacy_sqlite",
            "gs_file": str(args.gs),
            "db_file": str(args.db),
            "total_questions": len(load_gold_standard(args.gs)),
            "total_triplets": len(triplets),
            "conversion_rate": round(
                len(triplets) / max(len(load_gold_standard(args.gs)), 1), 4
            ),
            "min_score": args.min_score,
            "max_score": args.max_score,
            "timestamp": get_timestamp(),
        }
    else:
        parser.error(
            "Either provide Mode B args (--gs-fr/--chunks-fr) or "
            "legacy args (--gs/--db)"
        )

    save_json(report, args.output.with_suffix(".report.json"))
    logger.info(f"Done: {len(triplets)} GS triplets extracted")


if __name__ == "__main__":
    main()
