#!/usr/bin/env python3
"""
Fix Chunk Mapping - Correction complète.

Problèmes à corriger:
1. 31 questions avec "table-summary" -> vrais chunks
2. 72% answer_text non trouvé dans chunk -> remapper
3. 12 page [3] non vérifiées -> vérifier docling
4. Incohérence doc/article (FR-ANN-UVC-109) -> corriger

ISO 42001 A.6.2.2: Provenance traçable
ISO 29119: Données test valides
"""

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Paths
GS_PATH = Path("tests/data/gold_standard_annales_fr.json")
DB_PATH = Path("corpus/processed/corpus_mode_b_fr.db")
DOCLING_DIR = Path("corpus/processed/docling_fr")


@dataclass
class ChunkMatch:
    """Result of chunk matching."""

    chunk_id: str
    score: float
    method: str
    answer_overlap: float


def load_all_chunks(conn: sqlite3.Connection) -> dict[str, dict]:
    """Load all chunks from DB, excluding table-summary."""
    cur = conn.execute(
        """SELECT id, text, source, page, metadata FROM chunks
           WHERE id NOT LIKE '%table%'"""
    )
    chunks = {}
    for row in cur.fetchall():
        chunks[row[0]] = {
            "id": row[0],
            "text": row[1],
            "source": row[2],
            "page": row[3],
            "metadata": row[4],
        }
    return chunks


def load_table_chunks(conn: sqlite3.Connection) -> dict[str, dict]:
    """Load table chunks separately for fallback."""
    cur = conn.execute(
        """SELECT id, text, source, page FROM chunks
           WHERE id LIKE '%table%'"""
    )
    chunks = {}
    for row in cur.fetchall():
        chunks[row[0]] = {
            "id": row[0],
            "text": row[1],
            "source": row[2],
            "page": row[3],
        }
    return chunks


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r"[àâä]", "a", text)
    text = re.sub(r"[éèêë]", "e", text)
    text = re.sub(r"[îï]", "i", text)
    text = re.sub(r"[ôö]", "o", text)
    text = re.sub(r"[ùûü]", "u", text)
    text = re.sub(r"[ç]", "c", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def extract_keywords(text: str, min_len: int = 4) -> set[str]:
    """Extract significant keywords from text."""
    stopwords = {
        "dans",
        "pour",
        "avec",
        "sans",
        "cette",
        "etre",
        "avoir",
        "fait",
        "faire",
        "plus",
        "moins",
        "tout",
        "tous",
        "toute",
        "peut",
        "doit",
        "sont",
        "ete",
        "elle",
        "nous",
        "vous",
        "leur",
        "leurs",
        "meme",
        "autre",
        "entre",
        "sous",
        "aussi",
        "ainsi",
        "donc",
        "alors",
        "comme",
        "mais",
        "avant",
        "apres",
        "article",
        "regles",
        "regle",
        "joueur",
        "joueurs",
        "joueuse",
        "partie",
        "parties",
        "echecs",
        "echec",
        "coup",
        "coups",
    }
    text = normalize_text(text)
    words = text.split()
    return {w for w in words if len(w) >= min_len and w not in stopwords}


def compute_overlap(answer: str, chunk_text: str) -> float:
    """Compute keyword overlap between answer and chunk."""
    answer_keywords = extract_keywords(answer)
    if not answer_keywords:
        return 0.0

    chunk_normalized = normalize_text(chunk_text)
    matches = sum(1 for kw in answer_keywords if kw in chunk_normalized)
    return matches / len(answer_keywords)


def extract_article_numbers(text: str) -> list[str]:
    """Extract article numbers from text."""
    patterns = [
        r"article\s*([a-d]?\d+(?:\.\d+)*)",
        r"(\d+\.\d+(?:\.\d+)*)",
        r"annexe\s*([a-d])",
    ]
    found = []
    for p in patterns:
        found.extend(re.findall(p, text.lower()))
    return list(set(found))


def find_best_chunk(
    question: dict,
    chunks: dict[str, dict],
    source_filter: str,
) -> ChunkMatch | None:
    """
    Find the best chunk for a question.

    Priority:
    1. Chunk contains answer_text keywords (>30% overlap)
    2. Chunk contains article_reference
    3. Chunk on same page as expected
    """
    answer = question.get("answer_text", "")
    article_ref = question.get("article_reference", "")
    expected_pages = question.get("expected_pages", [])

    article_nums = extract_article_numbers(article_ref)

    candidates = []

    for chunk_id, chunk in chunks.items():
        # Filter by source
        if source_filter and source_filter not in chunk["source"]:
            continue

        chunk_text = chunk["text"]
        score = 0.0
        method = "none"

        # Score 1: Answer text overlap (most important)
        if answer and len(answer) > 10:
            overlap = compute_overlap(answer, chunk_text)
            if overlap >= 0.3:
                score += overlap * 100  # High weight
                method = "answer_overlap"

        # Score 2: Article number match
        chunk_text_lower = chunk_text.lower()
        for art_num in article_nums:
            if art_num in chunk_text_lower:
                score += 20
                if method == "none":
                    method = "article_match"
                break

        # Score 3: Page match
        if chunk["page"] in expected_pages:
            score += 10
            if method == "none":
                method = "page_match"

        if score > 0:
            candidates.append(
                ChunkMatch(
                    chunk_id=chunk_id,
                    score=score,
                    method=method,
                    answer_overlap=compute_overlap(answer, chunk_text) if answer else 0,
                )
            )

    if not candidates:
        return None

    # Sort by score descending
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[0]


def load_docling_pages(doc_name: str) -> dict[int, str]:
    """Load docling JSON and build page -> text mapping."""
    # Map document names
    mappings = {
        "A02": "A02_2025_26_Championnat_de_France_des_Clubs",
        "R01": "R01_2025_26_Regles_generales",
        "R02": "R02_2025_26_Regles_generales_Annexes",
        "R03": "R03_2025_26_Competitions_homologuees",
        "C01": "C01_2025_26_Coupe_de_France",
        "C03": "C03_2025_26_Coupe_Jean_Claude_Loubatiere",
        "C04": "C04_2025_26_Coupe_de_la_parité",
        "J01": "J01_2025_26_Championnat_de_France_Jeunes",
        "J02": "J02_2025_26_Championnat_de_France_Interclubs_Jeunes",
        "LA": "LA-octobre2025",
    }

    # Extract prefix
    prefix = doc_name.replace(".pdf", "").split("_")[0]
    if "-" in prefix:
        prefix = prefix.split("-")[0]

    docling_name = mappings.get(prefix, doc_name.replace(".pdf", ""))
    docling_path = DOCLING_DIR / f"{docling_name}.json"

    if not docling_path.exists():
        return {}

    with open(docling_path, encoding="utf-8") as f:
        data = json.load(f)

    page_texts: dict[int, list[str]] = defaultdict(list)
    doc = data.get("docling_document", data)
    texts = doc.get("texts", [])

    for text_obj in texts:
        prov = text_obj.get("prov", [])
        if prov:
            page_no = prov[0].get("page_no", 0)
            text = text_obj.get("text", "")
            if page_no > 0 and text:
                page_texts[page_no].append(text)

    # Join texts per page
    return {p: " ".join(texts) for p, texts in page_texts.items()}


def verify_page_in_docling(
    article_ref: str,
    answer_text: str,
    page_texts: dict[int, str],
    current_page: int,
) -> int | None:
    """
    Verify the correct page in docling for an article.

    Returns corrected page if different, None if current is OK.
    """
    if not page_texts:
        return None

    article_nums = extract_article_numbers(article_ref)
    answer_keywords = extract_keywords(answer_text)

    best_page = None
    best_score = 0

    for page_no, text in page_texts.items():
        text_lower = text.lower()
        score = 0

        # Article number match
        for art_num in article_nums:
            if art_num in text_lower:
                score += 10

        # Answer keyword match
        for kw in answer_keywords:
            if kw in normalize_text(text):
                score += 1

        if score > best_score:
            best_score = score
            best_page = page_no

    if best_page and best_page != current_page and best_score > 5:
        return best_page

    return None


def fix_all_questions():
    """Fix all chunk mappings."""
    # Load data
    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))
    chunks = load_all_chunks(conn)
    table_chunks = load_table_chunks(conn)

    print(f"Loaded {len(chunks)} chunks (excluding tables)")
    print(f"Loaded {len(table_chunks)} table chunks")
    print(f"Processing {len(gs['questions'])} questions...")

    # Stats
    stats = {
        "total": len(gs["questions"]),
        "fixed_chunk": 0,
        "fixed_page": 0,
        "fixed_doc": 0,
        "answer_in_chunk": 0,
        "no_chunk_found": 0,
        "table_replaced": 0,
    }

    # Cache docling per document
    docling_cache: dict[str, dict] = {}

    # Document corrections
    doc_corrections = {
        "FR-ANN-UVC-109": {
            "expected_docs": [
                "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf"
            ],
        },
        "FR-ANN-UVC-128": {
            "expected_docs": [
                "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf"
            ],
        },
    }

    for i, q in enumerate(gs["questions"]):
        qid = q["id"]

        # Fix document inconsistencies first
        if qid in doc_corrections:
            for key, value in doc_corrections[qid].items():
                q[key] = value
            stats["fixed_doc"] += 1

        source = q.get("expected_docs", [""])[0]
        source_prefix = source.replace(".pdf", "").split("_")[0]

        # Check if current chunk is table-summary
        current_chunk = q.get("expected_chunk_id", "")
        if "table" in current_chunk.lower():
            stats["table_replaced"] += 1

        # Find best chunk
        match = find_best_chunk(q, chunks, source_prefix)

        if match:
            if match.chunk_id != current_chunk:
                q["expected_chunk_id"] = match.chunk_id
                stats["fixed_chunk"] += 1

            if match.answer_overlap >= 0.3:
                stats["answer_in_chunk"] += 1

            # Update page from chunk
            chunk_data = chunks[match.chunk_id]
            new_page = chunk_data["page"]
            if q.get("expected_pages") != [new_page]:
                q["expected_pages"] = [new_page]
                stats["fixed_page"] += 1
        else:
            # Fallback: try to find any chunk on expected page
            expected_pages = q.get("expected_pages", [])
            if expected_pages:
                for chunk_id, chunk in chunks.items():
                    if (
                        source_prefix in chunk["source"]
                        and chunk["page"] == expected_pages[0]
                    ):
                        q["expected_chunk_id"] = chunk_id
                        break
                else:
                    stats["no_chunk_found"] += 1
            else:
                stats["no_chunk_found"] += 1

        # Verify page against docling for page [3] cases
        if q.get("expected_pages") == [3]:
            if source not in docling_cache:
                docling_cache[source] = load_docling_pages(source)

            page_texts = docling_cache[source]
            corrected = verify_page_in_docling(
                q.get("article_reference", ""),
                q.get("answer_text", ""),
                page_texts,
                3,
            )

            if corrected:
                q["expected_pages"] = [corrected]
                # Also update chunk
                for chunk_id, chunk in chunks.items():
                    if source_prefix in chunk["source"] and chunk["page"] == corrected:
                        q["expected_chunk_id"] = chunk_id
                        stats["fixed_page"] += 1
                        break

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(gs['questions'])}...")

    conn.close()

    # Save
    with open(GS_PATH, "w", encoding="utf-8") as f:
        json.dump(gs, f, ensure_ascii=False, indent=2)

    return stats


def validate_final():
    """Final validation."""
    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))

    stats = {
        "total": len(gs["questions"]),
        "with_chunk": 0,
        "chunk_exists": 0,
        "chunk_is_table": 0,
        "answer_in_chunk": 0,
        "page_3_count": 0,
        "single_page": 0,
    }

    for q in gs["questions"]:
        chunk_id = q.get("expected_chunk_id", "")

        if chunk_id:
            stats["with_chunk"] += 1

            if "table" in chunk_id.lower():
                stats["chunk_is_table"] += 1

            cur = conn.execute("SELECT text FROM chunks WHERE id = ?", (chunk_id,))
            row = cur.fetchone()
            if row:
                stats["chunk_exists"] += 1

                # Check answer overlap
                answer = q.get("answer_text", "")
                if answer and len(answer) > 10:
                    overlap = compute_overlap(answer, row[0])
                    if overlap >= 0.3:
                        stats["answer_in_chunk"] += 1

        pages = q.get("expected_pages", [])
        if len(pages) == 1:
            stats["single_page"] += 1
        if pages == [3]:
            stats["page_3_count"] += 1

    conn.close()
    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("  FIX CHUNK MAPPING")
    print("  Correction complète ISO 42001/29119")
    print("=" * 60)
    print()

    stats = fix_all_questions()

    print()
    print("Corrections effectuées:")
    print(f"  Chunks remappés:        {stats['fixed_chunk']}")
    print(f"  Tables remplacées:      {stats['table_replaced']}")
    print(f"  Pages corrigées:        {stats['fixed_page']}")
    print(f"  Documents corrigés:     {stats['fixed_doc']}")
    print(f"  Answer dans chunk:      {stats['answer_in_chunk']}")
    print(f"  Chunks non trouvés:     {stats['no_chunk_found']}")

    print()
    print("Validation finale...")
    val = validate_final()

    print()
    print("=" * 60)
    print("  VALIDATION FINALE")
    print("=" * 60)
    total = val["total"]
    print(f"  Chunks présents:        {val['with_chunk']}/{total}")
    print(f"  Chunks existent en DB:  {val['chunk_exists']}/{total}")
    print(f"  Chunks table (erreur):  {val['chunk_is_table']}")
    print(f"  Answer dans chunk:      {val['answer_in_chunk']}/{total}")
    print(f"  Page unique:            {val['single_page']}/{total}")
    print(f"  Page [3] restantes:     {val['page_3_count']}")
    print("=" * 60)
