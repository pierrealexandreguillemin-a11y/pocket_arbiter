#!/usr/bin/env python3
"""
Fix REAL Mapping - Trouve le chunk contenant la RÉPONSE.

Critère: Le chunk DOIT contenir les mots-clés de answer_text.
C'est la seule façon qu'un RAG puisse retrouver la réponse.

ISO 42001 A.6.2.2: Provenance traçable = chunk contient réponse
"""

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

GS_PATH = Path("tests/data/gold_standard_annales_fr.json")
DB_PATH = Path("corpus/processed/corpus_mode_b_fr.db")


def normalize(text: str) -> str:
    """Normalize text for matching."""
    text = text.lower()
    text = re.sub(r"[àâä]", "a", text)
    text = re.sub(r"[éèêë]", "e", text)
    text = re.sub(r"[îï]", "i", text)
    text = re.sub(r"[ôö]", "o", text)
    text = re.sub(r"[ùûü]", "u", text)
    text = re.sub(r"[ç]", "c", text)
    return text


def get_keywords(text: str) -> set[str]:
    """Extract significant keywords."""
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
        "peut",
        "doit",
        "sont",
        "elle",
        "nous",
        "vous",
        "leur",
        "meme",
        "autre",
        "entre",
        "aussi",
        "article",
        "regles",
        "regle",
        "joueur",
        "joueurs",
        "partie",
        "echecs",
        "coup",
        "coups",
        "blanc",
        "blancs",
        "noir",
        "noirs",
    }
    words = normalize(text).split()
    return {w for w in words if len(w) > 3 and w not in stopwords}


def score_chunk(chunk_text: str, answer_text: str, article_ref: str) -> float:
    """Score how well a chunk matches answer + article."""
    chunk_norm = normalize(chunk_text)
    score = 0.0

    # Answer keywords (most important)
    if answer_text and len(answer_text) > 5:
        answer_kw = get_keywords(answer_text)
        if answer_kw:
            matches = sum(1 for w in answer_kw if w in chunk_norm)
            score += (matches / len(answer_kw)) * 100

    # Article number
    art_match = re.search(r"(\d+\.\d+(?:\.\d+)?)", article_ref)
    if art_match:
        art_num = art_match.group(1)
        if art_num in chunk_norm:
            score += 20

    return score


def find_best_chunk(
    question: dict,
    chunks_by_source: dict[str, list[dict]],
) -> tuple[str, int, float]:
    """
    Find the chunk that best contains the answer.

    Returns (chunk_id, page, score).
    """
    source = question.get("expected_docs", [""])[0]
    answer_text = question.get("answer_text", "")
    article_ref = question.get("article_reference", "")

    # Get source prefix for matching
    source_prefix = source.replace(".pdf", "").split("_")[0]
    if "-" in source_prefix:
        source_prefix = source_prefix.split("-")[0]

    best_chunk = None
    best_page = 0
    best_score = 0.0

    for src, chunks in chunks_by_source.items():
        if source_prefix.lower() not in src.lower():
            continue

        for chunk in chunks:
            score = score_chunk(chunk["text"], answer_text, article_ref)
            if score > best_score:
                best_score = score
                best_chunk = chunk["id"]
                best_page = chunk["page"]

    return best_chunk, best_page, best_score


def main():
    print("=" * 60)
    print("  FIX REAL MAPPING")
    print("  Trouve chunks contenant LA RÉPONSE")
    print("=" * 60)

    # Load data
    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))

    # Load all chunks by source
    cur = conn.execute(
        "SELECT id, text, source, page FROM chunks WHERE id NOT LIKE '%table%'"
    )
    chunks_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in cur:
        chunks_by_source[row[2]].append(
            {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "page": row[3],
            }
        )

    print(f"Loaded {sum(len(c) for c in chunks_by_source.values())} chunks")
    print(f"Processing {len(gs['questions'])} questions...")

    # Stats
    fixed = 0
    high_score = 0
    low_score = 0
    no_match = 0

    for i, q in enumerate(gs["questions"]):
        chunk_id, page, score = find_best_chunk(q, chunks_by_source)

        if chunk_id:
            old_chunk = q.get("expected_chunk_id", "")
            if chunk_id != old_chunk:
                fixed += 1

            q["expected_chunk_id"] = chunk_id
            q["expected_pages"] = [page]

            if score >= 30:
                high_score += 1
            else:
                low_score += 1
        else:
            no_match += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(gs['questions'])}...")

    conn.close()

    # Save
    with open(GS_PATH, "w", encoding="utf-8") as f:
        json.dump(gs, f, ensure_ascii=False, indent=2)

    print()
    print("Résultats:")
    print(f"  Chunks modifiés:    {fixed}")
    print(f"  Score >= 30:        {high_score}")
    print(f"  Score < 30:         {low_score}")
    print(f"  Pas de match:       {no_match}")

    # Verify
    print()
    print("Vérification finale...")

    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))

    both_ok = 0
    for q in gs["questions"]:
        chunk_id = q.get("expected_chunk_id", "")
        answer_text = q.get("answer_text", "")
        article_ref = q.get("article_reference", "")

        cur = conn.execute("SELECT text FROM chunks WHERE id=?", (chunk_id,))
        row = cur.fetchone()
        if not row:
            continue

        chunk_text = normalize(row[0])

        # Check answer
        answer_ok = False
        if answer_text and len(answer_text) > 5:
            answer_kw = get_keywords(answer_text)
            if answer_kw:
                matches = sum(1 for w in answer_kw if w in chunk_text)
                answer_ok = (matches / len(answer_kw)) >= 0.3

        # Check article
        art_ok = False
        art_match = re.search(r"(\d+\.\d+(?:\.\d+)?)", article_ref)
        if art_match and art_match.group(1) in chunk_text:
            art_ok = True

        if answer_ok and art_ok:
            both_ok += 1

    conn.close()

    print(f"  Article + Réponse OK: {both_ok}/477 ({100 * both_ok / 477:.1f}%)")


if __name__ == "__main__":
    main()
