#!/usr/bin/env python3
"""
Fix Optimal Chunks - Trouve le MEILLEUR chunk pour chaque question.

Pour chaque question:
1. Cherche TOUS les chunks du document attendu
2. Score par overlap answer_text + article_reference
3. Sélectionne le meilleur chunk (pas de seuil minimum)

ISO 42001 A.6.2.2: Provenance traçable = meilleur chunk disponible
"""

import json
import sqlite3
import re
from pathlib import Path
from collections import defaultdict

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


def get_keywords(text: str, min_len: int = 3) -> set[str]:
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
        "une",
        "est",
        "les",
        "des",
        "qui",
        "que",
        "par",
        "sur",
        "soit",
        "quel",
        "quoi",
        "donc",
        "mais",
        "comme",
        "toute",
        "cela",
        "celle",
        "celui",
        "ceux",
        "apres",
        "avant",
        "cette",
        "lors",
        "lorsque",
    }
    words = normalize(text).split()
    return {w for w in words if len(w) >= min_len and w not in stopwords}


def score_chunk(
    chunk_text: str, answer_text: str, article_ref: str, question_text: str = ""
) -> tuple[float, float, float]:
    """
    Score a chunk for relevance.

    Returns (total_score, answer_overlap, article_match).
    """
    chunk_norm = normalize(chunk_text)
    answer_overlap = 0.0
    article_score = 0.0
    question_overlap = 0.0

    # Answer keywords (most important for RAG)
    if answer_text and len(answer_text) > 5:
        answer_kw = get_keywords(answer_text)
        if answer_kw:
            matches = sum(1 for w in answer_kw if w in chunk_norm)
            answer_overlap = matches / len(answer_kw)

    # Article number match
    art_match = re.search(r"(\d+\.?\d*\.?\d*)", article_ref)
    if art_match:
        art_num = art_match.group(1)
        if art_num in chunk_norm:
            article_score = 1.0

    # Question keywords (secondary)
    if question_text and len(question_text) > 10:
        q_kw = get_keywords(question_text)
        if q_kw:
            matches = sum(1 for w in q_kw if w in chunk_norm)
            question_overlap = matches / len(q_kw)

    # Weight: answer(60%) + article(25%) + question(15%)
    total_score = (answer_overlap * 60) + (article_score * 25) + (question_overlap * 15)

    return total_score, answer_overlap, article_score


def get_doc_prefix(doc_name: str) -> str:
    """Extract document prefix for matching."""
    prefix = doc_name.replace(".pdf", "").split("_")[0]
    if "-" in prefix:
        prefix = prefix.split("-")[0]
    return prefix.lower()


def main():
    print("=" * 60)
    print("  FIX OPTIMAL CHUNKS")
    print("  Meilleur chunk pour chaque question")
    print("=" * 60)

    # Load data
    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))

    # Load all chunks by source prefix
    cur = conn.execute(
        "SELECT id, text, source, page FROM chunks WHERE id NOT LIKE '%table%'"
    )
    chunks_by_prefix: dict[str, list[dict]] = defaultdict(list)
    for row in cur:
        prefix = get_doc_prefix(row[2])
        chunks_by_prefix[prefix].append(
            {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "page": row[3],
            }
        )

    print(
        f"Loaded chunks by prefix: {dict((k, len(v)) for k, v in chunks_by_prefix.items())}"
    )
    print(f"Processing {len(gs['questions'])} questions...")

    # Stats
    updated = 0
    improved = 0
    high_score = 0
    low_score = 0
    no_chunks = 0

    for i, q in enumerate(gs["questions"]):
        doc = q.get("expected_docs", [""])[0]
        doc_prefix = get_doc_prefix(doc)

        answer_text = q.get("answer_text", "")
        article_ref = q.get("article_reference", "")
        question_text = q.get("question_text", "")
        current_chunk = q.get("expected_chunk_id", "")

        # Get chunks for this document
        doc_chunks = chunks_by_prefix.get(doc_prefix, [])
        if not doc_chunks:
            no_chunks += 1
            continue

        # Score all chunks
        best_chunk = None
        best_page = 0
        best_score = -1
        best_answer_overlap = 0

        for chunk in doc_chunks:
            score, ans_overlap, art_match = score_chunk(
                chunk["text"], answer_text, article_ref, question_text
            )
            if score > best_score:
                best_score = score
                best_chunk = chunk["id"]
                best_page = chunk["page"]
                best_answer_overlap = ans_overlap

        if best_chunk:
            if best_chunk != current_chunk:
                updated += 1
                # Check if it's actually better
                if current_chunk:
                    for chunk in doc_chunks:
                        if chunk["id"] == current_chunk:
                            old_score, _, _ = score_chunk(
                                chunk["text"], answer_text, article_ref, question_text
                            )
                            if best_score > old_score:
                                improved += 1
                            break
                else:
                    improved += 1

            q["expected_chunk_id"] = best_chunk
            q["expected_pages"] = [best_page]

            if best_answer_overlap >= 0.3:
                high_score += 1
            else:
                low_score += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(gs['questions'])}...")

    conn.close()

    # Save
    with open(GS_PATH, "w", encoding="utf-8") as f:
        json.dump(gs, f, ensure_ascii=False, indent=2)

    print()
    print("Résultats:")
    print(f"  Chunks modifiés:    {updated}")
    print(f"  Améliorés:          {improved}")
    print(f"  Answer >= 30%:      {high_score}")
    print(f"  Answer < 30%:       {low_score}")
    print(f"  Pas de chunks:      {no_chunks}")

    # Final verification
    print()
    print("=" * 60)
    print("  VÉRIFICATION FINALE")
    print("=" * 60)

    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))

    answer_ok = 0
    article_ok = 0
    both_ok = 0
    neither = 0

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
        has_answer = False
        if answer_text and len(answer_text) > 5:
            answer_kw = get_keywords(answer_text)
            if answer_kw:
                matches = sum(1 for w in answer_kw if w in chunk_text)
                has_answer = (matches / len(answer_kw)) >= 0.3

        # Check article
        has_article = False
        art_match = re.search(r"(\d+\.?\d*\.?\d*)", article_ref)
        if art_match and art_match.group(1) in chunk_text:
            has_article = True

        if has_answer:
            answer_ok += 1
        if has_article:
            article_ok += 1
        if has_answer and has_article:
            both_ok += 1
        if not has_answer and not has_article:
            neither += 1

    conn.close()

    print(f"  Answer >= 30%:      {answer_ok}/477 ({100*answer_ok/477:.1f}%)")
    print(f"  Article trouvé:     {article_ok}/477 ({100*article_ok/477:.1f}%)")
    print(f"  BOTH (cible):       {both_ok}/477 ({100*both_ok/477:.1f}%)")
    print(f"  Aucun:              {neither}/477 ({100*neither/477:.1f}%)")


if __name__ == "__main__":
    main()
