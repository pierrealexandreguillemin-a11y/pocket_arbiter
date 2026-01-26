#!/usr/bin/env python3
"""
Correction des chunk_ids pour questions à réponses courtes.

Pour les réponses courtes (nombres, montants), le keyword matching
sur la réponse échoue. Ce script utilise les mots-clés de la QUESTION
combinés avec l'article_reference pour trouver le bon chunk.

Stratégie:
1. Identifier les questions avec réponses courtes (< 4 chars ou numériques)
2. Extraire les mots-clés de la question
3. Chercher chunks contenant: article_reference + mots-clés question + réponse
4. Valider que la réponse exacte est présente

ISO 42001: Correction anti-hallucination avec traçabilité
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime


def load_json(path: str) -> dict:
    """Load JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    """Save JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    replacements = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'î': 'i', 'ï': 'i', 'ô': 'o', 'ö': 'o',
        'ç': 'c', 'œ': 'oe', 'æ': 'ae',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def is_short_answer(answer: str) -> bool:
    """Check if answer is short (numeric, monetary, or < 4 chars)."""
    answer = answer.strip()
    # Numeric (with optional unit)
    if re.match(r'^\d+[\s]*(€|euros?|minutes?|heures?|jours?|points?)?\.?$', answer, re.IGNORECASE):
        return True
    # Very short text
    if len(answer) < 10:
        return True
    return False


def extract_question_keywords(question: str, min_length: int = 4) -> list[str]:
    """Extract meaningful keywords from question text."""
    text = normalize_text(question)
    stopwords = {
        'pour', 'dans', 'avec', 'cette', 'celui', 'celle', 'sont', 'etre',
        'quelle', 'quel', 'quels', 'quelles', 'comment', 'combien', 'quel',
        'propositions', 'proposition', 'suivantes', 'suivante', 'parmi',
        'correspond', 'correspond', 'lequel', 'laquelle', 'lesquels',
        'avoir', 'fait', 'faire', 'peut', 'doit', 'tous', 'tout', 'plus',
        'moins', 'entre', 'autres', 'autre', 'comme', 'ainsi', 'donc',
        'lors', 'apres', 'avant', 'depuis', 'pendant', 'selon', 'sans',
        'sous', 'vers', 'chez', 'contre', 'entre', 'parmi', 'sauf',
        'seront', 'aurons', 'nous', 'vous', 'leur', 'leurs', 'notre',
    }
    words = re.findall(r'\b[a-z]+\b', text)
    return [w for w in words if len(w) >= min_length and w not in stopwords]


def extract_article_patterns(article_ref: str) -> list[str]:
    """Extract search patterns from article_reference."""
    patterns = []

    # Extract article numbers (e.g., "3.8", "12.4")
    article_nums = re.findall(r'\b(\d+\.?\d*)\b', article_ref)
    patterns.extend(article_nums)

    # Extract "Article X" patterns
    article_match = re.findall(r'Article\s+(\d+\.?\d*)', article_ref, re.IGNORECASE)
    for m in article_match:
        patterns.append(f"Article {m}")

    # Extract section titles
    titles = re.findall(r'[A-ZÉÈÊÀÂÇ][a-zéèêàâùûîïôöç]+(?:\s+[A-ZÉÈÊÀÂÇ]?[a-zéèêàâùûîïôöç]+)*', article_ref)
    for t in titles:
        if len(t) > 5 and t not in ['Article', 'Chapitre', 'Section']:
            patterns.append(t)

    return list(set(patterns))


def find_chunk_for_short_answer(
    question: str,
    answer: str,
    article_ref: str,
    chunks: list[dict],
    source_docs: list[str] = None
) -> dict | None:
    """
    Find best chunk for question with short answer.

    Strategy:
    1. Filter by source document
    2. Must contain the answer (exact match)
    3. Score by question keywords + article patterns
    """
    answer_norm = normalize_text(answer)
    # Also check raw number for numeric answers
    answer_num = re.search(r'\d+', answer)
    answer_num = answer_num.group() if answer_num else None

    question_keywords = extract_question_keywords(question)
    article_patterns = extract_article_patterns(article_ref)

    candidates = []

    for chunk in chunks:
        # Filter by source document
        if source_docs:
            if not any(doc in chunk["id"] for doc in source_docs):
                continue

        chunk_text = chunk["text"]
        chunk_norm = normalize_text(chunk_text)

        # MUST contain the answer
        has_answer = answer_norm in chunk_norm
        if not has_answer and answer_num:
            # For numeric, check with word boundaries
            has_answer = bool(re.search(rf'\b{answer_num}\b', chunk_text))

        if not has_answer:
            continue

        # Score by article patterns
        article_score = 0
        for pattern in article_patterns:
            if pattern.lower() in chunk_norm:
                article_score += 1

        # Score by question keywords
        keyword_score = 0
        for kw in question_keywords:
            if kw in chunk_norm:
                keyword_score += 1

        if article_score > 0 or keyword_score >= 2:
            total_score = article_score * 2 + keyword_score
            candidates.append({
                "chunk_id": chunk["id"],
                "article_score": article_score,
                "keyword_score": keyword_score,
                "total_score": total_score,
                "text_preview": chunk_text[:150],
            })

    if not candidates:
        return None

    # Sort by total score
    candidates.sort(key=lambda x: x["total_score"], reverse=True)
    return candidates[0]


def fix_short_answer_chunks(gs: dict, chunks: list[dict], chunk_index: dict) -> dict:
    """
    Fix chunk_ids for questions with short answers.
    """
    results = {
        "total_short_answers": 0,
        "fixes_applied": 0,
        "already_ok": 0,
        "no_match_found": 0,
        "fixes": [],
        "remaining_issues": [],
        "reclassify_arithmetic": [],
    }

    for q in gs["questions"]:
        qid = q["id"]
        answer = q.get("expected_answer", "")
        question = q.get("question", "")

        # Only process short answers
        if not is_short_answer(answer):
            continue

        results["total_short_answers"] += 1

        current_chunk_id = q.get("expected_chunk_id", "")
        article_ref = q.get("metadata", {}).get("article_reference", "")
        expected_docs = q.get("expected_docs", [])
        reasoning_class = q.get("metadata", {}).get("reasoning_class", "")

        # Check if needs reclassification to arithmetic
        if re.match(r'^\d+\.?$', answer.strip()):
            if reasoning_class == "fact_single":
                results["reclassify_arithmetic"].append({
                    "id": qid,
                    "answer": answer,
                    "current_class": reasoning_class,
                })

        # Check current chunk
        answer_in_current = False
        if current_chunk_id in chunk_index:
            chunk_text = chunk_index[current_chunk_id]
            answer_norm = normalize_text(answer)
            chunk_norm = normalize_text(chunk_text)
            answer_in_current = answer_norm in chunk_norm
            if not answer_in_current:
                answer_num = re.search(r'\d+', answer)
                if answer_num:
                    answer_in_current = bool(re.search(rf'\b{answer_num.group()}\b', chunk_text))

        if answer_in_current:
            results["already_ok"] += 1
            continue

        # Find better chunk
        best = find_chunk_for_short_answer(
            question, answer, article_ref, chunks, expected_docs
        )

        if best:
            old_id = current_chunk_id
            new_id = best["chunk_id"]

            q["expected_chunk_id"] = new_id
            q["audit"] = (q.get("audit", "") + f" [SHORT_FIX] {old_id[:30]} -> {new_id[:30]} (art:{best['article_score']}, kw:{best['keyword_score']})").strip()

            results["fixes_applied"] += 1
            results["fixes"].append({
                "id": qid,
                "answer": answer,
                "old_chunk_id": old_id,
                "new_chunk_id": new_id,
                "article_score": best["article_score"],
                "keyword_score": best["keyword_score"],
            })
        else:
            results["no_match_found"] += 1
            results["remaining_issues"].append({
                "id": qid,
                "answer": answer,
                "article_ref": article_ref,
                "reason": "no_chunk_contains_answer",
            })

    return results


def main():
    """Main fix pipeline for short answers."""
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    base_path = Path(__file__).parent.parent.parent.parent
    gs_path = base_path / "tests" / "data" / "gold_standard_annales_fr_v7.json"
    chunks_path = base_path / "corpus" / "processed" / "chunks_mode_b_fr.json"
    output_path = base_path / "tests" / "data" / "short_answer_fix_report.json"

    print("=" * 70)
    print("FIX SHORT ANSWERS - Réponses courtes/numériques")
    print("=" * 70)
    print("""
Stratégie:
  1. Identifier réponses courtes (nombres, montants, < 10 chars)
  2. Chercher chunks contenant: réponse + article_ref + mots-clés question
  3. Reclassifier les questions arithmetic
""")

    # Load data
    print("Chargement des données...")
    gs = load_json(str(gs_path))
    chunks_data = load_json(str(chunks_path))
    chunks = chunks_data.get("chunks", chunks_data)
    chunk_index = {c["id"]: c["text"] for c in chunks}

    print(f"  GS version: {gs.get('version', 'unknown')}")
    print(f"  Questions: {len(gs['questions'])}")
    print(f"  Chunks: {len(chunks)}")

    # Run fix
    print(f"\n{'=' * 70}")
    print("EXÉCUTION")
    print("=" * 70)

    results = fix_short_answer_chunks(gs, chunks, chunk_index)

    # Display results
    print(f"\nRéponses courtes identifiées: {results['total_short_answers']}")
    print(f"Déjà OK: {results['already_ok']}")
    print(f"Corrections appliquées: {results['fixes_applied']}")
    print(f"Sans match trouvé: {results['no_match_found']}")

    if results["reclassify_arithmetic"]:
        print(f"\n{'=' * 70}")
        print(f"RECLASSIFICATION ARITHMETIC: {len(results['reclassify_arithmetic'])} questions")
        print("=" * 70)
        for item in results["reclassify_arithmetic"][:10]:
            print(f"  - {item['id']}: '{item['answer']}' ({item['current_class']} -> arithmetic)")

        # Apply reclassification
        for item in results["reclassify_arithmetic"]:
            for q in gs["questions"]:
                if q["id"] == item["id"]:
                    q["metadata"]["reasoning_class"] = "arithmetic"
                    q["audit"] = (q.get("audit", "") + " [RECLASS] fact_single -> arithmetic").strip()

    if results["fixes"]:
        print(f"\n{'=' * 70}")
        print("CORRECTIONS APPLIQUÉES")
        print("=" * 70)
        for fix in results["fixes"][:15]:
            print(f"\n{fix['id']}:")
            print(f"  Answer: {fix['answer']}")
            print(f"  Art: {fix['article_score']}, KW: {fix['keyword_score']}")

    # Update version
    old_version = gs.get("version", "7.4.3")
    parts = old_version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    new_version = ".".join(parts)
    gs["version"] = new_version

    # Add methodology
    if "methodology" not in gs:
        gs["methodology"] = {}
    gs["methodology"]["short_answer_fix"] = {
        "date": datetime.now().isoformat(),
        "method": "question_keyword_matching",
        "fixes_applied": results["fixes_applied"],
        "arithmetic_reclassified": len(results["reclassify_arithmetic"]),
        "remaining_issues": len(results["remaining_issues"]),
    }

    # Save
    save_json(gs, str(gs_path))
    print(f"\n[OK] Sauvegardé v{new_version} -> {gs_path}")

    # Save report
    report = {
        "date": datetime.now().isoformat(),
        "from_version": old_version,
        "to_version": new_version,
        **results,
    }
    save_json(report, str(output_path))
    print(f"Rapport détaillé: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
