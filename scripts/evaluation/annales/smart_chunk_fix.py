#!/usr/bin/env python3
"""
Correction intelligente des chunk_ids basée sur article_reference.

Stratégie:
1. Extraire l'article_reference de chaque question
2. Chercher les chunks contenant cet article
3. Parmi ceux-ci, trouver celui avec le meilleur score de mots-clés
4. Mettre à jour le chunk_id si le nouveau est meilleur

ISO 42001: Correction anti-hallucination avec traçabilité
ISO 25010: Amélioration qualité données
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict


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


def extract_keywords(text: str, min_length: int = 4) -> list[str]:
    """Extract meaningful keywords from text."""
    text = normalize_text(text)
    stopwords = {
        'pour', 'dans', 'avec', 'cette', 'celui', 'celle', 'sont', 'etre',
        'avoir', 'fait', 'faire', 'peut', 'doit', 'tous', 'tout', 'plus',
        'moins', 'entre', 'autres', 'autre', 'comme', 'ainsi', 'donc',
    }
    words = re.findall(r'\b[a-z]+\b', text)
    return [w for w in words if len(w) >= min_length and w not in stopwords]


def keyword_score(answer: str, chunk_text: str) -> float:
    """Compute keyword matching score."""
    keywords = extract_keywords(answer)
    if not keywords:
        return 0.0
    chunk_norm = normalize_text(chunk_text)
    found = sum(1 for kw in keywords if kw in chunk_norm)
    return found / len(keywords)


def extract_article_patterns(article_ref: str) -> list[str]:
    """
    Extract search patterns from article_reference.

    Examples:
    - "Article 3.2 du RIDNA" -> ["Article 3", "3.2"]
    - "LA - Chapitre 1.3" -> ["Chapitre 1.3", "1.3"]
    - "12.4. Arbitres Fédéraux" -> ["12.4", "Arbitres Fédéraux"]
    """
    patterns = []

    # Extract article numbers (e.g., "3.2", "12.4")
    article_nums = re.findall(r'\b(\d+\.?\d*)\b', article_ref)
    patterns.extend(article_nums)

    # Extract "Article X" patterns
    article_match = re.findall(r'Article\s+(\d+\.?\d*)', article_ref, re.IGNORECASE)
    for m in article_match:
        patterns.append(f"Article {m}")

    # Extract "Chapitre X" patterns
    chap_match = re.findall(r'Chapitre\s+(\d+\.?\d*)', article_ref, re.IGNORECASE)
    for m in chap_match:
        patterns.append(f"Chapitre {m}")

    # Extract section titles (capitalized words)
    titles = re.findall(r'[A-ZÉÈÊÀÂÇ][a-zéèêàâùûîïôöç]+(?:\s+[A-ZÉÈÊÀÂÇ]?[a-zéèêàâùûîïôöç]+)*', article_ref)
    for t in titles:
        if len(t) > 5 and t not in ['Article', 'Chapitre', 'Section']:
            patterns.append(t)

    return list(set(patterns))


def find_chunks_by_article(article_ref: str, chunks: list[dict], source_docs: list[str] = None) -> list[dict]:
    """
    Find chunks matching article reference.

    Returns list of candidate chunks with scores.
    """
    if not article_ref:
        return []

    patterns = extract_article_patterns(article_ref)
    if not patterns:
        return []

    candidates = []

    for chunk in chunks:
        # Filter by source document if specified
        if source_docs:
            if not any(doc in chunk["id"] for doc in source_docs):
                continue

        text = chunk["text"]

        # Count pattern matches
        matches = 0
        for pattern in patterns:
            if pattern.lower() in text.lower():
                matches += 1

        if matches > 0:
            candidates.append({
                "chunk_id": chunk["id"],
                "matches": matches,
                "text_preview": text[:200],
            })

    # Sort by number of matches
    candidates.sort(key=lambda x: x["matches"], reverse=True)
    return candidates[:10]


def smart_fix_chunk_ids(gs: dict, chunks: list[dict], chunk_index: dict) -> dict:
    """
    Fix chunk_ids using article_reference metadata.

    For each question with low answerability:
    1. Extract article_reference
    2. Find chunks containing that article
    3. Score by keyword match with answer
    4. Update if better match found
    """
    results = {
        "total_processed": 0,
        "fixes_applied": 0,
        "no_article_ref": 0,
        "no_better_match": 0,
        "fixes": [],
        "remaining_issues": [],
    }

    for q in gs["questions"]:
        qid = q["id"]
        current_chunk_id = q.get("expected_chunk_id", "")
        answer = q.get("expected_answer", "")
        article_ref = q.get("metadata", {}).get("article_reference", "")
        expected_docs = q.get("expected_docs", [])

        # Compute current score
        current_score = 0.0
        if current_chunk_id in chunk_index:
            current_score = keyword_score(answer, chunk_index[current_chunk_id])

        # Skip if already good enough
        if current_score >= 0.5:
            continue

        results["total_processed"] += 1

        # Check if we have article reference
        if not article_ref:
            results["no_article_ref"] += 1
            results["remaining_issues"].append({
                "id": qid,
                "current_score": current_score,
                "reason": "no_article_reference",
            })
            continue

        # Find candidate chunks by article reference
        candidates = find_chunks_by_article(article_ref, chunks, expected_docs)

        if not candidates:
            results["remaining_issues"].append({
                "id": qid,
                "current_score": current_score,
                "article_ref": article_ref,
                "reason": "no_chunks_match_article",
            })
            continue

        # Score candidates by keyword match with answer
        best_candidate = None
        best_score = current_score

        for cand in candidates:
            cand_id = cand["chunk_id"]
            if cand_id in chunk_index:
                score = keyword_score(answer, chunk_index[cand_id])
                if score > best_score:
                    best_score = score
                    best_candidate = {
                        "chunk_id": cand_id,
                        "score": score,
                        "article_matches": cand["matches"],
                    }

        if best_candidate and best_score > current_score:
            # Apply fix
            old_id = current_chunk_id
            new_id = best_candidate["chunk_id"]

            q["expected_chunk_id"] = new_id
            q["audit"] = (q.get("audit", "") + f" [SMART_FIX] {article_ref}: {old_id[:40]} -> {new_id[:40]} (score:{best_score:.2f})").strip()

            results["fixes_applied"] += 1
            results["fixes"].append({
                "id": qid,
                "article_ref": article_ref,
                "old_chunk_id": old_id,
                "new_chunk_id": new_id,
                "old_score": current_score,
                "new_score": best_score,
            })
        else:
            results["no_better_match"] += 1
            results["remaining_issues"].append({
                "id": qid,
                "current_score": current_score,
                "article_ref": article_ref,
                "reason": "no_better_match_found",
                "candidates_checked": len(candidates),
            })

    return results


def main():
    """Main smart fix pipeline."""
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    base_path = Path(__file__).parent.parent.parent.parent
    gs_path = base_path / "tests" / "data" / "gold_standard_annales_fr_v7.json"
    chunks_path = base_path / "corpus" / "processed" / "chunks_mode_b_fr.json"
    output_path = base_path / "tests" / "data" / "smart_fix_report.json"

    print("=" * 70)
    print("SMART CHUNK FIX - Basé sur article_reference")
    print("=" * 70)
    print("""
Stratégie:
  1. Pour chaque question avec score < 0.5
  2. Extraire l'article_reference des métadonnées
  3. Chercher les chunks contenant cet article
  4. Sélectionner celui avec le meilleur score mots-clés
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

    # Run smart fix
    print(f"\n{'=' * 70}")
    print("EXÉCUTION")
    print("=" * 70)

    results = smart_fix_chunk_ids(gs, chunks, chunk_index)

    # Display results
    print(f"\nQuestions traitées: {results['total_processed']}")
    print(f"Corrections appliquées: {results['fixes_applied']}")
    print(f"Sans article_reference: {results['no_article_ref']}")
    print(f"Pas de meilleur match: {results['no_better_match']}")

    if results["fixes"]:
        print(f"\n{'=' * 70}")
        print("CORRECTIONS APPLIQUÉES")
        print("=" * 70)
        for fix in results["fixes"][:20]:
            print(f"\n{fix['id']}:")
            print(f"  Article: {fix['article_ref'][:50]}...")
            print(f"  Score: {fix['old_score']:.2f} -> {fix['new_score']:.2f}")

    # Update version
    old_version = gs.get("version", "7.4.2")
    parts = old_version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    new_version = ".".join(parts)
    gs["version"] = new_version

    # Add methodology
    if "methodology" not in gs:
        gs["methodology"] = {}
    gs["methodology"]["smart_chunk_fix"] = {
        "date": datetime.now().isoformat(),
        "method": "article_reference_matching",
        "fixes_applied": results["fixes_applied"],
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
