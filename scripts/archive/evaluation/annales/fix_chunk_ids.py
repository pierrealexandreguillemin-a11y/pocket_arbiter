#!/usr/bin/env python3
"""
Corrige les chunk_ids du Gold Standard Annales - Approche multi-passes.

Stratégie de correction rigoureuse:
1. Passe 1 (score >= 0.95): Corrections certaines
2. Passe 2 (score >= 0.90): Haute confiance
3. Passe 3 (score >= 0.85): Confiance moyenne-haute
4. Passe 4 (score >= 0.80): Confiance acceptable

Pour chaque question, recherche également dans les chunks adjacents
(même parent, child différent) car le contenu peut être fragmenté.

Exemple structure chunk_id:
  LA-octobre2025.pdf-p038-parent157-child00
  LA-octobre2025.pdf-p038-parent157-child01  <- adjacent
  LA-octobre2025.pdf-p038-parent157-child02  <- adjacent

ISO 42001: Correction anti-hallucination avec traçabilité
ISO 25010: Amélioration qualité données
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scripts.pipeline.utils import load_json


def save_json(data: dict, path: str) -> None:
    """Save JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "ä": "a",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ç": "c",
        "œ": "oe",
        "æ": "ae",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def extract_keywords(text: str, min_length: int = 4) -> list[str]:
    """Extract meaningful keywords from text."""
    text = normalize_text(text)
    stopwords = {
        "pour",
        "dans",
        "avec",
        "cette",
        "celui",
        "celle",
        "sont",
        "etre",
        "avoir",
        "fait",
        "faire",
        "peut",
        "doit",
        "tous",
        "tout",
        "plus",
        "moins",
        "entre",
        "autres",
        "autre",
        "comme",
        "ainsi",
        "donc",
        "lors",
        "apres",
        "avant",
        "depuis",
        "pendant",
        "selon",
        "sans",
    }
    words = re.findall(r"\b[a-z]+\b", text)
    return [w for w in words if len(w) >= min_length and w not in stopwords]


def get_parent_id(chunk_id: str) -> str:
    """
    Extract parent ID from chunk_id.
    Example: LA-octobre2025.pdf-p038-parent157-child00 -> LA-octobre2025.pdf-p038-parent157
    """
    match = re.match(r"(.+-parent\d+)", chunk_id)
    return match.group(1) if match else chunk_id


def find_adjacent_chunks(chunk_id: str, chunk_index: dict) -> list[str]:
    """
    Find adjacent chunks (same parent, different child).
    Returns list of chunk_ids including the original.
    """
    parent = get_parent_id(chunk_id)
    if not parent:
        return [chunk_id]

    adjacent = []
    for cid in chunk_index:
        if cid.startswith(parent):
            adjacent.append(cid)

    return sorted(adjacent)


def compute_answer_score(answer: str, chunk_text: str) -> float:
    """
    Compute matching score between answer and chunk.
    Returns ratio of answer keywords found in chunk.
    """
    answer_keywords = extract_keywords(answer)
    if not answer_keywords:
        return 0.0

    chunk_norm = normalize_text(chunk_text)
    found = sum(1 for kw in answer_keywords if kw in chunk_norm)
    return found / len(answer_keywords)


def find_best_chunk_with_adjacents(
    answer: str,
    current_chunk_id: str,
    chunks: list[dict],
    chunk_index: dict,
    source_doc: str = None,
) -> dict | None:
    """
    Find best matching chunk, prioritizing adjacent chunks.

    Strategy:
    1. First check adjacent chunks (same parent)
    2. Then search in same document
    3. Finally search all chunks

    Returns best candidate or None.
    """
    answer_keywords = set(extract_keywords(answer))
    if not answer_keywords:
        return None

    candidates = []

    # Phase 1: Check adjacent chunks first (highest priority)
    adjacent_ids = find_adjacent_chunks(current_chunk_id, chunk_index)
    for adj_id in adjacent_ids:
        if adj_id in chunk_index:
            score = compute_answer_score(answer, chunk_index[adj_id])
            if score > 0:
                candidates.append(
                    {
                        "chunk_id": adj_id,
                        "score": score,
                        "source": "adjacent",
                    }
                )

    # Phase 2: Search in same document
    if source_doc:
        for chunk in chunks:
            if source_doc in chunk["id"] and chunk["id"] not in adjacent_ids:
                score = compute_answer_score(answer, chunk["text"])
                if score >= 0.5:  # Only consider reasonable matches
                    candidates.append(
                        {
                            "chunk_id": chunk["id"],
                            "score": score,
                            "source": "same_doc",
                        }
                    )

    # Phase 3: Search all chunks (only if no good match found)
    if not candidates or max(c["score"] for c in candidates) < 0.8:
        for chunk in chunks:
            if chunk["id"] not in [c["chunk_id"] for c in candidates]:
                score = compute_answer_score(answer, chunk["text"])
                if score >= 0.7:
                    candidates.append(
                        {
                            "chunk_id": chunk["id"],
                            "score": score,
                            "source": "global_search",
                        }
                    )

    if not candidates:
        return None

    # Sort by score descending, prefer adjacent chunks on tie
    candidates.sort(key=lambda x: (x["score"], x["source"] == "adjacent"), reverse=True)
    return candidates[0]


def fix_chunk_ids_multipass(
    gs: dict, chunks: list[dict], chunk_index: dict, thresholds: list[float]
) -> dict:
    """
    Fix chunk_ids using multiple passes with decreasing thresholds.

    Args:
        gs: Gold Standard data
        chunks: List of all chunks
        chunk_index: Dict mapping chunk_id to text
        thresholds: List of score thresholds (e.g., [0.95, 0.90, 0.85, 0.80])

    Returns:
        Dictionary with fix statistics per pass
    """
    results = {
        "passes": [],
        "total_fixed": 0,
        "all_fixes": [],
        "remaining_issues": [],
    }

    # Track which questions have been fixed
    fixed_ids = set()

    for threshold in thresholds:
        pass_fixes = []

        for q in gs["questions"]:
            qid = q["id"]

            # Skip if already fixed in previous pass
            if qid in fixed_ids:
                continue

            current_chunk_id = q.get("expected_chunk_id", "")
            answer = q.get("expected_answer", "")
            expected_docs = q.get("expected_docs", [])
            source_doc = expected_docs[0] if expected_docs else None

            # Check current chunk score
            current_score = 0.0
            if current_chunk_id in chunk_index:
                current_score = compute_answer_score(
                    answer, chunk_index[current_chunk_id]
                )

            # Skip if current chunk is already good enough
            if current_score >= threshold:
                continue

            # Find better chunk
            best = find_best_chunk_with_adjacents(
                answer, current_chunk_id, chunks, chunk_index, source_doc
            )

            if best and best["score"] >= threshold and best["score"] > current_score:
                # Apply fix
                old_id = current_chunk_id
                new_id = best["chunk_id"]

                q["expected_chunk_id"] = new_id

                # Update audit trail
                audit_note = f"[P{thresholds.index(threshold) + 1}:{threshold}] {old_id} -> {new_id} (score:{best['score']:.2f}, src:{best['source']})"
                q["audit"] = (q.get("audit", "") + " " + audit_note).strip()

                fix_info = {
                    "id": qid,
                    "old_chunk_id": old_id,
                    "new_chunk_id": new_id,
                    "score": best["score"],
                    "source": best["source"],
                    "threshold": threshold,
                    "pass": thresholds.index(threshold) + 1,
                }
                pass_fixes.append(fix_info)
                results["all_fixes"].append(fix_info)
                fixed_ids.add(qid)

        results["passes"].append(
            {
                "threshold": threshold,
                "pass_number": thresholds.index(threshold) + 1,
                "fixes_count": len(pass_fixes),
                "fixes": pass_fixes,
            }
        )
        results["total_fixed"] += len(pass_fixes)

    # Identify remaining issues
    for q in gs["questions"]:
        if q["id"] not in fixed_ids:
            chunk_id = q.get("expected_chunk_id", "")
            if chunk_id in chunk_index:
                score = compute_answer_score(
                    q.get("expected_answer", ""), chunk_index[chunk_id]
                )
                if score < 0.8:
                    results["remaining_issues"].append(
                        {
                            "id": q["id"],
                            "chunk_id": chunk_id,
                            "current_score": score,
                        }
                    )
            else:
                results["remaining_issues"].append(
                    {
                        "id": q["id"],
                        "chunk_id": chunk_id,
                        "current_score": 0.0,
                        "issue": "invalid_chunk_id",
                    }
                )

    return results


def main():
    """Main fix pipeline with multiple passes."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base_path = Path(__file__).parent.parent.parent.parent
    gs_path = base_path / "tests" / "data" / "gold_standard_annales_fr_v7.json"
    chunks_path = base_path / "corpus" / "processed" / "chunks_mode_b_fr.json"
    output_path = base_path / "tests" / "data" / "chunk_id_fix_report.json"

    print("=" * 70)
    print("FIX CHUNK_IDS - Approche Multi-Passes avec Adjacents")
    print("=" * 70)
    print("""
Stratégie:
  - Passe 1: score >= 0.95 (certitude)
  - Passe 2: score >= 0.90 (haute confiance)
  - Passe 3: score >= 0.85 (confiance moyenne-haute)
  - Passe 4: score >= 0.80 (acceptable)

Recherche prioritaire dans chunks adjacents (même parent).
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

    # Run multi-pass fix
    print(f"\n{'=' * 70}")
    print("EXÉCUTION DES PASSES")
    print("=" * 70)

    thresholds = [0.95, 0.90, 0.85, 0.80]
    results = fix_chunk_ids_multipass(gs, chunks, chunk_index, thresholds)

    # Display results per pass
    for p in results["passes"]:
        print(f"\nPasse {p['pass_number']} (seuil >= {p['threshold']}):")
        print(f"  Corrections: {p['fixes_count']}")
        if p["fixes"]:
            # Show breakdown by source
            by_source = defaultdict(int)
            for f in p["fixes"]:
                by_source[f["source"]] += 1
            for src, cnt in by_source.items():
                print(f"    - {src}: {cnt}")

    print(f"\n{'=' * 70}")
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"\nTotal corrections: {results['total_fixed']}")
    print(f"Problèmes restants: {len(results['remaining_issues'])}")

    if results["remaining_issues"]:
        print("\nÉchantillon problèmes non résolus:")
        for issue in results["remaining_issues"][:5]:
            print(f"  - {issue['id']}: score={issue.get('current_score', 0):.2f}")

    # Update version
    old_version = gs.get("version", "7.4.0")
    parts = old_version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    new_version = ".".join(parts)
    gs["version"] = new_version

    # Add methodology
    gs["methodology"]["chunk_id_correction"] = {
        "date": datetime.now().isoformat(),
        "method": "multipass_with_adjacent_search",
        "thresholds": thresholds,
        "total_fixes": results["total_fixed"],
        "remaining_issues": len(results["remaining_issues"]),
    }

    # Save
    save_json(gs, str(gs_path))
    print(f"\n[OK] Sauvegardé v{new_version} -> {gs_path}")

    # Save detailed report
    report = {
        "date": datetime.now().isoformat(),
        "from_version": old_version,
        "to_version": new_version,
        "thresholds": thresholds,
        "passes": results["passes"],
        "total_fixed": results["total_fixed"],
        "remaining_issues": results["remaining_issues"],
    }
    save_json(report, str(output_path))
    print(f"Rapport détaillé: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
