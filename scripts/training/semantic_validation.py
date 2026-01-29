#!/usr/bin/env python3
"""
Validation sémantique des Gold Standards.

Vérifie que chaque question answerable a sa réponse sur les expected_pages.
Extrait les keywords du CORPUS (pas de la question).

ISO 42001 A.6.2.4 - Data Validation
"""

import json
import re
from collections import defaultdict
from pathlib import Path


def load_chunks(path: str) -> dict:
    """Charge les chunks et indexe par (source, page)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    index = defaultdict(str)
    for c in data["chunks"]:
        key = (c["source"], c["page"])
        index[key] += " " + c["text"]

    return index


def load_gs(path: str) -> dict:
    """Charge le Gold Standard."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_corpus_keywords(text: str, min_len: int = 3) -> list[str]:
    """Extrait des mots-clés significatifs d'un texte de corpus."""
    # Mots vides courants en anglais et français
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "who",
        "what",
        "which",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "but",
        "and",
        "or",
        "if",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "any",
        "also",
        "see",
        "les",
        "le",
        "la",
        "un",
        "une",
        "des",
        "du",
        "de",
        "et",
        "ou",
        "que",
        "qui",
        "dans",
        "sur",
        "pour",
        "par",
        "avec",
        "est",
        "sont",
        "être",
        "avoir",
        "fait",
        "faire",
        "plus",
        "moins",
        "très",
        "bien",
        "mal",
        "player",
        "players",
        "game",
        "games",
        "arbiter",
        "move",
        "moves",
        "piece",
        "pieces",
        "chess",
        "clock",
        "time",
    }

    # Extraire les mots
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    # Filtrer
    keywords = []
    seen = set()
    for word in words:
        if len(word) >= min_len and word not in stopwords and word not in seen:
            keywords.append(word)
            seen.add(word)

    return keywords[:20]  # Max 20 keywords


def validate_question(
    question: dict,
    index: dict,
    default_source: str = "",
) -> dict:
    """Valide une question answerable contre le corpus.

    Returns:
        Dict avec status, issues, suggested_keywords, page_content_preview
    """
    _q_id = question.get("id", "unknown")
    _q_text = question.get("question", "")
    expected_pages = question.get("expected_pages", [])
    expected_docs = question.get("expected_docs", [])
    current_keywords = question.get("keywords", [])

    if not expected_pages:
        return {
            "status": "NO_PAGES",
            "issues": ["No expected_pages defined"],
            "suggested_keywords": [],
            "page_content": "",
        }

    # Récupérer le contenu des pages - essayer expected_docs d'abord
    source = expected_docs[0] if expected_docs else default_source
    page_content = ""
    for page in expected_pages:
        content = index.get((source, page), "")
        if not content and expected_docs:
            # Essayer sans extension ou avec variantes
            for alt_source in index:
                if alt_source[1] == page:
                    # Vérifier si le nom de source est similaire
                    if (
                        source.replace(".pdf", "") in alt_source[0]
                        or alt_source[0] in source
                    ):
                        content = index.get(alt_source, "")
                        break
        page_content += content

    if not page_content.strip():
        return {
            "status": "PAGES_NOT_FOUND",
            "issues": [
                f"Pages {expected_pages} not found in corpus for source {source}"
            ],
            "suggested_keywords": [],
            "page_content": "",
        }

    page_content_lower = page_content.lower()

    # Extraire keywords du corpus
    corpus_keywords = extract_corpus_keywords(page_content)

    # Vérifier les keywords actuels
    issues = []
    valid_keywords = []
    invalid_keywords = []

    if current_keywords:
        for kw in current_keywords:
            if kw and kw.lower() in page_content_lower:
                valid_keywords.append(kw)
            else:
                invalid_keywords.append(kw)

        if invalid_keywords:
            issues.append(f"Keywords not in corpus: {invalid_keywords}")
    else:
        issues.append("No keywords defined")

    # Déterminer le statut
    if not issues:
        status = "VALIDATED"
    elif len(valid_keywords) >= 2:
        status = "PARTIAL"
    else:
        status = "INVALID"

    return {
        "status": status,
        "issues": issues,
        "valid_keywords": valid_keywords,
        "invalid_keywords": invalid_keywords,
        "suggested_keywords": corpus_keywords[:8],
        "page_content_preview": page_content[:500] + "..."
        if len(page_content) > 500
        else page_content,
    }


def validate_gs(gs_path: str, chunks_path: str, default_source: str = "") -> dict:
    """Valide un Gold Standard complet."""
    gs = load_gs(gs_path)
    index = load_chunks(chunks_path)

    results = {
        "total": 0,
        "answerable": 0,
        "validated": 0,
        "partial": 0,
        "invalid": 0,
        "no_pages": 0,
        "pages_not_found": 0,
        "questions": [],
    }

    for q in gs["questions"]:
        results["total"] += 1

        # Skip unanswerable
        hard_type = q.get("metadata", {}).get("hard_type", "ANSWERABLE")
        if hard_type != "ANSWERABLE":
            continue

        results["answerable"] += 1

        validation = validate_question(q, index, default_source)
        validation["id"] = q.get("id", "unknown")
        validation["question"] = q.get("question", "")[:100]

        if validation["status"] == "VALIDATED":
            results["validated"] += 1
        elif validation["status"] == "PARTIAL":
            results["partial"] += 1
        elif validation["status"] == "NO_PAGES":
            results["no_pages"] += 1
        elif validation["status"] == "PAGES_NOT_FOUND":
            results["pages_not_found"] += 1
        else:
            results["invalid"] += 1

        results["questions"].append(validation)

    return results


def fix_keywords_from_corpus(
    gs_path: str, chunks_path: str, default_source: str, output_path: str
) -> dict:
    """Corrige les keywords en les extrayant du corpus et met à jour le GS."""
    gs = load_gs(gs_path)
    index = load_chunks(chunks_path)

    stats = {
        "total_fixed": 0,
        "already_valid": 0,
        "cannot_fix": 0,
        "pages_not_found": 0,
    }

    for q in gs["questions"]:
        # Skip unanswerable
        hard_type = q.get("metadata", {}).get("hard_type", "ANSWERABLE")
        if hard_type != "ANSWERABLE":
            continue

        expected_pages = q.get("expected_pages", [])
        expected_docs = q.get("expected_docs", [])

        if not expected_pages:
            stats["cannot_fix"] += 1
            continue

        # Récupérer le contenu des pages
        source = expected_docs[0] if expected_docs else default_source
        page_content = ""
        for page in expected_pages:
            content = index.get((source, page), "")
            if not content:
                # Essayer de trouver la page avec un nom de source proche
                for key in index.keys():
                    if key[1] == page and (source in key[0] or key[0] in source):
                        content = index.get(key, "")
                        break
            page_content += content

        if not page_content.strip():
            stats["cannot_fix"] += 1
            continue

        page_content_lower = page_content.lower()

        # Vérifier keywords actuels
        current_keywords = q.get("keywords", []) or []
        valid_count = sum(
            1 for kw in current_keywords if kw and kw.lower() in page_content_lower
        )

        if valid_count >= 3:
            # Déjà bon
            stats["already_valid"] += 1
            q["validation"] = {
                "status": "VALIDATED",
                "method": "corpus_verified",
            }
        else:
            # Extraire nouveaux keywords du corpus
            corpus_keywords = extract_corpus_keywords(page_content)

            # Garder les keywords valides existants + ajouter du corpus
            new_keywords = [
                kw for kw in current_keywords if kw and kw.lower() in page_content_lower
            ]
            for kw in corpus_keywords:
                if (
                    kw not in [k.lower() for k in new_keywords]
                    and len(new_keywords) < 6
                ):
                    new_keywords.append(kw)

            if len(new_keywords) >= 2:
                q["keywords"] = new_keywords[:6]
                q["validation"] = {
                    "status": "VALIDATED",
                    "method": "corpus_verified",
                    "auto_fixed": True,
                }
                stats["total_fixed"] += 1
            else:
                stats["cannot_fix"] += 1

    # Sauvegarder
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2, ensure_ascii=False)

    return stats


def main():
    """Point d'entrée."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic validation of Gold Standards"
    )
    parser.add_argument("--gs", type=Path, required=True, help="Gold Standard JSON")
    parser.add_argument("--chunks", type=Path, required=True, help="Chunks JSON")
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Default source name (optional for multi-source)",
    )
    parser.add_argument("--fix", action="store_true", help="Fix keywords from corpus")
    parser.add_argument("--output", type=Path, help="Output path for fixed GS")
    args = parser.parse_args()

    if args.fix:
        if not args.output:
            args.output = args.gs
        stats = fix_keywords_from_corpus(
            str(args.gs), str(args.chunks), args.source, str(args.output)
        )
        print("\nFix Results:")
        print(f"  Already valid: {stats['already_valid']}")
        print(f"  Fixed: {stats['total_fixed']}")
        print(f"  Cannot fix: {stats['cannot_fix']}")
    else:
        results = validate_gs(str(args.gs), str(args.chunks), args.source)

        pct = (
            results["validated"] / results["answerable"] * 100
            if results["answerable"] > 0
            else 0
        )
        print(f"\n{'=' * 60}")
        print("SEMANTIC VALIDATION REPORT")
        print(f"{'=' * 60}")
        print(f"Total questions: {results['total']}")
        print(f"Answerable: {results['answerable']}")
        print(f"Validated: {results['validated']} ({pct:.1f}%)")
        print(f"Partial: {results['partial']}")
        print(f"Invalid: {results['invalid']}")
        print(f"No pages: {results['no_pages']}")
        print(f"Pages not found: {results['pages_not_found']}")

        # Afficher les problèmes
        print(f"\n{'=' * 60}")
        print("ISSUES FOUND")
        print(f"{'=' * 60}")

        for q in results["questions"]:
            if q["status"] != "VALIDATED":
                print(f"\n{q['id']} [{q['status']}]")
                print(f"  Q: {q['question']}")
                for issue in q["issues"]:
                    print(f"  - {issue}")
                if q.get("suggested_keywords"):
                    print(f"  Suggested: {q['suggested_keywords'][:5]}")


if __name__ == "__main__":
    main()
