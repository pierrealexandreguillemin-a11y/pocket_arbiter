#!/usr/bin/env python3
"""Prepare Tier 1 adversarial questions for GS v7 import.

Output 2 files:
1. tier1_unanswerable.json - truly impossible questions (is_impossible=true)
2. tier1_hard_answerable.json - hard but answerable (is_impossible=false)

Fixes applied:
- Remove ANSWERABLE questions disguised as NEGATION
- Separate VOCABULARY_MISMATCH with existing answers from those without
- Add expected_docs where corpus reference exists
- Fix cognitive_level to Bloom's taxonomy
- Generate new URN IDs
- Add is_impossible, expected_answer fields
- Standardize validation metadata
"""

import hashlib
import json
from pathlib import Path


def generate_hash(content: str) -> str:
    """Generate 8-char hash for uniqueness."""
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def generate_new_id(source: str, category: str, seq: int, question: str) -> str:
    """Generate URN-like ID for adversarial question."""
    base = f"adversarial:{source}:{category}:{seq:03d}"
    hash_input = f"{base}:{question}"
    hash_suffix = generate_hash(hash_input)
    return f"{base}:{hash_suffix}"


def fix_cognitive_level(level: str) -> str:
    """Convert to Bloom's taxonomy title case."""
    mapping = {
        "REMEMBER": "Remember",
        "RECALL": "Remember",
        "UNDERSTAND": "Understand",
        "APPLY": "Apply",
        "ANALYZE": "Analyze",
    }
    return mapping.get(level.upper(), "Remember") if level else "Remember"


def extract_page_from_corpus_truth(corpus_truth: str) -> list[int]:
    """Extract page numbers from corpus_truth field."""
    import re

    pages = []
    # Pattern: p.123, p123, page 123
    matches = re.findall(r"p\.?(\d+)|page\s*(\d+)", corpus_truth.lower())
    for m in matches:
        page = m[0] or m[1]
        if page:
            pages.append(int(page))
    return pages


# Questions to EXCLUDE (actually answerable - explain true rules)
EXCLUDE_IDS = [
    "FR-ADV-077",  # Roque roi bouge - CORRECT rule
    "FR-ADV-078",  # Promotion roi - CORRECT rule
    "FR-ADV-079",  # Abandon adversaire - CORRECT rule
    "FR-ADV-080",  # Deux coups - CORRECT rule
    "FR-ADV-081",  # Pat victoire - CORRECT rule
    "FR-ADV-082",  # Roi capture protegee - CORRECT rule
]

# Expected docs mapping for questions that reference corpus
EXPECTED_DOCS_MAP = {
    # NUMBER_SWAP - reference real rules with wrong numbers
    "FR-ADV-051": ["LA-octobre2025.pdf"],  # 100 coups -> 75 coups
    "FR-ADV-052": ["LA-octobre2025.pdf"],  # 60 coups -> 50/75
    "FR-ADV-053": ["A02_2025_26_Championnat_de_France_des_Clubs.pdf"],  # retard 45->60
    "FR-ADV-054": ["LA-octobre2025.pdf"],  # increment blitz
    "FR-ADV-055": ["LA-octobre2025.pdf"],  # 2500 MI -> 2400
    "FR-ADV-065": ["LA-octobre2025.pdf"],  # 3 repetitions
    "FR-ADV-066": ["LA-octobre2025.pdf"],  # 90 min classique
    # ANTONYM - reference real rules with opposite meaning
    "FR-ADV-067": ["LA-octobre2025.pdf"],  # interdire reclamation
    "FR-ADV-068": ["LA-octobre2025.pdf"],  # refuser signer
    "FR-ADV-069": ["LA-octobre2025.pdf"],  # ignorer coup illegal
    "FR-ADV-070": ["LA-octobre2025.pdf"],  # desactiver pendule
    "FR-ADV-071": ["LA-octobre2025.pdf"],  # spectateur intervenir
    "FR-ADV-072": ["LA-octobre2025.pdf"],  # telephone permis
    "FR-ADV-073": ["LA-octobre2025.pdf"],  # continuer position illegale
    "FR-ADV-074": ["LA-octobre2025.pdf"],  # nulle apres coup
    "FR-ADV-075": ["LA-octobre2025.pdf"],  # refuser nulle
    "FR-ADV-076": ["LA-octobre2025.pdf"],  # retirer piece
    # NEGATION (remaining - those with FAUX corpus_truth)
    "FR-ADV-083": ["LA-octobre2025.pdf"],  # Sofia rule
    "FR-ADV-084": ["LA-octobre2025.pdf"],  # zeitnot parties longues
    # ENTITY_SWAP - some reference FFE/FIDE rules
    "FR-ADV-043": [
        "A02_2025_26_Championnat_de_France_des_Clubs.pdf"
    ],  # Top16 FIDE->FFE
    "FR-ADV-045": ["A02_2025_26_Championnat_de_France_des_Clubs.pdf"],  # N1 FIDE->FFE
    "FR-ADV-046": ["J01_2025_26_Championnat_de_France_Jeunes.pdf"],  # Poussin/Pupille
    "FR-ADV-048": ["LA-octobre2025.pdf"],  # Rapide vs Blitz
    "FR-ADV-049": ["LA-octobre2025.pdf"],  # arbitre adjoint
}

# Category mapping for ID generation
CATEGORY_MAP = {
    "classement": "rating",
    "tournoi": "clubs",
    "temps": "rules",
    "regles_jeu": "rules",
    "discipline": "rules",
    "arbitrage": "rules",
    "cadences": "rules",
    "titres": "rules",
    "jeunes": "youth",
    "online": "rules",  # For OUT_OF_SCOPE
}


def main() -> None:
    review_path = Path("tests/data/adversarial_review.json")
    unanswerable_path = Path("tests/data/tier1_unanswerable.json")
    hard_answerable_path = Path("tests/data/tier1_hard_answerable.json")

    with open(review_path, encoding="utf-8") as f:
        review = json.load(f)

    tier1_types = [
        "VOCABULARY_MISMATCH",
        "ENTITY_SWAP",
        "ANTONYM",
        "NEGATION",
        "NUMBER_SWAP",
    ]

    # VOCABULARY_MISMATCH questions where answer EXISTS (hard but answerable)
    HARD_ANSWERABLE_IDS = [
        "FR-Q77",  # 18 mois vs 12 mois - answer exists
        "FR-Q94",  # 18 mois vs 12 mois - answer exists
        "FR-Q141",  # noyau jargon - answer exists in A02
        "FR-Q144",  # joueurs etrangers - answer exists in A02
        "FR-Q149",  # flag anglicism - drapeau exists in LA
        "FR-Q150",  # Art 12.9 vs 9.6.2 - 75 coups exists
    ]

    # Filter and process
    unanswerable = []
    hard_answerable = []
    excluded = []
    seq_unanswerable: dict[str, int] = {}
    seq_answerable: dict[str, int] = {}

    for q in review["questions"]:
        if q["hard_type"] not in tier1_types:
            continue

        legacy_id = q["legacy_id"]

        # Exclude questions that explain true rules (NEGATION type CORRECT)
        if legacy_id in EXCLUDE_IDS:
            excluded.append(
                {"id": legacy_id, "reason": "Actually answerable (explains true rule)"}
            )
            continue

        # Get category for ID
        cat = CATEGORY_MAP.get(q.get("category", ""), "rules")

        # Fix expected_docs
        expected_docs = q.get("expected_docs", [])
        if not expected_docs and legacy_id in EXPECTED_DOCS_MAP:
            expected_docs = EXPECTED_DOCS_MAP[legacy_id]

        # Extract pages from corpus_truth
        expected_pages = extract_page_from_corpus_truth(q.get("corpus_truth", ""))

        # Fix cognitive_level
        meta = q.get("metadata", {})
        cognitive = fix_cognitive_level(meta.get("cognitive_level", ""))

        # Determine if truly unanswerable or hard-but-answerable
        is_hard_answerable = legacy_id in HARD_ANSWERABLE_IDS

        if is_hard_answerable:
            # Hard but answerable - sequence for ffe:human
            key = f"ffe:human:{cat}"
            seq_answerable[key] = seq_answerable.get(key, 0) + 1
            seq = seq_answerable[key]
            new_id = f"ffe:human:{cat}:{seq:03d}:{generate_hash(q['question'])}"

            gs7_question = {
                "id": new_id,
                "question": q["question"],
                "expected_answer": q.get("corpus_truth", ""),
                "is_impossible": False,
                "expected_chunk_id": "",  # TBD
                "expected_docs": expected_docs,
                "expected_pages": expected_pages if expected_pages else [],
                "category": q.get("category", ""),
                "keywords": q.get("keywords", []),
                "validation": {
                    "status": "VALIDATED",
                    "method": "adversarial_adapted",
                    "reviewer": "human",
                },
                "audit": "",
                "metadata": {
                    "answer_type": "extractive",
                    "reasoning_type": "hard-vocabulary",
                    "cognitive_level": cognitive,
                    "difficulty": 0.8,  # Hard due to vocabulary
                    "original_hard_type": q["hard_type"],
                    "corpus_truth": q.get("corpus_truth", ""),
                },
                "legacy_id": legacy_id,
            }
            hard_answerable.append(gs7_question)
        else:
            # Truly unanswerable
            key = f"adversarial:human:{cat}"
            seq_unanswerable[key] = seq_unanswerable.get(key, 0) + 1
            seq = seq_unanswerable[key]
            new_id = f"adversarial:human:{cat}:{seq:03d}:{generate_hash(q['question'])}"

            gs7_question = {
                "id": new_id,
                "question": q["question"],
                "expected_answer": f"Question impossible: {q.get('corpus_truth', '')}",
                "is_impossible": True,
                "expected_chunk_id": "",
                "expected_docs": expected_docs,
                "expected_pages": expected_pages if expected_pages else [],
                "category": q.get("category", ""),
                "keywords": q.get("keywords", []),
                "validation": {
                    "status": "VALIDATED",
                    "method": "adversarial_manual",
                    "reviewer": "human",
                },
                "audit": "",
                "metadata": {
                    "answer_type": "unanswerable",
                    "reasoning_type": "adversarial",
                    "cognitive_level": cognitive,
                    "hard_type": q["hard_type"],
                    "corpus_truth": q.get("corpus_truth", ""),
                    "test_purpose": meta.get("test_purpose", ""),
                },
                "legacy_id": legacy_id,
            }
            unanswerable.append(gs7_question)

    # Save unanswerable questions
    output_unanswerable = {
        "description": "Tier 1 UNANSWERABLE questions for GS v7 adversarial import",
        "source": "tests/data/adversarial_review.json",
        "preparation_date": "2026-01-25",
        "status": "READY_FOR_IMPORT",
        "is_impossible": True,
        "total_questions": len(unanswerable),
        "questions": unanswerable,
    }

    with open(unanswerable_path, "w", encoding="utf-8") as f:
        json.dump(output_unanswerable, f, indent=2, ensure_ascii=False)

    # Save hard-answerable questions
    output_hard = {
        "description": "Tier 1 HARD-BUT-ANSWERABLE questions for GS v7 regular import",
        "source": "tests/data/adversarial_review.json",
        "preparation_date": "2026-01-25",
        "status": "READY_FOR_IMPORT",
        "is_impossible": False,
        "note": "Ces questions testent la robustesse vocabulaire mais SONT answerable",
        "total_questions": len(hard_answerable),
        "questions": hard_answerable,
    }

    with open(hard_answerable_path, "w", encoding="utf-8") as f:
        json.dump(output_hard, f, indent=2, ensure_ascii=False)

    # Summary
    print("=" * 70)
    print("TIER 1 PREPARATION COMPLETE")
    print("=" * 70)
    print(f"UNANSWERABLE (is_impossible=true): {len(unanswerable)}")
    print(f"  -> {unanswerable_path}")
    print(f"HARD-ANSWERABLE (is_impossible=false): {len(hard_answerable)}")
    print(f"  -> {hard_answerable_path}")
    print(f"EXCLUDED (truly answerable): {len(excluded)}")

    # Stats unanswerable
    by_type = {}
    for q in unanswerable:
        ht = q["metadata"]["hard_type"]
        by_type[ht] = by_type.get(ht, 0) + 1

    print("\nUnanswerable distribution:")
    for ht, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {ht}: {count}")


if __name__ == "__main__":
    main()
