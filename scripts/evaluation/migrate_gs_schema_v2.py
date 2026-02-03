#!/usr/bin/env python3
"""Migration du Gold Standard vers le schéma v2.0.

Réorganise les 46+ champs en 8 groupes fonctionnels sans perte de données.
Ref: docs/specs/GS_SCHEMA_V2.md

Usage:
    python scripts/evaluation/migrate_gs_schema_v2.py
"""

import json
from datetime import datetime
from pathlib import Path


def migrate_question_to_v2(q: dict) -> dict:
    """Migre une question du schéma v1 (flat metadata) vers v2 (groupes)."""
    meta = q.get("metadata", {})
    validation = q.get("validation", {})

    return {
        # Racine
        "id": q.get("id"),
        "legacy_id": q.get("legacy_id"),
        # Groupe: content
        "content": {
            "question": q.get("question"),
            "expected_answer": q.get("expected_answer"),
            "is_impossible": q.get("is_impossible", False),
        },
        # Groupe: mcq
        "mcq": {
            "original_question": meta.get("original_question"),
            "choices": meta.get("choices"),
            "mcq_answer": meta.get("mcq_answer"),
            "correct_answer": meta.get("correct_answer"),
            "original_answer": meta.get("original_answer"),
        },
        # Groupe: provenance (ISO 42001)
        "provenance": {
            "chunk_id": q.get("expected_chunk_id"),
            "docs": q.get("expected_docs"),
            "pages": q.get("expected_pages"),
            "article_reference": meta.get("article_reference"),
            "answer_explanation": meta.get("answer_explanation"),
            "annales_source": meta.get("annales_source"),
        },
        # Groupe: classification
        "classification": {
            "category": q.get("category"),
            "keywords": q.get("keywords"),
            "difficulty": meta.get("difficulty"),
            "question_type": meta.get("question_type"),
            "cognitive_level": meta.get("cognitive_level"),
            "reasoning_type": meta.get("reasoning_type"),
            "reasoning_class": meta.get("reasoning_class"),
            "answer_type": meta.get("answer_type"),
        },
        # Groupe: validation (ISO 29119)
        "validation": {
            "status": validation.get("status"),
            "method": validation.get("method"),
            "reviewer": validation.get("reviewer"),
            "answer_current": validation.get("answer_current"),
            "verified_date": validation.get("verified_date"),
            "pages_verified": validation.get("pages_verified"),
            "batch": validation.get("batch"),
        },
        # Groupe: processing
        "processing": {
            "chunk_match_score": meta.get("chunk_match_score"),
            "chunk_match_method": meta.get("chunk_match_method"),
            "reasoning_class_method": meta.get("reasoning_class_method"),
            "triplet_ready": meta.get("triplet_ready"),
            "extraction_flags": meta.get("extraction_flags", []),
            "answer_source": meta.get("answer_source"),
            "quality_score": meta.get("quality_score"),
        },
        # Groupe: audit
        "audit": {
            "history": q.get("audit"),
            "qat_revalidation": meta.get("qat_revalidation"),
            "requires_inference": meta.get("requires_inference"),
        },
    }


def validate_coherence(q: dict) -> list[str]:
    """Vérifie les contraintes de cohérence du schéma v2."""
    errors = []

    # C1: correct_answer == choices[mcq_answer]
    mcq = q.get("mcq", {})
    choices = mcq.get("choices") or {}
    mcq_answer = mcq.get("mcq_answer")
    correct_answer = mcq.get("correct_answer")

    if mcq_answer and choices and correct_answer:
        expected = choices.get(mcq_answer)
        if expected and expected != correct_answer:
            errors.append(f"C1: correct_answer != choices[{mcq_answer}]")

    # C6: question finit par ?
    content = q.get("content", {})
    question = content.get("question") or ""
    if not question.endswith("?"):
        errors.append("C6: question ne finit pas par ?")

    # C7: expected_answer > 5 chars
    expected_answer = content.get("expected_answer") or ""
    if len(expected_answer) <= 5:
        errors.append(f"C7: expected_answer <= 5 chars ({len(expected_answer)})")

    # C8: difficulty in [0, 1]
    classification = q.get("classification", {})
    difficulty = classification.get("difficulty")
    if difficulty is not None and not (0 <= difficulty <= 1):
        errors.append(f"C8: difficulty {difficulty} not in [0, 1]")

    return errors


def main() -> None:
    """Point d'entrée principal."""
    gs_path = Path("tests/data/gold_standard_annales_fr_v7.json")

    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    # Mise à jour metadata
    gs["schema"] = "unified-v2"
    gs["description"] = "Gold standard v8.0 - Schema v2.0 (8 groupes fonctionnels)"
    gs["methodology"]["schema_migration"] = {
        "date": datetime.now().isoformat(),
        "from_version": "unified-v1",
        "to_version": "unified-v2",
        "spec": "docs/specs/GS_SCHEMA_V2.md",
    }

    # Migration des questions
    migrated = []
    all_errors = []

    for i, q in enumerate(gs["questions"]):
        q_v2 = migrate_question_to_v2(q)
        errors = validate_coherence(q_v2)

        if errors:
            all_errors.append((i + 1, q.get("id"), errors))

        migrated.append(q_v2)

    gs["questions"] = migrated

    # Rapport
    print(f"Questions migrées: {len(migrated)}")
    print(f"Erreurs de cohérence: {len(all_errors)}")

    if all_errors:
        print("\nErreurs détectées:")
        for qnum, qid, errs in all_errors[:10]:
            print(f"  Q{qnum} ({qid}): {errs}")
        if len(all_errors) > 10:
            print(f"  ... et {len(all_errors) - 10} autres")

    # Sauvegarde
    with open(gs_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, ensure_ascii=False, indent=2)

    print(f"\nFichier mis à jour: {gs_path}")


if __name__ == "__main__":
    main()
