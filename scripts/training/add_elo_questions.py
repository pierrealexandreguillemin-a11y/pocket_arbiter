#!/usr/bin/env python3
"""Add 8 ELO questions to gold standard."""

import json
from pathlib import Path


def main() -> None:
    gs_path = Path("tests/data/gold_standard_annales_fr_v7.json")

    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    elo_questions = [
        {
            "id": "FR-ELO-001",
            "question": "Un joueur classé 1923 avec K=20 bat un adversaire classé 1812. Combien de points Elo gagne-t-il ?",
            "expected_answer": "7 points. Différence +111 → score théorique 0.65. Score réel 1. Gain = (1 - 0.65) × 20 = 7",
            "is_impossible": False,
            "expected_chunk_id": "LA-octobre2025.pdf-p186-parent552-child00",
            "expected_docs": ["LA-octobre2025.pdf"],
            "expected_pages": [186],
            "category": "classement",
            "keywords": ["elo", "coefficient", "K", "calcul", "gain"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "NUMERICAL",
                "reasoning_type": "multi-hop",
                "cognitive_level": "APPLY",
                "article_reference": "Article 8.3 Règles Classement FIDE",
            },
        },
        {
            "id": "FR-ELO-002",
            "question": "Un joueur de 17 ans classé 2250 joue sa 25ème partie officielle. Quel est son coefficient K ?",
            "expected_answer": "K=40. Moins de 18 ans ET classement < 2300 → K=40 jusqu'à fin de l'année de ses 18 ans.",
            "is_impossible": False,
            "expected_chunk_id": "LA-octobre2025.pdf-p185-parent546-child00",
            "expected_docs": ["LA-octobre2025.pdf"],
            "expected_pages": [185],
            "category": "classement",
            "keywords": ["coefficient", "K", "développement", "jeune", "18 ans"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "NUMERICAL",
                "reasoning_type": "multi-hop",
                "cognitive_level": "APPLY",
                "article_reference": "Article 8.3.3 Règles Classement FIDE",
            },
        },
        {
            "id": "FR-ELO-003",
            "question": "Un joueur classé 1923 (K=20) joue 5 parties : victoire vs 1812, défaite vs 2148, nulle vs 2515, victoire vs 1413, défaite vs 2109. Score total 2.5/5. Combien de points gagne-t-il ?",
            "expected_answer": "7,4 points (arrondi à 7). Score théorique total = 2,13. Score réel = 2,5. Différence = 0,37. Gain = 0,37 × 20 = 7,4 pts.",
            "is_impossible": False,
            "expected_chunk_id": "LA-octobre2025.pdf-p186-parent552-child01",
            "expected_docs": ["LA-octobre2025.pdf"],
            "expected_pages": [186],
            "category": "classement",
            "keywords": ["elo", "calcul", "tournoi", "variation", "plusieurs parties"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "NUMERICAL",
                "reasoning_type": "multi-hop",
                "cognitive_level": "APPLY",
                "article_reference": "Article 8.3 Règles Classement FIDE",
            },
        },
        {
            "id": "FR-ELO-004",
            "question": "Un joueur U14 prend sa première licence FFE. Quel Elo estimé standard lui sera attribué ?",
            "expected_answer": "1299. Les U20, U18, U16, U14, U12, U10 et U8 reçoivent un Elo estimé standard de 1299.",
            "is_impossible": False,
            "expected_chunk_id": "R01_2025_26_Regles_generales.pdf-p004-parent022-child00",
            "expected_docs": ["R01_2025_26_Regles_generales.pdf"],
            "expected_pages": [4],
            "category": "classement",
            "keywords": ["elo", "estimé", "premier", "jeune", "U14"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "NUMERICAL",
                "reasoning_type": "single-hop",
                "cognitive_level": "REMEMBER",
                "article_reference": "Article 5 R01 Règles générales",
            },
        },
        {
            "id": "FR-ELO-005",
            "question": "Un joueur reprend la compétition après 7 ans d'arrêt. Son ancien Elo était 1650. Que se passe-t-il ?",
            "expected_answer": "Il récupère normalement son ancien Elo (1650). Mais son club peut demander à la FFE un Elo estimé différent avec pièces justificatives, car l'ancien Elo a plus de 5 saisons.",
            "is_impossible": False,
            "expected_chunk_id": "R01_2025_26_Regles_generales.pdf-p004-parent021-child01",
            "expected_docs": ["R01_2025_26_Regles_generales.pdf"],
            "expected_pages": [4],
            "category": "classement",
            "keywords": ["elo", "ancien", "reprise", "5 saisons", "estimé"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "PROCEDURAL",
                "reasoning_type": "multi-hop",
                "cognitive_level": "UNDERSTAND",
                "article_reference": "Article 5 R01 Règles générales",
            },
        },
        {
            "id": "FR-ELO-006",
            "question": "Un joueur senior dont l'Elo descend à 1350 (sous le plancher FIDE). Quel Elo lui sera attribué ?",
            "expected_answer": "1399 obligatoirement. Un joueur dont l'Elo descend en dessous du plancher FIDE aura un Elo estimé à 1399 quel que soit son âge.",
            "is_impossible": False,
            "expected_chunk_id": "R01_2025_26_Regles_generales.pdf-p004-parent023-child00",
            "expected_docs": ["R01_2025_26_Regles_generales.pdf"],
            "expected_pages": [4],
            "category": "classement",
            "keywords": ["elo", "plancher", "FIDE", "estimé", "1399"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "NUMERICAL",
                "reasoning_type": "single-hop",
                "cognitive_level": "REMEMBER",
                "article_reference": "Article 5 R01 Règles générales",
            },
        },
        {
            "id": "FR-ELO-007",
            "question": "Un joueur U10 prend sa première licence FFE. Quel Elo estimé standard lui sera attribué ?",
            "expected_answer": "1299. En Elo Standard : 1299 pour tous les jeunes (U20, U18, U16, U14, U12, U10, U8), 1399 pour les vétérans, seniors plus et seniors.",
            "is_impossible": False,
            "expected_chunk_id": "R01_2025_26_Regles_generales.pdf-p004-parent022-child00",
            "expected_docs": ["R01_2025_26_Regles_generales.pdf"],
            "expected_pages": [4],
            "category": "classement",
            "keywords": ["elo", "estimé", "standard", "U10", "jeune", "catégorie"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "NUMERICAL",
                "reasoning_type": "single-hop",
                "cognitive_level": "REMEMBER",
                "article_reference": "Article 5 R01 Règles générales",
            },
        },
        {
            "id": "FR-ELO-008",
            "question": "Un senior prend sa première licence FFE. Quelle est la différence entre son Elo estimé standard et son Elo estimé rapide ?",
            "expected_answer": "200 points. Elo standard = 1399, Elo rapide = 1199. Différence = 1399 - 1199 = 200 points.",
            "is_impossible": False,
            "expected_chunk_id": "R01_2025_26_Regles_generales.pdf-p004-parent022-child00",
            "expected_docs": ["R01_2025_26_Regles_generales.pdf"],
            "expected_pages": [4],
            "category": "classement",
            "keywords": ["elo", "estimé", "standard", "rapide", "différence", "senior"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "NUMERICAL",
                "reasoning_type": "multi-hop",
                "cognitive_level": "ANALYZE",
                "article_reference": "Article 5 R01 Règles générales",
            },
        },
    ]

    gs["questions"].extend(elo_questions)

    if "coverage" in gs:
        gs["coverage"]["total_questions"] = len(gs["questions"])

    gs["version"] = "7.0.0_dedup_elo8"

    with open(gs_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2, ensure_ascii=False)

    print("Questions ajoutées: 8")
    print(f"Total questions: {len(gs['questions'])}")


if __name__ == "__main__":
    main()
