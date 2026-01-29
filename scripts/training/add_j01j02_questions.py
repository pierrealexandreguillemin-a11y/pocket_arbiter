#!/usr/bin/env python3
"""Add 7 J01/J02 questions to gold standard."""

import json
from pathlib import Path


def main() -> None:
    gs_path = Path("tests/data/gold_standard_annales_fr_v7.json")

    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    j01j02_questions = [
        {
            "id": "FR-J01-001",
            "question": "Quelle est la cadence minimum pour les U8 et U10 lors des qualifications départementales du Championnat de France Jeunes ?",
            "expected_answer": "Minimum 15 min + 5 sec/coup pour les catégories U8, U8F, U10, U10F.",
            "is_impossible": False,
            "expected_chunk_id": "J01_2025_26_Championnat_de_France_Jeunes.pdf-p004-parent018-child00",
            "expected_docs": ["J01_2025_26_Championnat_de_France_Jeunes.pdf"],
            "expected_pages": [4],
            "category": "jeunes",
            "keywords": ["cadence", "U8", "U10", "départemental", "jeunes"],
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
                "article_reference": "Article 3.2.1 J01",
            },
        },
        {
            "id": "FR-J01-002",
            "question": "Quelle est la cadence pour les U8 et U10 lors du tournoi national du Championnat de France Jeunes ?",
            "expected_answer": "50 minutes pour toute la partie avec ajout de 10 sec/coup.",
            "is_impossible": False,
            "expected_chunk_id": "J01_2025_26_Championnat_de_France_Jeunes.pdf-p004-parent020-child00",
            "expected_docs": ["J01_2025_26_Championnat_de_France_Jeunes.pdf"],
            "expected_pages": [4],
            "category": "jeunes",
            "keywords": ["cadence", "U8", "U10", "national", "jeunes", "50 minutes"],
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
                "article_reference": "Article 3.2.3 J01",
            },
        },
        {
            "id": "FR-J01-003",
            "question": "Quel Elo minimum doit atteindre un joueur U12 pour être qualifié d'office au Championnat de France Jeunes ?",
            "expected_answer": "1900 pour les U12, 1700 pour les U12F.",
            "is_impossible": False,
            "expected_chunk_id": "J01_2025_26_Championnat_de_France_Jeunes.pdf-p003-parent012-child00",
            "expected_docs": ["J01_2025_26_Championnat_de_France_Jeunes.pdf"],
            "expected_pages": [3],
            "category": "jeunes",
            "keywords": ["elo", "qualification", "U12", "jeunes", "office"],
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
                "article_reference": "Article 2.6 J01",
            },
        },
        {
            "id": "FR-J02-001",
            "question": "Combien de joueurs composent une équipe en Top Jeunes et Nationale 1 du Championnat Interclubs Jeunes ?",
            "expected_answer": "8 joueurs et/ou joueuses (Top Jeunes, Nationales 1 et 2) ou 4 joueurs et/ou joueuses (Nationale 3).",
            "is_impossible": False,
            "expected_chunk_id": "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf-p004-parent016-child00",
            "expected_docs": [
                "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf"
            ],
            "expected_pages": [4],
            "category": "jeunes",
            "keywords": ["équipe", "composition", "interclubs", "jeunes", "8 joueurs"],
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
                "article_reference": "Article 3.7.a J02",
            },
        },
        {
            "id": "FR-J02-002",
            "question": "Quelle est la cadence pour les échiquiers 1 à 6 en Top Jeunes et Nationale 1 du Championnat Interclubs Jeunes ?",
            "expected_answer": "1h30 min et 30 secondes par coup pour les échiquiers 1 à 6.",
            "is_impossible": False,
            "expected_chunk_id": "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf-p003-parent012-child00",
            "expected_docs": [
                "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf"
            ],
            "expected_pages": [3],
            "category": "jeunes",
            "keywords": ["cadence", "interclubs", "jeunes", "échiquier", "1h30"],
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
                "article_reference": "Article 3.3 J02",
            },
        },
        {
            "id": "FR-J02-003",
            "question": "Quelle est la cadence pour les échiquiers 7 et 8 en Interclubs Jeunes Top Jeunes et N1 ?",
            "expected_answer": "50 min et 10 secondes par coup pour les échiquiers 7 et 8.",
            "is_impossible": False,
            "expected_chunk_id": "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf-p003-parent012-child00",
            "expected_docs": [
                "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf"
            ],
            "expected_pages": [3],
            "category": "jeunes",
            "keywords": [
                "cadence",
                "interclubs",
                "jeunes",
                "échiquier 7",
                "échiquier 8",
                "50 min",
            ],
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
                "article_reference": "Article 3.3 J02",
            },
        },
        {
            "id": "FR-J02-004",
            "question": "Comment sont ordonnés les joueurs sur les échiquiers en Interclubs Jeunes ?",
            "expected_answer": "Par catégorie d'âge décroissante (le plus âgé à l'échiquier 1). L'échiquier 8 peut être placé devant l'échiquier 7 quels que soient les Elo. Un joueur placé après un plus jeune que lui entraîne un forfait administratif.",
            "is_impossible": False,
            "expected_chunk_id": "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf-p004-parent018-child01",
            "expected_docs": [
                "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf"
            ],
            "expected_pages": [4],
            "category": "jeunes",
            "keywords": ["ordre", "échiquier", "âge", "interclubs", "jeunes"],
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
                "article_reference": "Article 3.7.c J02",
            },
        },
    ]

    gs["questions"].extend(j01j02_questions)

    if "coverage" in gs:
        gs["coverage"]["total_questions"] = len(gs["questions"])

    gs["version"] = "7.0.0_dedup_elo8_j7"

    with open(gs_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2, ensure_ascii=False)

    print("Questions ajoutées: 7 (3 J01 + 4 J02)")
    print(f"Total questions: {len(gs['questions'])}")


if __name__ == "__main__":
    main()
