#!/usr/bin/env python3
"""Add 8 coverage questions for uncovered P0/P1 documents.

answer_type taxonomy (GOLD_STANDARD_V6_ANNALES.md):
- extractive: answer directly in chunk
- abstractive: requires synthesis
- yes_no: yes/no questions
- list: enumeration answers
- multiple_choice: MCQ
"""

import json
from pathlib import Path


def main() -> None:
    gs_path = Path("tests/data/gold_standard_annales_fr_v7.json")

    with open(gs_path, "r", encoding="utf-8") as f:
        gs = json.load(f)

    coverage_questions = [
        {
            "id": "FR-J03-001",
            "question": "Quelle est la cadence de jeu lors de la phase departementale du Championnat de France scolaire ?",
            "expected_answer": "15 min KO ou 12 min + 3 s par coup. Les appariements sont faits au systeme Suisse en 5 rondes minimum.",
            "is_impossible": False,
            "expected_chunk_id": "J03_2025_26_Championnat_de_France_scolaire.pdf-p002-parent009-child00",
            "expected_docs": ["J03_2025_26_Championnat_de_France_scolaire.pdf"],
            "expected_pages": [2],
            "category": "jeunes",
            "keywords": ["cadence", "scolaire", "departemental", "15 min"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "REMEMBER",
                "article_reference": "Article 2.2.1 J03",
            },
        },
        {
            "id": "FR-F01-001",
            "question": "Combien de divisions comporte le Championnat de France Feminin des Clubs ?",
            "expected_answer": "3 divisions : Top 12F (12 equipes), Nationale 1 feminine (4 groupes de 8 equipes), et Nationale 2 feminine.",
            "is_impossible": False,
            "expected_chunk_id": "F01_2025_26_Championnat_de_France_des_clubs_Feminin.pdf-p001-parent000-child00",
            "expected_docs": [
                "F01_2025_26_Championnat_de_France_des_clubs_Feminin.pdf"
            ],
            "expected_pages": [1],
            "category": "feminin",
            "keywords": ["structure", "division", "Top 12F", "N1F", "N2F"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "list",
                "reasoning_type": "single-hop",
                "cognitive_level": "REMEMBER",
                "article_reference": "Article 1.1 F01",
            },
        },
        {
            "id": "FR-F02-001",
            "question": "Combien de joueuses se qualifient pour la finale du Championnat individuel feminin rapides si 25 joueuses participent a la phase ZID ?",
            "expected_answer": "3 joueuses. Pour 21 a 30 participantes, 3 joueuses se qualifient.",
            "is_impossible": False,
            "expected_chunk_id": "F02_2025_26_Championnat_individuel_Feminin_parties_rapides.pdf-p001-parent001-child00",
            "expected_docs": [
                "F02_2025_26_Championnat_individuel_Feminin_parties_rapides.pdf"
            ],
            "expected_pages": [1],
            "category": "feminin",
            "keywords": ["qualification", "finale", "feminin", "rapides", "ZID"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "APPLY",
                "article_reference": "Article 1.2 F02",
            },
        },
        {
            "id": "FR-H01-001",
            "question": "Un joueur peut-il refuser de jouer contre un adversaire en raison de son handicap ?",
            "expected_answer": "Non. Personne n'a le droit de refuser de jouer avec une personne contre laquelle elle a ete appariee correctement, notamment si sa raison est fondee sur un handicap de l'adversaire.",
            "is_impossible": False,
            "expected_chunk_id": "H01_2025_26_Conduite_pour_joueur_handicapes.pdf-p001-parent000-child00",
            "expected_docs": ["H01_2025_26_Conduite_pour_joueur_handicapes.pdf"],
            "expected_pages": [1],
            "category": "handicap",
            "keywords": ["handicap", "refus", "appariement"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "yes_no",
                "reasoning_type": "single-hop",
                "cognitive_level": "UNDERSTAND",
                "article_reference": "Article 2 H01",
            },
        },
        {
            "id": "FR-H02-001",
            "question": "Que se passe-t-il si une equipe recoit dans des locaux inaccessibles alors qu'un joueur adverse est handicape ?",
            "expected_answer": "La rencontre sera deplacee dans un local accessible propose par l'equipe accueillante, sinon par le comite departemental, et a defaut par la Ligue. L'equipe fautive perdra la partie concernee si le principe n'est pas respecte.",
            "is_impossible": False,
            "expected_chunk_id": "H02_2025_26_Joueurs_a_mobilite_reduite.pdf-p001-parent002-child00",
            "expected_docs": ["H02_2025_26_Joueurs_a_mobilite_reduite.pdf"],
            "expected_pages": [1],
            "category": "handicap",
            "keywords": ["accessibilite", "mobilite reduite", "locaux"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "multi-hop",
                "cognitive_level": "UNDERSTAND",
                "article_reference": "Phase II H02",
            },
        },
        {
            "id": "FR-E02-001",
            "question": "Quelle est la regle de compatibilite appliquee pour le calcul de l'Elo Rapide ?",
            "expected_answer": "+-320 points. Le seuil d'entree au Rapide est de 7 parties compatibles et le plancher est de 800.",
            "is_impossible": False,
            "expected_chunk_id": "E02-Le_classement_rapide.pdf-p001-parent001-child00",
            "expected_docs": ["E02-Le_classement_rapide.pdf"],
            "expected_pages": [1],
            "category": "classement",
            "keywords": ["compatibilite", "elo", "rapide", "320 points"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "REMEMBER",
                "article_reference": "Article 1 E02",
            },
        },
        {
            "id": "FR-A01-001",
            "question": "Combien de tournois constituent le Championnat de France individuel ?",
            "expected_answer": "10 tournois : National, National Feminin, Accession Roger FERRY, Veterans, Seniors Plus, Open A, Open B, Open C, Open D et Open E.",
            "is_impossible": False,
            "expected_chunk_id": "A01_2025_26_Championnat_de_France.pdf-p001-parent001-child00",
            "expected_docs": ["A01_2025_26_Championnat_de_France.pdf"],
            "expected_pages": [1],
            "category": "competitions",
            "keywords": ["structure", "tournoi", "national", "open"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "list",
                "reasoning_type": "single-hop",
                "cognitive_level": "REMEMBER",
                "article_reference": "Article 1.2 A01",
            },
        },
        {
            "id": "FR-A03-001",
            "question": "Quelles sont les categories par moyenne Elo au Championnat de France des Clubs rapides ?",
            "expected_answer": "3 categories : Categorie A (moyenne > 2000), Categorie B (moyenne > 1700), Categorie C (moyenne <= 1700).",
            "is_impossible": False,
            "expected_chunk_id": "A03_2025_26_Championnat_de_France_des_Clubs_rapides.pdf-p004-parent000-child00",
            "expected_docs": [
                "A03_2025_26_Championnat_de_France_des_Clubs_rapides.pdf"
            ],
            "expected_pages": [4],
            "category": "competitions",
            "keywords": ["categorie", "moyenne", "elo", "clubs", "rapides"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "list",
                "reasoning_type": "single-hop",
                "cognitive_level": "REMEMBER",
                "article_reference": "Article 1.1 A03",
            },
        },
    ]

    gs["questions"].extend(coverage_questions)

    if "coverage" in gs:
        gs["coverage"]["total_questions"] = len(gs["questions"])

    gs["version"] = "7.1.0_cov8"

    with open(gs_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2, ensure_ascii=False)

    print("Questions ajoutees: 8 (J03, F01, F02, H01, H02, E02, A01, A03)")
    print(f"Total questions: {len(gs['questions'])}")


if __name__ == "__main__":
    main()
