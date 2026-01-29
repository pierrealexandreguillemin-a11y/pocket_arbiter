#!/usr/bin/env python3
"""Add 11 coverage questions for remaining uncovered documents.

One question per document + 2 questions on compositions and noyaux.
answer_type taxonomy: extractive, abstractive, yes_no, list, multiple_choice
cognitive_level taxonomy (Bloom's): Remember, Understand, Apply, Analyze
"""

import json
from pathlib import Path


def main() -> None:
    gs_path = Path("tests/data/gold_standard_annales_fr_v7.json")

    with open(gs_path, "r", encoding="utf-8") as f:
        gs = json.load(f)

    coverage_questions = [
        {
            "id": "FR-MED-001",
            "question": "Qui nomme le medecin federal de la FFE ?",
            "expected_answer": "Le comite directeur de la FFE parmi les membres pris en son sein. Il doit etre obligatoirement docteur en medecine et licencie a la FFE.",
            "is_impossible": False,
            "expected_chunk_id": "2022_Reglement_medical_19082022.pdf-p002-parent002-child00",
            "expected_docs": ["2022_Reglement_medical_19082022.pdf"],
            "expected_pages": [2],
            "category": "medical",
            "keywords": ["medecin", "federal", "nomination", "comite directeur"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "Remember",
                "article_reference": "Article 2.2.1 Reglement medical",
            },
        },
        {
            "id": "FR-FIN-001",
            "question": "Qui certifie la sincerite des comptes de la FFE chaque annee ?",
            "expected_answer": "Le commissaire aux comptes certifie tous les ans la sincerite des comptes, du resultat, et de la situation patrimoniale de la federation.",
            "is_impossible": False,
            "expected_chunk_id": "2023_Reglement_Financier20230610.pdf-p001-parent000-child00",
            "expected_docs": ["2023_Reglement_Financier20230610.pdf"],
            "expected_pages": [1],
            "category": "administratif",
            "keywords": ["commissaire", "comptes", "certification", "finances"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "Remember",
                "article_reference": "Article 2 Reglement Financier",
            },
        },
        {
            "id": "FR-STAT-001",
            "question": "Quelle est l'adresse du siege social de la FFE ?",
            "expected_answer": "Chateau d'Asnieres - 6, rue de l'Eglise - 92 600 Asnieres-sur-Seine. Il peut etre transfere dans ce departement par simple decision du comite directeur.",
            "is_impossible": False,
            "expected_chunk_id": "2024_Statuts20240420.pdf-p002-parent001-child00",
            "expected_docs": ["2024_Statuts20240420.pdf"],
            "expected_pages": [2],
            "category": "administratif",
            "keywords": ["siege", "social", "adresse", "Asnieres"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "Remember",
                "article_reference": "Article 1.2 Statuts FFE",
            },
        },
        {
            "id": "FR-RI-001",
            "question": "Combien de licencies A minimum sont necessaires pour l'affiliation d'un club a la FFE ?",
            "expected_answer": "L'affiliation d'une association ne vaut que si elle compte au moins 5 licencies A.",
            "is_impossible": False,
            "expected_chunk_id": "2025_Reglement_Interieur_20250503.pdf-p002-parent001-child00",
            "expected_docs": ["2025_Reglement_Interieur_20250503.pdf"],
            "expected_pages": [2],
            "category": "administratif",
            "keywords": ["affiliation", "club", "licencies A", "minimum"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "Remember",
                "article_reference": "Article 1.1 Reglement Interieur",
            },
        },
        {
            "id": "FR-DEL-001",
            "question": "Quelles formes de jeu d'echecs sont reconnues dans le contrat de delegation de la FFE avec l'Etat ?",
            "expected_answer": "Cinq specialites : Classique, Rapide, Blitz, Chess960 et e-Chess.",
            "is_impossible": False,
            "expected_chunk_id": "Contrat_de_delegation_15032022.pdf-p005-parent004-child00",
            "expected_docs": ["Contrat_de_delegation_15032022.pdf"],
            "expected_pages": [5],
            "category": "administratif",
            "keywords": [
                "delegation",
                "specialites",
                "disciplines",
                "e-Chess",
                "contrat",
            ],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "list",
                "reasoning_type": "single-hop",
                "cognitive_level": "Remember",
                "article_reference": "Article 1 Contrat de delegation",
            },
        },
        {
            "id": "FR-N6BDR-001",
            "question": "Combien de joueurs composent une equipe en Nationale VI departementale des Bouches-du-Rhone ?",
            "expected_answer": "4 joueurs qui possedent une licence A et affilies a un club des Bouches-du-Rhone.",
            "is_impossible": False,
            "expected_chunk_id": "Interclubs_DepartementalBdr.pdf-p005-parent008-child00",
            "expected_docs": ["Interclubs_DepartementalBdr.pdf"],
            "expected_pages": [5],
            "category": "regional",
            "keywords": ["equipe", "composition", "N6", "Bouches-du-Rhone"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "Remember",
                "article_reference": "Article 3.7 Reglement N6 BDR",
            },
        },
        {
            "id": "FR-IJBDR-001",
            "question": "Quelle est la cadence pour l'echiquier 4 en Interclubs Jeunes des Bouches-du-Rhone ?",
            "expected_answer": "50 minutes + 10 secondes par coup.",
            "is_impossible": False,
            "expected_chunk_id": "InterclubsJeunes_PACABdr.pdf-p002-parent001-child00",
            "expected_docs": ["InterclubsJeunes_PACABdr.pdf"],
            "expected_pages": [2],
            "category": "regional",
            "keywords": ["cadence", "echiquier 4", "interclubs", "jeunes"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "Remember",
                "article_reference": "Article 7 Interclubs Jeunes BDR",
            },
        },
        {
            "id": "FR-N4PACA-001",
            "question": "Combien de groupes de Nationale 4 sont crees en ligue PACA et de combien d'equipes chacun ?",
            "expected_answer": "5 groupes de 8 equipes chacun. La ligue dispose de 5 accessions en N3.",
            "is_impossible": False,
            "expected_chunk_id": "règlement_n4_2024_2025__1_.pdf-p001-parent000-child00",
            "expected_docs": ["règlement_n4_2024_2025__1_.pdf"],
            "expected_pages": [1],
            "category": "regional",
            "keywords": ["N4", "groupes", "PACA", "structure"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "Remember",
                "article_reference": "Article 1.1 Reglement N4 PACA",
            },
        },
        {
            "id": "FR-REGPACA-001",
            "question": "Combien de joueurs composent une equipe en Regionale de la ligue PACA ?",
            "expected_answer": "5 joueurs licencies A.",
            "is_impossible": False,
            "expected_chunk_id": "règlement_régionale_2024_2025.pdf-p002-parent001-child00",
            "expected_docs": ["règlement_régionale_2024_2025.pdf"],
            "expected_pages": [2],
            "category": "regional",
            "keywords": ["equipe", "composition", "regionale", "PACA"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "Remember",
                "article_reference": "Article 3.7.a Reglement Regionale PACA",
            },
        },
        {
            "id": "FR-COMP-001",
            "question": "Quel est le nombre de joueurs par equipe selon le niveau de championnat : Top16, N1-N3, N4, Regionale PACA et Departementale BDR ?",
            "expected_answer": "Top16/N1/N2/N3 : 8 joueurs. N4 : 8 joueurs (ou 6 si la Ligue le choisit). Regionale PACA : 5 joueurs. Departementale N6 BDR : 4 joueurs.",
            "is_impossible": False,
            "expected_chunk_id": "A02_2025_26_Championnat_de_France_des_Clubs.pdf-p005-parent004-child00",
            "expected_docs": [
                "A02_2025_26_Championnat_de_France_des_Clubs.pdf",
                "règlement_régionale_2024_2025.pdf",
                "Interclubs_DepartementalBdr.pdf",
            ],
            "expected_pages": [5, 2, 5],
            "category": "interclubs",
            "keywords": [
                "composition",
                "equipe",
                "joueurs",
                "championnat",
                "national",
                "regional",
            ],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "list",
                "reasoning_type": "multi-hop",
                "cognitive_level": "Understand",
                "article_reference": "Article 3.7.a A02, Article 3.7.a Regionale PACA, Article 3.7 N6 BDR",
            },
        },
        {
            "id": "FR-NOYAU-001",
            "question": "Quelle est la regle du noyau pour les equipes en N1, N2, N3 et N4 ?",
            "expected_answer": "Chaque equipe doit aligner a chaque ronde au moins 50% de personnes ayant deja participe au moins une fois pour le compte de cette equipe depuis le debut de la saison (sauf ronde 1). En cas d'infraction, forfait administratif sur l'echiquier hors noyau et tous ceux qui le suivent.",
            "is_impossible": False,
            "expected_chunk_id": "A02_2025_26_Championnat_de_France_des_Clubs.pdf-p006-parent005-child00",
            "expected_docs": ["A02_2025_26_Championnat_de_France_des_Clubs.pdf"],
            "expected_pages": [6],
            "category": "interclubs",
            "keywords": ["noyau", "equipe", "50%", "composition", "N1", "N2", "N3"],
            "validation": {
                "status": "PENDING",
                "method": "manual_creation",
                "reviewer": "human",
            },
            "audit": "",
            "metadata": {
                "answer_type": "extractive",
                "reasoning_type": "single-hop",
                "cognitive_level": "Remember",
                "article_reference": "Article 3.7.f A02",
            },
        },
    ]

    gs["questions"].extend(coverage_questions)

    if "coverage" in gs:
        gs["coverage"]["total_questions"] = len(gs["questions"])

    gs["version"] = "7.2.0"

    with open(gs_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2, ensure_ascii=False)

    print("Questions ajoutees: 11")
    print("- FR-MED-001: Reglement medical (medecin federal)")
    print("- FR-FIN-001: Reglement Financier (commissaire aux comptes)")
    print("- FR-STAT-001: Statuts (siege social)")
    print("- FR-RI-001: Reglement Interieur (affiliation club)")
    print("- FR-DEL-001: Contrat delegation (formes de jeu)")
    print("- FR-N6BDR-001: Interclubs Departemental BDR (composition)")
    print("- FR-IJBDR-001: Interclubs Jeunes BDR (cadence)")
    print("- FR-N4PACA-001: N4 PACA (structure)")
    print("- FR-REGPACA-001: Regionale PACA (composition)")
    print("- FR-COMP-001: Compositions par championnat (multi-hop)")
    print("- FR-NOYAU-001: Regle du noyau (50%)")
    print(f"Total questions: {len(gs['questions'])}")


if __name__ == "__main__":
    main()
