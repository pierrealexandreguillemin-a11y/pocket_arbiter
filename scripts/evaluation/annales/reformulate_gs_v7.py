#!/usr/bin/env python3
"""
Reformulate Gold Standard v7 questions into natural language queries.

Transforms MCQ exam questions into direct questions suitable for RAG triplets.

Standards:
- Know Your RAG (arXiv:2411.19710) - Diverse question types
- BEIR - Natural query diversity
- Sentence Transformers - Training data quality

ISO Reference: ISO 42001, ISO 25010
"""

import json
import re
from datetime import datetime
from pathlib import Path

# MCQ patterns to transform
MCQ_PATTERNS = [
    # "Quelle proposition parmi les suivantes..."
    (r"^Quelle proposition parmi les suivantes ne correspond pas [àa] (.+)\s*\?$",
     r"Quelles sont \1 ?"),
    (r"^Quelle proposition parmi les suivantes correspond [àa] (.+)\s*\?$",
     r"Quelles sont \1 ?"),
    (r"^Quelle proposition parmi les suivantes (.+)\s*\?$",
     r"Quelle est \1 ?"),

    # "Quelle personne parmi les suivantes..."
    (r"^[Qq]uelle personne parmi les suivantes (.+)\s*\?$",
     r"Qui \1 ?"),

    # "Quelle sanction parmi les suivantes..."
    (r"^[Qq]uelle sanction parmi les suivantes (.+)\s*\?$",
     r"Quelles sanctions \1 ?"),

    # "Quel cas parmi les suivants..."
    (r"^[Qq]uel cas parmi les suivants (.+)\s*\?$",
     r"Dans quels cas \1 ?"),

    # "Pour quelle compétition parmi les suivantes..."
    (r"^Pour quelle comp[ée]tition parmi les suivantes (.+)\s*\?$",
     r"Pour quelles compétitions \1 ?"),

    # "Quel niveau d'arbitre minimum parmi les suivants..."
    (r"^[Ee]n .+, quel niveau d'arbitre minimum parmi les suivants (.+)\s*\?$",
     r"Quel niveau d'arbitre minimum \1 ?"),

    # Generic "parmi les suivant(e)s"
    (r"parmi les suivant(?:e)?s?", ""),
]

# Scenario patterns to simplify
SCENARIO_SIMPLIFICATIONS = [
    # "Vous êtes l'arbitre..." → "L'arbitre..."
    (r"^Vous [êe]tes (?:l')?arbitre[^.]*\.\s*", ""),
    (r"^Vous arbitrez[^.]*\.\s*", ""),

    # Remove verbose context
    (r"^(?:Dans|Lors d')[^,]+,\s*", ""),

    # Simplify "le conducteur des blancs/noirs"
    (r"[Ll]e conducteur des (blancs|noirs)", r"les \1"),
    (r"[Ll]a conductrice des (blancs|noirs)", r"les \1"),

    # Remove timestamps like "15:15"
    (r"\d{1,2}\s*[h:]\s*\d{2}", "l'heure prévue"),
]

# Questions to reformulate completely (by ID prefix or full ID)
COMPLETE_REFORMULATIONS = {
    # Example: questions about missions
    "ffe:annales:clubs:001": "Quelles sont les missions de l'arbitre aux échecs ?",
    "ffe:annales:clubs:002": "Qui peut siéger dans un jury d'appel lors d'une compétition d'échecs ?",
    "ffe:annales:clubs:003": "Quelles sanctions l'arbitre peut-il appliquer aux capitaines d'équipe ?",
    "ffe:annales:clubs:004": "Quelles compétitions sont ouvertes aux ententes de clubs ?",
    "ffe:annales:clubs:005": "Quelles sont les missions du Directeur Régional de l'Arbitrage ?",
    "ffe:annales:clubs:006": "Comment un arbitre inactif peut-il redevenir actif ?",
    "ffe:annales:clubs:007": "Quelles sont les conditions administratives pour obtenir un titre d'arbitre fédéral ?",
    "ffe:annales:clubs:008": "Un arbitre international étranger peut-il arbitrer en France ?",
    "ffe:annales:clubs:009": "Quelle instance donne son accord pour qualifier un joueur dans un club ?",
    "ffe:annales:clubs:010": "Comment déterminer la catégorie d'âge d'un joueur ?",
    "ffe:annales:clubs:011": "Qu'est-ce qu'un forfait en compétition d'échecs ?",
    "ffe:annales:clubs:012": "Quand un résultat de rencontre est-il homologué ?",
    "ffe:annales:clubs:013": "Quelles sont les règles sur les droits d'inscription aux tournois ?",
    "ffe:annales:clubs:014": "Un joueur peut-il participer à plusieurs phases d'un tournoi découpé ?",
    "ffe:annales:clubs:015": "Qui bénéficie d'une exonération des droits d'inscription ?",
    "ffe:annales:clubs:016": "Quelles sont les règles de participation au championnat de France des clubs ?",
    "ffe:annales:clubs:017": "Qui nomme les directions de groupe en Nationale 3 ?",
    "ffe:annales:clubs:018": "Quel niveau d'arbitre faut-il pour éviter une sanction en N4 ?",
    "ffe:annales:clubs:019": "Quelles sont les règles de participation dans plusieurs équipes ?",
    "ffe:annales:clubs:020": "À partir de quand un joueur est-il forfait en championnat des clubs ?",
    "ffe:annales:clubs:021": "Qui envoie les documents en cas de litige technique ?",
    "ffe:annales:clubs:022": "Qu'est-ce que le noyau d'équipe en Nationale 3 ?",
    "ffe:annales:clubs:023": "Comment sont appariées les équipes en coupe Loubatière ?",
    "ffe:annales:clubs:024": "Comment sont attribuées les couleurs en coupe de France ?",
    "ffe:annales:clubs:025": "Quelle est l'amende pour forfait en coupe de France ?",
    "ffe:annales:clubs:026": "Comment départager deux équipes à égalité en coupe de France ?",
    "ffe:annales:clubs:027": "Quelles sont les règles de composition d'équipe en coupe de la parité ?",
    "ffe:annales:clubs:028": "Comment sont comptés les points de match en coupe de la parité ?",
    "ffe:annales:clubs:029": "Quel grade d'arbitre faut-il pour arbitrer un rapide FIDE ?",
    "ffe:annales:clubs:030": "Comment demander l'agrément FIDE pour un arbitre ?",
    # Additional reformulations for unchanged questions
    "ffe:annales:clubs:041": "Que doit faire l'arbitre quand un joueur signale une irrégularité vue dans un autre tournoi ?",
    "ffe:annales:clubs:062": "Comment gérer une réclamation sur les appariements avant leur publication ?",
    "ffe:annales:clubs:066": "Comment expliquer l'attribution des couleurs aux joueurs selon le système suisse ?",
    "ffe:annales:clubs:085": "Que faire si un joueur inscrit un résultat incorrect sur la feuille de partie ?",
    "ffe:annales:clubs:115": "Un joueur peut-il roquer avec un roi qui a déjà bougé ?",
    "ffe:annales:clubs:126": "Quand les Noirs peuvent-ils réclamer le gain au temps en Blitz ?",
    "ffe:annales:clubs:134": "Comment appliquer les règles de la pièce touchée en Blitz ?",
    "ffe:annales:clubs:150": "Que faire quand un joueur refuse d'arrêter d'écrire les coups en zeitnot ?",
    "ffe:annales:clubs:151": "Comment saisir les résultats avec des forfaits dans PAPI ?",
    "ffe:annales:clubs:158": "Que doit faire l'arbitre après un forfait en système suisse ?",
    "ffe:annales:clubs:161": "Comment expliquer un appariement inhabituel au système suisse ?",
    "ffe:annales:clubs:163": "Comment gérer un retrait de joueur en milieu de tournoi ?",
    "ffe:annales:open:005": "Quelles sont les obligations de formation continue pour les arbitres ?",
    "ffe:annales:open:010": "Comment gérer une joueuse étrangère non licenciée dans un tournoi FIDE ?",
    "ffe:annales:open:011": "Quelles règles spécifiques s'appliquent au système suisse en compétitions fédérales ?",
    "ffe:annales:open:016": "Quel est le délai d'envoi des résultats après un tournoi ?",
    "ffe:annales:open:019": "Comment calculer les départages au système suisse ?",
    "ffe:annales:open:028": "Quand une rencontre de championnat des clubs est-elle homologuée ?",
    "ffe:annales:open:030": "Quelles sont les obligations de licence pour participer à un tournoi ?",
    "ffe:annales:open:031": "Quels arbitres doivent être présents dans un club avec équipe en N3 ?",
    # Image-based question (rules:064)
    "ffe:annales:rules:064": "Quelle est la règle de la pièce touchée quand un joueur touche plusieurs pièces sans appuyer sur la pendule ?",
    # Truncated questions
    "ffe:annales:rules:087": "Quand l'arbitre peut-il déclarer une partie nulle pour répétition de position ou règle des 50 coups ?",
    "ffe:annales:rules:088": "Quelles sont les responsabilités de l'arbitre versus l'organisateur pour un tournoi ?",
}


def apply_mcq_patterns(text: str) -> str:
    """Apply MCQ pattern transformations."""
    result = text
    for pattern, replacement in MCQ_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def apply_scenario_simplifications(text: str) -> str:
    """Apply scenario simplification patterns."""
    result = text
    for pattern, replacement in SCENARIO_SIMPLIFICATIONS:
        result = re.sub(pattern, replacement, result)
    return result


def clean_question(text: str) -> str:
    """Clean and normalize question text."""
    # Fix common encoding issues
    text = text.replace("é", "é").replace("à", "à").replace("è", "è")
    text = text.replace("ê", "ê").replace("ô", "ô").replace("û", "û")
    text = text.replace("€", "€").replace("œ", "œ")

    # Remove markdown artifacts
    text = re.sub(r"<!--.*?-->", "", text)
    text = re.sub(r"\|[^|]+\|", "", text)
    text = re.sub(r"##?\s*", "", text)
    text = re.sub(r"```[^`]*```", "", text)

    # Remove table fragments
    text = re.sub(r"Echiquier N°\d+.*?(?=\s|$)", "", text)
    text = re.sub(r"\d+\s*-\s*\d+\s*\|?", "", text)

    # Clean whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Ensure ends with ?
    if text and not text.endswith("?"):
        text = text.rstrip(".,;:") + " ?"

    return text


def reformulate_question(question: dict) -> tuple[str, str]:
    """
    Reformulate a question into natural language.
    Returns (reformulated_question, method_used).
    """
    qid = question.get("id", "")
    original = question.get("question", "")
    answer = question.get("expected_answer", "")
    article = question.get("metadata", {}).get("article_reference", "")

    # Check for complete reformulation first
    for prefix, reformulated in COMPLETE_REFORMULATIONS.items():
        if qid.startswith(prefix):
            return reformulated, "manual_mapping"

    # Apply transformations
    result = original

    # Clean first
    result = clean_question(result)

    # Apply MCQ patterns
    transformed = apply_mcq_patterns(result)
    if transformed != result:
        return clean_question(transformed), "mcq_pattern"

    # Apply scenario simplifications
    transformed = apply_scenario_simplifications(result)
    if transformed != result:
        return clean_question(transformed), "scenario_pattern"

    # If already a clean question, return as-is
    if result.endswith("?") and len(result) < 200:
        return result, "already_clean"

    # For very long questions, try to extract the core question
    if len(result) > 200:
        # Look for question keywords at end
        match = re.search(
            r"(?:Que|Quel|Quelle|Comment|Pourquoi|Quand|Où|Combien)[^?]+\?$",
            result
        )
        if match:
            return match.group(0), "extracted_tail"

    return result, "unchanged"


def main():
    """Main reformulation pipeline."""
    input_path = Path("tests/data/gold_standard_annales_fr_v7.json")
    output_path = Path("tests/data/gold_standard_annales_fr_v7.json")

    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    print(f"Processing {len(questions)} questions...")

    # Track statistics
    methods = {}
    reformulated_count = 0

    for q in questions:
        original = q.get("question", "")
        reformulated, method = reformulate_question(q)

        # Store original if reformulated
        if reformulated != original:
            q["original_annales"] = original
            q["question"] = reformulated
            reformulated_count += 1

        # Track method
        q["metadata"]["reformulation_method"] = method
        methods[method] = methods.get(method, 0) + 1

    # Update version
    data["version"] = "7.5.0"
    data["reformulation"] = {
        "date": datetime.now().isoformat(),
        "from_version": "7.4.0",
        "to_version": "7.5.0",
        "reformulated_count": reformulated_count,
        "methods": methods,
    }

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Reformulation complete: v7.4.0 -> v7.5.0")
    print(f"   Reformulated: {reformulated_count}/{len(questions)}")
    print(f"\nMethods used:")
    for method, count in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"   {method}: {count}")


if __name__ == "__main__":
    main()
