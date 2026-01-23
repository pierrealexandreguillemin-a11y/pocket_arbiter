#!/usr/bin/env python3
"""Fix Gold Standard issues identified in audit.

Issues:
1. FR-Q80 doesn't end with ?
2. Missing SQuAD2-CR categories: ANTONYM, NEGATION, MUTUAL_EXCLUSION
3. Add to adversarial_questions.json first, then merge

References:
- arXiv:2004.14004 (SQuAD2-CR)
- ISO 42001: Data quality requirements
"""

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    """Load JSON file with UTF-8 encoding."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """Save JSON file with UTF-8 encoding and formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fix_fr_q80():
    """Fix FR-Q80 question format."""
    gs_path = Path("tests/data/gold_standard_fr.json")
    gs = load_json(gs_path)

    for q in gs["questions"]:
        if q["id"] == "FR-Q80":
            # Original: "Un joueur peut-il obtenir... (CM) ? Expliquez les voies possibles."
            # Fix: Make it a proper question
            q["question"] = (
                "Un joueur peut-il obtenir le titre de Maitre FIDE (FM) sans passer "
                "par le titre de Candidat Maitre (CM), et quelles sont les voies possibles ?"
            )
            print(f"Fixed FR-Q80: {q['question'][:60]}...")
            break

    save_json(gs_path, gs)
    print("Saved gold_standard_fr.json")


def add_squad2cr_questions():
    """Add missing SQuAD2-CR categories to adversarial questions."""
    adv_path = Path("tests/data/adversarial_questions.json")
    adv = load_json(adv_path)

    # NEW ANTONYM questions (target ~10-15)
    antonym_questions = [
        {
            "id": "FR-ADV-067",
            "question": "Quand l'arbitre doit-il interdire une reclamation de nulle par repetition ?",
            "category": "arbitrage",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["interdire", "reclamation", "nulle"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "L'arbitre AUTORISE (pas interdit) les reclamations valides",
                "corpus_truth": "ANTONYM - Art 9.2: arbitre VERIFIE et ACCEPTE si valide",
                "test_purpose": "Test detection antonym (SQuAD2-CR)",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-068",
            "question": "Quand un joueur doit-il refuser de signer la feuille de partie ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["refuser", "signer", "feuille"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "Un joueur doit SIGNER (pas refuser) la feuille",
                "corpus_truth": "ANTONYM - Art 8.7: les deux joueurs DOIVENT signer",
                "test_purpose": "Test detection antonym obligation",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-069",
            "question": "Quand l'arbitre peut-il ignorer un coup illegal ?",
            "category": "arbitrage",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["ignorer", "coup illegal"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "L'arbitre INTERVIENT (pas ignore) sur coup illegal",
                "corpus_truth": "ANTONYM - Art 7.5: arbitre DOIT intervenir",
                "test_purpose": "Test detection antonym action",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-070",
            "question": "Quand peut-on desactiver la pendule pendant une partie classique ?",
            "category": "temps",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["desactiver", "pendule"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "La pendule reste ACTIVE (pas desactivee) pendant le jeu",
                "corpus_truth": "ANTONYM - Art 6: pendule doit fonctionner",
                "test_purpose": "Test detection antonym etat",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-071",
            "question": "Quand un spectateur peut-il intervenir dans une partie en cours ?",
            "category": "discipline",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["spectateur", "intervenir"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "Spectateurs ne peuvent PAS intervenir (jamais)",
                "corpus_truth": "ANTONYM - Art 12.7: spectateurs NE DOIVENT PAS intervenir",
                "test_purpose": "Test detection antonym interdiction",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-072",
            "question": "Quand est-il permis d'utiliser son telephone en zone de jeu ?",
            "category": "discipline",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["permis", "telephone", "zone de jeu"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "Telephone INTERDIT (pas permis) en zone de jeu",
                "corpus_truth": "ANTONYM - Art 11.3.2: appareils electroniques INTERDITS",
                "test_purpose": "Test detection antonym interdiction",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-073",
            "question": "Quand doit-on continuer a jouer apres avoir revendique une position illegale ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["continuer", "position illegale"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "On doit ARRETER (pas continuer) pour reclamation",
                "corpus_truth": "ANTONYM - Art 7: arreter pendule et appeler arbitre",
                "test_purpose": "Test detection antonym procedure",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-074",
            "question": "Quand peut-on accepter une offre de nulle apres avoir joue son coup ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["accepter", "nulle", "apres coup"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "Offre AVANT le coup, pas apres",
                "corpus_truth": "ANTONYM - Art 9.1.2: offre en jouant, adversaire decide",
                "test_purpose": "Test detection antonym sequence",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-075",
            "question": "Quand un joueur peut-il refuser une nulle proposee correctement ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["refuser", "nulle"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "Un joueur PEUT toujours refuser (pas d'obligation)",
                "corpus_truth": "PARTIEL - Art 9.1: adversaire peut accepter OU refuser",
                "test_purpose": "Test nuance antonym (cas valide)",
                "expected_behavior": "ANSWER_CORRECTLY"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-076",
            "question": "Quand peut-on retirer une piece de l'echiquier sans la capturer ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["retirer", "piece", "sans capturer"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "Pieces retirees = CAPTUREES, pas juste retirees",
                "corpus_truth": "ANTONYM - Sauf promotion (retrait pion) ou irregularite",
                "test_purpose": "Test detection antonym exception",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        }
    ]

    # NEW NEGATION questions (target ~8-10)
    negation_questions = [
        {
            "id": "FR-ADV-077",
            "question": "Pourquoi le roque n'est-il pas possible si le roi a deja bouge ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["roque", "roi", "bouge"],
            "metadata": {
                "hard_type": "NEGATION",
                "hard_reason": "Question correcte - roque impossible si roi a bouge",
                "corpus_truth": "CORRECT - Art 3.8.2.1.a: roi ne doit pas avoir bouge",
                "test_purpose": "Test negation valide (SQuAD2-CR)",
                "expected_behavior": "ANSWER_CORRECTLY"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-078",
            "question": "Pourquoi ne peut-on pas promouvoir un pion en roi ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["promouvoir", "pion", "roi"],
            "metadata": {
                "hard_type": "NEGATION",
                "hard_reason": "Question correcte - promotion en roi interdite",
                "corpus_truth": "CORRECT - Art 3.7.5.1: Dame, Tour, Fou, Cavalier uniquement",
                "test_purpose": "Test negation valide",
                "expected_behavior": "ANSWER_CORRECTLY"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-079",
            "question": "Pourquoi un joueur ne peut-il pas abandonner au nom de son adversaire ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["abandonner", "adversaire"],
            "metadata": {
                "hard_type": "NEGATION",
                "hard_reason": "Question correcte - abandon = decision personnelle",
                "corpus_truth": "CORRECT - Art 5.1.1: abandon par le joueur lui-meme",
                "test_purpose": "Test negation logique",
                "expected_behavior": "ANSWER_CORRECTLY"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-080",
            "question": "Pourquoi n'est-il pas permis de jouer deux coups consecutifs ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["deux coups", "consecutifs"],
            "metadata": {
                "hard_type": "NEGATION",
                "hard_reason": "Question correcte - alternance obligatoire",
                "corpus_truth": "CORRECT - Art 1.1: joueurs jouent alternativement",
                "test_purpose": "Test negation regle fondamentale",
                "expected_behavior": "ANSWER_CORRECTLY"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-081",
            "question": "Pourquoi le pat n'est-il pas une victoire pour le joueur sans coup legal ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["pat", "victoire"],
            "metadata": {
                "hard_type": "NEGATION",
                "hard_reason": "Question correcte - pat = nulle, pas victoire",
                "corpus_truth": "CORRECT - Art 5.2.1: pat = partie nulle",
                "test_purpose": "Test negation resultat",
                "expected_behavior": "ANSWER_CORRECTLY"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-082",
            "question": "Un roi ne peut-il vraiment pas capturer une piece protegee ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["roi", "capturer", "protegee"],
            "metadata": {
                "hard_type": "NEGATION",
                "hard_reason": "Question correcte - roi ne peut se mettre en echec",
                "corpus_truth": "CORRECT - Art 3.9: roi ne peut aller en case attaquee",
                "test_purpose": "Test double negation",
                "expected_behavior": "ANSWER_CORRECTLY"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-083",
            "question": "Pourquoi ne peut-on pas annuler une partie officielle par accord mutuel apres 10 coups ?",
            "category": "tournoi",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["annuler", "accord", "10 coups"],
            "metadata": {
                "hard_type": "NEGATION",
                "hard_reason": "Faux - on PEUT proposer nulle apres x coups selon reglement",
                "corpus_truth": "FAUX - Nulles Sofia/x coups = regle tournoi, pas FIDE base",
                "test_purpose": "Test negation contextuelle",
                "expected_behavior": "CORRECT_NEGATION"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-084",
            "question": "Pourquoi n'y a-t-il pas de zeitnot en parties longues ?",
            "category": "temps",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["zeitnot", "parties longues"],
            "metadata": {
                "hard_type": "NEGATION",
                "hard_reason": "Faux - zeitnot possible dans toute cadence",
                "corpus_truth": "FAUX - Zeitnot = situation, pas dependant de cadence",
                "test_purpose": "Test negation incorrecte",
                "expected_behavior": "CORRECT_NEGATION"
            },
            "difficulty": "hard"
        }
    ]

    # NEW MUTUAL_EXCLUSION questions (target 2-3)
    mutual_exclusion_questions = [
        {
            "id": "FR-ADV-085",
            "question": "Une partie peut-elle etre simultanement gagnee par les Blancs et nulle ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["gagnee", "nulle", "simultanement"],
            "metadata": {
                "hard_type": "MUTUAL_EXCLUSION",
                "hard_reason": "Resultats mutuellement exclusifs",
                "corpus_truth": "EXCLUSION - Art 5: 1-0, 0-1, ou 1/2 mutuellement exclusifs",
                "test_purpose": "Test exclusion mutuelle (SQuAD2-CR)",
                "expected_behavior": "REJECT_CONTRADICTION"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-086",
            "question": "Un joueur peut-il etre a la fois en echec et avoir le trait sans coup legal ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["echec", "sans coup legal"],
            "metadata": {
                "hard_type": "MUTUAL_EXCLUSION",
                "hard_reason": "Echec + sans coup = mat, pas pat",
                "corpus_truth": "DISTINCTION - Echec + sans coup = MAT, sans echec = PAT",
                "test_purpose": "Test exclusion resultat",
                "expected_behavior": "CLARIFY_DISTINCTION"
            },
            "difficulty": "hard"
        },
        {
            "id": "FR-ADV-087",
            "question": "Une position peut-elle etre a la fois legale et avoir deux rois en echec ?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["deux rois", "echec", "legale"],
            "metadata": {
                "hard_type": "MUTUAL_EXCLUSION",
                "hard_reason": "Position impossible - un seul roi peut etre en echec",
                "corpus_truth": "EXCLUSION - Position illegale par definition",
                "test_purpose": "Test exclusion position",
                "expected_behavior": "REJECT_CONTRADICTION"
            },
            "difficulty": "hard"
        }
    ]

    # INTL versions
    intl_antonym = [
        {
            "id": "INTL-ADV-016",
            "question": "When must the arbiter reject a valid threefold repetition claim?",
            "category": "arbitrage",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["reject", "threefold repetition"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "Arbiter ACCEPTS (not rejects) valid claims",
                "corpus_truth": "ANTONYM - Art 9.2: arbiter VERIFIES and ACCEPTS if valid",
                "test_purpose": "Test antonym detection (SQuAD2-CR)",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        },
        {
            "id": "INTL-ADV-017",
            "question": "When can a spectator legally intervene in an ongoing game?",
            "category": "discipline",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["spectator", "intervene"],
            "metadata": {
                "hard_type": "ANTONYM",
                "hard_reason": "Spectators CANNOT intervene (never)",
                "corpus_truth": "ANTONYM - Art 12.7: spectators MUST NOT interfere",
                "test_purpose": "Test antonym prohibition",
                "expected_behavior": "CORRECT_ANTONYM"
            },
            "difficulty": "hard"
        }
    ]

    intl_negation = [
        {
            "id": "INTL-ADV-018",
            "question": "Why can't a pawn be promoted to a king?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["promote", "pawn", "king"],
            "metadata": {
                "hard_type": "NEGATION",
                "hard_reason": "Correct question - promotion to king forbidden",
                "corpus_truth": "CORRECT - Art 3.7.5.1: Queen, Rook, Bishop, Knight only",
                "test_purpose": "Test valid negation (SQuAD2-CR)",
                "expected_behavior": "ANSWER_CORRECTLY"
            },
            "difficulty": "hard"
        }
    ]

    intl_mutual = [
        {
            "id": "INTL-ADV-019",
            "question": "Can a game simultaneously be won by White and drawn?",
            "category": "regles_jeu",
            "expected_docs": [],
            "expected_pages": [],
            "keywords": ["won", "drawn", "simultaneously"],
            "metadata": {
                "hard_type": "MUTUAL_EXCLUSION",
                "hard_reason": "Mutually exclusive results",
                "corpus_truth": "EXCLUSION - Art 5: 1-0, 0-1, or 1/2 mutually exclusive",
                "test_purpose": "Test mutual exclusion (SQuAD2-CR)",
                "expected_behavior": "REJECT_CONTRADICTION"
            },
            "difficulty": "hard"
        }
    ]

    # Add all new questions
    adv["questions_fr"].extend(antonym_questions)
    adv["questions_fr"].extend(negation_questions)
    adv["questions_fr"].extend(mutual_exclusion_questions)
    adv["questions_intl"].extend(intl_antonym)
    adv["questions_intl"].extend(intl_negation)
    adv["questions_intl"].extend(intl_mutual)

    # Update statistics
    adv["statistics"]["total"] = len(adv["questions_fr"]) + len(adv["questions_intl"])
    adv["statistics"]["by_category"]["ANTONYM"] = 10 + 2  # FR + INTL
    adv["statistics"]["by_category"]["NEGATION"] = 8 + 1
    adv["statistics"]["by_category"]["MUTUAL_EXCLUSION"] = 3 + 1

    # Update version
    adv["version"] = "1.2"

    save_json(adv_path, adv)

    print("Added SQuAD2-CR categories:")
    print(f"  ANTONYM: 10 FR + 2 INTL = 12")
    print(f"  NEGATION: 8 FR + 1 INTL = 9")
    print(f"  MUTUAL_EXCLUSION: 3 FR + 1 INTL = 4")
    print(f"\nTotal adversarial: {adv['statistics']['total']}")


def main():
    """Fix all issues."""
    print("=" * 60)
    print("FIX GOLD STANDARD ISSUES")
    print("=" * 60)
    print()

    print("1. Fixing FR-Q80...")
    fix_fr_q80()
    print()

    print("2. Adding SQuAD2-CR categories...")
    add_squad2cr_questions()
    print()

    print("Done! Now run merge_adversarial.py to update gold standard.")


if __name__ == "__main__":
    main()
