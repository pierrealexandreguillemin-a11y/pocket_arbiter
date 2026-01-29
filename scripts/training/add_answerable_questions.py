"""
Add answerable questions to Gold Standard - Pocket Arbiter

Adds new answerable questions to reach target 33% unanswerable ratio.
Questions are based on actual corpus content with proper classifications.

ISO Reference: ISO/IEC 42001 A.6.2.2, ISO/IEC 25010 S4.2
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

# Question templates by category
QUESTION_TEMPLATES_FR = [
    # ARBITRAGE (10 questions)
    {
        "question": "Quelle est la composition de la Direction Nationale de l'Arbitrage (DNA) ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [227],
        "keywords": ["DNA", "direction", "nationale", "arbitrage", "composition"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelles sont les missions de la DNA ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [13],
        "keywords": ["DNA", "mission", "arbitrage", "direction"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment sont défraiés les arbitres lors d'un tournoi ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [10],
        "keywords": ["arbitre", "défraiement", "frais", "tournoi"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quand l'arbitre peut-il interrompre les parties en cours ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [10],
        "keywords": ["arbitre", "interrompre", "partie", "interruption"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "CONDITIONAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Comment l'arbitre gère-t-il les perturbateurs pendant une compétition ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [10],
        "keywords": ["arbitre", "perturbateur", "exclure", "compétition"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelles sont les responsabilités de l'arbitre concernant le confort des joueurs ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [10],
        "keywords": ["arbitre", "confort", "joueur", "chauffage", "éclairage"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Que doit vérifier l'arbitre concernant le matériel avant un tournoi ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [10],
        "keywords": ["arbitre", "matériel", "quantité", "remplacement"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "APPLY",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "Quelles informations l'arbitre doit-il afficher lors d'un tournoi ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [10],
        "keywords": ["arbitre", "affichage", "information", "règlement"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "Quelles sont les relations entre l'arbitre et les instances fédérales ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [10],
        "keywords": ["arbitre", "instances", "fédéral", "transmettre", "résultats"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment est organisé le secteur de l'arbitrage dans la FFE ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [13],
        "keywords": ["arbitrage", "FFE", "organisation", "DNA"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    # TOURNOI (10 questions)
    {
        "question": "Quelles sont les formules de jeu possibles pour un championnat ?",
        "category": "tournoi",
        "expected_docs": ["A01_2025_26_Championnat_de_France.pdf"],
        "expected_pages": [3],
        "keywords": ["formule", "championnat", "toutes-rondes", "suisse"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "Quels sont les droits d'engagement pour le Championnat de France ?",
        "category": "tournoi",
        "expected_docs": ["A01_2025_26_Championnat_de_France.pdf"],
        "expected_pages": [1],
        "keywords": ["droits", "engagement", "championnat", "France"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    {
        "question": "Comment fonctionne le système d'appariement dans un tournoi suisse ?",
        "category": "tournoi",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [102],
        "keywords": ["appariement", "suisse", "tournoi", "système"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quelles sont les règles de départage en cas d'égalité de points ?",
        "category": "tournoi",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [150, 151],
        "keywords": ["départage", "égalité", "points", "Buchholz"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "ANALYZE",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment sont organisées les compétitions toutes-rondes ?",
        "category": "tournoi",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [100, 101],
        "keywords": ["toutes-rondes", "round-robin", "compétition"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelles sont les conditions d'homologation d'un tournoi ?",
        "category": "tournoi",
        "expected_docs": ["R03_2025_26_Competitions_homologuees.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["homologation", "tournoi", "conditions", "FFE"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment fonctionne la Coupe de France des échecs ?",
        "category": "tournoi",
        "expected_docs": ["C01_2025_26_Coupe_de_France.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["coupe", "France", "équipe", "élimination"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quelles sont les règles de la Coupe de la parité ?",
        "category": "tournoi",
        "expected_docs": ["C04_2025_26_Coupe_de_la_parit_.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["coupe", "parité", "mixte", "équipe"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment s'organisent les interclubs départementaux ?",
        "category": "tournoi",
        "expected_docs": ["Interclubs_DepartementalBdr.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["interclub", "départemental", "équipe", "division"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelles sont les différentes divisions dans les championnats par équipes ?",
        "category": "tournoi",
        "expected_docs": ["A02_2025_26_Championnat_de_France_des_Clubs.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["division", "Top", "N1", "N2", "équipe"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    # REGLES_JEU (10 questions)
    {
        "question": "Comment se déroule le roque côté roi (petit roque) ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [23, 24],
        "keywords": ["roque", "roi", "tour", "petit"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quelles sont les conditions pour qu'une position soit déclarée nulle ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [28, 29],
        "keywords": ["nul", "nulle", "position", "match", "insuffisant"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "ANALYZE",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment fonctionne la prise en passant ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [22],
        "keywords": ["passant", "pion", "prise", "capture"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quand le roi est-il considéré en échec ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [41],
        "keywords": ["roi", "échec", "attaqué", "pièce"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "Qu'est-ce qu'un coup légal selon les règles FIDE ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [41],
        "keywords": ["coup", "légal", "règles", "conditions"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelles pièces peut-on obtenir lors de la promotion d'un pion ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [22],
        "keywords": ["promotion", "pion", "dame", "tour", "fou", "cavalier"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    {
        "question": "Comment se termine une partie par pat ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [28, 29],
        "keywords": ["pat", "nul", "mouvement", "légal"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "Quelle est la règle des 50 coups ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [40, 41],
        "keywords": ["50", "coups", "nulle", "pion", "prise"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Comment réclamer la nulle par triple répétition ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [40, 41],
        "keywords": ["triple", "répétition", "nulle", "position"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Qu'est-ce qu'une position illégale ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [41],
        "keywords": ["position", "illégale", "coup", "légal"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    # TEMPS (6 questions)
    {
        "question": "Qu'est-ce que le contrôle de temps Fischer ?",
        "category": "temps",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [30, 31],
        "keywords": ["Fischer", "incrément", "temps", "contrôle"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quelle est la cadence pour les parties rapides ?",
        "category": "temps",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [49, 50],
        "keywords": ["rapide", "cadence", "minutes", "temps"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    {
        "question": "Comment fonctionne le temps additionnel (Bronstein) ?",
        "category": "temps",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [30, 31],
        "keywords": ["Bronstein", "temps", "additionnel", "délai"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quelle est la cadence pour les parties blitz ?",
        "category": "temps",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [51],
        "keywords": ["blitz", "cadence", "minutes", "temps"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    {
        "question": "Que se passe-t-il quand le temps d'un joueur est écoulé (drapeau tombé) ?",
        "category": "temps",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [33],
        "keywords": ["drapeau", "temps", "écoulé", "perdre"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "CONDITIONAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment gère-t-on le zeitnot (manque de temps) ?",
        "category": "temps",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [38, 39],
        "keywords": ["zeitnot", "temps", "notation", "trouble"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    # CLASSEMENT (6 questions)
    {
        "question": "Comment est calculé le classement Elo ?",
        "category": "classement",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [164, 165],
        "keywords": ["Elo", "classement", "calcul", "rating"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "ANALYZE",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Qu'est-ce que le classement rapide ?",
        "category": "classement",
        "expected_docs": ["E02-Le_classement_rapide.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["rapide", "classement", "Elo", "parties"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "Comment sont homologuées les parties pour le classement ?",
        "category": "classement",
        "expected_docs": ["R03_2025_26_Competitions_homologuees.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["homologation", "classement", "partie", "FFE"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelles sont les conditions pour obtenir un classement initial ?",
        "category": "classement",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [164, 165],
        "keywords": ["initial", "classement", "Elo", "première"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelle est la différence entre Elo FFE et Elo FIDE ?",
        "category": "classement",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [164, 165],
        "keywords": ["Elo", "FFE", "FIDE", "classement"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "ANALYZE",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Comment évolue le coefficient K selon le classement ?",
        "category": "classement",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [164, 165],
        "keywords": ["coefficient", "K", "Elo", "calcul"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    # JEUNES (6 questions)
    {
        "question": "Quelles sont les catégories d'âge pour les jeunes ?",
        "category": "jeunes",
        "expected_docs": ["J01_2025_26_Championnat_de_France_Jeunes.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["catégorie", "âge", "jeunes", "pupille", "benjamin"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    {
        "question": "Comment se qualifie-t-on pour le Championnat de France Jeunes ?",
        "category": "jeunes",
        "expected_docs": ["J01_2025_26_Championnat_de_France_Jeunes.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["qualification", "championnat", "jeunes", "France"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment fonctionnent les interclubs jeunes ?",
        "category": "jeunes",
        "expected_docs": ["J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["interclub", "jeunes", "équipe", "championnat"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelles sont les règles du championnat scolaire ?",
        "category": "jeunes",
        "expected_docs": ["J03_2025_26_Championnat_de_France_scolaire.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["scolaire", "championnat", "école", "jeunes"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "À quel âge peut-on participer à la catégorie Poussin ?",
        "category": "jeunes",
        "expected_docs": ["J01_2025_26_Championnat_de_France_Jeunes.pdf"],
        "expected_pages": [1],
        "keywords": ["poussin", "âge", "catégorie", "année"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    {
        "question": "Quelles sont les cadences utilisées pour les compétitions jeunes ?",
        "category": "jeunes",
        "expected_docs": ["J01_2025_26_Championnat_de_France_Jeunes.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["cadence", "jeunes", "temps", "contrôle"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    # FEMININ (4 questions)
    {
        "question": "Comment fonctionne le championnat féminin par équipes ?",
        "category": "feminin",
        "expected_docs": ["F01_2025_26_Championnat_de_France_des_clubs_Feminin.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["féminin", "championnat", "équipe", "clubs"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelles sont les conditions de participation au championnat féminin individuel ?",
        "category": "feminin",
        "expected_docs": [
            "F02_2025_26_Championnat_individuel_Feminin_parties_rapides.pdf"
        ],
        "expected_pages": [1, 2],
        "keywords": ["féminin", "championnat", "individuel", "participation"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelle est la cadence pour le championnat féminin rapide ?",
        "category": "feminin",
        "expected_docs": [
            "F02_2025_26_Championnat_individuel_Feminin_parties_rapides.pdf"
        ],
        "expected_pages": [1],
        "keywords": ["féminin", "rapide", "cadence", "championnat"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    {
        "question": "Comment la FFE promeut-elle les échecs féminins ?",
        "category": "feminin",
        "expected_docs": ["F01_2025_26_Championnat_de_France_des_clubs_Feminin.pdf"],
        "expected_pages": [1],
        "keywords": ["féminin", "FFE", "promotion", "développement"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    # HANDICAP (4 questions)
    {
        "question": "Quelles sont les adaptations pour les joueurs handicapés ?",
        "category": "handicap",
        "expected_docs": ["H01_2025_26_Conduite_pour_joueur_handicapes.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["handicap", "adaptation", "joueur", "conduite"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelles sont les règles pour les joueurs à mobilité réduite ?",
        "category": "handicap",
        "expected_docs": ["H02_2025_26_Joueurs_a_mobilite_reduite.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["mobilité", "réduite", "joueur", "adaptation"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment un joueur malvoyant peut-il participer à une compétition ?",
        "category": "handicap",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [55, 56],
        "keywords": ["malvoyant", "aveugle", "assistant", "échiquier"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quel matériel spécifique est prévu pour les joueurs non-voyants ?",
        "category": "handicap",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [55, 56],
        "keywords": ["non-voyant", "matériel", "échiquier", "pièces"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    # DISCIPLINE (5 questions)
    {
        "question": "Quelles sont les sanctions disciplinaires possibles ?",
        "category": "discipline",
        "expected_docs": ["2018_Reglement_Disciplinaire20180422.pdf"],
        "expected_pages": [7],
        "keywords": ["sanction", "disciplinaire", "suspension", "exclusion"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment fonctionne la Commission Fédérale de Discipline ?",
        "category": "discipline",
        "expected_docs": ["2018_Reglement_Disciplinaire20180422.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["commission", "discipline", "fédéral", "CFD"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quels comportements sont considérés comme fautes disciplinaires ?",
        "category": "discipline",
        "expected_docs": ["2018_Reglement_Disciplinaire20180422.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["faute", "disciplinaire", "comportement", "interdit"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "ANALYZE",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Comment faire appel d'une décision disciplinaire ?",
        "category": "discipline",
        "expected_docs": ["2018_Reglement_Disciplinaire20180422.pdf"],
        "expected_pages": [2, 3],
        "keywords": ["appel", "décision", "discipline", "recours"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelle est la durée maximale d'une suspension ?",
        "category": "discipline",
        "expected_docs": ["2018_Reglement_Disciplinaire20180422.pdf"],
        "expected_pages": [7],
        "keywords": ["suspension", "durée", "maximum", "sanction"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    # ADMINISTRATION (5 questions)
    {
        "question": "Quels sont les statuts de la FFE ?",
        "category": "administration",
        "expected_docs": ["2024_Statuts20240420.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["statut", "FFE", "fédération", "français"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment fonctionne le règlement intérieur de la FFE ?",
        "category": "administration",
        "expected_docs": ["2025_Reglement_Interieur_20250503.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["règlement", "intérieur", "FFE", "fonctionnement"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Qu'est-ce que le contrat de délégation FFE ?",
        "category": "administration",
        "expected_docs": ["Contrat_de_delegation_15032022.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["contrat", "délégation", "FFE", "ministère"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quelles sont les règles financières de la FFE ?",
        "category": "administration",
        "expected_docs": ["2023_Reglement_Financier20230610.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["financier", "règlement", "FFE", "budget"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quelles sont les obligations médicales pour les compétitions ?",
        "category": "administration",
        "expected_docs": ["2022_Reglement_medical_19082022.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["médical", "règlement", "compétition", "surveillance"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    # NOTATION (5 questions)
    {
        "question": "Comment noter une partie d'échecs selon la notation algébrique ?",
        "category": "notation",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [38, 39],
        "keywords": ["notation", "algébrique", "coup", "écrire"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quand peut-on arrêter de noter les coups ?",
        "category": "notation",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [38, 39],
        "keywords": ["noter", "coups", "arrêter", "temps"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "CONDITIONAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quels symboles utilise-t-on pour les pièces en notation ?",
        "category": "notation",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [38],
        "keywords": ["symbole", "pièce", "notation", "lettre"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    {
        "question": "Comment noter le roque dans la notation algébrique ?",
        "category": "notation",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [38, 39],
        "keywords": ["roque", "notation", "O-O", "algébrique"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "Comment noter une prise en notation algébrique ?",
        "category": "notation",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [38, 39],
        "keywords": ["prise", "notation", "x", "capture"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    # TITRES (5 questions)
    {
        "question": "Comment obtenir le titre d'Arbitre Fédéral 1 (AF1) ?",
        "category": "titres",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [14, 15],
        "keywords": ["AF1", "arbitre", "fédéral", "titre"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Quels sont les différents titres d'arbitre en France ?",
        "category": "titres",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [14, 15],
        "keywords": ["titre", "arbitre", "AF1", "AF2", "AF3"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "Comment obtenir le titre d'Arbitre International (AI) ?",
        "category": "titres",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [168, 169],
        "keywords": ["arbitre", "international", "AI", "FIDE"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quelles sont les conditions pour le titre de Maître FIDE (FM) ?",
        "category": "titres",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [168, 169],
        "keywords": ["FM", "FIDE", "maître", "titre", "Elo"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment sont attribués les titres de Grand Maître ?",
        "category": "titres",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [168, 169],
        "keywords": ["GM", "grand", "maître", "titre", "norme"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    # 5 QUESTIONS SUPPLEMENTAIRES
    {
        "question": "Comment fonctionne le système Buchholz pour le départage ?",
        "category": "tournoi",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [150, 151],
        "keywords": ["Buchholz", "départage", "adversaire", "points"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "ANALYZE",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quelles sont les règles pour les parties ajournées ?",
        "category": "regles_jeu",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [57, 58],
        "keywords": ["ajournement", "scellé", "enveloppe", "reprise"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment l'arbitre vérifie-t-il le matériel électronique des joueurs ?",
        "category": "arbitrage",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [44, 45],
        "keywords": ["électronique", "téléphone", "vérification", "triche"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "Quelles sont les pénalités pour comportement antisportif ?",
        "category": "discipline",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [45, 46],
        "keywords": ["antisportif", "pénalité", "comportement", "sanction"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "Comment sont organisées les rondes dans un tournoi suisse ?",
        "category": "tournoi",
        "expected_docs": ["LA-octobre2025.pdf"],
        "expected_pages": [102, 103],
        "keywords": ["ronde", "suisse", "appariement", "tournoi"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
]

# INTL Question templates (31 questions based on FIDE Laws)
QUESTION_TEMPLATES_INTL = [
    # FIDE Laws - Articles 1-5 (10 questions)
    {
        "question": "What is the initial position of the pieces on the chessboard?",
        "category": "laws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [18],
        "keywords": ["initial", "position", "pieces", "chessboard"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "How does the knight move according to FIDE Laws?",
        "category": "laws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [20, 21],
        "keywords": ["knight", "move", "squares", "L-shape"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "What are the rules for castling in FIDE Laws?",
        "category": "laws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [23, 24],
        "keywords": ["castling", "king", "rook", "conditions"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "What happens when a pawn reaches the eighth rank?",
        "category": "laws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [21, 22],
        "keywords": ["pawn", "promotion", "eighth", "rank"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "What is the touch-move rule according to Article 4?",
        "category": "laws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [25, 26],
        "keywords": ["touch", "move", "piece", "Article 4"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "When is a move considered completed according to FIDE Laws?",
        "category": "laws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [26],
        "keywords": ["move", "completed", "clock", "press"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "CONDITIONAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "What are the conditions for checkmate?",
        "category": "laws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [28],
        "keywords": ["checkmate", "king", "escape", "game"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "What is stalemate and when does it occur?",
        "category": "laws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [28, 29],
        "keywords": ["stalemate", "draw", "legal", "move"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "How does en passant capture work?",
        "category": "laws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [21, 22],
        "keywords": ["en passant", "pawn", "capture", "adjacent"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "What is the purpose of the 'j'adoube' or 'I adjust' declaration?",
        "category": "laws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [25],
        "keywords": ["adjust", "j'adoube", "piece", "touch"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    # Time control - Article 6 (6 questions)
    {
        "question": "How should the chess clock be handled according to Article 6?",
        "category": "time",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [29, 30],
        "keywords": ["clock", "handle", "press", "hand"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "What is the default time for arriving late to a game?",
        "category": "time",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [32],
        "keywords": ["default", "time", "late", "forfeit"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    {
        "question": "What happens when a flag falls on the chess clock?",
        "category": "time",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [31],
        "keywords": ["flag", "falls", "time", "expire"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "CONDITIONAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "What are the requirements for chess clock settings?",
        "category": "time",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [33],
        "keywords": ["clock", "settings", "increment", "time"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "What is Fischer time control?",
        "category": "time",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [29, 30],
        "keywords": ["Fischer", "increment", "time", "control"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "What is Bronstein time delay?",
        "category": "time",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [29, 30],
        "keywords": ["Bronstein", "delay", "time", "control"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    # Irregularities - Article 7 (4 questions)
    {
        "question": "What should happen when an illegal move is discovered?",
        "category": "irregularities",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [35, 36],
        "keywords": ["illegal", "move", "penalty", "discovered"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "What happens if the initial position was incorrect?",
        "category": "irregularities",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [35],
        "keywords": ["initial", "position", "incorrect", "restart"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "What is the penalty for making an illegal move?",
        "category": "irregularities",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [35, 36],
        "keywords": ["illegal", "move", "penalty", "time"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    {
        "question": "How does the arbiter handle displacement of pieces?",
        "category": "irregularities",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [36, 37],
        "keywords": ["displacement", "pieces", "arbiter", "position"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    # Recording - Article 8 (3 questions)
    {
        "question": "When is move recording required according to FIDE Laws?",
        "category": "recording",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [38, 39],
        "keywords": ["recording", "moves", "scoresheet", "required"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "CONDITIONAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "What notation system is used for recording moves?",
        "category": "recording",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [38],
        "keywords": ["notation", "algebraic", "recording", "moves"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "FACTUAL",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "LEXICAL_MATCH",
        },
    },
    {
        "question": "When can a player stop recording moves?",
        "category": "recording",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [38, 39],
        "keywords": ["stop", "recording", "time", "trouble"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "CONDITIONAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    # Drawn game - Article 9 (4 questions)
    {
        "question": "What is the 50-move rule for claiming a draw?",
        "category": "draws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [40, 41],
        "keywords": ["50", "moves", "draw", "claim"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "DEFINITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "How does one claim a draw by threefold repetition?",
        "category": "draws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [40],
        "keywords": ["threefold", "repetition", "draw", "claim"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "What are the conditions for a draw by insufficient material?",
        "category": "draws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [28, 29],
        "keywords": ["insufficient", "material", "draw", "checkmate"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "ANALYZE",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
    {
        "question": "When does the 75-move rule apply for automatic draw?",
        "category": "draws",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [40, 41],
        "keywords": ["75", "moves", "automatic", "draw"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "CONDITIONAL",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "SINGLE_SENTENCE",
        },
    },
    # Arbiter duties - Article 11-12 (4 questions)
    {
        "question": "What are the main duties of the arbiter according to Article 12?",
        "category": "arbiter",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [47],
        "keywords": ["arbiter", "duties", "Article 12", "shall"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "What behavior is forbidden for players according to Article 11?",
        "category": "conduct",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [45],
        "keywords": ["forbidden", "distract", "annoy", "opponent"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "REMEMBER",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "What powers does the arbiter have for penalizing players?",
        "category": "arbiter",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [47, 48],
        "keywords": ["arbiter", "penalty", "power", "sanction"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "LIST",
            "cognitive_level": "UNDERSTAND",
            "reasoning_type": "MULTI_SENTENCE",
        },
    },
    {
        "question": "How does the arbiter handle electronic device violations?",
        "category": "arbiter",
        "expected_docs": ["FIDE_Arbiters_Manual_2025.pdf"],
        "expected_pages": [45],
        "keywords": ["electronic", "device", "phone", "violation"],
        "metadata": {
            "hard_type": "ANSWERABLE",
            "answer_type": "PROCEDURAL",
            "cognitive_level": "APPLY",
            "reasoning_type": "DOMAIN_KNOWLEDGE",
        },
    },
]


def main():
    parser = argparse.ArgumentParser(description="Add answerable questions")
    parser.add_argument("--gs", type=Path, required=True, help="Gold Standard JSON")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output JSON")
    parser.add_argument("--corpus", type=str, default="fr", help="Corpus (fr or intl)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write output")
    args = parser.parse_args()

    # Load existing GS
    with open(args.gs, encoding="utf-8") as f:
        gs = json.load(f)

    # Get last question number
    existing_ids = [q["id"] for q in gs["questions"]]
    if args.corpus == "fr":
        last_num = max(
            [
                int(id.split("-")[1][1:])
                for id in existing_ids
                if id.startswith("FR-Q") and not id.startswith("FR-Q-")
            ]
            + [0]
        )
        templates = QUESTION_TEMPLATES_FR
        id_prefix = "FR-Q"
    else:
        last_num = max(
            [int(id.split("_")[1]) for id in existing_ids if id.startswith("intl_")]
            + [0]
        )
        templates = QUESTION_TEMPLATES_INTL
        id_prefix = "intl_"

    print(f"Corpus: {args.corpus.upper()}")
    print(f"Existing questions: {len(gs['questions'])}")
    print(f"Last {id_prefix} number: {last_num}")
    print(f"Questions to add: {len(templates)}")

    # Create new questions
    new_questions = []
    for i, template in enumerate(templates, start=1):
        q = template.copy()
        if args.corpus == "fr":
            q["id"] = f"FR-Q{last_num + i:03d}"
        else:
            q["id"] = f"intl_{last_num + i:03d}"
        q["validation"] = {"status": "PENDING", "method": "auto_generated"}
        q["audit"] = f"auto_{datetime.now().strftime('%Y-%m-%d')}"
        new_questions.append(q)

    print(f"Created {len(new_questions)} new questions")

    if args.dry_run:
        print("\nDry run - not writing output")
        print(
            f"Sample question: {json.dumps(new_questions[0], indent=2, ensure_ascii=False)}"
        )
        return

    # Add to GS
    gs["questions"].extend(new_questions)

    # Update metadata
    old_version = gs.get("version", "5.29")
    new_version = "5.30" if args.corpus == "fr" else "2.4"
    gs["version"] = new_version
    gs["description"] = (
        f"Gold standard v{new_version} - {len(gs['questions'])} questions (added {len(new_questions)} answerable)"
    )

    # Update statistics
    answerable = sum(
        1
        for q in gs["questions"]
        if q.get("metadata", {}).get("hard_type", "ANSWERABLE") == "ANSWERABLE"
    )
    unanswerable = len(gs["questions"]) - answerable
    if "statistics" not in gs:
        gs["statistics"] = {}
    gs["statistics"]["total_questions"] = len(gs["questions"])
    gs["statistics"]["adversarial_questions"] = unanswerable
    gs["statistics"]["adversarial_ratio"] = (
        f"{100 * unanswerable / len(gs['questions']):.1f}%"
    )

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {args.output}")
    print(f"Version: {old_version} -> {new_version}")
    print(f"Total questions: {len(gs['questions'])}")
    print(f"Answerable: {answerable} ({100*answerable/len(gs['questions']):.1f}%)")
    print(
        f"Unanswerable: {unanswerable} ({100*unanswerable/len(gs['questions']):.1f}%)"
    )


if __name__ == "__main__":
    main()
