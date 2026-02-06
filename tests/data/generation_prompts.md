# Gold Standard BY DESIGN - Generation Prompts

Generated: 2026-02-05T20:13:32.261211

Total chunks to process: 897
Target answerable ratio: 70%

---

## Chunk 1: J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf-p005-parent021-child00

### ANSWERABLE Prompt:

```

CHUNK (ID: J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf-p005-parent021-child00):
## 3.7.f) Participation dans plusieurs équipes
Lorsqu'un club a plusieurs équipes engagées dans une même nationale, tout joueur ou joueuse ayant participé pour le compte d'une équipe ne peut plus jouer pour le compte d'une autre.
Lorsqu'un club a plusieurs équipes engagées dans différentes nationales, un joueur ou joueuse ne peut participer dans une nationale s'il a déjà joué 4 fois en nationales supérieures.
Un joueur ou une joueuse ne peut disputer plus de 11 rondes dans le championnat.
Un joueur ou une joueuse ne peut participer à plus de 7 rondes au total en Nationale 1 ou 2. Pour disputer une ronde de Nationale 1 ou 2, un joueur ou une joueuse ne doit pas avoir joué un nombre de rondes égal ou supérieur au nombre de rondes le plus élevé que comportent sur l'ensemble de la saison les différentes divisions auquel le club a déjà participé.
Exemple : dans une zone interdépartementale, la Nationale 3 se joue en 11 rondes. Un joueur ou une joueuse a déjà participé à 9 rondes dans cette division, avant de jouer en Nationale 1. Il ou elle a le droit (malgré les 7 rondes jouées) car le club a une équipe dans une division où il y a 11 rondes.
Pour tout le paragraphe 3.7.f, tout membre d'une équipe figurant sur la feuille de match est considéré comme ayant participé au match ; les parties jouées aux échiquiers 7 et 8 comptent pour 1⁄2 partie.
Le non-respect de cet article entraine un forfait administratif sur l'échiquier du joueur ou la joueuse en infraction, avec la

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.

```

### UNANSWERABLE Prompt:

```

CHUNK CONTEXT (ID: J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf-p005-parent021-child00):
## 3.7.f) Participation dans plusieurs équipes
Lorsqu'un club a plusieurs équipes engagées dans une même nationale, tout joueur ou joueuse ayant participé pour le compte d'une équipe ne peut plus jouer pour le compte d'une autre.
Lorsqu'un club a plusieurs équipes engagées dans différentes nationales, un joueur ou joueuse ne peut participer dans une nationale s'il a déjà joué 4 fois en nationales supérieures.
Un joueur ou une joueuse ne peut disputer plus de 11 rondes dans le championnat.
Un joueur ou une joueuse ne peut participer à plus de 7 rondes au total en Nationale 1 ou 2. Pour disputer une ronde de Nationale 1 ou 2, un joueur ou une joueuse ne doit pas avoir joué un nombre de rondes égal ou supérieur au nombre de rondes le plus élevé que comportent sur l'ensemble de la saison les différentes divisions auquel le club a déjà participé.
Exemple : dans une zone interdépartementale, la Nationale 3 se joue en 11 rondes. Un joueur ou une joueuse a déjà participé à 9 rondes d

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre FFE?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles FIDE en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche pour les pendules?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait etre pris?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|FALSE_PREMISE|TEMPORAL_MISMATCH|AMBIGUOUS|COUNTERFACTUAL",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}

La question doit etre realiste et trompeuse pour un systeme RAG.

```

---

## Chunk 2: LA-octobre2025.pdf-p132-parent389-child02

### ANSWERABLE Prompt:

```

CHUNK (ID: LA-octobre2025.pdf-p132-parent389-child02):
Appariement pondéré : m éthode d'appariement dans laquelle un poids numérique est attribué à chaque paire ou flotteur descendant, conformément aux critères d'appariement, dans le but d'avoir le meilleur appariement possible. Le meilleur appariement possible est celui avec le poids le plus faible.
Avant dernier niveau à apparier (ADN) : un niveau qui a échoué au test de complétion et qui nécessite un nouvel appariement en y ajoutant des flotteurs descendants afin de compléter l'appariement du niveau des joueurs au score effondré. Pour ce groupe, un critère particulier s'applique (C.4) tandis que l'habituel critère C7 ne s'applique pas.

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.

```

### UNANSWERABLE Prompt:

```

CHUNK CONTEXT (ID: LA-octobre2025.pdf-p132-parent389-child02):
Appariement pondéré : m éthode d'appariement dans laquelle un poids numérique est attribué à chaque paire ou flotteur descendant, conformément aux critères d'appariement, dans le but d'avoir le meilleur appariement possible. Le meilleur appariement possible est celui avec le poids le plus faible.
Avant dernier niveau à apparier (ADN) : un niveau qui a échoué au test de complétion et qui nécessite un nouvel appariement en y ajoutant des flotteurs descendants afin de compléter l'appariement du niveau des joueurs au score effondré. Pour ce groupe, un critère particulier s'applique (C.4) tandis que l'habituel critère C7 ne s'applique pas.

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre FFE?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles FIDE en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche pour les pendules?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait etre pris?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|FALSE_PREMISE|TEMPORAL_MISMATCH|AMBIGUOUS|COUNTERFACTUAL",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}

La question doit etre realiste et trompeuse pour un systeme RAG.

```

---

## Chunk 3: LA-octobre2025.pdf-p124-parent358-child00

### ANSWERABLE Prompt:

```

CHUNK (ID: LA-octobre2025.pdf-p124-parent358-child00):
Si le signalement intervient après la fin de la ronde suivante, la correction aura lieu après le tournoi et uniquement dans le cadre du calcul du classement Elo du joueur/de la joueuse.
9. Lorsque l'appariement est achevé, les paires sont classées avant leur publication. Les critères de tri, dans l'ordre de priorité décroissante, sont:
2. o Le score du joueur le plus fort de la paire concernée
3. o La somme des scores des deux joueurs de la paire concernée
4. o Le rang du joueur le plus fort de l'appariement concerné, conformément à l'ordre initial (C.04.2.B)
Explication : On indique ici comment est choisi l'ordre des numéros de tables.
10. Une fois publiés, les appariements ne doivent plus être modifiés, sauf s'ils violent l'article C.04.1.b (deux joueurs ne doivent pas jouer l'un contre l'autre plus d'une fois).

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.

```

### UNANSWERABLE Prompt:

```

CHUNK CONTEXT (ID: LA-octobre2025.pdf-p124-parent358-child00):
Si le signalement intervient après la fin de la ronde suivante, la correction aura lieu après le tournoi et uniquement dans le cadre du calcul du classement Elo du joueur/de la joueuse.
9. Lorsque l'appariement est achevé, les paires sont classées avant leur publication. Les critères de tri, dans l'ordre de priorité décroissante, sont:
2. o Le score du joueur le plus fort de la paire concernée
3. o La somme des scores des deux joueurs de la paire concernée
4. o Le rang du joueur le plus fort de l'appariement concerné, conformément à l'ordre initial (C.04.2.B)
Explication : On indique ici comment est choisi l'ordre des numéros de tables.
10. Une fois publiés, les appariements ne doivent plus être modifiés, sauf s'ils violent l'article C.04.1.b (deux joueurs ne doivent pas jouer l'un contre l'autre plus d'une fois).

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre FFE?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles FIDE en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche pour les pendules?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait etre pris?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|FALSE_PREMISE|TEMPORAL_MISMATCH|AMBIGUOUS|COUNTERFACTUAL",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}

La question doit etre realiste et trompeuse pour un systeme RAG.

```

---

## Chunk 4: LA-octobre2025.pdf-p030-parent135-child00

### ANSWERABLE Prompt:

```

CHUNK (ID: LA-octobre2025.pdf-p030-parent135-child00):
## 11.7. Infractions impliquant la malhonnêteté :
- Tout officiel qui sciemment fait un faux rapport ou fournit des informations trompeuses
- Toute personne qui offre, promet, accepte ou accorde un avantage injustifié à un officiel, un arbitre, un joueur ou toute autre partie liée à la compétition dans le but de se procurer un avantage par quelque moyen que ce soit (incluant la violence, les menaces, l'intimidation, le harcèlement, la coercition, l'offre de tout type de prestation ou d'avantage financier ou autre)
- Tout joueur, ou toute personne aidant un joueur, qui délibérément - et en toute connaissance de cause -triche :
- o Par le biais de dispositifs électroniques ou d'autres sources d'informations et de conseils en cours de jeu.
- o Fausse une compétition échiquéenne par exemple par la manipulation des résultats, le sabotage, le trucage des matchs, la fraude au classement, la fausse identité, la falsification ou l'altération d'actes de naissance, et la participation délibérée à des tournois et des parties fictifs ou à des événements de ce type ou toute autre fausse information dans le but d'obtenir un avantage déloyal pour un joueur ou une équipe.
- Toute accusation ou allégation infondée de tricherie, qu'elles soient publiques ou privées, à l'encontre d'un joueur ou d'un officiel.

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.

```

### UNANSWERABLE Prompt:

```

CHUNK CONTEXT (ID: LA-octobre2025.pdf-p030-parent135-child00):
## 11.7. Infractions impliquant la malhonnêteté :
- Tout officiel qui sciemment fait un faux rapport ou fournit des informations trompeuses
- Toute personne qui offre, promet, accepte ou accorde un avantage injustifié à un officiel, un arbitre, un joueur ou toute autre partie liée à la compétition dans le but de se procurer un avantage par quelque moyen que ce soit (incluant la violence, les menaces, l'intimidation, le harcèlement, la coercition, l'offre de tout type de prestation ou d'avantage financier ou autre)
- Tout joueur, ou toute personne aidant un joueur, qui délibérément - et en toute connaissance de cause -triche :
- o Par le biais de dispositifs électroniques ou d'autres sources d'informations et de conseils en cours de jeu.
- o Fausse une compétition échiquéenne par exemple par la manipulation des résultats, le sabotage, le trucage des matchs, la fraude au classement, la fausse identité, la falsification ou l'altération d'actes de naissance, et la participation délibérée

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre FFE?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles FIDE en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche pour les pendules?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait etre pris?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|FALSE_PREMISE|TEMPORAL_MISMATCH|AMBIGUOUS|COUNTERFACTUAL",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}

La question doit etre realiste et trompeuse pour un systeme RAG.

```

---

## Chunk 5: LA-octobre2025.pdf-p046-parent171-child01

### ANSWERABLE Prompt:

```

CHUNK (ID: LA-octobre2025.pdf-p046-parent171-child01):
Autrement, une horloge doit être accrochée au mur à l'intérieur du lieu de la compétition et indiquer l'heure officielle du tournoi.
Si le délai de forfait n'est pas 0(zéro), il est conseillé que l'arbitre annonce publiquement l'heure du début de la ronde et qu'il/elle note l'heure de départ. Si le délai de forfait est par exemple de 30 minutes et que la ronde devait commencer à 15h00, mais a effectivement commencé à 15h15, les joueurs et joueuses ne perdent par forfait qu'à 15h45.
- 6.8. On considère qu'un drapeau est tombé quand l'arbitre constate le fait ou que l'un ou l'autre des deux adversaires a fait une réclamation valide à ce sujet.
Un drapeau est considéré comme tombé quand le fait est noté ou demandé, pas quand il est effectivement tombé. Si un résultat est atteint avant que l'on remarque la chute du drapeau alors ce résultat est conservé. L'arbitre doit annoncer la chute du drapeau dès qu'il/elle le remarque.
6.9. En dehors de l'application de l'un des articles 5.1.1, 5.1.2, 5.2.1, 5.2.2 ou 5.2.3, si un joueur ou une joueuse n'a pas achevé le nombre de coups prescrits dans le temps imparti, la partie est perdue par ce joueur ou cette joueuse. Toutefois, la partie est déclarée nulle, si la position est telle que son adversaire ne peut mater son roi par aucune suite de coups légaux.

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.

```

### UNANSWERABLE Prompt:

```

CHUNK CONTEXT (ID: LA-octobre2025.pdf-p046-parent171-child01):
Autrement, une horloge doit être accrochée au mur à l'intérieur du lieu de la compétition et indiquer l'heure officielle du tournoi.
Si le délai de forfait n'est pas 0(zéro), il est conseillé que l'arbitre annonce publiquement l'heure du début de la ronde et qu'il/elle note l'heure de départ. Si le délai de forfait est par exemple de 30 minutes et que la ronde devait commencer à 15h00, mais a effectivement commencé à 15h15, les joueurs et joueuses ne perdent par forfait qu'à 15h45.
- 6.8. On considère qu'un drapeau est tombé quand l'arbitre constate le fait ou que l'un ou l'autre des deux adversaires a fait une réclamation valide à ce sujet.
Un drapeau est considéré comme tombé quand le fait est noté ou demandé, pas quand il est effectivement tombé. Si un résultat est atteint avant que l'on remarque la chute du drapeau alors ce résultat est conservé. L'arbitre doit annoncer la chute du drapeau dès qu'il/elle le remarque.
6.9. En dehors de l'application de l'un des articles 5.1.

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre FFE?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles FIDE en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche pour les pendules?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait etre pris?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|FALSE_PREMISE|TEMPORAL_MISMATCH|AMBIGUOUS|COUNTERFACTUAL",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}

La question doit etre realiste et trompeuse pour un systeme RAG.

```

---

## Chunk 6: LA-octobre2025.pdf-p084-parent253-child00

### ANSWERABLE Prompt:

```

CHUNK (ID: LA-octobre2025.pdf-p084-parent253-child00):
## 10.1. Cas de force majeure
Recommandations pour les arbitres et l'équipe d'organisation d'événements échiquéens : ils doivent indiquer aux joueurs/joueuses s'il est nécessaire d'interrompre la ronde ou le tournoi en cas de force majeure. (Alerte incendie, incident grave pendant le t ournoi,...)
10.1.1. Les parties peuvent être interrompues avec la possibilité de mettre un coup sous enveloppe (ajournement), il est toutefois préférable de reprendre les parties, si possible, même après une longue pause où le risque est présent que des joueurs/joueuses puissent avoir eu accès à une aide extérieure.
Dans la mesure du possible, arbitre et organisateur mettront tout en place pour éviter les risques d'accès à une aide extérieure pendant la pause.
- 10.1.2. Si nécessaire, l'arbitre est autorisé à modifier la cadence de jeu (c'est -à-dire réduire le temps restant pour terminer les parties)
- 10.1.3. Si nécessaire, il est permis de dépasser les 12 heures de jeu dans la journée pour terminer les parties interrompues et atteindre le nombre prévu de rondes.
- 10.1.4. L'arbitre en chef et l'équipe d'organisation peuvent organiser, si nécessaire, une ronde avec une cadence de jeu différente pour terminer la journée/le tournoi avec le nombre de rondes requis.
- 10.2. Autres cas (panne de courant, température basse, bruit, conditions de jeu difficiles, absence d'arbitre, ... )

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.

```

### UNANSWERABLE Prompt:

```

CHUNK CONTEXT (ID: LA-octobre2025.pdf-p084-parent253-child00):
## 10.1. Cas de force majeure
Recommandations pour les arbitres et l'équipe d'organisation d'événements échiquéens : ils doivent indiquer aux joueurs/joueuses s'il est nécessaire d'interrompre la ronde ou le tournoi en cas de force majeure. (Alerte incendie, incident grave pendant le t ournoi,...)
10.1.1. Les parties peuvent être interrompues avec la possibilité de mettre un coup sous enveloppe (ajournement), il est toutefois préférable de reprendre les parties, si possible, même après une longue pause où le risque est présent que des joueurs/joueuses puissent avoir eu accès à une aide extérieure.
Dans la mesure du possible, arbitre et organisateur mettront tout en place pour éviter les risques d'accès à une aide extérieure pendant la pause.
- 10.1.2. Si nécessaire, l'arbitre est autorisé à modifier la cadence de jeu (c'est -à-dire réduire le temps restant pour terminer les parties)
- 10.1.3. Si nécessaire, il est permis de dépasser les 12 heures de jeu dans la journée pour ter

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre FFE?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles FIDE en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche pour les pendules?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait etre pris?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|FALSE_PREMISE|TEMPORAL_MISMATCH|AMBIGUOUS|COUNTERFACTUAL",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}

La question doit etre realiste et trompeuse pour un systeme RAG.

```

---

## Chunk 7: LA-octobre2025.pdf-p196-parent580-child00

### ANSWERABLE Prompt:

```

CHUNK (ID: LA-octobre2025.pdf-p196-parent580-child00):
## 1.4.6. Classement des adversaires
- a) La liste de classement en vigueur au début du tournoi sera utilisée (voir les exceptions en 1.1.4.) Le classement de joueurs ou joueuses appartenant à des fédérations temporairement exclues au début de la manifestation peut être déterminé en faisant appel au secrétariat de la FIDE.
- b) Pour les besoins des normes, le classement minimum (plancher ajusté de classement) des adversaires sera comme suit :
- c) Au plus un/une adversaire peut avoir son classement augmenté jusqu'au plancher ajusté de classement. Quand plus d'un /une adversaire est en dessous du plancher, le classement de l'adversaire le plus faible sera augmenté.
- d) Les joueurs et joueuses non classés et non couverts par 1.4.6b doivent être considérés comme classés 1400.
Norme de Grand Maître : 2200
Norme de Maître International : 2050
Norme de Grand Maître féminin : 2000
Norme de Maître International féminin : 1850

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.

```

### UNANSWERABLE Prompt:

```

CHUNK CONTEXT (ID: LA-octobre2025.pdf-p196-parent580-child00):
## 1.4.6. Classement des adversaires
- a) La liste de classement en vigueur au début du tournoi sera utilisée (voir les exceptions en 1.1.4.) Le classement de joueurs ou joueuses appartenant à des fédérations temporairement exclues au début de la manifestation peut être déterminé en faisant appel au secrétariat de la FIDE.
- b) Pour les besoins des normes, le classement minimum (plancher ajusté de classement) des adversaires sera comme suit :
- c) Au plus un/une adversaire peut avoir son classement augmenté jusqu'au plancher ajusté de classement. Quand plus d'un /une adversaire est en dessous du plancher, le classement de l'adversaire le plus faible sera augmenté.
- d) Les joueurs et joueuses non classés et non couverts par 1.4.6b doivent être considérés comme classés 1400.
Norme de Grand Maître : 2200
Norme de Maître International : 2050
Norme de Grand Maître féminin : 2000
Norme de Maître International féminin : 1850

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre FFE?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles FIDE en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche pour les pendules?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait etre pris?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|FALSE_PREMISE|TEMPORAL_MISMATCH|AMBIGUOUS|COUNTERFACTUAL",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}

La question doit etre realiste et trompeuse pour un systeme RAG.

```

---

## Chunk 8: LA-octobre2025.pdf-p209-parent623-child02

### ANSWERABLE Prompt:

```

CHUNK (ID: LA-octobre2025.pdf-p209-parent623-child02):
- 5.6. Des droits de titre devront être versés conformément au règlement financier de la FIDE.
- 5.6.1. La fédération nationale est responsable du paiement de ces frais.
- 5.6.2. Dans le cas décrit à l'article 5.2.2, le candidat est responsable du paiement de ces frais.
- 5.7. Un délai de 45 jours est fixé pour que les candidatures soient examinées correctement.
- 5.8. Toutes les candidatures seront diffusées en détail sur le site internet de la FIDE pendant un minimum de 60 jours avant leur validation. Ceci afin de permettre d'éventuelles objections.

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.

```

### UNANSWERABLE Prompt:

```

CHUNK CONTEXT (ID: LA-octobre2025.pdf-p209-parent623-child02):
- 5.6. Des droits de titre devront être versés conformément au règlement financier de la FIDE.
- 5.6.1. La fédération nationale est responsable du paiement de ces frais.
- 5.6.2. Dans le cas décrit à l'article 5.2.2, le candidat est responsable du paiement de ces frais.
- 5.7. Un délai de 45 jours est fixé pour que les candidatures soient examinées correctement.
- 5.8. Toutes les candidatures seront diffusées en détail sur le site internet de la FIDE pendant un minimum de 60 jours avant leur validation. Ceci afin de permettre d'éventuelles objections.

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre FFE?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles FIDE en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche pour les pendules?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait etre pris?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|FALSE_PREMISE|TEMPORAL_MISMATCH|AMBIGUOUS|COUNTERFACTUAL",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}

La question doit etre realiste et trompeuse pour un systeme RAG.

```

---

## Chunk 9: LA-octobre2025.pdf-p208-parent622-child00

### ANSWERABLE Prompt:

```

CHUNK (ID: LA-octobre2025.pdf-p208-parent622-child00):
## 4.7. A partir du 1er janvier 2024 : Participation à un (1) Séminaire d' Arbitre International et avec une évaluation d'assiduité positive.
- 4.8. Expérience en tant qu'arbitre dans quatre (4) événements, en conformité avec les articles 2.3 à 2.8, qui seront considérés valides pour une norme dans l'une des conditions suivantes :
- 4.8.1. La finale du championnat individuel (adulte, open ou féminin) national. (Deux normes maximum).
- 4.8.2. Tout tournoi ou matchs officiels de la FIDE.
- 4.8.3. Tout tournoi international dont la composition est telle qu'un joueur peut théoriquement obtenir une norme (FIDE Handbook B01).
- 4.8.4. Tout championnat Rapide ou Blitz, Mondial ou Continental (Une norme maximum).
- 4.9. Chacune des possibilités suivantes pourra être utilisée une (1) seule fois dans le cadre d'une demande de titre d'AI :
- 4.9.1. Être arbitre d'un tournoi international homologué pour le classement FIDE avec au moins 100 joueurs représentant au moins 3 fédérations, au moins 30% de joueurs classés FIDE, et au moins 7 rondes.
- 4.9.2. Être arbitre dans au moins sept (7) rondes de la plus haute division d'un championnat par équipe national vérifiant les conditions suivantes :
- a) Un minimum de 4 échiquiers par équipes.
- b) Un minimum de 10 équipes (6 en cas de toutes rondes aller-retour).

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.

```

### UNANSWERABLE Prompt:

```

CHUNK CONTEXT (ID: LA-octobre2025.pdf-p208-parent622-child00):
## 4.7. A partir du 1er janvier 2024 : Participation à un (1) Séminaire d' Arbitre International et avec une évaluation d'assiduité positive.
- 4.8. Expérience en tant qu'arbitre dans quatre (4) événements, en conformité avec les articles 2.3 à 2.8, qui seront considérés valides pour une norme dans l'une des conditions suivantes :
- 4.8.1. La finale du championnat individuel (adulte, open ou féminin) national. (Deux normes maximum).
- 4.8.2. Tout tournoi ou matchs officiels de la FIDE.
- 4.8.3. Tout tournoi international dont la composition est telle qu'un joueur peut théoriquement obtenir une norme (FIDE Handbook B01).
- 4.8.4. Tout championnat Rapide ou Blitz, Mondial ou Continental (Une norme maximum).
- 4.9. Chacune des possibilités suivantes pourra être utilisée une (1) seule fois dans le cadre d'une demande de titre d'AI :
- 4.9.1. Être arbitre d'un tournoi international homologué pour le classement FIDE avec au moins 100 joueurs représentant au moins 3 fédérations, au moins 30

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre FFE?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles FIDE en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche pour les pendules?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait etre pris?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|FALSE_PREMISE|TEMPORAL_MISMATCH|AMBIGUOUS|COUNTERFACTUAL",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}

La question doit etre realiste et trompeuse pour un systeme RAG.

```

---

## Chunk 10: LA-octobre2025.pdf-p075-parent229-child00

### ANSWERABLE Prompt:

```

CHUNK (ID: LA-octobre2025.pdf-p075-parent229-child00):
## 8. Départage et parties non jouées
Note de traduction : L'article de la FIDE n'étant pas à jour, la direction des règlements conseille la lecture de l'article 14 du C.07 Tie-Break Régulations.

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.

```

### UNANSWERABLE Prompt:

```

CHUNK CONTEXT (ID: LA-octobre2025.pdf-p075-parent229-child00):
## 8. Départage et parties non jouées
Note de traduction : L'article de la FIDE n'étant pas à jour, la direction des règlements conseille la lecture de l'article 14 du C.07 Tie-Break Régulations.

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre FFE?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles FIDE en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche pour les pendules?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait etre pris?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|FALSE_PREMISE|TEMPORAL_MISMATCH|AMBIGUOUS|COUNTERFACTUAL",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}

La question doit etre realiste et trompeuse pour un systeme RAG.

```

---
