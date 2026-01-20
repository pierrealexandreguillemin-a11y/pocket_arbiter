# Hard Cases Vérifiés - Validation Manuelle

> 6 premiers hard cases avec citations EXACTES du corpus
> À valider avant intégration au gold standard

---

## Hard Case #1 - Pat (LA-octobre2025.pdf p.43)

**Question**: Mon roi peut plus bouger mais il est pas en échec, c'est quoi ?

**Citation exacte corpus**:
> 5.2.1. La partie est nulle lorsque le joueur ou la joueuse au trait ne dispose d'aucun coup légal et que son roi n'est pas en échec. On dit alors que la partie se termine par le "pat".

**Hard reason**: Langage oral "peut plus bouger" ≠ terme corpus "ne dispose d'aucun coup légal"

---

## Hard Case #2 - Téléphone éteint (LA-octobre2025.pdf p.54)

**Question**: Mon adversaire a son téléphone dans sa poche mais il est éteint, c'est grave ?

**Citation exacte corpus**:
> Si un téléphone mobile (même éteint) est découvert sur un joueur, sa partie est immédiatement perdue et son adversaire gagne.

**Hard reason**: Question sur cas limite "éteint" - règle stricte souvent mal comprise

---

## Hard Case #3 - Couleurs inversées (LA-octobre2025.pdf p.48)

**Question**: On a joué 12 coups avec les mauvaises couleurs, on recommence la partie ?

**Citation exacte corpus**:
> 7.3. Si une partie a débuté avec des couleurs inversées et que les deux adversaires ont joué moins de dix coups, alors la partie est interrompue et une nouvelle partie est jouée avec les couleurs correctes. Après dix coups ou plus, la partie continue.

**Hard reason**: Question numérique "12 coups" vs seuil "dix coups" - réponse = NON, on continue

---

## Hard Case #4 - Cadence jeunes (J01 p.4)

**Question**: C'est quoi la cadence pour les poussins au championnat de France ?

**Citation exacte corpus**:
> U8, U8F, U10, U10F : 50 minutes pour toute la partie avec ajout de 10 sec./coup.

**Hard reason**: Synonyme "poussins" vs "U8/U10" - terminologie différente

---

## Hard Case #5 - Accessibilité handicap (H02 p.1)

**Question**: Si un joueur en fauteuil roulant ne peut pas accéder à la salle, qui perd la partie ?

**Citation exacte corpus**:
> Aucun match impliquant des joueuses et joueurs handicapés ne doit se dérouler dans des locaux inaccessibles. Faute de respecter ce dernier principe, l'équipe fautive perdra la partie concernée.

**Hard reason**: Question pratique terrain + terme "fauteuil roulant" vs "mobilité réduite"

---

## Hard Case #6 - Descente N2 (A02 p.2)

**Question**: Une équipe finit 9ème en Nationale 2, elle descend ou pas ?

**Citation exacte corpus**:
> Les équipes classées 9ème et 10ème de chaque groupe de N2 descendent en division inférieure.

**Hard reason**: Question numérique directe "9ème" + langage oral "elle descend ou pas"

---

## Diversité des sources

| Source | Count |
|--------|-------|
| LA-octobre2025.pdf | 3 |
| J01_Championnat_Jeunes | 1 |
| H02_Mobilite_reduite | 1 |
| A02_Championnat_Clubs | 1 |

---

---

## Hard Case #7 - Négation roque (LA-octobre2025.pdf p.42) - THÈME 1

**Question**: Si je touche ma tour AVANT mon roi, j'ai pas le droit de roquer ?

**Citation exacte corpus**:
> 4.4.2. touche délibérément une de ses tours puis son roi, il/elle n'a pas l'autorisation de roquer de ce côté lors de ce coup, et la situation doit être traitée selon l'Article 4.3.1

**Hard reason**: NÉGATION "pas le droit" + ordre inverse tour/roi - retriever doit comprendre la négation

---

## Hard Case #8 - Question composée pat/répétition (p.43 + p.51) - THÈME 5

**Question**: C'est quoi la différence entre le pat et la nulle par répétition ?

**Citations exactes corpus**:
> Page 43: 5.2.1. La partie est nulle lorsque le joueur ou la joueuse au trait ne dispose d'aucun coup légal et que son roi n'est pas en échec. On dit alors que la partie se termine par le "pat".

> Page 51: 9.2. La partie est nulle, sur une demande correcte du joueur/de la joueuse ayant le trait, lorsque la même position, pour la troisième fois au moins

**Hard reason**: QUESTION COMPOSÉE - nécessite retrieval sur 2 pages différentes (43 ET 51)

---

## Hard Case #9 - Synonyme zeitnot (p.50) - THÈME 3

**Question**: Qu'est-ce qui se passe quand un joueur est en zeitnot ?

**Citation exacte corpus**:
> Page 50: Il arrive assez souvent qu'un joueur ou une joueuse en manque de temps demande à l'arbitre combien de coups il lui reste à jouer avant le contrôle de temps

**Hard reason**: SYNONYME "zeitnot" absent du corpus - terme officiel = "manque de temps"

**Note audit**: Corrigé le 2026-01-20. "Fischer" était présent dans corpus p.45/69/147

---

## Couverture des 5 thèmes

| Thème | Hard Case | Couvert |
|-------|-----------|---------|
| 1. Négations | #7 (roque) | ✓ |
| 2. Langage oral | #1, #6 | ✓ |
| 3. Synonymes | #4, #5, #9 | ✓ |
| 4. Numériques | #3, #6 | ✓ |
| 5. Composées | #8 | ✓ |

---

*À valider avant intégration - Pocket Arbiter*
