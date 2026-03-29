# Retrieval Table Gap Analysis — 2026-03-29

## Finding

Le retrieval pipeline retrouve les tables pour seulement 21/298 questions (7%).
Mais l'analyse du corpus montre que BEAUCOUP plus de questions necessitent des
donnees tabulaires pour repondre correctement.

## Root cause

Le GS pointe vers la page PROSE qui decrit la regle. Le tableau recapitulatif
est souvent sur une AUTRE page du meme document. Le search trouve la prose
mais pas le tableau associe.

Exemples :
- Q00-Q07 (Elo) : la prose decrit les regles de calcul, le tableau p.150/184/189/196/197 LA
  contient les tables de conversion dp → probabilite
- Q08-Q09 (cadences U8/U10) : la prose mentionne les cadences, le tableau p.5 R01
  liste TOUTES les cadences par categorie
- Q10 (Elo U12) : la prose decrit les qualifications, le tableau p.2 R01 liste
  les categories d'age avec seuils
- Q11-Q14 (composition equipes) : la prose decrit les regles, les tableaux p.106-117 LA
  donnent les grilles Berger exactes

## Tables cles dans la DB (111 total, sous-ensemble critique)

| Table | Source | Page | Contenu | Questions impactees |
|-------|--------|------|---------|-------------------|
| Categories age | R01 | 2 | U8-S65 avec ages | Q03, Q06, Q08-Q10 |
| Equivalence cadences | R01 | 5 | A-D avec temps | Q08, Q09, Q12-Q15 |
| Conversion Elo | LA | 150, 184, 189, 197 | dp → probabilite | Q00-Q02, Q07, Q20 |
| Baremes titres FIDE | LA | 186 | GM 2500, MI 2400 | Q20 |
| Arrondis Elo | LA | 196 | Niveaux min avant/apres | Q03-Q07 |
| Berger equipes | LA | 106-117 | Grilles appariements | Q11, Q32 |
| Qualif arbitres | LA | 179-180 | Par type competition | — |
| Bareme frais | Reglement Financier | 5 | 8 tranches distance | — |

## Etat des table_rows (row-as-chunk)

1355 table_rows avec embeddings EXISTENT dans corpus_v2_fr.db.
DESACTIVEES dans search.py ligne 34 :
```python
# "table_rows" disabled: -6pp R@5 regression (see row_as_chunk_experiment.md)
```

L'experiment row-as-chunk a ete fait avec le 270M (garbage). Le 1B pourrait
reagir differemment aux rows. A re-tester.

## Piste d'amelioration

Le probleme n'est pas que les tables ne sont pas dans la DB — elles y sont,
embeddees, avec structured_cells et FTS5. Le probleme est que le search
ne les retrouve pas quand la question porte sur le SUJET du tableau
plutot que sur son CONTENU exact.

Options a explorer :
1. Re-activer table_rows dans cosine_search pour le 1B
2. Augmenter le poids des table_summaries dans le RRF
3. Ajouter un matching par sujet (question sur "cadence" → table cadences)
4. Injecter systematiquement les tables de reference pour certains types de questions
5. Enrichir le GS avec les pages tableaux en expected_pages secondaires

## Relation avec row_as_chunk_experiment.md

L'experiment row-as-chunk (data/benchmarks/row_as_chunk_experiment.md) a montre
-6pp recall@5 avec les 1355 rows. Mais :
- Teste uniquement avec 270M (garbage sur tout)
- Le probleme identifie etait la pollution du top-k par des rows courtes
- Avec un modele plus capable (1B, 3n), les rows pourraient AIDER au lieu de polluer
- A re-evaluer dans le contexte du 1B

## Impact sur l'eval

Les evals v5, v5b, et 1B mesurent la generation sur un contexte APPAUVRI
(prose sans tableaux associes). Les resultats sous-estiment potentiellement
la qualite du 1B si on lui donnait le bon contexte.
