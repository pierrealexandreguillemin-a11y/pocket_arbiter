---
generated: 2026-03-18T07:22:10.973414+00:00
pipeline: hybrid cosine+BM25 RRF, adaptive-k largest-gap
model: google/embeddinggemma-300m-qat-q4_0-unquantized
db: corpus_v2_fr.db
gs_version: 9.0.0
match_level: page
settings: {'min_score': 0.005, 'max_k': 10, 'rrf_k': 60}
questions_total: 298
---

# Recall Baseline — Pipeline v2

## Global

| Metrique | Score |
|----------|-------|
| recall@1 | 35.9% |
| recall@3 | 52.7% |
| recall@5 | 56.0% |
| recall@10 | 59.1% |
| MRR | 0.445 |

## Par reasoning_class

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| arithmetic | 25 | 44.0% | 60.0% | 64.0% | 72.0% |
| fact_single | 136 | 36.8% | 58.1% | 62.5% | 64.0% |
| reasoning | 4 | 25.0% | 50.0% | 75.0% | 75.0% |
| summary | 133 | 33.8% | 45.9% | 47.4% | 51.1% |

## Par difficulty

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| easy | 201 | 39.3% | 55.2% | 58.2% | 60.7% |
| hard | 16 | 37.5% | 56.2% | 62.5% | 62.5% |
| medium | 81 | 27.2% | 45.7% | 49.4% | 54.3% |

## Top echecs (recall@10 = 0)

| # | Question | Expected | Class |
|---|----------|----------|-------|
| 1 | Quelle tâche ne fait pas partie des missions de l' | LA-octobre2025.pdf p[10] | summary |
| 2 | Quelle sanction l'arbitre ne peut-il pas appliquer | R01_2025_26_Regles_generales.pdf p[5] | fact_single |
| 3 | A partir de quel jour minimum apres une rencontre  | R01_2025_26_Regles_generales.pdf p[4] | arithmetic |
| 4 | Quel tarif d'inscription s'applique a une personne | R03_2025_26_Competitions_homologuees.pdf p[2] | fact_single |
| 5 | Un joueur avec licence B peut-il participer a un t | R03_2025_26_Competitions_homologuees.pdf p[2] | summary |
| 6 | Quel est le niveau d'arbitre minimum requis pour u | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[9] | reasoning |
| 7 | A quelle heure un joueur est-il forfait si la rond | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 8 | En Coupe de France, si l'equipe A a les noirs au 1 | C01_2025_26_Coupe_de_France.pdf p[3] | summary |
| 9 | Quelle proposition sur le systeme de points en Cou | C04_2025_26_Coupe_de_la_parité.pdf p[5] | fact_single |
| 10 | Qui nomme les superviseurs a la DNA ? | LA-octobre2025.pdf p[227] | summary |
| 11 | Vous êtes arbitre-adjoint lors de la phase départe | R01_2025_26_Regles_generales.pdf p[2] | fact_single |
| 12 | Pour jouer un match par équipe en championnat de F | R01_2025_26_Regles_generales.pdf p[1] | fact_single |
| 13 | Lors d'un match de Nationale 2, un capitaine vous  | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 14 | En cours de partie, face à une décision de l'arbit | R02_2025_26_Regles_generales_Annexes.pdf p[1] | fact_single |
| 15 | En coupe « Jean-Claude Loubatière », quelle affirm | C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf p[2] | summary |
| 16 | Avant le commencement du tournoi, vous avez des ob | LA-octobre2025.pdf p[17] | fact_single |
| 17 | Comme tout arbitre consciencieux, vous vérifiez qu | LA-octobre2025.pdf p[9] | fact_single |
| 18 | Pendant que vous rentrez les scores et que les jou | LA-octobre2025.pdf p[165] | fact_single |
| 19 | Une fois l'appariement réalisé, Rosana vous indiqu | R01_2025_26_Regles_generales.pdf p[3] | fact_single |
| 20 | Au cours de l'appariement de la ronde 3, dans le g | R01_2025_26_Regles_generales.pdf p[2] | summary |

## Decision

recall@5 = 56.0% → **Fine-tuning embeddings justifie**
