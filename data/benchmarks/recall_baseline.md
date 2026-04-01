---
generated: 2026-03-31T21:48:26.642225+00:00
pipeline: hybrid cosine+BM25 RRF, adaptive-k largest-gap
model: google/embeddinggemma-300m
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
| recall@1 | 41.9% |
| recall@3 | 58.1% |
| recall@5 | 63.4% |
| recall@10 | 67.4% |
| MRR | 0.511 |

## Par reasoning_class

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| arithmetic | 25 | 48.0% | 68.0% | 76.0% | 80.0% |
| fact_single | 136 | 47.8% | 63.2% | 69.9% | 73.5% |
| reasoning | 4 | 50.0% | 50.0% | 50.0% | 50.0% |
| summary | 133 | 34.6% | 51.1% | 54.9% | 59.4% |

## Par difficulty

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| easy | 201 | 42.3% | 57.7% | 64.7% | 68.2% |
| hard | 16 | 50.0% | 68.8% | 68.8% | 68.8% |
| medium | 81 | 39.5% | 56.8% | 59.3% | 65.4% |

## Top echecs (recall@10 = 0)

| # | Question | Expected | Class |
|---|----------|----------|-------|
| 1 | Quelle tâche ne fait pas partie des missions de l' | LA-octobre2025.pdf p[10] | summary |
| 2 | Un joueur avec licence B peut-il participer a un t | R03_2025_26_Competitions_homologuees.pdf p[2] | summary |
| 3 | Quel est le niveau d'arbitre minimum requis pour u | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[9] | reasoning |
| 4 | A quelle heure un joueur est-il forfait si la rond | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 5 | Qui nomme les superviseurs a la DNA ? | LA-octobre2025.pdf p[227] | summary |
| 6 | Vous êtes arbitre-adjoint lors de la phase départe | R01_2025_26_Regles_generales.pdf p[2] | fact_single |
| 7 | Lors d'un match de Nationale 2, un capitaine vous  | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 8 | En coupe « Jean-Claude Loubatière », quelle affirm | C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf p[2] | summary |
| 9 | Avant le commencement du tournoi, vous avez des ob | LA-octobre2025.pdf p[17] | fact_single |
| 10 | Comme tout arbitre consciencieux, vous vérifiez qu | LA-octobre2025.pdf p[9] | fact_single |
| 11 | Pendant que vous rentrez les scores et que les jou | LA-octobre2025.pdf p[165] | fact_single |
| 12 | Dans le cadre de la gestion d'un match du champion | LA-octobre2025.pdf p[165] | fact_single |
| 13 | Une fois l'appariement réalisé, Rosana vous indiqu | R01_2025_26_Regles_generales.pdf p[3] | fact_single |
| 14 | Au cours de l'appariement de la ronde 3, dans le g | R01_2025_26_Regles_generales.pdf p[2] | summary |
| 15 | Pour un tournoi, le montant de l'inscription est f | R03_2025_26_Competitions_homologuees.pdf p[2] | fact_single |
| 16 | Avant même que vous ayez publié les résultats de l | R03_2025_26_Competitions_homologuees.pdf p[2] | fact_single |
| 17 | Votre club organise un tournoi en cadence rapide.  | R03_2025_26_Competitions_homologuees.pdf p[2] | fact_single |
| 18 | Avant l'appariement de la ronde 4, Isabelle a eu c | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[1] | arithmetic |
| 19 | Lors d'un match par équipe, le capitaine de l'équi | R01_2025_26_Regles_generales.pdf p[2] | fact_single |
| 20 | Dans le cadre d'un tournoi, un enfant malvoyant ut | R03_2025_26_Competitions_homologuees.pdf p[2] | fact_single |

## Decision

recall@5 = 63.4% → **Optimisations retrieval necessaires**
