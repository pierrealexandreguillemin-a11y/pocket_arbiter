---
generated: 2026-03-18T21:58:36.470842+00:00
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
| recall@1 | 35.6% |
| recall@3 | 49.0% |
| recall@5 | 56.7% |
| recall@10 | 63.1% |
| MRR | 0.441 |

## Par reasoning_class

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| arithmetic | 25 | 44.0% | 52.0% | 60.0% | 64.0% |
| fact_single | 136 | 41.9% | 55.1% | 62.5% | 66.9% |
| reasoning | 4 | 50.0% | 50.0% | 50.0% | 75.0% |
| summary | 133 | 27.1% | 42.1% | 50.4% | 58.6% |

## Par difficulty

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| easy | 201 | 38.3% | 50.7% | 58.2% | 64.7% |
| hard | 16 | 43.8% | 62.5% | 62.5% | 62.5% |
| medium | 81 | 27.2% | 42.0% | 51.9% | 59.3% |

## Top echecs (recall@10 = 0)

| # | Question | Expected | Class |
|---|----------|----------|-------|
| 1 | Quelle instance fédérale donne son accord pour qua | R01_2025_26_Regles_generales.pdf p[1] | summary |
| 2 | Quel est le niveau d'arbitre minimum requis pour u | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[9] | reasoning |
| 3 | Quelle restriction s'applique a une joueuse ayant  | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | summary |
| 4 | A quelle heure un joueur est-il forfait si la rond | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 5 | En Nationale 3, quelle composition d'equipe pour l | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | arithmetic |
| 6 | En Coupe de France, si l'equipe A a les noirs au 1 | C01_2025_26_Coupe_de_France.pdf p[3] | summary |
| 7 | Quelle proposition sur le systeme de points en Cou | C04_2025_26_Coupe_de_la_parité.pdf p[5] | fact_single |
| 8 | Qui nomme les superviseurs a la DNA ? | LA-octobre2025.pdf p[227] | summary |
| 9 | Vous êtes arbitre-adjoint lors de la phase départe | R01_2025_26_Regles_generales.pdf p[2] | fact_single |
| 10 | Lors d'un match de Nationale 2, un capitaine vous  | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 11 | Lors d'une rencontre de N2, le procès-verbal de la | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | arithmetic |
| 12 | Afin de pouvoir arbitrer les tournois homologués p | LA-octobre2025.pdf p[17] | fact_single |
| 13 | Avant le commencement du tournoi, vous avez des ob | LA-octobre2025.pdf p[17] | fact_single |
| 14 | Comme tout arbitre consciencieux, vous vérifiez qu | LA-octobre2025.pdf p[9] | fact_single |
| 15 | Pendant que vous rentrez les scores et que les jou | LA-octobre2025.pdf p[165] | fact_single |
| 16 | Dans le cadre de la gestion d'un match du champion | LA-octobre2025.pdf p[165] | fact_single |
| 17 | Une fois l'appariement réalisé, Rosana vous indiqu | R01_2025_26_Regles_generales.pdf p[3] | fact_single |
| 18 | Au cours de l'appariement de la ronde 3, dans le g | R01_2025_26_Regles_generales.pdf p[2] | summary |
| 19 | Avant même que vous ayez publié les résultats de l | R03_2025_26_Competitions_homologuees.pdf p[2] | fact_single |
| 20 | Votre club organise un tournoi en cadence rapide.  | R03_2025_26_Competitions_homologuees.pdf p[2] | fact_single |

## Decision

recall@5 = 56.7% → **Fine-tuning embeddings justifie**
