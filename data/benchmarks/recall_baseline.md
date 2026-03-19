---
generated: 2026-03-19T12:22:22.335145+00:00
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
| recall@1 | 35.2% |
| recall@3 | 52.7% |
| recall@5 | 59.1% |
| recall@10 | 63.1% |
| MRR | 0.448 |

## Par reasoning_class

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| arithmetic | 25 | 44.0% | 56.0% | 60.0% | 68.0% |
| fact_single | 136 | 41.2% | 58.8% | 65.4% | 69.1% |
| reasoning | 4 | 50.0% | 50.0% | 50.0% | 50.0% |
| summary | 133 | 27.1% | 45.9% | 52.6% | 56.4% |

## Par difficulty

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| easy | 201 | 38.3% | 54.2% | 60.2% | 64.7% |
| hard | 16 | 37.5% | 56.2% | 56.2% | 56.2% |
| medium | 81 | 27.2% | 48.1% | 56.8% | 60.5% |

## Top echecs (recall@10 = 0)

| # | Question | Expected | Class |
|---|----------|----------|-------|
| 1 | Quelle instance fédérale donne son accord pour qua | R01_2025_26_Regles_generales.pdf p[1] | summary |
| 2 | Quelle situation n'est pas autorisee dans le champ | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[1] | fact_single |
| 3 | Quel est le niveau d'arbitre minimum requis pour u | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[9] | reasoning |
| 4 | Quelle restriction s'applique a une joueuse ayant  | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | summary |
| 5 | A quelle heure un joueur est-il forfait si la rond | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 6 | En Nationale 3, quelle composition d'equipe pour l | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | arithmetic |
| 7 | En Coupe de France, si l'equipe A a les noirs au 1 | C01_2025_26_Coupe_de_France.pdf p[3] | summary |
| 8 | Quelle proposition sur le systeme de points en Cou | C04_2025_26_Coupe_de_la_parité.pdf p[5] | fact_single |
| 9 | Qui nomme les superviseurs a la DNA ? | LA-octobre2025.pdf p[227] | summary |
| 10 | Vous êtes arbitre-adjoint lors de la phase départe | R01_2025_26_Regles_generales.pdf p[2] | fact_single |
| 11 | Lors d'un match de Nationale 2, un capitaine vous  | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 12 | Lors d'une rencontre de N2, le procès-verbal de la | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | arithmetic |
| 13 | En coupe « Jean-Claude Loubatière », quelle affirm | C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf p[2] | summary |
| 14 | Afin de pouvoir arbitrer les tournois homologués p | LA-octobre2025.pdf p[17] | fact_single |
| 15 | Avant le commencement du tournoi, vous avez des ob | LA-octobre2025.pdf p[17] | fact_single |
| 16 | Comme tout arbitre consciencieux, vous vérifiez qu | LA-octobre2025.pdf p[9] | fact_single |
| 17 | Pendant que vous rentrez les scores et que les jou | LA-octobre2025.pdf p[165] | fact_single |
| 18 | Dans le cadre de la gestion d'un match du champion | LA-octobre2025.pdf p[165] | fact_single |
| 19 | Une fois l'appariement réalisé, Rosana vous indiqu | R01_2025_26_Regles_generales.pdf p[3] | fact_single |
| 20 | Au cours de l'appariement de la ronde 3, dans le g | R01_2025_26_Regles_generales.pdf p[2] | summary |

## Decision

recall@5 = 59.1% → **Fine-tuning embeddings justifie**
