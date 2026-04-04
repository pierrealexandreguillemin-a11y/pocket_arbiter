---
generated: 2026-04-04T21:02:47.622185+00:00
pipeline: hybrid cosine+BM25 RRF, adaptive-k largest-gap
model: google/embeddinggemma-300m
db: corpus_v2_fr.db
gs_version: 9.0.0
match_level: page
settings: {'min_score': 0.005, 'max_k': 5, 'rrf_k': 60}
questions_total: 298
---

# Recall Baseline — Pipeline v2

## Global

| Metrique | Score |
|----------|-------|
| recall@1 | 36.9% |
| recall@3 | 53.4% |
| recall@5 | 55.4% |
| recall@10 | 55.7% |
| MRR | 0.449 |

## Par reasoning_class

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| arithmetic | 25 | 40.0% | 56.0% | 60.0% | 60.0% |
| fact_single | 136 | 42.6% | 61.0% | 62.5% | 62.5% |
| reasoning | 4 | 50.0% | 50.0% | 50.0% | 50.0% |
| summary | 133 | 30.1% | 45.1% | 47.4% | 48.1% |

## Par difficulty

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| easy | 201 | 38.8% | 55.7% | 57.7% | 57.7% |
| hard | 16 | 31.2% | 56.2% | 62.5% | 62.5% |
| medium | 81 | 33.3% | 46.9% | 48.1% | 49.4% |

## Top echecs (recall@10 = 0)

| # | Question | Expected | Class |
|---|----------|----------|-------|
| 1 | Quelle tâche ne fait pas partie des missions de l' | LA-octobre2025.pdf p[10] | summary |
| 2 | Quelle instance fédérale donne son accord pour qua | R01_2025_26_Regles_generales.pdf p[1] | summary |
| 3 | Un joueur avec licence B peut-il participer a un t | R03_2025_26_Competitions_homologuees.pdf p[2] | summary |
| 4 | Quelle situation n'est pas autorisee dans le champ | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[1] | fact_single |
| 5 | Quel est le niveau d'arbitre minimum requis pour u | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[9] | reasoning |
| 6 | Quelle restriction s'applique a une joueuse ayant  | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | summary |
| 7 | A quelle heure un joueur est-il forfait si la rond | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 8 | En Nationale 3, quelle composition d'equipe pour l | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | arithmetic |
| 9 | En Coupe de France, si l'equipe A a les noirs au 1 | C01_2025_26_Coupe_de_France.pdf p[3] | summary |
| 10 | Qui nomme les superviseurs a la DNA ? | LA-octobre2025.pdf p[227] | summary |
| 11 | Vous êtes arbitre-adjoint lors de la phase départe | R01_2025_26_Regles_generales.pdf p[2] | fact_single |
| 12 | Lors d'un match de Nationale 2, un capitaine vous  | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 13 | Lors d'une rencontre de N2, le procès-verbal de la | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | arithmetic |
| 14 | En coupe « Jean-Claude Loubatière », quelle affirm | C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf p[2] | summary |
| 15 | Avant le commencement du tournoi, vous avez des ob | LA-octobre2025.pdf p[17] | fact_single |
| 16 | Comme tout arbitre consciencieux, vous vérifiez qu | LA-octobre2025.pdf p[9] | fact_single |
| 17 | Pendant que vous rentrez les scores et que les jou | LA-octobre2025.pdf p[165] | fact_single |
| 18 | Dans le cadre de la gestion d'un match du champion | LA-octobre2025.pdf p[165] | fact_single |
| 19 | Un joueur a pris sa licence dans votre club le 13  | R01_2025_26_Regles_generales.pdf p[2] | summary |
| 20 | Une fois l'appariement réalisé, Rosana vous indiqu | R01_2025_26_Regles_generales.pdf p[3] | fact_single |

## Decision

recall@5 = 55.4% → **Fine-tuning embeddings justifie**
