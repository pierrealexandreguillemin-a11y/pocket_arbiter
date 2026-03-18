---
generated: 2026-03-18T07:03:20.379166+00:00
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
| recall@1 | 28.2% |
| recall@3 | 28.2% |
| recall@5 | 28.2% |
| recall@10 | 28.2% |
| MRR | 0.282 |

## Par reasoning_class

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| arithmetic | 25 | 36.0% | 36.0% | 36.0% | 36.0% |
| fact_single | 136 | 32.4% | 32.4% | 32.4% | 32.4% |
| reasoning | 4 | 25.0% | 25.0% | 25.0% | 25.0% |
| summary | 133 | 22.6% | 22.6% | 22.6% | 22.6% |

## Par difficulty

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| easy | 201 | 29.9% | 29.9% | 29.9% | 29.9% |
| hard | 16 | 25.0% | 25.0% | 25.0% | 25.0% |
| medium | 81 | 24.7% | 24.7% | 24.7% | 24.7% |

## Top echecs (recall@10 = 0)

| # | Question | Expected | Class |
|---|----------|----------|-------|
| 1 | Quelle tâche ne fait pas partie des missions de l' | LA-octobre2025.pdf p[10] | summary |
| 2 | Quelle sanction l'arbitre ne peut-il pas appliquer | R01_2025_26_Regles_generales.pdf p[5] | fact_single |
| 3 | Quelle instance fédérale donne son accord pour qua | R01_2025_26_Regles_generales.pdf p[1] | summary |
| 4 | Quelle situation ne constitue pas un forfait selon | R01_2025_26_Regles_generales.pdf p[3] | arithmetic |
| 5 | A partir de quel jour minimum apres une rencontre  | R01_2025_26_Regles_generales.pdf p[4] | arithmetic |
| 6 | Quel tarif d'inscription s'applique a une personne | R03_2025_26_Competitions_homologuees.pdf p[2] | fact_single |
| 7 | Un joueur avec licence B peut-il participer a un t | R03_2025_26_Competitions_homologuees.pdf p[2] | summary |
| 8 | L'inscription est-elle gratuite pour une championn | R03_2025_26_Competitions_homologuees.pdf p[2] | summary |
| 9 | Quelle situation n'est pas autorisee dans le champ | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[1] | fact_single |
| 10 | Quel est le niveau d'arbitre minimum requis pour u | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[9] | reasoning |
| 11 | Quelle restriction s'applique a une joueuse ayant  | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | summary |
| 12 | A quelle heure un joueur est-il forfait si la rond | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[4] | arithmetic |
| 13 | En Nationale 3, quelle composition d'equipe pour l | A02_2025_26_Championnat_de_France_des_Clubs.pdf p[6] | arithmetic |
| 14 | Quel systeme d'appariement utiliser pour 9 equipes | C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf p[6] | summary |
| 15 | En Coupe de France, si l'equipe A a les noirs au 1 | C01_2025_26_Coupe_de_France.pdf p[3] | summary |
| 16 | Quelle amende pour forfait au 4e tour si le 1er to | C01_2025_26_Coupe_de_France.pdf p[4] | arithmetic |
| 17 | Comment sont departagees deux equipes a egalite en | C01_2025_26_Coupe_de_France.pdf p[5] | fact_single |
| 18 | Quelle proposition sur le systeme de points en Cou | C04_2025_26_Coupe_de_la_parité.pdf p[5] | fact_single |
| 19 | Qui nomme les superviseurs a la DNA ? | LA-octobre2025.pdf p[227] | summary |
| 20 | Vous êtes arbitre-adjoint lors de la phase départe | R01_2025_26_Regles_generales.pdf p[2] | fact_single |

## Decision

recall@5 = 28.2% → **Fine-tuning embeddings justifie**
