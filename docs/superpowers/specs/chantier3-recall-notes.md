# Chantier 3 : Re-mesure Recall — Notes

> Notes accumulees pendant les chantiers precedents. A transformer en spec complete quand le chantier demarre.

## A calibrer sur les 298 questions GS

- **Adaptive k** : parametres `min_score`, `max_gap`, `max_k` exposes dans search.py avec valeurs conservatrices par defaut. Calibrer sur distribution des scores cosine du GS pour trouver le sweet spot recall/precision.
