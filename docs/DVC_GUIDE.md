# Guide DVC - Pocket Arbiter

> **Document ID**: DOC-GUIDE-001
> **ISO Reference**: ISO/IEC 12207:2017 - Gestion de configuration
> **Version**: 2.0
> **Date**: 2026-02-11

DVC (Data Version Control) versionne les fichiers volumineux sans les stocker dans Git.

## Pourquoi DVC dans ce projet ?

| Type de fichier | Taille | Ou ? |
|-----------------|--------|------|
| Code Python/Kotlin | ~KB | Git |
| Documentation | ~KB | Git |
| Corpus PDF (29 FFE + 1 FIDE) | ~75 MB | DVC |
| Embeddings (.npy) | ~11 MB | DVC |
| Training data (triplets, BEIR, RAGAS) | ~34 MB | DVC |
| Gold Standard annales | ~2 MB | DVC |
| **Total DVC** | **~122 MB** | |

## Configuration actuelle

```
Remote: local (C:/Dev/dvc_remote)
Cache:  .dvc/cache (local)
```

### Fichiers .dvc trackes par Git

| Fichier .dvc | Cible | Taille |
|---|---|---|
| `corpus/fr.dvc` | 29 PDF FFE | 68.6 MB |
| `corpus/intl.dvc` | 1 PDF FIDE | 6.6 MB |
| `corpus/processed/embeddings_fr.npy.dvc` | Embeddings FR | 7.9 MB |
| `corpus/processed/embeddings_intl.npy.dvc` | Embeddings INTL | 3.1 MB |
| `data/training.dvc` | Triplets, BEIR, RAGAS (37 fichiers) | 33.7 MB |
| `tests/data/gold_standard_annales_fr_v8_adversarial.json.dvc` | GS annales | 1.7 MB |

## Workflow quotidien

```bash
# Recuperer les donnees (nouveau clone)
python -m dvc pull

# Apres modification des donnees
python -m dvc add data/training
git add data/training.dvc
git commit -m "chore(dvc): update training data"
python -m dvc push
git push
```

## Commandes essentielles

| Commande | Usage |
|----------|-------|
| `python -m dvc pull` | Telecharger les donnees |
| `python -m dvc push` | Uploader les donnees |
| `python -m dvc add <fichier>` | Tracker un fichier |
| `python -m dvc status` | Voir les changements |
| `python -m dvc diff` | Comparer versions |

## Points d'attention

1. **Toujours commiter les .dvc** : ils lient code et donnees
2. **dvc push avant git push** : sinon les donnees sont perdues pour les autres
3. **Ne jamais editer les .dvc manuellement**
4. **Venv obligatoire** : `python -m dvc` depuis le venv du projet

## Migration future vers cloud

Le remote local est suffisant pour un developpeur unique. Pour collaborer :

```bash
# DagsHub (gratuit, 10 GB)
python -m dvc remote add -d dagshub dagshub://<user>/pocket_arbiter

# Google Cloud Storage
python -m dvc remote add -d gcs gs://pocket-arbiter-dvc
```

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-18 | Creation initiale, setup Google Drive |
| 2.0 | 2026-02-11 | Remote local (122 MB), inventaire complet fichiers .dvc, venv |
