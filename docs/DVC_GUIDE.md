# 📦 Guide DVC - Pocket Arbiter

> DVC (Data Version Control) permet de versionner les fichiers volumineux sans les stocker dans Git.

## 🎯 Pourquoi DVC dans ce projet ?

| Type de fichier | Taille | Où ? |
|-----------------|--------|------|
| Code Python/Kotlin | ~KB | Git ✅ |
| Documentation | ~KB | Git ✅ |
| Embeddings (.npy) | ~50-200 MB | DVC ✅ |
| Vector DB (.db) | ~50-100 MB | DVC ✅ |
| PDFs règlements | ~5 MB | Git ou DVC |

## 🚀 Setup initial (une seule fois)

```bash
# 1. Installer DVC
pip install dvc dvc-gdrive  # ou dvc-s3, dvc-gs

# 2. Initialiser DVC dans le projet
cd C:\Dev\pocket_arbiter
dvc init

# 3. Configurer le remote (Google Drive recommandé)
# Créer un dossier dans Drive, copier l'ID depuis l'URL
dvc remote add -d storage gdrive://TON_FOLDER_ID
```

## 📁 Fichiers à tracker avec DVC

```bash
# Quand tu génères des embeddings
dvc add corpus/processed/embeddings_fr.npy
dvc add corpus/processed/embeddings_intl.npy

# Quand tu crées les bases vectorielles (SqliteVectorStore)
dvc add corpus/processed/corpus_mode_b_fr.db
# Note: INTL a reconstruire (voir VISION.md Dual-RAG)

# Commiter les fichiers .dvc dans Git
git add corpus/processed/*.dvc corpus/processed/.gitignore
git commit -m "Add embeddings and vector DBs to DVC"

# Pousser vers le remote
dvc push
```

## 🔄 Workflow quotidien

```bash
# Récupérer les données (nouveau clone ou mise à jour)
dvc pull

# Après modification des données
dvc add data/embeddings/
git add data/embeddings.dvc
git commit -m "Update embeddings"
dvc push
git push
```

## 📋 Commandes essentielles

| Commande | Usage |
|----------|-------|
| `dvc pull` | Télécharger les données |
| `dvc push` | Uploader les données |
| `dvc add <fichier>` | Tracker un fichier |
| `dvc status` | Voir les changements |
| `dvc diff` | Comparer versions |

## ⚠️ Points d'attention

1. **Toujours commiter les .dvc** → Ils lient code et données
2. **dvc push avant git push** → Sinon les données sont perdues
3. **Ne jamais éditer les .dvc manuellement**

## 🔗 Ressources

- [Documentation DVC](https://dvc.org/doc)
- [DVC avec Google Drive](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive)
