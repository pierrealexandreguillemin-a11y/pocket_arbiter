# Stratégie de Fallback Embeddings - QLoRA vs Base Model

> **Document ID**: SPEC-EFS-001
> **ISO Reference**: ISO 42001 A.6.2.2, ISO 12207
> **Version**: 1.0
> **Date**: 2026-01-25
> **Statut**: APPROVED

---

## 1. Objectif

Définir une architecture permettant un fallback sans effort si le fine-tuning QLoRA échoue, en retournant au modèle de base EmbeddingGemma.

## 2. Point de Divergence

```
                    PIPELINE CORPUS
                         │
    ┌────────────────────┴────────────────────┐
    │                                          │
    │   chunks_mode_b_fr.json                 │  ◄── GIT (partagé)
    │   (texte pur, 1857 chunks)              │      Ne change JAMAIS
    │                                          │
    └────────────────────┬────────────────────┘
                         │
                         │  ◄── POINT DE DIVERGENCE
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼────┐                    ┌────▼────┐
    │  BASE   │                    │  QLORA  │
    │ MODEL   │                    │ FINETUNED│
    └────┬────┘                    └────┬────┘
         │                               │
    ┌────▼───────────────┐    ┌────▼───────────────┐
    │ embeddings_base/   │    │ embeddings_qlora/  │
    │ embeddings_fr.npy  │    │ embeddings_fr.npy  │
    │ (768D, 1857 vecs)  │    │ (768D, 1857 vecs)  │
    └────────┬───────────┘    └────────┬───────────┘
             │                         │
             │    DVC TAGS             │
             │                         │
    ┌────────▼───────────┐    ┌────────▼───────────┐
    │ v1.0-base          │    │ v1.1-qlora         │
    │ corpus_mode_b_fr.db│    │ corpus_mode_b_fr.db│
    │ model_id: base     │    │ model_id: qlora    │
    └────────────────────┘    └────────────────────┘
```

## 3. Versioning Strategy

### 3.1 Git (Code + Chunks)

| Fichier | Versionné par | Raison |
|---------|---------------|--------|
| `corpus/processed/chunks_mode_b_fr.json` | Git | Texte, petit, partagé |
| `scripts/pipeline/*.py` | Git | Code source |
| `docs/specs/*.md` | Git | Documentation |

### 3.2 DVC (Embeddings + DB)

| Fichier | Versionné par | Tag Convention |
|---------|---------------|----------------|
| `corpus/processed/embeddings_mode_b_fr.npy` | DVC | `v{major}.{minor}-{model}` |
| `corpus/processed/corpus_mode_b_fr.db` | DVC | `v{major}.{minor}-{model}` |

**Tags:**
- `v1.0-base` → Modèle base `google/embeddinggemma-300m`
- `v1.1-qlora` → Modèle finetuné `embeddinggemma-chess-fr`

## 4. Commandes de Fallback

### 4.1 État Normal (QLoRA fonctionne)

```bash
# Utiliser embeddings QLoRA finetuned
git checkout v1.1-qlora -- data/training/unified.dvc corpus/processed.dvc
dvc checkout
```

### 4.2 Fallback (QLoRA échoue)

```bash
# Retour au modèle de base en 2 commandes
git checkout v1.0-base -- corpus/processed.dvc
dvc checkout

# Vérifier le modèle actif
python -m scripts.pipeline.update_db_model_id --db corpus/processed/corpus_mode_b_fr.db --show
# Output: model_id: google/embeddinggemma-300m
```

### 4.3 Régénérer Embeddings (si nécessaire)

```bash
# Avec modèle de base
python -m scripts.pipeline.embeddings \
    --chunks corpus/processed/chunks_mode_b_fr.json \
    --output corpus/processed/embeddings_mode_b_fr.npy \
    --model google/embeddinggemma-300m

# Reconstruire DB avec traçabilité
python -m scripts.pipeline.export_sdk \
    --chunks corpus/processed/chunks_mode_b_fr.json \
    --embeddings corpus/processed/embeddings_mode_b_fr.npy \
    --output corpus/processed/corpus_mode_b_fr.db \
    --model-id google/embeddinggemma-300m

# Tracker avec DVC
dvc add corpus/processed/embeddings_mode_b_fr.npy corpus/processed/corpus_mode_b_fr.db
git add corpus/processed/*.dvc
git commit -m "chore: regenerate embeddings with base model"
git tag v1.0-base
dvc push && git push --tags
```

## 5. Métadonnées DB (ISO 42001 Traçabilité)

Chaque DB contient maintenant `model_id` dans les métadonnées:

```sql
SELECT * FROM metadata;
-- schema_version | 2.0
-- embedding_dim  | 768
-- total_chunks   | 1857
-- model_id       | google/embeddinggemma-300m  ← NOUVEAU
```

**Modèles possibles:**
| model_id | Description | Usage |
|----------|-------------|-------|
| `google/embeddinggemma-300m` | Modèle de base | Fallback, baseline |
| `google/embeddinggemma-300m-qat-q4_0-unquantized` | Variante QAT | QLoRA training input |
| `embeddinggemma-chess-fr` | Modèle finetuné local | Production (si QLoRA OK) |

## 6. Arborescence Cible

```
corpus/processed/
├── chunks_mode_b_fr.json           # GIT - jamais change
├── chunks_mode_b_intl.json         # GIT - jamais change
├── embeddings_mode_b_fr.npy        # DVC - model-specific
├── embeddings_mode_b_intl.npy      # DVC - model-specific
├── corpus_mode_b_fr.db             # DVC - model-specific
├── corpus_mode_b_intl.db           # DVC - model-specific
├── embeddings_mode_b_fr.npy.dvc    # Git tracks DVC pointer
└── corpus_mode_b_fr.db.dvc         # Git tracks DVC pointer

models/  (après QLoRA)
└── embeddinggemma-chess-fr/        # DVC - trained model
    ├── config.json
    ├── model.safetensors
    └── training_args.json
```

## 7. Workflow Complet

```bash
# 1. BASELINE: Créer version base (une seule fois)
dvc add corpus/processed/embeddings_mode_b_fr.npy corpus/processed/corpus_mode_b_fr.db
git add corpus/processed/*.dvc
git commit -m "feat: baseline embeddings with base model"
git tag v1.0-base
dvc push && git push --tags

# 2. QLORA: Fine-tuning
python -m scripts.training.finetune_embeddinggemma \
    --triplets data/training/unified/triplets_train.jsonl \
    --output models/embeddinggemma-chess-fr

# 3. RE-EMBED: Avec modèle finetuné
python -m scripts.pipeline.embeddings \
    --chunks corpus/processed/chunks_mode_b_fr.json \
    --output corpus/processed/embeddings_mode_b_fr.npy \
    --model models/embeddinggemma-chess-fr

python -m scripts.pipeline.export_sdk \
    --chunks corpus/processed/chunks_mode_b_fr.json \
    --embeddings corpus/processed/embeddings_mode_b_fr.npy \
    --output corpus/processed/corpus_mode_b_fr.db \
    --model-id embeddinggemma-chess-fr

# 4. VERSION: Créer tag QLoRA
dvc add corpus/processed/embeddings_mode_b_fr.npy corpus/processed/corpus_mode_b_fr.db
git add corpus/processed/*.dvc
git commit -m "feat: embeddings with QLoRA finetuned model"
git tag v1.1-qlora
dvc push && git push --tags

# 5. FALLBACK: Si problème, retour en arrière
git checkout v1.0-base -- corpus/processed/*.dvc
dvc checkout
# → Retour instantané au modèle de base
```

## 8. Décision: Git vs DVC

| Critère | Git | DVC | Décision |
|---------|-----|-----|----------|
| Chunks JSON (~5MB) | ✓ | - | Git (petit, texte) |
| Embeddings NPY (~11MB) | - | ✓ | DVC (binaire) |
| DB SQLite (~15MB) | - | ✓ | DVC (binaire) |
| Code Python | ✓ | - | Git |
| Modèles finetuned (~1GB) | - | ✓ | DVC (gros binaire) |

**Règle simple:**
- **Git** = code + config + chunks texte (< 10MB, texte)
- **DVC** = embeddings + DB + modèles (binaire ou > 10MB)

## 9. Vérification Fallback

```bash
# Script de vérification rapide
python -c "
import sqlite3
db = sqlite3.connect('corpus/processed/corpus_mode_b_fr.db')
model = db.execute(\"SELECT value FROM metadata WHERE key='model_id'\").fetchone()[0]
print(f'Active model: {model}')
expected = 'google/embeddinggemma-300m'  # ou 'embeddinggemma-chess-fr'
assert model == expected, f'Model mismatch: {model} != {expected}'
print('✓ Fallback verification passed')
"
```

---

*Document ISO 42001/12207 - Pocket Arbiter Project*
*Stratégie de fallback pour continuité de service RAG*
