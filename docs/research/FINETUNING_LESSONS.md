# Lecons Fine-tuning EmbeddingGemma - Checklist & Erreurs

> **Document ID**: RES-FT-LESSONS-001
> **ISO Reference**: ISO 42001, ISO 25010
> **Version**: 1.0
> **Date**: 2026-01-24
> **Statut**: Reference
> **Classification**: Interne
> **Origine**: Extrait de LORA_FINETUNING_GUIDE.md (sections 9-12)
> **Mots-cles**: fine-tuning, lessons learned, checklist, errors, benchmarks

---

## 1. Checklist Pre-Notebook

### 1.1 Prerequis

- [ ] GPU disponible (T4 minimum, A100 recommande)
- [ ] `sentence-transformers>=3.0` installe
- [ ] `transformers>=4.56.0` installe
- [ ] `wandb` configure (optionnel mais recommande)
- [ ] Acces Kaggle/HuggingFace pour EmbeddingGemma

### 1.2 Donnees

- [ ] Gold Standard valide (GS v6.7.0: 477 questions)
- [ ] `corpus_mode_b_fr.db` accessible avec chunks
- [ ] Script generation hard negatives teste
- [ ] Split train/val prepare (80/20)

### 1.3 Configuration

- [ ] Hyperparametres choisis (voir FINETUNING_RESOURCES.md)
- [ ] Evaluator configure
- [ ] Early stopping callback pret
- [ ] Logging W&B/TensorBoard active

### 1.4 Post-training

- [ ] Script export modele pret
- [ ] Script re-generation embeddings pret
- [ ] Tests regression prepares

---

## 2. Erreurs a eviter

| Erreur | Consequence | Solution |
|--------|-------------|----------|
| Batch size trop petit | MNR Loss inefficace | Minimum 32, ideal 64-128 |
| Pas de prompts | Embeddings sous-optimaux | Toujours utiliser prompts EmbeddingGemma |
| LR trop eleve | Overfitting rapide | Commencer a 2e-5, reduire si necessaire |
| Pas d'early stopping | Overfitting | Callback + patience 3 |
| Hard negatives random | Training signal faible | Utiliser retrieval-based negatives |
| Oublier NO_DUPLICATES | False negatives dans batch | `batch_sampler=BatchSamplers.NO_DUPLICATES` |
| Pas de val split | Impossible detecter overfitting | 80/20 split obligatoire |

---

## 3. Variantes EmbeddingGemma et Deploiement Mobile

### 3.1 Variantes disponibles

| Variante | Source | Taille | Format | Usage |
|----------|--------|--------|--------|-------|
| `google/embeddinggemma-300m` | HuggingFace | ~1.2 GB | PyTorch | **Fine-tuning LoRA** |
| `google/embeddinggemma-300m-qat-q4_0-unquantized` | HuggingFace | ~600 MB | PyTorch Q4 | **Fine-tuning QLoRA** |
| `litert-community/embeddinggemma-300m` | HuggingFace | 179-196 MB | .tflite | **Deploiement mobile** |

### 3.2 Comparaison QLoRA vs LoRA sur TFLite

| Critere | QLoRA (HuggingFace) | LoRA sur TFLite |
|---------|---------------------|-----------------|
| **Fichier** | `.safetensors` PyTorch | `.tflite` LiteRT |
| **Fine-tuning** | OUI | NON (frozen) |
| **GPU requis** | Oui (training) | Non (inference) |
| **Taille modele** | ~600 MB | ~180 MB |

**Conclusion**: Le modele TFLite est le **resultat final** apres conversion, pas le point de depart.

### 3.3 Performances EmbeddingGemma (benchmarks officiels)

| Metrique | Valeur | Notes |
|----------|--------|-------|
| **MTEB Rank** | #1 (<500M params) | Meilleur multilingual open <500M |
| **Latence inference** | <15 ms | 256 tokens sur EdgeTPU |
| **RAM quantifie** | <200 MB | Avec QAT int4/int8 |
| **Dimensions** | 768D (ou 128/256/512 MRL) | Truncation sans perte majeure |
| **Contexte** | 2048 tokens | vs 512 pour concurrents |
| **Langues** | 100+ | Francais inclus |

### 3.4 Workflow Fine-tuning → Deploiement

```
PHASE 1: FINE-TUNING (Kaggle/Colab)
├── google/embeddinggemma-300m (HuggingFace)
├── QLoRA fine-tuning avec triplets (~5000)
└── embeddinggemma-chess-fr/ (adapters fusionnes)
            ↓
PHASE 2: EXPORT MOBILE
├── Conversion TFLite + quantization int4/int8
└── embeddinggemma-chess-fr.tflite (~110-180 MB)
            ↓
PHASE 3: DEPLOIEMENT ANDROID (AI Edge RAG SDK)
├── LiteRT runtime → inference on-device <15ms
└── Devices cibles: Pixel 8/9, Samsung S23/S24
```

### 3.5 Comparaison approches

| Approche | Avantages | Inconvenients | Recommandation |
|----------|-----------|---------------|----------------|
| **QLoRA sur HuggingFace** | Adapte au domaine, meilleur recall | GPU requis, conversion TFLite | Recommande |
| **Export TFLite direct** | Simple, rapide | Performance baseline seulement | Fallback |
| **LoRA sur TFLite** | N/A | Non supporte | Impossible |

---

## 4. Benchmarks de reference

### 4.1 Gains observes (domain-specific fine-tuning)

| Domaine | Avant | Apres | Gain | Source |
|---------|-------|-------|------|--------|
| Legal | 59.5% nDCG | 82.5% nDCG | +38% | Voyage AI |
| Mortgage | 59.9% nDCG | 62.1% nDCG | +4% | SugiV |
| Medical (MIRIAD) | 83.4% nDCG | 88.6% nDCG | +5% | HuggingFace |

### 4.2 Projection Pocket Arbiter

| Metrique | Baseline | Projection conservative | Projection optimiste |
|----------|----------|------------------------|---------------------|
| Recall@5 FR | 91.56% | 94% | 97% |
| Hard cases | 46/150 | 25/150 | 15/150 |
| nDCG@10 | ~60% | 70% | 80% |

---

## 5. Lecons du fiasco precedent (Jan 2026)

### 5.1 Ce qui a mal tourne

1. **Notebook de mauvaise qualite** - Source non officielle
2. **Dataset trop petit** - 193 questions (ancien GS)
3. **Pas de validation split** - Overfitting non detecte
4. **Hard negatives random** - Signal d'apprentissage faible
5. **Recall degrade** - 65.69% vs 91.56% baseline

### 5.2 Ce qu'il faut faire maintenant

1. **Utiliser sources officielles** - Google AI Dev, Nilay Chauhan (Google)
2. **GS v6.7.0** - 477 questions officielles DNA
3. **Validation BY DESIGN** - Chunk visible lors reformulation
4. **Hard negatives intelligents** - same_doc_diff_page strategy
5. **Early stopping** - Patience 3, threshold 0.001

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-24 | Extraction depuis LORA_FINETUNING_GUIDE.md (sections 9-12) + lecons ajoutees |

---

*Document ISO 42001/25010 - Pocket Arbiter Project*
