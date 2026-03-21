# ADR-001: Selection du modele de generation pour CPT

> **Date**: 2026-03-21
> **Statut**: PROPOSE
> **Decideurs**: Pierre Guillemin
> **Contexte**: Chantier 4 — Continued Pre-Training modele generation

---

## Contexte

Le pipeline RAG Pocket Arbiter a deux modeles :
- **Retrieval** : EmbeddingGemma-300M base (fige, recall@5 = 60.1%)
- **Generation** : a choisir pour CPT sur corpus FFE (28 PDFs, ~500K tokens)

La spec VISION.md impose RAM < 500MB et latence < 5s sur mid-range Android.
La spec a ete ecrite en janvier 2026 avant la sortie de Gemma 3n (juin 2025).

Le CPT est un causal language modeling (CLM) sur le corpus brut avec full fine-tuning
(FFT > LoRA pour CPT — Biderman et al. TMLR 2024).
L'objectif est que le modele connaisse le vocabulaire et les regles FFE/FIDE
pour mieux exploiter les chunks retrouves par le retrieval.

---

## Options

### Option A : Gemma 3 270M IT

| Critere | Valeur |
|---------|--------|
| Params | 270M |
| RAM runtime | ~150 MB |
| Taille disque | 200 MB |
| VRAM training (4-bit LoRA) | ~3 GB |
| Kaggle T4 | Largement OK |
| Spec RAM 500MB | DANS la spec |
| Qualite generation | Faible (plus petit LLM Gemma 3) |
| LiteRT support | litert-community/gemma-3-270m-it |
| Latence estimee | 1-2s |

**Pour** : respect strict de la spec, training trivial, latence excellente.
**Contre** : qualite de generation probablement insuffisante pour synthese RAG
(reformulation, citation structuree, aveu d'ignorance). Risque de devoir
recommencer avec un plus gros modele.

### Option B : Gemma 3 1B IT

| Critere | Valeur |
|---------|--------|
| Params | 1B |
| RAM runtime | ~400 MB |
| Taille disque | 657 MB |
| VRAM training (4-bit LoRA) | ~5 GB |
| Kaggle T4 | OK |
| Spec RAM 500MB | DEPASSE (~400 MB RAM, mais dans la marge avec embeddings) |
| Qualite generation | Correcte (bon ratio taille/qualite) |
| LiteRT support | litert-community/gemma-3-1b-it |
| Latence estimee | 2-3s |

**Pour** : meilleur compromis qualite/taille, training confortable sur T4,
deja prevu comme backup dans model_card.json.
**Contre** : RAM ~400 MB + EmbeddingGemma ~110 MB = ~510 MB total, depasse
legerement la spec 500MB. Necessite de relever la contrainte RAM.

### Option C : Gemma 3n E2B IT

| Critere | Valeur |
|---------|--------|
| Params | 5B total (2B effectifs, MatFormer) |
| RAM runtime | ~2 GB |
| Taille disque | ~1.5 GB |
| VRAM training (4-bit LoRA) | ~10-12 GB |
| Kaggle T4 | Serre (16 GB) |
| Spec RAM 500MB | EXPLOSE (4x la spec) |
| Qualite generation | Bonne (architecture mobile-first) |
| LiteRT support | A verifier (Gemma 3n recent) |
| Latence estimee | 3-5s |

**Pour** : meilleure qualite, concu pour mobile (MatFormer selective activation).
**Contre** : 2 GB RAM = incompatible mid-range (6 GB total), training serre sur T4,
LiteRT support incertain, taille assets ~1.5 GB a telecharger.

---

## Analyse

### RAM budget realiste (runtime Android)

```
Device mid-range : 6 GB RAM total
- Android OS + apps : ~3 GB
- Disponible pour Pocket Arbiter : ~3 GB
- EmbeddingGemma-300M : 110 MB
- SQLite DB + index : ~50 MB
- App overhead : ~100 MB
- Budget restant pour LLM : ~2.7 GB
```

Les 3 options entrent dans le budget reel du device (2.7 GB disponibles).
La contrainte spec 500MB etait conservative.

### Qualite vs taille : existe-t-il un seuil ?

La spec VISION.md prevoit : "Gemma 3 270M IT, backup Gemma 3 1B si qualite < 70%".
La qualite n'a jamais ete evaluee car le pipeline generation n'a jamais ete construit.

Le CPT ne change pas fondamentalement la capacite de generation — il ajoute du
vocabulaire domaine. Si le 270M ne sait pas structurer une reponse RAG de base,
le CPT ne le sauvera pas.

### Training feasibility

Les 3 options sont faisables sur Kaggle T4. Le E2B est le plus serre mais reste
dans les 16 GB avec Unsloth 4-bit + gradient checkpointing.

### Deploiement LiteRT

- 270M et 1B : .tflite disponibles sur litert-community (verifies model_card.json)
- E2B : Gemma 3n est recent, support LiteRT-LM v0.1.0 (early preview, API instable)

---

## Recommandation

**Option B (Gemma 3 1B IT)** avec mise a jour de la spec RAM.

Raisons :
1. Le 270M est probablement trop faible pour la synthese RAG — risque de devoir refaire
2. Le E2B a un support LiteRT incertain et un training serre
3. Le 1B est le sweet spot : qualite correcte, training confortable, LiteRT prouve
4. La contrainte RAM 500MB est artificielle — le budget reel device est ~2.7 GB
5. Deja prevu comme backup dans model_card.json — pas une surprise architecturale

### Changements requis si Option B

- VISION.md : relever contrainte RAM de 500MB a 1GB (ou supprimer, documenter le budget reel)
- model_card.json : passer 1B de "backup" a "primary"
- ARCHITECTURE.md : mettre a jour taille assets

---

## Decision

**[X] Option A — Gemma 3 270M IT** (respect spec, qualite incertaine)
**[ ] Option B — Gemma 3 1B IT** (relever spec RAM, meilleur compromis)
**[ ] Option C — Gemma 3n E2B IT** (meilleure qualite, risques deploiement)

Date decision : 2026-03-21
Decideur : Pierre Guillemin

Justification : commencer par le plus petit, respecter la spec, evaluer la qualite.
Gate de rollback : si qualite generation < 70% (eval humaine), passer au 1B (Option B).
