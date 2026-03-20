# SimCSE + ICT Self-Supervised Fine-tuning — Design Spec

> **Date**: 2026-03-20
> **Statut**: En revue (post-review v2)
> **Baseline**: recall@5 = 60.1%, recall@1 = 38.9%, MRR = 0.479
> **Target**: recall@5 ≥ 65.1% (+5pp minimum pour justifier conversion TFLite)
> **Contrainte**: corpus-only (GS = evaluation uniquement), Android LiteRT compatible
> **Methode**: LoRA (rank 8-16) — PAS de full fine-tuning (1116 ex = overfitting certain)

---

## Contexte

### Pourquoi self-supervised

- 8 optimisations testees (chantier 3) : +3.4pp total sur recall@5
- GS = faux-ami potentiel (264/298 = QCM annales, chunk_ids corriges 6x)
- Precedent fine-tune supervise a DEGRADE le recall (82.84% → 65.69%)
- Signal corpus = ground truth incontestable (28 PDFs FFE)

### Pourquoi SimCSE + ICT

- SimCSE : ameliore uniformite + alignement espace vectoriel (Gao et al. EMNLP 2021)
- ICT : apprend directement la tache retrieval query→document (Lee et al. ACL 2019)
- Zero modele externe, zero dependance, zero cout API
- sentence-transformers natif, Kaggle T4 suffisant

### Ce chantier est le premier de deux

1. **Retrieval** : SimCSE + ICT sur EmbeddingGemma-300M (ce document)
2. **Generation** : GRPO sur Gemma 3 (voir project_lora_without_regret.md)

### Avertissement echelle (honnetete intellectuelle)

SimCSE a ete concu pour 1M phrases (Wikipedia). ICT/ORQA pour 409M exemples
(batch 4096 × 100K steps). Notre corpus = 1116 chunks.

C'est une reduction de 900x (SimCSE) a 73000x (ICT) par rapport aux regimes
originaux des papers. **Aucune litterature ne valide ces techniques a cette echelle.**

De plus :
- LoRA + SimCSE = combinaison non testee dans la litterature
- mean_tokens pooling (EmbeddingGemma) ≠ CLS pooling (SimCSE paper)
- batch 64 in-batch negatives ≠ batch 4096 (ORQA)

**C'est un essai exploratoire, pas une application standard.**
Le filet de securite = early stopping sur GS recall@5, rollback si ≤ baseline.
Budget = ~10 min Kaggle T4, risque = faible (LoRA preserve les poids base).

---

## Architecture

```
corpus/processed/docling_v2_fr/*.json (1116 children)
    |
    ├──> simcse_pairs.jsonl (chunk, chunk)  — dropout = augmentation
    └──> ict_pairs.jsonl (pseudo-query, chunk) — first sentence extraction
         |
         v
    Kaggle T4 16GB
    ├── Stage 1: SimCSE (warmup, uniformite)
    └── Stage 2: ICT (task-specific retrieval)
         |
         v
    embeddinggemma-simcse-ict/ (checkpoint)
         |
         v
    Local: rebuild corpus_v2_fr.db (1 seul rebuild)
         |
         v
    recall@5 mesure → gate T1/T2
```

---

## Data pipeline

### Format documents (CRITIQUE — alignement training/inference)

Les documents d'entrainement DOIVENT etre pre-formates avec le CCH title
exactement comme le pipeline d'inference (indexer_embed.format_document) :

```
"title: {CCH_title} | text: {chunk_text}"
```

Le CCH title est construit depuis source + section de la DB (make_cch_title).
Le trainer ajoute le query prompt (`"task: search result | query: "`) via
`prompts={"anchor": QUERY_PROMPT}` mais PAS de prompt document — le document
est deja formate dans le JSONL.

### SimCSE

Chaque chunk pre-formate passe en double. Le dropout du transformer cree
deux embeddings differents pour la meme phrase. In-batch negatives = les
autres phrases du batch.

```python
cch = make_cch_title(c["source"], c["section"], SOURCE_TITLES)
formatted = format_document(c["text"], cch)  # "title: CCH | text: chunk"
simcse_data.append((formatted, formatted))
# 1116 paires
```

### ICT (Inverse Cloze Task)

Pour chaque chunk, extraire une phrase **aleatoire** non-triviale (>20 chars)
comme pseudo-query (standard ORQA : "q is a random sentence"). Le document =
chunk pre-formate avec CCH. La phrase est retiree du chunk 90% du temps (§9.2).

```python
sentence = extract_random_sentence(c["text"])  # query brute
masked = mask_sentence_from_chunk(c["text"], sentence)  # 90% masking
cch = make_cch_title(c["source"], c["section"], SOURCE_TITLES)
formatted_doc = format_document(masked, cch)  # "title: CCH | text: masked_chunk"
ict_data.append((sentence, formatted_doc))
# ~1067 paires (chunks sans phrase valide sont skippes, assert <30%)
```

Pas de hard negatives explicites — `CachedMultipleNegativesRankingLoss` utilise
les autres elements du batch comme negatifs. Sampler `NO_DUPLICATES` pour les
deux stages (evite les self-negatives qui corrompent la loss).

### Validation donnees

- simcse_pairs : 1116 paires, documents pre-formates avec CCH
- ict_pairs : >= 1000 paires, pseudo-query >20 chars, skip rate <30%
- Stats en chars ET tokens (seq_length decisions)
- Sample 30 pseudo-queries verifie manuellement

---

## Training pipeline

### Methode : LoRA (pas full fine-tuning)

Full fine-tuning de 308M params sur 1116 exemples = overfitting certain.
Le precedent fine-tune full a degrade recall de 82.84% a 65.69%.
LoRA rank 8 + Dense layers = ~1.7% params entrainables (5.2M sur 308M) :
- LoRA adapters q_proj+v_proj : ~0.5M params
- Dense projection layers (modules 2-3, defrosted par sentence-transformers) : ~4.7M params

| Param LoRA | Valeur | Source |
|------------|--------|--------|
| rank | 8 | FINETUNING_RESOURCES.md §4.2, Unsloth default |
| alpha | 8 | alpha = rank (stabilite, FINETUNING_RESOURCES.md §4.2) |
| dropout | 0.1 | FINETUNING_RESOURCES.md §4.2 |
| target_modules | q_proj, v_proj | FINETUNING_RESOURCES.md §4.2, attention layers |

### Stage 1 — SimCSE

| Param | Valeur | Source |
|-------|--------|--------|
| Batch size | 64 | SimCSE paper Table D.1, BERT-base unsup |
| Learning rate | 2e-5 | Adapte pour LoRA (paper SimCSE = 3e-5 full fine-tuning, LoRA standard = 2e-5) |
| Epochs | 3 | Adapte (1116 ex vs 1M original — regime exploratoire, early stopping obligatoire) |
| Temperature | 0.05 | SimCSE paper §3 (valeur BERT, EmbeddingGemma peut differer — tester 0.02-0.1) |
| Pooling | mean_tokens | EmbeddingGemma default (verifie : model[1].pooling_mode_mean_tokens=True). Note : SimCSE utilise CLS, mais EmbeddingGemma impose mean pooling. |
| Loss | CachedMultipleNegativesRankingLoss | FINETUNING_RESOURCES.md §3.4.1, mini_batch_size=16 |
| mini_batch_size | 16 | Permet batch effectif plus grand, plus de in-batch negatives |
| Batch sampler | NO_DUPLICATES | sbert.net training overview |
| Warmup | 10% | sbert.net default |
| Weight decay | 0.01 | FINETUNING_RESOURCES.md §4.1 |
| Max grad norm | 1.0 | finetune_embeddinggemma.py (archive) |
| Precision | fp32 | T4 = Turing (compute 7.5), PAS de support bf16 (Ampere+ requis). Model card EmbeddingGemma interdit fp16. fp32 = seul choix safe. ~4-6 GB VRAM, T4 16GB OK. |
| Seed | 42 | Reproductibilite |
| Early stopping | patience 3, sur GS recall@5 | GS = seul signal retrieval significatif |

**Note** : pas de eval split sur les données de training (inutile pour SimCSE).
Le seul signal significatif est recall@5 sur GS apres chaque stage.

### Stage 2 — ICT

| Param | Valeur | Source |
|-------|--------|--------|
| Batch size | 64 | Adapte ORQA 4096 (TPU) → T4 |
| Learning rate | 1e-5 | Adapte pour LoRA stage 2 (ORQA paper = 1e-4 full pre-training, reduit pour LoRA continuation) |
| Epochs | 5 | Adapte (1116 ex vs Wikipedia) |
| Masking rate | 90% | ORQA paper §9.2 |
| Loss | CachedMultipleNegativesRankingLoss | Coherent avec stage 1 |
| mini_batch_size | 16 | Coherent avec stage 1 |
| Weight decay | 0.01 | Coherent avec stage 1 |
| Max grad norm | 1.0 | Coherent avec stage 1 |
| Seed | 42 | Reproductibilite |
| Early stopping | patience 3, sur GS recall@5 | Meme signal que stage 1 |

Reprend le checkpoint de stage 1. LR 1e-5 (< stage 1 2e-5) pour ne pas
detruire les poids appris en SimCSE.

### Post-stage : merge et validation (CRITIQUE)

Apres chaque stage :
1. `merge_and_unload()` — merge LoRA dans les poids base (standalone checkpoint)
2. `_hf_peft_config_loaded = False` — reset flag peft (sinon stage 2 a 0 params, issue #3246)
3. `model.save(output_dir)` — sauvegarde sentence-transformers (modules.json preservé)
4. Validation embedding : charger checkpoint, encoder test, verifier shape (1,768) + norm ~1.0 + no NaN

### Evaluation

- **Apres chaque stage** : recall@5 sur GS (298 questions, eval only)
- **Gate rollback** : recall@5 < 60.1% → rollback au checkpoint precedent
- **Training loss** : NaN/Inf assertion, logging_nan_inf_filter=True

---

## Rebuild pipeline

```
DVC snapshot corpus_v2_fr.db (fait 2026-03-20)
    |
    v
Dry-run : chunk + enrichment sans embedding (~30s)
    → Verifier schema, counts, token distribution
    |
    v
Rebuild complet : build_index(..., model_id="models/embeddinggemma-simcse-ict")
    → children + table summaries embedding
    → ~12 min
    |
    v
Integrity gates I1-I9 (9/9 PASS requis)
    |
    v
recall.py → recall@5, MRR
    |
    v
DVC snapshot post-rebuild
```

---

## Fichiers et responsabilites

| Fichier | Action | SRP | Lignes max |
|---------|--------|-----|------------|
| `scripts/training/train_simcse_ict.py` | CREATE | Script Kaggle : config, data, training, eval, save | 250 |
| `scripts/training/ict_data.py` | CREATE | Extraction pseudo-queries ICT depuis chunks | 80 |
| `scripts/training/config.py` | CREATE | Hyperparams, seeds, paths — un seul endroit | 50 |
| `scripts/training/tests/test_ict_data.py` | CREATE | Tests extraction pseudo-queries | 100 |
| `scripts/pipeline/indexer.py` | VERIFY | Accepte model path local (deja parametre) | — |
| `models/model_card.json` | UPDATE | Section finetuning avec valeurs mesurees | — |
| `corpus/processed/corpus_v2_fr.db` | REBUILD | Un seul rebuild apres fine-tuning | — |

### Standards notebook Kaggle

| Standard | Implementation |
|----------|---------------|
| Reproductibilite | Seeds 42 pour random, numpy, torch dans config.py |
| Structure | Setup → Data → Stage 1 → Stage 2 → Eval → Save |
| Config cell | Tous hyperparams importes depuis config.py |
| Versioning | Script dans git, donnees DVC, checkpoint HF Hub |
| Deps explicites | pip install avec versions pinnees en tete |
| Idempotent | Executable top-to-bottom, meme resultat |

### Workflow local ↔ Kaggle

```
Local (git)                          Kaggle (GPU T4)
─────────                            ──────────────
scripts/training/*.py   ──upload──>  Notebook cells
corpus/processed/*.json ──upload──>  Dataset Kaggle
                                     |
                                     v
                                     Training (~10 min : 17 steps/epoch × 8 epochs × ~1s/step)
                                     |
                                     v
                        <──download── embeddinggemma-simcse-ict/
                                     (ou push HF Hub)
```

Le script est le meme fichier .py en local et sur Kaggle.

---

## Artefacts par etape

| Etape | Artefact | Versionning |
|-------|----------|-------------|
| 0. Pre-training | corpus_v2_fr.db | DVC (snapshotte 2026-03-20) |
| 1. Data prep | data/training/simcse_pairs.jsonl | git |
| 1b. Data prep | data/training/ict_pairs.jsonl | git |
| 2. Post-SimCSE | models/embeddinggemma-simcse/ | HF Hub ou DVC |
| 2b. Eval SimCSE | data/benchmarks/recall_post_simcse.json | git |
| 3. Post-ICT | models/embeddinggemma-simcse-ict/ | HF Hub ou DVC |
| 3b. Eval ICT | data/benchmarks/recall_post_ict.json | git |
| 4. Rebuild DB | corpus_v2_fr.db | DVC (nouveau snapshot) |
| 4b. Integrity | I1-I9 PASS | stdout |

Chaque etape est independante. Rollback possible a tout point.

---

## Model card (mise a jour)

Section ajoutee a models/model_card.json :

```json
{
  "finetuning": {
    "method": "SimCSE + ICT (self-supervised, corpus-only, LoRA)",
    "base_model": "google/embeddinggemma-300m",
    "adapter": {
      "type": "LoRA",
      "rank": 8, "alpha": 8, "dropout": 0.1,
      "target_modules": ["q_proj", "v_proj"]
    },
    "training_data": {
      "simcse": "1116 chunk texts (dropout augmentation)",
      "ict": "1116 pairs (first sentence -> full chunk, 90% masking)",
      "source": "corpus/processed/docling_v2_fr/",
      "labels": "NONE — self-supervised"
    },
    "hyperparameters": {
      "simcse": {
        "batch_size": 64, "lr": "2e-5", "epochs": 3,
        "temperature": 0.05, "pooling": "mean_tokens (EmbeddingGemma default, not SimCSE CLS)",
        "loss": "CachedMultipleNegativesRankingLoss",
        "mini_batch_size": 16, "weight_decay": 0.01,
        "max_grad_norm": 1.0, "precision": "fp32 (T4 Turing, no bf16)", "seed": 42
      },
      "ict": {
        "batch_size": 64, "lr": "1e-5", "masking_rate": 0.9,
        "sentence_selection": "random (ORQA standard)",
        "epochs": 5, "loss": "CachedMultipleNegativesRankingLoss",
        "mini_batch_size": 16, "weight_decay": 0.01,
        "max_grad_norm": 1.0, "precision": "fp32 (T4 Turing, no bf16)", "seed": 42
      }
    },
    "evaluation": {
      "metric": "recall@5 page-level",
      "gs": "gold_standard_annales_fr_v8_adversarial.json (eval only, NOT training)",
      "baseline": 0.601,
      "post_simcse": null,
      "post_ict": null
    },
    "infrastructure": {
      "gpu": "Kaggle T4 16GB",
      "framework": "sentence-transformers >= 3.0, peft",
      "precision": "fp32 (T4=Turing no bf16, model card interdit fp16)"
    },
    "standards": {
      "SimCSE": "Gao et al. EMNLP 2021 (arXiv:2104.08821)",
      "ICT": "Lee et al. ACL 2019 (arXiv:1906.00300)",
      "LoRA": "Hu et al. ICLR 2022 (arXiv:2106.09685)"
    },
    "date": null,
    "checkpoints": {
      "post_simcse": "models/embeddinggemma-simcse/",
      "post_ict": "models/embeddinggemma-simcse-ict/"
    }
  }
}
```

Champs null remplis apres chaque etape avec valeurs mesurees.

---

## Contrainte Android / LiteRT

Le modele fine-tune doit etre convertible en .tflite pour le RAG Android.

| Option | Description | Retenu |
|--------|-------------|--------|
| Merge LoRA + reconversion TFLite | merge LoRA → full model → ai-edge-torch → .tflite Mixed Precision | Si gate T1 passe |
| Export embeddings only | DB pre-calculee desktop, query .tflite original | NON — mismatch espaces vectoriels |
| LoRA separe runtime | Base .tflite + LoRA FlatBuffers a l'inference | LiteRT-LM v0.1.0 early preview, instable |

**Option B eliminee** : query embeddings (Android .tflite original) ≠ corpus embeddings
(desktop fine-tune) → cosine search casse.

### Gates TFLite

| Gate | Condition | Action |
|------|-----------|--------|
| T1 | recall@5 ≥ 65.1% (≥+5pp) | Planifier conversion TFLite (chantier separe) |
| T2 | recall@5 < 65.1% (<+5pp) | Rollback, .tflite original conserve |

La conversion TFLite est un chantier separe apres validation du gain.

---

## Risques et mitigations

| Risque | Probabilite | Impact | Mitigation |
|--------|-------------|--------|------------|
| Overfitting (1116 exemples) | Moyenne | Recall degrade | LoRA rank 8 (~0.5% params), weight_decay 0.01, early stopping |
| Gain marginal (<2pp) | Haute | Effort gaspille | Gate : si ≤60.1% apres SimCSE, stop. Budget <1h Kaggle. |
| Pseudo-queries triviales | Moyenne | Pas d'apprentissage | Verifier sample 30, min 20 chars, diversite |
| EmbeddingGemma diverge fp16 | Faible | Training crash | Verifie OK (Tom Aarsen). Fallback bf16/fp32 |
| Catastrophic forgetting | Faible | Embedding degrade | LoRA preserve base weights. Verif 10 queries avant/apres |
| Kaggle timeout | Faible | Perte checkpoint | ~10 min total, save apres chaque stage |
| Temperature 0.05 mal adaptee | Moyenne | Signal contrastif faible | Issu de BERT, pas de EmbeddingGemma. Tester 0.02-0.1 si loss stagne |

---

## Definition of Done

- [ ] simcse_pairs.jsonl : 1116 paires verifiees
- [ ] ict_pairs.jsonl : 1116 paires, pseudo-queries >20 chars, sample 30 OK
- [ ] Stage 1 termine : checkpoint sauve, recall mesure
- [ ] Stage 2 termine : checkpoint sauve, recall mesure
- [ ] Model card mis a jour avec valeurs reelles
- [ ] recall_post_ict.json dans git
- [ ] DB rebuilt, I1-I9 PASS
- [ ] DVC snapshot post-rebuild
- [ ] Gate T1/T2 evaluee, decision documentee

---

## Standards

| Standard | Application |
|----------|-------------|
| SimCSE (EMNLP 2021) | Hyperparams Table D.1, dropout contrastif |
| ICT/ORQA (ACL 2019) | Masking 90%, batch negatives, §7.3 + §9.2 |
| LoRA (ICLR 2022) | Rank 8, alpha 8, target q_proj/v_proj (Hu et al. arXiv:2106.09685) |
| sentence-transformers v3+ + peft | CachedMultipleNegativesRankingLoss, NO_DUPLICATES sampler |
| sbert.net training overview | Warmup 10%, weight_decay 0.01, max_grad_norm 1.0 |
| FINETUNING_RESOURCES.md | Hyperparams §4.1-4.2, LoRA config, CachedMNRL recommande |
| HF blog Tom Aarsen | fp16 compatible EmbeddingGemma, pooling mean_tokens verifie |
| Kaggle best practices | Seeds, structure, deps pinnees, idempotent |
| ISO 42001 | Tracabilite (model card, artefacts, rollback) |
| ISO 29119 | Eval sur GS (pas training), quality gates |
| ISO 25010 | Fichiers ≤ 300 lignes |

---

## Sources

- [SimCSE (EMNLP 2021)](https://arxiv.org/abs/2104.08821)
- [SimCSE GitHub](https://github.com/princeton-nlp/SimCSE)
- [ICT/ORQA (ACL 2019)](https://arxiv.org/abs/1906.00300)
- [LoRA (ICLR 2022)](https://arxiv.org/abs/2106.09685)
- [GTE multi-stage (2023)](https://arxiv.org/abs/2308.03281)
- [sentence-transformers training](https://sbert.net/docs/sentence_transformer/training_overview.html)
- [sentence-transformers unsupervised](https://sbert.net/examples/sentence_transformer/unsupervised_learning/README.html)
- [HF blog EmbeddingGemma](https://huggingface.co/blog/embeddinggemma)
- [Train sentence-transformers v3](https://huggingface.co/blog/train-sentence-transformers)
- [Kaggle notebook best practices](https://mljourney.com/kaggle-notebooks-best-practices-for-ml-experiments/)
- [Kaggle fine-tuning Gemma 3 Unsloth](https://www.kaggle.com/code/kingabzpro/fine-tuning-gemma-3-unsloth)
