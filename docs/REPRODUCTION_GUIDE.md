# Pocket Arbiter — Guide de Reproduction Pipeline

> **Document ID**: DOC-REPRO-001
> **ISO References**: ISO/IEC 42001:2023 (AI lifecycle), ISO/IEC 12207:2017 (Software lifecycle),
>   ISO/IEC 5338:2023 (AI lifecycle processes), CRISP-DM 1.0
> **Version**: 1.0
> **Date**: 2026-03-23
> **Statut**: Approuve
> **Classification**: Interne
> **Objectif**: Permettre la reconstruction complete du pipeline chaque saison (nouveaux reglements FFE)

---

## 0. Conformite standards

Ce guide respecte les standards suivants pour la reproductibilite ML/AI :

| Standard | Clause | Application |
|----------|--------|-------------|
| ISO/IEC 42001:2023 | A.6.2.3 Data lineage | Tracabilite complete des donnees (DVC, checksums) |
| ISO/IEC 42001:2023 | A.6.2.6 Model lifecycle | 4 phases documentees, gates a chaque etape |
| ISO/IEC 42001:2023 | A.7.2 Verification | Gates automatiques (I1-I9, G1, R1, P1-P4) |
| ISO/IEC 42001:2023 | A.8.2 Documentation | Configurations, hyperparametres, seeds documentes |
| ISO/IEC 5338:2023 | 6.2.3 Data management | DVC versioning, dataset composition |
| ISO/IEC 5338:2023 | 6.3.2 Model training | Hyperparametres, metriques, reproductibilite |
| ISO/IEC 5338:2023 | 6.3.4 Model evaluation | Gates quantitatives, eval humaine |
| ISO/IEC 12207:2017 | 6.4.8 Configuration mgmt | DVC + Git, conventional commits |
| CRISP-DM 1.0 | All phases | Data → Modeling → Evaluation → Deployment |
| ML Reproducibility (Pineau 2021) | Checklist | Seeds, configs, environments documentes |

---

## 1. Vue d'ensemble

Le pipeline se reproduit en 4 phases independantes (CRISP-DM: Data Preparation → Modeling → Evaluation).
Chaque phase a ses entrees, sorties, et gates de validation (ISO 42001 A.7.2).

```
Phase 1: Extraction + Chunking + Enrichment + Indexation  (~2h)
    -> corpus_v2_fr.db (SQLite, DVC tracked)
    -> Gates: I1-I9 (integrite)

Phase 2: Validation recall                                 (~10 min)
    -> recall@5 >= 60% (gate R1)
    -> Gates: R1-R4 (qualite retrieval)

Phase 3: Generation fine-tuning (Kaggle T4)                (~1h30)
    -> TAPT checkpoint + SFT checkpoint (DVC tracked)
    -> Gates: G1 (perplexity), overfit ratio < 2.0

Phase 4: Evaluation comparative                            (~30 min)
    -> generation_eval_{base,tapt,sft}.json
    -> Gates: P1-P4 (format), G4a-G4b (qualite generation)
```

---

## 2. Prerequis

### 2.1 Environnement local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
pip install pre-commit xenon pip-audit
python -m pre_commit install
python -m pre_commit install --hook-type commit-msg --hook-type pre-push
```

### 2.2 Comptes externes

| Service | Compte | Usage |
|---------|--------|-------|
| Kaggle | pguillemin | GPU T4, datasets, kernels |
| HuggingFace | Pierrax | Modele base (google/gemma-3-270m-it) |
| DVC remote | C:/Dev/dvc_remote (local) | Versioning DB et modeles |

### 2.3 Fichiers source (a mettre a jour chaque saison)

```
corpus/fr/   # 28 PDFs FFE (telecharger depuis ffe.com)
corpus/intl/ # 1 PDF FIDE (telecharger depuis fide.com)
```

---

## 3. Phase 1 : Extraction + Indexation

### 3.1 Extraction PDF

```bash
python scripts/pipeline/extract.py
```

| Parametre | Valeur | Notes |
|-----------|--------|-------|
| Backend | Docling ML + docling-hierarchical-pdf | Heading levels reels |
| Input | corpus/fr/*.pdf (28 fichiers) | ~300 pages total |
| Output | JSON avec markdown, tables, heading_pages | Intermediaire |
| Duree | ~1h (LA = 222 pages) | Le plus lent |

### 3.2 Chunking

Pipeline 6 etapes dans `chunker.py` + `chunker_utils.py` :

| Etape | Description | Standard |
|-------|-------------|----------|
| 1 | Extraction tables AVANT header split | 98 vs 22 tables sinon |
| 2 | MarkdownHeaderTextSplitter [#, ##, ###, ####] | LangChain |
| 3 | RecursiveCharacterTextSplitter (tiktoken) | Azure 2026 |
| 4 | Parent assembly (group par h1+h2, cap 2048) | GraphRAG |
| 5 | Page interpolation (line-level) | Custom |
| 6 | Table linkage | LangChain multi-vector |

**Configuration critique** (dans `chunker_utils.py`) :

```python
CHUNK_SIZE = 450       # Firecrawl 2026 benchmark
CHUNK_OVERLAP = 50     # 11% overlap (teste 0/50/100)
PARENT_MAX_TOKENS = 2048
MERGE_THRESHOLD = 200  # Fusionner children < 200 tokens
TABLE_MIN_LINES = 3
```

**Output attendu** :
- 1116 children (median ~350 tokens)
- 282 parents (cap 2048 tokens)
- 111 tables extraites

### 3.3 Enrichissement

Module `enrichment.py`, 3 optimisations :

| OPT | Description | Impact mesure |
|-----|-------------|---------------|
| OPT-1 | Contextual retrieval (Anthropic 2024) | +3.7pp R@1 |
| OPT-2 | Abbreviation expansion (12 termes) | +0.3pp |
| OPT-4 | Chapter overrides LA (5 ranges pages) | +0.3pp |

**Fichier contextes** : `corpus/processed/chunk_contexts.json` (1116 entries)
- Genere par LLM, median 54 tokens par contexte
- A regenerer si le corpus change (les contextes referencent des sections specifiques)

**Abbreviations** (dictionnaire fixe) :

```
AFC, AFJ, AI, CDJE, CM, DNA, FFE, FIDE, FM, GMI, MI, UV
```

**Chapter overrides** (LA-octobre2025.pdf uniquement) :

```
pages 56-57  → "Annexe A - Cadence Rapide"
pages 58-66  → "Annexe B - Cadence Blitz"
pages 182-186 → "Classement Elo Standard FIDE"
pages 187-191 → "Classement Rapide et Blitz FIDE"
pages 192-205 → "Titres FIDE"
```

### 3.4 Indexation

Module `indexer.py` → produit `corpus_v2_fr.db` (SQLite) :

| Table | Rows | Contenu |
|-------|------|---------|
| children | 1116 | Chunks enrichis + embeddings 768D |
| parents | 282 | Texte parent (pas d'embedding) |
| table_summaries | 111 | Summaries + raw_table_text + embeddings |
| structured_cells | 4308 | Cellules parsees (col_name, cell_value) |
| children_fts | 1116 | Index FTS5 BM25 |
| table_summaries_fts | 111 | Index FTS5 BM25 |

**Modele embedding** : EmbeddingGemma-300M base (`google/embeddinggemma-300m`)
- Dimensions : 768
- Prompt : `title: {cch_title} | text: {enriched_text}`
- Batch size : 128
- Normalisation : L2

**Gates integrite (I1-I9)** : toutes doivent passer apres le build.

### 3.5 DVC versioning

```bash
python -m dvc add corpus/processed/corpus_v2_fr.db
python -m dvc push
git add corpus/processed/corpus_v2_fr.db.dvc
git commit -m "data: rebuild corpus v2 FR (saison 20XX-20YY)"
```

---

## 4. Phase 2 : Validation recall

```bash
python scripts/pipeline/recall.py \
  tests/data/gold_standard_annales_fr_v8_adversarial.json \
  corpus/processed/corpus_v2_fr.db
```

**Metriques attendues** (baseline mars 2026) :

| Metrique | Valeur | Gate |
|----------|--------|------|
| recall@1 | 38.9% | — |
| recall@5 | 60.1% | R1 (seuil 70% — FAIL connu) |
| recall@10 | 63.8% | — |
| MRR | 0.479 | R4 (seuil 0.50 — FAIL connu) |

**Note** : le GS devra etre mis a jour si les reglements changent significativement.

---

## 5. Phase 3 : Generation fine-tuning (Kaggle)

### 5.1 Preparation des donnees

```bash
# Generer corpus_paragraphs.jsonl (TAPT)
# → depuis les chunks enrichis, regrouper en paragraphes
# Fichier: kaggle/dataset-generation/corpus_paragraphs.jsonl

# Generer reading_tasks.jsonl (SFT)
# → AdaptLLM regex mining avec connecteurs FR
# Fichier: kaggle/dataset-generation/reading_tasks.jsonl

# Uploader le dataset
kaggle datasets create -p kaggle/dataset-generation/
```

### 5.2 TAPT (Kernel 1)

**Script** : `kaggle/kernel-generation/train_generation.py`
**Input** : base model + corpus_paragraphs.jsonl

| Parametre | Valeur |
|-----------|--------|
| Modele base | google/gemma-3-270m-it |
| Methode | Full Fine-Tuning fp32+AMP |
| Epochs | 5 |
| Batch (effective) | 16 (batch 1 × grad_accum 16) |
| LR | 5e-6 |
| Scheduler | cosine |
| Warmup | 10% |
| NEFTune alpha | 5 |
| Seq length | 2048 |
| Seed | 42 |
| GPU | Tesla T4 (15 GB) |
| Duree | ~37 min |

**Gate G1** : perplexity post-TAPT < perplexity baseline

**Output** : checkpoint dans `/kaggle/working/gemma-270m-cpt/`

```bash
# Telecharger le checkpoint
kaggle kernels output pguillemin/pocket-arbiter-cpt-sft-generation -p models/kaggle-output/

# Uploader comme dataset pour le kernel SFT
cp models/kaggle-output/gemma-270m-cpt/{model.safetensors,config.json,tokenizer.json,tokenizer_config.json,generation_config.json,chat_template.jinja} kaggle/tapt-checkpoint/
kaggle datasets create -p kaggle/tapt-checkpoint/
```

### 5.3 SFT (Kernel 2)

**Script** : `kaggle/kernel-sft/train_sft.py`
**Input** : TAPT checkpoint + reading_tasks.jsonl

| Parametre | Valeur |
|-----------|--------|
| Modele base | TAPT checkpoint |
| Methode | SFTTrainer (trl==0.16.0) fp16+gradient_checkpointing |
| Epochs | 3 |
| Batch (effective) | 16 (batch 1 × grad_accum 16) |
| LR | 2e-5 |
| Scheduler | cosine |
| Warmup | 10% |
| NEFTune alpha | 5 |
| Seq length | 1024 (OOM mitigation, vocab 262K) |
| Eval split | 10% holdout |
| eval_strategy | "no" (eval manuelle post-training, OOM sinon) |
| Seed | 42 |
| GPU | Tesla T4 (15 GB) |
| Duree | ~20 min |

**Gates** :
- Loss finale < 2.0
- Overfit ratio (eval_loss / train_loss_manual) < 2.0

**Output** : checkpoint dans `/kaggle/working/gemma-270m-cpt-sft/`

```bash
# Telecharger le checkpoint
kaggle kernels output pguillemin/pocket-arbiter-sft-generation -p models/kaggle-sft-output/

# DVC tracking
python -m dvc add models/kaggle-output models/kaggle-sft-output
python -m dvc push
git add models/kaggle-output.dvc models/kaggle-sft-output.dvc
```

### 5.4 Piege connu : OOM eval

Le vocabulaire Gemma (262K tokens) cause un OOM pendant l'eval Trainer :
- model (~1 GB) + optimizer (~7 GB) + eval logits fp32 (~5 GB) = 13 GB > T4 15 GB
- **Solution** : `eval_strategy="no"`, eval manuelle post-training apres `del trainer`
- La fonction `compute_clm_loss()` dans le script SFT gere cette eval manuelle

---

## 6. Phase 4 : Evaluation comparative

### 6.1 Kernel eval (Kaggle)

**Script** : `kaggle/kernel-eval/eval_generation_kaggle.py`

**Input** :
- 3 modeles : base, TAPT, SFT (depuis Kaggle datasets)
- DB : corpus_v2_fr.db (depuis Kaggle dataset eval-data)
- GS : gold_standard_annales_fr_v8_adversarial.json

**Prompt systeme** (dans `kaggle/eval-data/generation_prompt.py`) :

```
Tu es un assistant pour arbitres d'echecs. Tu reponds aux questions
en te basant UNIQUEMENT sur le contexte fourni (extraits des
reglements FFE/FIDE).

REGLES:
- Cite TOUJOURS le document source et l'article/section.
- Si le contexte ne contient pas la reponse, dis
  'Information non trouvee dans les extraits fournis.'
- Ne reponds JAMAIS avec des informations hors contexte.
- Reponds en francais.
- Sois concis et actionnable.
```

| Parametre | Valeur |
|-----------|--------|
| Temperature | 0.7 |
| Max tokens | 256 |
| do_sample | True |
| Questions | 298 testables (34 human + 264 annales) |

**Output** : 3 fichiers JSON dans `data/benchmarks/`

```bash
kaggle kernels output pguillemin/pocket-arbiter-eval-generation-3-models -p data/benchmarks/
```

### 6.2 Gates evaluation

| Gate | Verification | Seuil |
|------|-------------|-------|
| P1 | 3 fichiers, meme nombre de questions | = |
| P2 | Reponses non vides | SFT < base |
| P3 | 264 annales presentes dans chaque fichier | 264 |
| P4 | Longueur moyenne reponses | > 20 tokens |
| G4a | Qualite globale (eval humaine) | >= 70% |
| G4b | Citations automatiques | >= 80% |

### 6.3 Resultats mars 2026 (reference)

| Modele | Empty | Citations auto | Temps |
|--------|-------|---------------|-------|
| Base | 71/298 (24%) | 21.6% | 9.8 min |
| TAPT | 9/298 (3%) | 34.1% | 33.8 min |
| SFT | **0/298 (0%)** | 33.0% | 7.1 min |

---

## 7. Matrice de configuration

Tous les parametres critiques en un seul endroit.

### 7.1 Retrieval

| Parametre | Valeur | Fichier | Justification |
|-----------|--------|---------|---------------|
| chunk_size | 450 | chunker_utils.py | Firecrawl 2026 |
| chunk_overlap | 50 | chunker_utils.py | 11%, teste 0/50/100 |
| parent_max_tokens | 2048 | chunker_utils.py | GraphRAG standard |
| embedding_model | google/embeddinggemma-300m | indexer_embed.py | MTEB #1 < 500M |
| embedding_dim | 768 | indexer_embed.py | EmbeddingGemma natif |
| rrf_k | 60 | search.py | Standard RRF |
| structured_weight | 1.5 | search.py | Boost tables |
| synonymes | 70 entries | query_expansion.py | FR chess + Snowball |

### 7.2 Generation

| Parametre | Valeur TAPT | Valeur SFT | Fichier |
|-----------|-------------|------------|---------|
| epochs | 5 | 3 | kernel scripts |
| batch_size | 1 | 1 | kernel scripts |
| grad_accum | 16 | 16 | kernel scripts |
| lr | 5e-6 | 2e-5 | kernel scripts |
| seq_length | 2048 | 1024 | kernel scripts |
| neftune_alpha | 5 | 5 | kernel scripts |
| seed | 42 | 42 | kernel scripts |

---

## 8. Arbre des fichiers critiques

```
pocket_arbiter/
├── corpus/
│   ├── fr/                     # 28 PDFs FFE (INPUT — a mettre a jour)
│   ├── intl/                   # 1 PDF FIDE (INPUT)
│   └── processed/
│       ├── corpus_v2_fr.db     # DB finale (DVC tracked)
│       ├── chunk_contexts.json # Contextes enrichissement (a regenerer)
│       └── *.json              # Chunks, parents, tables intermediaires
│
├── scripts/pipeline/
│   ├── extract.py              # Phase 1: extraction PDF
│   ├── chunker.py              # Phase 1: chunking
│   ├── chunker_utils.py        # Phase 1: config + utils chunking
│   ├── enrichment.py           # Phase 1: OPT 1-2-4
│   ├── indexer.py              # Phase 1: build DB
│   ├── indexer_embed.py        # Phase 1: embeddings
│   ├── indexer_db.py           # Phase 1: schema SQLite
│   ├── search.py               # Phase 2: recherche hybride
│   ├── recall.py               # Phase 2: mesure recall
│   ├── integrity.py            # Phase 1: gates I1-I9
│   └── tests/                  # 312 tests, 80% coverage
│
├── kaggle/
│   ├── kernel-generation/      # Kernel TAPT
│   │   └── train_generation.py
│   ├── kernel-sft/             # Kernel SFT
│   │   └── train_sft.py
│   ├── kernel-eval/            # Kernel eval 3 modeles
│   │   └── eval_generation_kaggle.py
│   ├── eval-data/              # Dataset eval (DB + GS + scripts)
│   ├── dataset-generation/     # Dataset training (corpus + tasks)
│   ├── tapt-checkpoint/        # TAPT model (Kaggle dataset)
│   └── sft-checkpoint/         # SFT model (Kaggle dataset)
│
├── models/
│   ├── model_card.json         # Documentation modeles (ce fichier)
│   ├── kaggle-output.dvc       # DVC: TAPT outputs (7.1 GB)
│   └── kaggle-sft-output.dvc   # DVC: SFT outputs (11 GB)
│
├── data/benchmarks/
│   ├── generation_eval_base.json  # Eval base model
│   ├── generation_eval_tapt.json  # Eval TAPT model
│   └── generation_eval.json       # Eval SFT model
│
└── tests/data/
    └── gold_standard_annales_fr_v8_adversarial.json  # 403 questions
```

---

## 9. Checklist reproduction annuelle

### Avant de commencer

- [ ] Telecharger les nouveaux PDFs FFE depuis ffe.com
- [ ] Placer dans `corpus/fr/` (remplacer les anciens)
- [ ] Verifier l'environnement Python (`pip install -r requirements.txt`)
- [ ] Verifier DVC (`python -m dvc status`)

### Phase 1 : Extraction + Indexation

- [ ] `python scripts/pipeline/extract.py` (1h)
- [ ] Verifier : ~1000+ children, ~250+ parents, ~100+ tables
- [ ] Regenerer `chunk_contexts.json` si le corpus a change
- [ ] Verifier les chapter overrides (pages LA) — peuvent changer si LA repagine
- [ ] `python scripts/pipeline/build_pipeline.py` ou script equivalent
- [ ] Gates I1-I9 PASS
- [ ] `python -m dvc add corpus/processed/corpus_v2_fr.db && python -m dvc push`

### Phase 2 : Validation recall

- [ ] `python scripts/pipeline/recall.py` sur GS existant
- [ ] Si recall@5 < 50% → probleme extraction ou chunking
- [ ] Si recall@5 > 55% → acceptable (le GS sera probablement a mettre a jour aussi)

### Phase 3 : Fine-tuning

- [ ] Mettre a jour `corpus_paragraphs.jsonl` (depuis nouveaux chunks)
- [ ] Mettre a jour `reading_tasks.jsonl` (re-miner avec AdaptLLM regex)
- [ ] Upload dataset Kaggle : `kaggle datasets create -p kaggle/dataset-generation/`
- [ ] Lancer kernel TAPT → verifier Gate G1 (perplexity decrease)
- [ ] Telecharger + uploader TAPT checkpoint comme dataset
- [ ] Lancer kernel SFT → verifier overfit ratio < 2.0
- [ ] Telecharger SFT checkpoint
- [ ] DVC track : `python -m dvc add models/kaggle-output models/kaggle-sft-output && python -m dvc push`

### Phase 4 : Evaluation

- [ ] Mettre a jour `kaggle/eval-data/` (nouvelle DB + GS)
- [ ] Upload dataset eval : `kaggle datasets create -p kaggle/eval-data/`
- [ ] Lancer kernel eval → telecharger outputs
- [ ] Verifier : SFT empty_responses < TAPT < base
- [ ] Eval humaine 34 questions (optionnel mais recommande)

### Post-reproduction

- [ ] Commit conventionnel : `data: rebuild corpus saison 20XX-20YY`
- [ ] Mettre a jour `models/model_card.json` (dates, metriques)
- [ ] Mettre a jour `docs/PROJECT_HISTORY.md` (nouvelle ere)
- [ ] Pre-commit hooks PASS : `python -m pre_commit run --all-files`
- [ ] Tests PASS : `python -m pytest scripts/iso/ scripts/pipeline/tests/ -m "not slow" -v`

---

## 10. Lineage et tracabilite (ISO 42001 A.6.2.3)

### 10.1 Data lineage

```
PDF sources (corpus/fr/, corpus/intl/)
  │ hash: sha256 de chaque PDF
  ▼
Extraction (Docling ML)
  │ version: docling>=2.68.0, docling-hierarchical-pdf>=0.1.5
  ▼
Chunking (LangChain)
  │ config: CHUNK_SIZE=450, OVERLAP=50, PARENT_MAX=2048
  │ version: langchain-text-splitters>=0.3.0
  ▼
Enrichment (OPT 1-2-4)
  │ contextes: chunk_contexts.json (1116 entries, LLM-generated)
  │ abbreviations: 12 termes fixes
  │ overrides: 5 ranges pages LA
  ▼
Indexation (EmbeddingGemma-300M)
  │ model: google/embeddinggemma-300m (base, NOT QAT)
  │ version: sentence-transformers>=5.2.0
  │ dim: 768, prompt: "title: {cch} | text: {enriched}"
  ▼
corpus_v2_fr.db (DVC tracked)
  │ hash: dans models/kaggle-output.dvc
  ▼
TAPT (Full FT, 5 epochs)
  │ base: google/gemma-3-270m-it
  │ data: corpus_paragraphs.jsonl (5529 para, 381K tokens)
  │ seed: 42
  ▼
SFT (SFTTrainer, 3 epochs)
  │ base: TAPT checkpoint
  │ data: reading_tasks.jsonl (1802 tasks, AdaptLLM)
  │ seed: 42
  ▼
Eval (3 modeles × 298 questions)
  │ prompt: generation_prompt.py
  │ temp: 0.7, max_tokens: 256
  ▼
generation_eval_{base,tapt,sft}.json
```

### 10.2 Reproducibilite (Pineau 2021 checklist)

| Critere | Implementation | Verification |
|---------|---------------|--------------|
| Seeds fixes | SEED=42 dans tous les scripts | grep SEED kaggle/ |
| Versions packages | requirements.txt (pinned) | pip freeze |
| Configs documentes | Section 7 (matrice) + model_card.json | Ce document |
| Donnees versionnees | DVC (corpus, modeles, training data) | dvc status |
| Hardware documente | Tesla T4, 15 GB VRAM | sft_metrics.json |
| Metriques reproductibles | Loss curves dans trainer_state.json | 306 entries |
| Environment | Python 3.10+, CUDA 12.x | Kaggle T4 standard |

### 10.3 Verification d'integrite

```bash
# Verifier les hash DVC (donnees non corrompues)
python -m dvc status

# Verifier la DB
python -c "
import sqlite3
conn = sqlite3.connect('corpus/processed/corpus_v2_fr.db')
print('children:', conn.execute('SELECT COUNT(*) FROM children').fetchone()[0])
print('parents:', conn.execute('SELECT COUNT(*) FROM parents').fetchone()[0])
print('tables:', conn.execute('SELECT COUNT(*) FROM table_summaries').fetchone()[0])
print('cells:', conn.execute('SELECT COUNT(*) FROM structured_cells').fetchone()[0])
conn.close()
"

# Verifier les modeles (taille attendue)
# TAPT: model.safetensors ~1023 MB
# SFT:  model.safetensors ~1023 MB
```

---

## 11. Risques et mitigations (ISO 42001 6.1)

| Risque | Probabilite | Impact | Mitigation |
|--------|-------------|--------|------------|
| FFE change le format PDF | Moyenne | Eleve | Tester extraction sur 1 PDF avant batch complet |
| Docling change d'API | Faible | Moyen | Version pinned dans requirements.txt |
| Kaggle supprime T4 gratuit | Faible | Eleve | Scripts compatibles Colab/local avec GPU 16GB |
| EmbeddingGemma deprecie | Faible | Critique | Pas de remplacement <500MB connu, garder .tflite |
| GS desaligne post-update | Moyenne | Moyen | Verifier chunk_match sur 30 questions echantillon |
| Gemma 3 270M deprecie | Faible | Moyen | Checkpoint sauve localement + DVC |
| OOM Kaggle (vocab 262K) | Connu | Moyen | eval_strategy="no", eval manuelle post-training |

---

## 12. References

| Reference | Usage |
|-----------|-------|
| [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) | AI management system, data lineage, model lifecycle |
| [ISO/IEC 5338:2023](https://www.iso.org/standard/81118.html) | AI system lifecycle processes |
| [ISO/IEC 12207:2017](https://www.iso.org/standard/63712.html) | Software lifecycle, configuration management |
| [CRISP-DM 1.0](https://www.datascience-pm.com/crisp-dm-2/) | Data mining process model |
| [ML Reproducibility Checklist (Pineau 2021)](https://arxiv.org/abs/2003.12206) | Seeds, configs, environments |
| [Anthropic Contextual Retrieval 2024](https://www.anthropic.com/news/contextual-retrieval) | OPT-1 |
| [AdaptLLM (Cheng ICLR 2024)](https://arxiv.org/abs/2309.09530) | Reading comprehension SFT |
| [NEFTune (Jain ICLR 2024)](https://arxiv.org/abs/2310.05914) | Noise embedding fine-tuning |
| [Firecrawl Chunking Benchmark 2026](https://www.firecrawl.dev/blog/best-chunking-strategies-rag) | chunk_size=450 |
| [DVC Documentation](https://dvc.org/doc) | Data versioning |

---

## 13. Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-03-23 | Creation initiale — pipeline complet, standards ISO 42001/5338/12207 |
