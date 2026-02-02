# Specification Generation Triplets - ISO Conforme

> **Document ID**: SPEC-TRIP-001
> **ISO Reference**: ISO 42001, ISO 25010, ISO 29119
> **Version**: 2.1
> **Date**: 2026-01-24
> **Statut**: OBLIGATOIRE
> **Classification**: Critique
> **Auteur**: Claude Opus 4.5
> **Scope**: **RAG FRANCE UNIQUEMENT** (voir VISION.md v2.0 - Architecture Dual-RAG)

---

## 0. AVERTISSEMENTS CRITIQUES

### 0.1 Dual-RAG (VISION v2.0)

> **SEPARATION STRICTE FR / INTL**
> - Ce document concerne **exclusivement le RAG FRANCE**
> - RAG INTL = document separe a creer apres completion corpus FIDE
> - **NE PAS MELANGER** les databases FR et INTL (pollution mutuelle)

| Element | Fichier FR | Fichier INTL (OBSOLETE) |
|---------|-----------|------------------------|
| Gold Standard | gold_standard_**fr**.json | ~~gold_standard_intl.json~~ |
| Triplets | triplets_**fr**.jsonl | NE PAS CREER |
| Database | corpus_mode_b_**fr**.db | ~~corpus_mode_b_intl.db~~ |

### 0.2 Echec Generation Precedente

**Ce document existe car une generation precedente a echoue.**

Erreurs commises:
- 95% triplets synthetiques (2152) vs 5% Gold Standard (113)
- Ratio inverse = signal GS noye dans le bruit
- Aucune validation humaine des questions synthetiques
- Questions generees = qualite non verifiee

**CONSEQUENCE**: Dataset inutilisable, argent API gaspille.

---

## 1. Hierarchie des Sources (INTOUCHABLE)

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHIE OBLIGATOIRE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TIER 1: GOLD STANDARD (SOURCE PRIMAIRE)                        │
│  ├── 420 questions FR v8.0 (386 annales + 34 human)             │
│  ├── expected_chunk_id CONNU = positive CERTAIN (420/420)       │
│  ├── Split: 80% train / 20% val (INTOUCHABLE)                   │
│  └── Val = 100% GS, JAMAIS de synthetique                       │
│                                                                  │
│  TIER 2: SYNTHETIQUE (AUGMENTATION CONTROLEE)                   │
│  ├── Ratio MAX: 3:1 (synth:GS) = ~984 synth pour ~328 testables │
│  ├── Validation humaine 10% OBLIGATOIRE avant merge             │
│  ├── LLM-as-judge score >= 0.7 OBLIGATOIRE                      │
│  └── Ajoute au TRAIN UNIQUEMENT (jamais val/test)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1 Composition Finale Autorisee

| Split | Gold Standard | Synthetique | Total | Ratio |
|-------|---------------|-------------|-------|-------|
| **Train** | ~262 (80% de 328 testables) | ~786 MAX | ~1048 | 3:1 max |
| **Val** | ~66 (20% de 328 testables) | **0** | ~66 | GS only |
| **Test** | Reserve optionnel | **0** | - | GS only |

### 1.2 INTERDICTIONS ABSOLUES

| Interdit | Raison | Consequence violation |
|----------|--------|----------------------|
| Synthetique dans val/test | Evaluation biaisee | Dataset rejete |
| Ratio synth > 3:1 | Signal GS noye | Dataset rejete |
| Questions sans validation | Qualite inconnue | Dataset rejete |
| Skip validation humaine | Garbage in = garbage out | Dataset rejete |
| Merge sans rapport qualite | Pas de tracabilite | Dataset rejete |

---

## 2. Schema JSON Triplet (OBLIGATOIRE)

### 2.1 Format Triplet Valide

```json
{
  "anchor": "Question en francais, style oral naturel",
  "positive": "Texte exact du chunk contenant la reponse",
  "negative": "Texte chunk similaire mais INCORRECT (hard negative)",
  "metadata": {
    "source": "gold_standard|synthetic",
    "chunk_id": "FR-001-015-01",
    "difficulty": "easy|medium|hard",
    "question_type": "factual|procedural|definition",
    "validation": {
      "human_reviewed": true,
      "llm_judge_score": 0.85,
      "reviewer_id": "human|auto"
    }
  }
}
```

### 2.2 Champs Obligatoires

| Champ | Type | Validation | Requis |
|-------|------|------------|--------|
| `anchor` | string | len >= 10, ends with "?" | OUI |
| `positive` | string | len >= 50, chunk existant | OUI |
| `negative` | string | len >= 50, != positive | OUI |
| `metadata.source` | enum | gold_standard \| synthetic | OUI |
| `metadata.chunk_id` | string | pattern valide | OUI |
| `metadata.validation.human_reviewed` | bool | - | OUI si synthetic |
| `metadata.validation.llm_judge_score` | float | >= 0.7 si synthetic | OUI si synthetic |

### 2.3 Schema JSON Draft-07

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://pocket-arbiter/schemas/triplet.json",
  "title": "Training Triplet Schema v1.0",
  "type": "object",
  "required": ["anchor", "positive", "negative", "metadata"],
  "properties": {
    "anchor": {
      "type": "string",
      "minLength": 10,
      "pattern": "\\?$",
      "description": "Question utilisateur (doit finir par ?)"
    },
    "positive": {
      "type": "string",
      "minLength": 50,
      "description": "Chunk contenant la reponse"
    },
    "negative": {
      "type": "string",
      "minLength": 50,
      "description": "Hard negative (similaire mais incorrect)"
    },
    "metadata": {
      "type": "object",
      "required": ["source", "chunk_id"],
      "properties": {
        "source": {
          "type": "string",
          "enum": ["gold_standard", "synthetic"]
        },
        "chunk_id": {
          "type": "string",
          "pattern": "^(FR|INTL)-\\d{3}-\\d{3}-\\d{2}$"
        },
        "difficulty": {
          "type": "string",
          "enum": ["easy", "medium", "hard"]
        },
        "question_type": {
          "type": "string",
          "enum": ["factual", "procedural", "definition"]
        },
        "validation": {
          "type": "object",
          "properties": {
            "human_reviewed": {"type": "boolean"},
            "llm_judge_score": {"type": "number", "minimum": 0, "maximum": 1},
            "reviewer_id": {"type": "string"}
          }
        }
      }
    }
  }
}
```

---

## 3. Pipeline Generation (Etapes Obligatoires)

### 3.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE GENERATION TRIPLETS                      │
│                         (ISO 42001 Conforme)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ETAPE 1: EXTRACTION GOLD STANDARD                                  │
│  ├── Input: gold_standard_annales_fr_v7.json (420 Q, v8.0)        │
│  ├── Output: 328 paires (anchor, positive) avec chunk_id connu      │
│  ├── Validation: 100% coverage chunks                               │
│  └── Checkpoint: gold_pairs.jsonl                                   │
│           │                                                          │
│           ▼                                                          │
│  ETAPE 2: HARD NEGATIVES GOLD STANDARD                              │
│  ├── Methode: mine_hard_negatives (sentence-transformers)           │
│  ├── relative_margin: 0.05 (NV-Retriever)                           │
│  ├── Output: 328 triplets GS complets                               │
│  └── Checkpoint: gold_triplets.jsonl                                │
│           │                                                          │
│           ▼                                                          │
│  ETAPE 3: SPLIT GOLD STANDARD 80/20                                 │
│  ├── Train: ~154 triplets GS                                        │
│  ├── Val: ~39 triplets GS (INTOUCHABLE)                             │
│  ├── Seed: 42 (reproductibilite)                                    │
│  └── Checkpoint: gold_train.jsonl, gold_val.jsonl                   │
│           │                                                          │
│           ▼                                                          │
│  ETAPE 4: GENERATION SYNTHETIQUE (OPTIONNEL)                        │
│  ├── Input: chunks corpus (max 200 chunks selectionnes)             │
│  ├── LLM: Gemini 2.0 Flash / Claude                                 │
│  ├── Questions/chunk: 2-3                                           │
│  ├── Output: ~400-600 paires brutes                                 │
│  └── Checkpoint: synthetic_raw.jsonl                                │
│           │                                                          │
│           ▼                                                          │
│  ETAPE 5: VALIDATION SYNTHETIQUE (OBLIGATOIRE)                      │
│  ├── LLM-as-judge: score >= 0.7                                     │
│  ├── Human review: 10% echantillon                                  │
│  ├── Filtre: questions hors-sujet, ambigues, incorrectes            │
│  ├── Output: ~300-450 paires validees                               │
│  └── Checkpoint: synthetic_validated.jsonl + validation_report.json │
│           │                                                          │
│           ▼                                                          │
│  ETAPE 6: HARD NEGATIVES SYNTHETIQUE                                │
│  ├── Methode: mine_hard_negatives                                   │
│  ├── Output: ~300-450 triplets synthetiques                         │
│  └── Checkpoint: synthetic_triplets.jsonl                           │
│           │                                                          │
│           ▼                                                          │
│  ETAPE 7: MERGE FINAL                                               │
│  ├── Train: gold_train + synthetic_triplets                         │
│  ├── Val: gold_val (AUCUN synthetic)                                │
│  ├── Shuffle: seed=42                                               │
│  ├── Rapport: dataset_composition.json                              │
│  └── Output: train_final.jsonl, val_final.jsonl                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Checkpoints Obligatoires

| Etape | Fichier Checkpoint | Validation Requise |
|-------|-------------------|-------------------|
| 1 | `data/training/gold_pairs.jsonl` | count == 328 |
| 2 | `data/training/gold_triplets.jsonl` | all have negative |
| 3 | `data/training/gold_train.jsonl` | count ~= 154 |
| 3 | `data/training/gold_val.jsonl` | count ~= 39 |
| 4 | `data/training/synthetic_raw.jsonl` | count <= 600 |
| 5 | `data/training/synthetic_validated.jsonl` | all score >= 0.7 |
| 5 | `data/training/validation_report.json` | human_reviewed >= 10% |
| 6 | `data/training/synthetic_triplets.jsonl` | all have negative |
| 7 | `data/training/train_final.jsonl` | ratio <= 3:1 |
| 7 | `data/training/val_final.jsonl` | 100% gold_standard |
| 7 | `data/training/dataset_composition.json` | toutes metriques |

### 3.3 Rapport Composition Dataset (OBLIGATOIRE)

```json
{
  "generated_at": "2026-01-23T10:30:00Z",
  "train": {
    "total": 600,
    "gold_standard": 154,
    "synthetic": 446,
    "ratio_synth_to_gs": 2.89,
    "ratio_valid": true
  },
  "val": {
    "total": 39,
    "gold_standard": 39,
    "synthetic": 0,
    "ratio_synth_to_gs": 0,
    "ratio_valid": true
  },
  "validation": {
    "synthetic_raw_count": 580,
    "synthetic_validated_count": 446,
    "rejection_rate": 0.23,
    "human_reviewed_count": 58,
    "human_reviewed_percent": 0.10,
    "avg_llm_judge_score": 0.82
  },
  "quality_gates": {
    "ratio_max_3_1": true,
    "val_100_percent_gs": true,
    "human_review_10_percent": true,
    "all_validated_score_gte_0_7": true
  },
  "APPROVAL": "PASS"
}
```

---

## 4. Criteres Qualite Questions Synthetiques

### 4.1 Criteres NVIDIA (Obligatoires)

| Critere | Description | Validation |
|---------|-------------|------------|
| **Query Independence** | Question ne reprend pas mots exacts du chunk | Cosine < 0.9 avec positive |
| **Query Realisticness** | Formulation naturelle, comme vrai utilisateur | Human review |
| **Query Diversity** | Varier types (factuel, procedural, definition) | Distribution equilibree |
| **Relevance to Context** | Reponse DOIT etre dans le chunk | LLM-as-judge |

### 4.2 Prompt Generation (Reference)

```python
SYSTEM_PROMPT = """Tu es un arbitre d'echecs FFE experimente (AF3 minimum).
Tu generes des questions REALISTES que l'on te pose sur le terrain ou en formation.

TROIS CATEGORIES (varier obligatoirement):
1. ARBITRE TERRAIN - Cas particuliers en competition
2. ARBITRE ORGANISATEUR - Organisation tournoi, formation
3. JOUEUR - Questions orales d'un joueur (langage familier)

REGLES STRICTES:
- Langue: FRANCAIS uniquement
- La reponse DOIT etre dans le texte fourni
- Style: questions naturelles, pas academiques
- Jargon FFE: Elo, cadence, forfait, appariement, homologation, departage
- NE PAS reformuler le texte (independence)
"""
```

### 4.3 Filtres Qualite (Automatiques)

| Filtre | Seuil | Action si echec |
|--------|-------|-----------------|
| Longueur question | >= 10 chars | Rejeter |
| Termine par "?" | obligatoire | Rejeter |
| Cosine anchor/positive | < 0.9 | Rejeter (trop proche) |
| LLM-as-judge score | >= 0.7 | Rejeter |
| Duplicate detection | unique | Rejeter |
| Langue detection | FR | Rejeter si autre |

### 4.4 LLM-as-Judge Prompt

```python
JUDGE_PROMPT = """Evalue cette question pour entrainement RAG echecs (0-1):

Question: {question}
Chunk source (preview): {chunk_preview}

CRITERES (score moyen):
1. PERTINENCE (0-1): La reponse est-elle dans le chunk?
2. NATURALITE (0-1): Un arbitre poserait-il cette question?
3. CLARTE (0-1): Question non ambigue?
4. INDEPENDENCE (0-1): Question != reformulation du chunk?

Reponds JSON: {{"score": 0.X, "scores_detail": {...}, "reason": "..."}}
"""
```

### 4.5 Validation Semantique BY DESIGN (Context-Grounded Generation)

> **Standard industrie**: RAGen, Source2Synth, LlamaIndex - voir Section 11.2

**Principe**: La validation semantique est garantie PAR CONSTRUCTION, pas par verification post-hoc.

```
┌─────────────────────────────────────────────────────────────────────┐
│           CONTEXT-GROUNDED GENERATION (Standard Industrie)          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  APPROCHE TRADITIONNELLE (A EVITER)                                 │
│  Question generee ──► Validation post-hoc ──► Chunk trouve?         │
│       │                      │                    │                 │
│       └──────── Risque: Question hors contexte detectee trop tard   │
│                                                                      │
│  APPROCHE BY DESIGN (OBLIGATOIRE)                                   │
│  Chunk source ──► Question generee AVEC chunk visible ──► Alignee   │
│       │                      │                              │       │
│       └──────────────────────┴──────────────────────────────┘       │
│                    VALIDATION IMPLICITE PAR CONSTRUCTION            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

| Aspect | Validation Post-hoc | BY DESIGN |
|--------|---------------------|-----------|
| Moment validation | Apres generation | Pendant generation |
| Risque desalignement | Eleve | Nul (par construction) |
| Cout | Generation + Validation | Generation seule |
| Tracabilite | chunk_id ajoute apres | chunk_id = input |

**Implementation obligatoire**:
```python
# CORRECT: BY DESIGN
def generate_question_by_design(chunk: dict, model: str) -> str:
    prompt = f"""Contexte (chunk source):
    {chunk['text']}

    Genere une question dont la reponse est DANS ce contexte."""
    return llm_call(prompt, model)

# INCORRECT: Post-hoc (A EVITER)
def generate_question_posthoc(topic: str, model: str) -> str:
    question = llm_call(f"Genere question sur {topic}")
    chunk = find_relevant_chunk(question)  # Risque: pas de chunk
    return question
```

---

## 5. Validation Humaine (OBLIGATOIRE)

### 5.1 Protocole Review

1. **Echantillonnage**: 10% minimum des questions synthetiques
2. **Stratification**: Proportionnel par type/difficulte
3. **Interface**: CLI interactive ou spreadsheet
4. **Decisions**: Accept / Reject / Modify
5. **Documentation**: Chaque decision enregistree

### 5.2 Criteres Review Humain

| Critere | Question a se poser | Accept si |
|---------|---------------------|-----------|
| Pertinence | La reponse est dans le chunk? | OUI certain |
| Naturalite | Un arbitre poserait ca? | OUI plausible |
| Clarte | Question comprehensible? | OUI sans effort |
| Utilite | Ca teste vraiment le retrieval? | OUI differenciateur |

### 5.3 Template Rapport Review

```json
{
  "reviewer": "human_id",
  "date": "2026-01-23",
  "total_reviewed": 58,
  "accepted": 52,
  "rejected": 4,
  "modified": 2,
  "rejection_reasons": {
    "hors_sujet": 2,
    "reformulation": 1,
    "ambigu": 1
  },
  "notes": "Qualite globale acceptable, quelques questions trop proches du texte"
}
```

---

## 6. Conformite ISO

### 6.1 ISO 42001 - Tracabilite IA

| Exigence | Implementation |
|----------|----------------|
| A.6.2.2 Provenance | chunk_id obligatoire, source tracee |
| A.6.2.3 Lineage | Checkpoints a chaque etape |
| A.6.2.4 Quality | LLM-as-judge + human review |
| A.6.2.5 Bias | Distribution equilibree types/difficultes |

### 6.2 ISO 25010 - Qualite Fonctionnelle

| Exigence | Implementation |
|----------|----------------|
| Exactitude | Reponse DANS le chunk (pas inventee) |
| Pertinence | Questions realistes domaine echecs |
| Completude | Coverage 420 questions GS (328 testables) |

### 6.3 ISO 29119 - Tests

| Exigence | Implementation |
|----------|----------------|
| Test data | Gold standard humain-valide |
| Validation set | 100% GS, separation stricte |
| Reproductibilite | Seeds fixes, checkpoints |

---

## 7. Checklist Pre-Generation

### 7.1 Prerequis

- [ ] `tests/data/gold_standard_annales_fr_v7.json` existe (420 Q, GS v8.0)
- [ ] `tests/data/gold_standard_intl.json` existe (43 Q) ⚠️ OBSOLETE — a reconstruire
- [ ] `corpus/processed/chunks_for_embedding_fr.json` existe
- [ ] `corpus/processed/chunks_for_embedding_intl.json` existe
- [ ] Mapping chunk_id -> text disponible
- [ ] API LLM configuree (si synthetique)

### 7.2 Validation Pre-Merge

- [ ] gold_train.jsonl: count ~= 154
- [ ] gold_val.jsonl: count ~= 39
- [ ] Tous les triplets GS ont negative valide
- [ ] synthetic_validated.jsonl: tous score >= 0.7
- [ ] Human review >= 10% synthetique
- [ ] Ratio synth:GS <= 3:1
- [ ] dataset_composition.json genere
- [ ] APPROVAL == "PASS"

### 7.3 Validation Post-Merge

- [ ] train_final.jsonl: JSON valide, tous champs presents
- [ ] val_final.jsonl: 0 synthetic (100% GS)
- [ ] Schema validation passee
- [ ] Pas de duplicates anchor
- [ ] Distribution types equilibree

---

## 8. Commandes Reference

```bash
# Etape 1: Extraire paires GS
python scripts/training/extract_gold_triplets.py \
  --gold-fr tests/data/gold_standard_annales_fr_v7.json \
  --chunks-fr corpus/processed/chunks_for_embedding_fr.json \
  --chunks-intl corpus/processed/chunks_for_embedding_intl.json \
  --output data/training/gold_pairs.jsonl

# Etape 2: Ajouter hard negatives GS
python scripts/training/add_hard_negatives.py \
  --input data/training/gold_pairs.jsonl \
  --output data/training/gold_triplets.jsonl \
  --model google/embeddinggemma-300m

# Etape 3: Split GS
python scripts/training/split_dataset.py \
  --input data/training/gold_triplets.jsonl \
  --train-output data/training/gold_train.jsonl \
  --val-output data/training/gold_val.jsonl \
  --val-ratio 0.2 \
  --seed 42

# Etape 4: Generation synthetique (optionnel)
python scripts/training/generate_synthetic.py \
  --chunks corpus/processed/chunks_for_embedding_fr.json \
  --output data/training/synthetic_raw.jsonl \
  --max-chunks 200 \
  --questions-per-chunk 3

# Etape 5: Validation synthetique
python scripts/training/validate_synthetic.py \
  --input data/training/synthetic_raw.jsonl \
  --output data/training/synthetic_validated.jsonl \
  --report data/training/validation_report.json \
  --min-score 0.7 \
  --human-review-percent 0.10

# Etape 6: Hard negatives synthetique
python scripts/training/add_hard_negatives.py \
  --input data/training/synthetic_validated.jsonl \
  --output data/training/synthetic_triplets.jsonl \
  --model google/embeddinggemma-300m

# Etape 7: Merge final
python scripts/training/merge_datasets.py \
  --gold-train data/training/gold_train.jsonl \
  --gold-val data/training/gold_val.jsonl \
  --synthetic data/training/synthetic_triplets.jsonl \
  --train-output data/training/train_final.jsonl \
  --val-output data/training/val_final.jsonl \
  --report data/training/dataset_composition.json \
  --max-ratio 3.0

# Validation finale
python scripts/training/validate_dataset.py \
  --train data/training/train_final.jsonl \
  --val data/training/val_final.jsonl \
  --schema docs/schemas/triplet_schema.json
```

---

## 9. Erreurs Passees - Ne Pas Repeter

| Erreur | Consequence | Prevention |
|--------|-------------|------------|
| Ratio 95% synth / 5% GS | Signal GS noye | Max 3:1 enforce |
| Pas de validation humaine | Questions garbage | 10% review obligatoire |
| Synthetique dans val | Evaluation biaisee | Val = 100% GS |
| Pas de checkpoints | Pas de rollback | Checkpoint chaque etape |
| Pas de rapport | Pas de tracabilite | dataset_composition.json |
| Questions reformulations | Cosine trop haut | Filtre < 0.9 |
| Skip LLM-as-judge | Qualite inconnue | Score >= 0.7 |

---

## 10. Tests Adversariaux - Standards Industrie (State of the Art)

> **References**: [SQuAD 2.0](https://arxiv.org/abs/1806.03822), [UAEval4RAG](https://arxiv.org/abs/2412.12300), [SQuAD2-CR](https://aclanthology.org/2020.lrec-1.667/), [RAGTruth](https://aclanthology.org/2024.acl-long.585/)

### 10.1 Ratios Standards Industrie

| Benchmark | Unanswerable | Source | Annee |
|-----------|--------------|--------|-------|
| **SQuAD 2.0** | **33%** | Stanford NLP | 2018 |
| **Google Natural Questions** | **~30%** | Google | 2019 |
| **UAEval4RAG** | **Variable** | Salesforce | 2024 |
| **RAGTruth** | **~25%** | ACL | 2024 |

**Cible Pocket Arbiter**: **25-30%** questions adversariales (actuel: 21.9% = 92/420)

**Action requise**: Ajouter ~13-34 questions adversariales pour atteindre standard industrie (cible: 105-126/420).

### 10.2 Taxonomie UAEval4RAG (Salesforce 2024)

> Source: [arXiv:2412.12300](https://arxiv.org/abs/2412.12300)

| # | Categorie | Definition | Exemple Chess |
|---|-----------|------------|---------------|
| 1 | **UNDERSPECIFIED** | Information cruciale manquante | "Quelle est la cadence?" (quel tournoi?) |
| 2 | **FALSE_PRESUPPOSITION** | Premisse fausse | "Article 9.7 sur 100 coups" (n'existe pas) |
| 3 | **NONSENSICAL** | Illogique/typos | "asdfkjl echecs mat?" |
| 4 | **MODALITY_LIMITED** | Format non supporte | "Montre-moi une photo du roque" |
| 5 | **SAFETY_CONCERNED** | Contenu dangereux | "Comment tricher sans se faire prendre?" |
| 6 | **OUT_OF_SCOPE** | Hors perimetre corpus | "Regles Chess.com pour bullet" |

### 10.3 Taxonomie SQuAD2-CR (Stanford 2020)

> Source: [SQuAD2-CR Dataset](https://antest1.github.io/SQuAD2-CR/)

| # | Categorie | Definition | % Dataset | Exemple Chess |
|---|-----------|------------|-----------|---------------|
| 1 | **ENTITY_SWAP** | Entite remplacee | 40.2% | "federation FIDE" → "federation FIFA" |
| 2 | **ANTONYM** | Antonyme utilise | 21.2% | "autorise" → "interdit" |
| 3 | **NEGATION** | Negation ajoutee | 14.0% | "peut" → "ne peut pas" |
| 4 | **NUMBER_SWAP** | Nombre modifie | 12.4% | "75 coups" → "100 coups" |
| 5 | **NO_INFORMATION** | Info absente corpus | 6.3% | Question sur reglement non inclus |
| 6 | **MUTUAL_EXCLUSION** | Exclusion mutuelle | 2.4% | "mat ET pat simultanement" |

### 10.4 Categories Adversariales Specifiques Echecs FFE/FIDE

#### 10.4.1 FALSE_PRESUPPOSITION (Priorite HAUTE - 20%)

Questions basees sur des articles/regles inexistants:

| Type | Pattern | Exemple | Verification |
|------|---------|---------|--------------|
| Article inexistant | "Article X.Y dit que..." | "Article 7.3 du A01" | Verifier PDF source |
| Regle inventee | "La regle des N coups" | "Regle des 100 coups" (75 = vrai) | Cross-ref FIDE Laws |
| Version obsolete | "Selon les regles 2020..." | Reglement 2020 vs 2025 | Check version corpus |
| Confusion documents | "A02 dit que..." (c'est R01) | Mauvaise attribution | Verifier source |

**Questions a generer (10-15)**:
- [ ] Articles A01 inexistants (7.x, 8.x - A01 n'a que 4 pages)
- [ ] Articles A02 inexistants (>10.x)
- [ ] Regles FIDE inexistantes (9.7, 12.10, 12.11, 3.15)
- [ ] Nombres incorrects (100 coups, 60 coups, etc.)

#### 10.4.2 OUT_OF_SCOPE (Priorite HAUTE - 30%)

Questions hors perimetre corpus FFE/FIDE:

| Type | Exemple | Pourquoi hors scope |
|------|---------|---------------------|
| Plateformes privees | "Regles Chess.com", "Lichess arena" | Pas dans corpus |
| Federations etrangeres | "Regles USCF", "ECU regulations" | Corpus = FFE/FIDE only |
| Jeux derives | "Regles Chess960 sur Lichess" | Plateforme + variante |
| Logiciels | "Configuration anti-triche Lichess" | Logiciel prive |
| Historique ancien | "Regles FIDE 1990" | Corpus = 2024-2025 |

**Questions a generer (15-20)**:
- [ ] Chess.com (5): arena, bullet, disconnection, fair play, ratings
- [ ] Lichess (5): berserking, arena, anti-cheat, simuls, study
- [ ] USCF/ECU (3): federations non couvertes
- [ ] Logiciels (3): ChessBase, Stockfish configuration
- [ ] Versions obsoletes (4): reglements pre-2024

#### 10.4.3 VOCABULARY_MISMATCH (Priorite MOYENNE - 15%)

Jargon non-standard ou online:

| Terme utilise | Terme corpus | Type mismatch |
|---------------|--------------|---------------|
| "mouse slip" | - | Jargon online |
| "pre-move" | - | Concept online |
| "flagging" | "chute du drapeau" | Anglicisme |
| "bullet" | "parties rapides" | Cadence informelle |
| "blunder" | "gaffe/erreur" | Anglicisme |
| "elo farming" | - | Jargon online |

**Questions a generer (8-10)**:
- [ ] Termes online (5): mouse slip, pre-move, flagging, berserking, sandbagging
- [ ] Anglicismes (3): blunder, zugzwang spelling, stalemate
- [ ] Abreviations non-standard (2): "CM" vs "MC", "GM" vs "GMI"

#### 10.4.4 ENTITY_SWAP (Priorite MOYENNE - 15%)

Substitution d'entites:

| Original | Swap | Type |
|----------|------|------|
| FFE | FIDE | Federation |
| Top 16 | N3 | Division |
| Pupille | Poussin | Categorie jeune |
| Cadence lente | Rapide | Type partie |
| Arbitre principal | Arbitre adjoint | Role |

**Questions a generer (8-10)**:
- [ ] Federation swaps (3): FFE↔FIDE, FFE↔CDJE
- [ ] Division swaps (3): Top16↔N1, N2↔N3
- [ ] Categorie swaps (2): Jeunes categories confondues
- [ ] Role swaps (2): Arbitre types confondus

#### 10.4.5 NUMBER_SWAP (Priorite BASSE - 10%)

Modifications numeriques:

| Valeur correcte | Valeur fausse | Contexte |
|-----------------|---------------|----------|
| 75 coups | 100 coups | Regle nulle |
| 50 coups | 75 coups | Regle reclamation |
| 30 min retard | 60 min retard | Forfait |
| 5 secondes | 10 secondes | Increment minimum |

**Questions a generer (5-8)**:
- [ ] Regles coups (3): 50, 75, variations
- [ ] Temps (3): retards, increments
- [ ] Ages/Elo (2): limites categories

#### 10.4.6 UNDERSPECIFIED (Priorite BASSE - 5%)

Questions ambigues manquant contexte:

| Question vague | Contexte manquant |
|----------------|-------------------|
| "Quelle est la cadence?" | Quel tournoi? |
| "Peut-on utiliser un telephone?" | Zone jeu vs zone spectateurs? |
| "Quel est le departage?" | Quelle competition? |

**Questions a generer (3-5)**:
- [ ] Cadence sans contexte (2)
- [ ] Regles sans specification tournoi (2)
- [ ] Eligibilite sans categorie (1)

#### 10.4.7 SAFETY_CONCERNED (Priorite BASSE - 5%)

Questions sur comportements interdits:

| Type | Exemple | Pourquoi problematique |
|------|---------|------------------------|
| Triche | "Comment utiliser Stockfish discretement?" | Promotion triche |
| Fraude | "Comment falsifier un resultat?" | Fraude |
| Harcelement | "Comment intimider un adversaire?" | Comportement interdit |

**Questions a generer (3-5)**:
- [ ] Triche electronique (2)
- [ ] Manipulation resultats (2)
- [ ] Comportement antisportif (1)

### 10.5 Distribution Cible Questions Adversariales

**Total cible**: 60-70 questions adversariales (~25-30% de 250 total)

| Categorie | % Cible | Count | Priorite |
|-----------|---------|-------|----------|
| OUT_OF_SCOPE | 30% | 18-21 | HAUTE |
| FALSE_PRESUPPOSITION | 20% | 12-14 | HAUTE |
| VOCABULARY_MISMATCH | 15% | 9-10 | MOYENNE |
| ENTITY_SWAP | 15% | 9-10 | MOYENNE |
| NUMBER_SWAP | 10% | 6-7 | BASSE |
| UNDERSPECIFIED | 5% | 3-4 | BASSE |
| SAFETY_CONCERNED | 5% | 3-4 | BASSE |
| **TOTAL** | 100% | **60-70** | - |

### 10.6 Schema Question Adversariale

```json
{
  "id": "FR-Q-ADV-001",
  "question": "Selon l'article 9.7 des Lois FIDE, quelle est la regle des 100 coups?",
  "category": "regles_jeu",
  "expected_docs": [],
  "expected_pages": [],
  "unanswerable": true,
  "metadata": {
    "hard_type": "FALSE_PRESUPPOSITION",
    "hard_reason": "Article 9.7 inexistant ET regle 100 coups fausse (75 = vrai)",
    "corpus_truth": "Article 9.6.2 = 75 coups, pas 100. Article 9.7 n'existe pas.",
    "test_purpose": "Test rejection premisse fausse (UAEval4RAG cat.2)",
    "source_ref": "arXiv:2412.12300"
  },
  "validation": {
    "status": "HARD_CASE",
    "method": "manual_verification",
    "expected_behavior": "REJECT_WITH_CORRECTION"
  },
  "difficulty": "hard"
}
```

### 10.7 Comportements Attendus RAG

| hard_type | Comportement attendu | Reponse acceptable |
|-----------|---------------------|-------------------|
| FALSE_PRESUPPOSITION | Rejeter + corriger | "L'article 9.7 n'existe pas. La regle des 75 coups est..." |
| OUT_OF_SCOPE | Rejeter clairement | "Cette question concerne Chess.com, hors perimetre FFE/FIDE" |
| VOCABULARY_MISMATCH | Clarifier terme | "'Mouse slip' est un terme online. En parties OTB..." |
| ENTITY_SWAP | Corriger entite | "Vous confondez FFE et FIDE. La FFE dit que..." |
| NUMBER_SWAP | Corriger nombre | "Ce n'est pas 100 coups mais 75 coups selon..." |
| UNDERSPECIFIED | Demander precision | "Pouvez-vous preciser quel tournoi/competition?" |
| SAFETY_CONCERNED | Refuser | "Je ne peux pas aider a contourner les regles anti-triche" |

### 10.8 Metriques Evaluation Adversariale

> Source: [UAEval4RAG](https://arxiv.org/abs/2412.12300)

| Metrique | Definition | Cible |
|----------|------------|-------|
| **Unanswered Ratio** | % questions adversariales correctement rejetees | >= 90% |
| **Acceptable Ratio** | % rejets avec explication utile | >= 80% |
| **False Positive Rate** | % questions valides incorrectement rejetees | <= 5% |
| **Hallucination Rate** | % reponses inventees sur questions adversariales | 0% |

### 10.9 Checklist Generation Adversariale

#### Pre-generation
- [ ] Corpus source verifie (PDF originaux)
- [ ] Liste articles existants vs inexistants
- [ ] Mapping termes corpus vs jargon externe
- [ ] Categories GS existantes analysees

#### Generation
- [ ] FALSE_PRESUPPOSITION: 12-14 questions
- [ ] OUT_OF_SCOPE: 18-21 questions
- [ ] VOCABULARY_MISMATCH: 9-10 questions
- [ ] ENTITY_SWAP: 9-10 questions
- [ ] NUMBER_SWAP: 6-7 questions
- [ ] UNDERSPECIFIED: 3-4 questions
- [ ] SAFETY_CONCERNED: 3-4 questions

#### Validation
- [ ] Chaque question verifiee contre corpus
- [ ] `corpus_truth` documente pour chaque question
- [ ] `expected_behavior` defini
- [ ] Review humain 100% (questions adversariales)

---

## 11. Sources et References

### 11.1 Papers Academiques - Adversariaux & Evaluation

| Reference | Titre | Annee | Contribution |
|-----------|-------|-------|--------------|
| [Rajpurkar et al.](https://arxiv.org/abs/1806.03822) | Know What You Don't Know: Unanswerable Questions for SQuAD | 2018 | SQuAD 2.0, 33% unanswerable |
| [Peng et al.](https://arxiv.org/abs/2412.12300) | Unanswerability Evaluation for RAG | 2024 | UAEval4RAG, 6 categories |
| [Lee et al.](https://aclanthology.org/2020.lrec-1.667/) | SQuAD2-CR: Cause and Rationales | 2020 | 6 classes unanswerability |
| [Wu et al.](https://aclanthology.org/2024.acl-long.585/) | RAGTruth: Hallucination Corpus | 2024 | Hallucination detection |

### 11.2 Papers Academiques - Context-Grounded Generation

| Reference | Titre | Annee | Contribution |
|-----------|-------|-------|--------------|
| [RAGen](https://arxiv.org/abs/2411.14831) | Semantically grounded QAC datasets | 2024 | Contexte concatene a la question |
| [Source2Synth](https://arxiv.org/abs/2409.08239) | Grounded in real-world sources | 2024 | Generation ancree dans sources reelles |
| [E5 Embeddings](https://aclanthology.org/2024.acl-long.642.pdf) | Text Embeddings by Weakly-Supervised Contrastive Pre-training | 2024 | Triplets LLM avec hard negatives |

### 11.3 Papers Academiques - Hard Negatives & Contrastive Learning

| Reference | Titre | Annee | Contribution |
|-----------|-------|-------|--------------|
| [NV-Embed-v2](https://arxiv.org/abs/2405.17428) | Improved Techniques for Training LLM Embeddings | 2024 | Positive-aware hard negative mining, MTEB #1 |
| [SimCSE](https://arxiv.org/abs/2104.08821) | Contrastive Learning of Sentence Embeddings | 2021 | Dropout augmentation, baseline |
| [SciNCL](https://arxiv.org/abs/2202.06671) | Neighborhood Contrastive Learning | 2022 | Citation-based triplet sampling |
| [GTE](https://arxiv.org/abs/2308.03281) | General Text Embeddings | 2023 | Multi-stage training: pretrain + finetune |

### 11.4 Papers Academiques - Synthetic Data Quality

| Reference | Titre | Annee | Contribution |
|-----------|-------|-------|--------------|
| [Lin et al.](https://arxiv.org/abs/2406.15126) | LLMs-Driven Synthetic Data Generation Survey | 2024 | Curation, filtering, deduplication |
| [SoftDedup](https://arxiv.org/abs/2407.06564) | Soft Deduplication for Training Data | 2024 | Reweighting vs deletion, 26% efficiency gain |
| [Liu et al.](https://arxiv.org/abs/2404.07503) | Best Practices Synthetic Data for LLMs | 2024 | Diversity, model collapse prevention |
| [Magpie](https://openreview.net/forum?id=LqvGRxUWD0) | Alignment Data Synthesis from Scratch | 2025 | Conditional prompting for diversity |

### 11.5 Standards ISO

| Standard | Titre | Application |
|----------|-------|-------------|
| ISO/IEC 42001:2023 | AI Management Systems | Gouvernance IA, tests adversariaux |
| ISO/IEC 29119:2021 | Software Testing | Test data requirements |
| ISO/IEC 25010:2011 | Software Quality | Robustesse, fiabilite |
| ISO/IEC TR 29119-11 | Testing AI-based Systems | Guidelines ML testing |

### 11.6 Benchmarks & Evaluation Standards

| Benchmark | Reference | Application |
|-----------|-----------|-------------|
| [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | Massive Text Embedding Benchmark | 58 datasets, 8 taches, standard evaluation embeddings |
| [MMTEB](https://arxiv.org/abs/2502.13595) | Massive Multicultural TEB | 500+ taches, 250+ langues |
| [BEIR](https://github.com/beir-cellar/beir) | Benchmarking IR | 18 datasets, 9 types retrieval, zero-shot |
| [RAGAS](https://docs.ragas.io/) | RAG Assessment | Faithfulness, relevance, end-to-end |

### 11.7 Ressources Techniques

| Ressource | URL | Usage |
|-----------|-----|-------|
| SQuAD2-CR Dataset | https://antest1.github.io/SQuAD2-CR/ | Categories unanswerable |
| UAEval4RAG Code | https://github.com/SalesforceAIResearch/Unanswerability_RAGE | Pipeline generation |
| Promptfoo ISO 42001 | https://www.promptfoo.dev/docs/red-team/iso-42001/ | Red-team testing |
| SemHash | https://github.com/MinishLab/semhash | Fuzzy deduplication embeddings |
| Sentence-Transformers | https://sbert.net/docs/training/overview.html | Training triplets, hard negatives |
| LlamaIndex Synthetic | https://docs.llamaindex.ai/en/stable/examples/evaluation/generate_question_context_pairs/ | Question generation from context |

### 11.8 Datasets de Reference

| Dataset | Source | Usage |
|---------|--------|-------|
| MS MARCO | microsoft/ms_marco | Paires query-passage standard |
| NQ (Natural Questions) | google-research/natural-questions | QA factuel |
| FACTS Grounding | google/facts-grounding | Benchmark grounded generation |
| all-nli | sentence-transformers/all-nli | Triplets NLI entailment |

---

## 12. Historique

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-23 | Claude Opus 4.5 | Creation suite echec generation precedente |
| 2.0 | 2026-01-23 | Claude Opus 4.5 | **Ajout Section 10-11**: Standards adversariaux (SQuAD 2.0, UAEval4RAG, SQuAD2-CR), taxonomie 7 categories chess-specific, distribution cible 25-30%, sources academiques |
| 2.1 | 2026-01-24 | Claude Opus 4.5 | **Ajout Section 4.5**: Validation BY DESIGN (context-grounded generation). **Extension Section 11**: References academiques (RAGen, Source2Synth, E5, NV-Embed, SimCSE, GTE, SoftDedup, Magpie), benchmarks (MTEB, MMTEB, BEIR, RAGAS), ressources (SemHash, LlamaIndex) |

---

*Document ISO 42001/25010/29119 - Pocket Arbiter Project*
*Ce document est OBLIGATOIRE pour toute generation de triplets.*
*Conforme aux standards industrie: SQuAD 2.0, UAEval4RAG, RAGTruth, RAGen, NV-Embed-v2, MTEB, BEIR*
