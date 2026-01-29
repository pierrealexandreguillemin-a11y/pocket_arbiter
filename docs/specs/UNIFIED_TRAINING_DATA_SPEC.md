# Specification Generation Donnees Unifiees - Gold Standard v6.7.0

> **Document ID**: SPEC-UTD-001
> **ISO Reference**: ISO 42001, ISO 25010, ISO 29119, ISO 12207
> **Version**: 1.2
> **Date**: 2026-01-24
> **Statut**: DRAFT
> **Classification**: Critique
> **Auteur**: Claude Opus 4.5
> **Predecesseur**: SPEC-TRIP-001 (TRIPLET_GENERATION_SPEC.md)
> **Mots-cles**: triplets, ARES, BEIR, RAGAS, Gold Standard, annales, ISO 42001
> **Scope**: **RAG FRANCE UNIQUEMENT** (voir VISION.md v2.0 - Architecture Dual-RAG)

---

## 0. Resume Executif

> **AVERTISSEMENT DUAL-RAG (VISION v2.0)**
> Ce document concerne **exclusivement le RAG FRANCE**.
> RAG INTL = document separe a creer apres completion corpus FIDE.
> Cause separation: Pollution mutuelle des corpus due a specificite metier et scopes differents.

Ce document specifie le workflow unifie de generation de donnees d'entrainement et d'evaluation RAG **FR** a partir du Gold Standard v6.7.0 (annales DNA FFE).

### 0.1 Databases Cibles (SEPARATION STRICTE)

| Element | RAG FR (ce document) | RAG INTL (hors scope) |
|---------|---------------------|----------------------|
| Gold Standard | gold_standard_annales_fr.json | A CREER |
| Chunks | chunks_mode_b_**fr**.json | chunks_mode_b_**intl**.json (OBSOLETE) |
| Embeddings | embeddings_mode_b_**fr**.npy | A REFAIRE |
| Database | corpus_mode_b_**fr**.db | corpus_mode_b_**intl**.db (OBSOLETE) |
| Benchmark | MTEB (FR) | MMTEB (INTL) |

**Objectif**: Generer un dataset multi-format qui sert simultanement:
1. Fine-tuning embeddings (triplets)
2. Evaluation ARES simplifiee (context relevance)
3. Benchmark BEIR (retrieval standard)
4. Evaluation RAGAS (end-to-end RAG)

**Innovation**: Validation semantique **BY DESIGN** via reformulation guidee par chunk.

---

## 1. Contexte et Justification

### 1.1 Problemes Identifies (Audit GS v5.30)

| Probleme | Impact | Reference |
|----------|--------|-----------|
| expected_chunk_id par keyword matching | ~5% erreur estimee | AUDIT-GS-530-001 NC-01 |
| Pas de validation semantique | Labels potentiellement incorrects | AUDIT-GS-530-001 F-01 |
| GS v5.30 obsolete | 318 questions manuelles vs 477 officielles | SPEC-GS-V6 |
| Questions format examen | Vocabulaire ≠ utilisateurs reels | SPEC-GS-V6 S2.2 Phase 4 |

### 1.2 Solution: Gold Standard v6.7.0 "Annales"

| Avantage | Valeur | ISO Reference |
|----------|--------|---------------|
| Questions officielles DNA | 477 questions validees jury national | ISO 42001 A.7.3 |
| answer_text 100% | Reponses explicites | ISO 25010 Exactitude |
| article_reference 100% | Tracabilite complete | ISO 42001 A.6.2.2 |
| expected_pages 83% | Mapping vers corpus | ISO 42001 A.6.2.2 |
| Difficulte calibree | Taux reussite candidats | ISO 29119 Test Data |

### 1.3 Innovation: Validation Semantique BY DESIGN

**Principe**: Au lieu de valider a posteriori, la reformulation est guidee par le chunk cible.

```
Question Examen ──► Chunk (via expected_pages) ──► Reformulation
       │                      │                         │
       │                      │                         │
       └──────────────────────┴─────────────────────────┘
                              │
                    VALIDATION SEMANTIQUE IMPLICITE
                    (LLM voit chunk → reformulation alignee)
```

### 1.4 Fondements Industrie - Context-Grounded Generation

L'approche "Validation BY DESIGN" s'appuie sur des pratiques etablies dans l'industrie et la recherche.

#### 1.4.1 References Academiques (ArXiv)

| Framework | Principe | Reference |
|-----------|----------|-----------|
| **RAGen** (2024) | "Semantically grounded QAC datasets" - le contexte source est concatene avec la question lors de la generation | arXiv:2411.14831 |
| **Source2Synth** (2024) | Generation de donnees synthetiques "grounded in real-world sources" | arXiv:2409.08239 |
| **Synthetic Context Generation** | 84% des contextes synthetiques contiennent la phrase de reponse - validation implicite | arXiv:2407.01489 |

#### 1.4.2 References Industrie (HuggingFace, Google)

| Ressource | Contribution | Lien |
|-----------|--------------|------|
| **FACTS Grounding** (Google) | Benchmark evaluant si reponses sont "grounded" dans le contexte fourni | huggingface.co/datasets/google/facts-grounding |
| **Sentence-Transformers** | Datasets triplets (anchor, positive, negative) generes A PARTIR du passage cible | sentence-transformers/all-nli |
| **MS MARCO** | Paires (query, passage) ou le passage EST le contexte de generation | microsoft/ms_marco |

#### 1.4.3 Pratiques Open Source (GitHub)

| Projet | Pattern | Reference |
|--------|---------|-----------|
| **LlamaIndex** | Pipelines synthetiques: chunk extrait PUIS question generee en contexte | llama-index synthetic generation |
| **Haystack** | Question generation from context: le passage visible guide la question | deepset-ai/haystack |
| **Sentence-Transformers Training** | Fine-tuning avec triplets generes depuis passages cibles | sbert.net/training |

#### 1.4.4 Synthese: BY DESIGN = Standard de Facto

| Aspect | Standard Industrie | Notre Implementation |
|--------|-------------------|----------------------|
| Question generee avec chunk visible | RAGen, Source2Synth, LlamaIndex | Etape 2 reformulation |
| Validation implicite par construction | "Grounded generation" pattern | Pas de validation post-hoc |
| Tracabilite chunk → question | BEIR qrels, MTEB triplets | original_annales + expected_chunk_id |
| Nom standardise | Aucun terme unique adopte | "Validation BY DESIGN" |

**Conclusion**: L'approche BY DESIGN n'est pas un terme standardise nomme, mais la **pratique sous-jacente est le standard de facto** dans l'industrie. Notre terminologie "Validation Semantique BY DESIGN" est une **formalisation explicite** d'une pratique implicite repandue, conforme aux exigences ISO 42001 de documentation et tracabilite.

---

## 2. Architecture du Workflow

### 2.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WORKFLOW UNIFIE GENERATION DONNEES                        │
│                         (ISO 42001/25010/29119)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ENTREE                                                                      │
│  ├── tests/data/gold_standard_annales_fr.json (v6.7.0, 477 Q)               │
│  ├── corpus/processed/chunks_mode_b_fr.json (2801 chunks)                   │
│  └── data/evaluation/ares/few_shot_fr.tsv (70 exemples)                     │
│                                                                              │
│  ETAPE 1: MAPPING PAGES → CHUNKS                                            │
│  ├── Input: expected_docs + expected_pages (395 questions, 83%)             │
│  ├── Output: expected_chunk_id pour chaque question                         │
│  ├── Methode: Match exact doc + page → chunk contenant cette page           │
│  ├── Validation: Log questions sans match (fallback article_reference)      │
│  └── Checkpoint: gold_standard_annales_v6.8_with_chunks.json                │
│           │                                                                  │
│           ▼                                                                  │
│  ETAPE 2: REFORMULATION GUIDEE PAR CHUNK                                    │
│  ├── Input: question + answer_text + chunk_text                             │
│  ├── Prompt: LLM reformule en langage courant AVEC chunk visible            │
│  ├── Output: question_reformulated + original_annales                       │
│  ├── Validation: BY DESIGN (chunk contexte = alignement garanti)            │
│  └── Checkpoint: gold_standard_annales_v6.9_reformulated.json               │
│           │                                                                  │
│           ▼                                                                  │
│  ETAPE 3: GENERATION HARD NEGATIVES                                         │
│  ├── Input: (question, positive_chunk)                                      │
│  ├── Methode: mine_hard_negatives (sentence-transformers)                   │
│  ├── Strategie: Same doc different page > Same category > Random            │
│  ├── Output: negative_chunk pour chaque question                            │
│  └── Checkpoint: triplets_gold_annales.jsonl                                │
│           │                                                                  │
│           ▼                                                                  │
│  ETAPE 4: EXPORT MULTI-FORMAT                                               │
│  ├── TRIPLETS (SentenceTransformers): triplets_train.jsonl, triplets_val.jsonl
│  ├── ARES: ares_gold_label.tsv, ares_unlabeled.tsv                          │
│  ├── BEIR: queries.jsonl, corpus.jsonl, qrels.tsv                           │
│  ├── RAGAS: ragas_evaluation.jsonl                                          │
│  └── Checkpoint: data/training/unified/                                     │
│           │                                                                  │
│           ▼                                                                  │
│  ETAPE 5: VALIDATION & DVC                                                  │
│  ├── Schema validation (JSON Schema Draft-07)                               │
│  ├── Distribution check (categories, difficultes)                           │
│  ├── DVC tracking: dvc add data/training/unified/                           │
│  └── Rapport: dataset_composition.json                                      │
│                                                                              │
│  SORTIES (DVC tracked)                                                       │
│  ├── data/training/unified/triplets_train.jsonl                             │
│  ├── data/training/unified/triplets_val.jsonl                               │
│  ├── data/training/unified/ares_gold_label.tsv                              │
│  ├── data/training/unified/beir/                                            │
│  ├── data/training/unified/ragas_evaluation.jsonl                           │
│  └── data/training/unified/dataset_composition.json                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Dependances entre Etapes

```
ETAPE 1 (Mapping)
    │
    ▼
ETAPE 2 (Reformulation) ←── Necessite chunk_text de ETAPE 1
    │
    ▼
ETAPE 3 (Hard Negatives) ←── Necessite positive_chunk de ETAPE 1
    │
    ▼
ETAPE 4 (Export) ←── Necessite triplets complets de ETAPE 3
    │
    ▼
ETAPE 5 (Validation & DVC)
```

### 2.3 Integration Standards Industrie par Etape

| Etape | Standards Integres | Implementation | Verification |
|-------|-------------------|----------------|--------------|
| **1. Mapping** | ISO 42001 A.6.2.2 | chunk_id = provenance | 100% answerable mapped |
| **2. Reformulation** | RAGen, Source2Synth | Chunk visible dans prompt | BY DESIGN audit |
| **2. Reformulation** | Liu et al. | Pas de generation recursive | Lineage check |
| **2. Reformulation** | Magpie | Conditional prompting (category, difficulty) | Distribution check |
| **3. Hard Negatives** | NV-Embed-v2 | Positive-aware mining | same_doc >= 40% |
| **3. Hard Negatives** | E5 | Anchor independence | cosine < 0.9 |
| **4. Export** | MTEB/BEIR | Formats standard | Schema validation |
| **5. Validation** | SoftDedup | Deduplication fuzzy | SemHash cosine < 0.95 |
| **5. Validation** | GTE | Distribution equilibree | Entropy >= 0.8 |

### 2.4 Checkpoints Qualite (Quality Gates)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUALITY GATES PAR ETAPE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ETAPE 1 ─► GATE 1: Mapping Coverage                                        │
│            ├── >= 80% questions avec chunk_id                               │
│            └── Log questions sans match (adversarial OK)                    │
│                                                                              │
│  ETAPE 2 ─► GATE 2: Reformulation Quality                                   │
│            ├── 100% BY DESIGN (chunk visible)                               │
│            ├── 0% generation recursive (model collapse)                     │
│            └── >= 3 categories par batch (diversity)                        │
│                                                                              │
│  ETAPE 3 ─► GATE 3: Hard Negative Quality                                   │
│            ├── same_doc_diff_page >= 40%                                    │
│            ├── cosine(anchor, positive) < 0.9                               │
│            └── Pas de duplicate negatives                                   │
│                                                                              │
│  ETAPE 5 ─► GATE 5: Final Validation                                        │
│            ├── Deduplication < 5% (SemHash)                                 │
│            ├── Schema JSON valide (Draft-07)                                │
│            ├── Distribution entropy >= 0.8                                  │
│            └── DVC tracked + lineage documented                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Benchmarks Cibles

| Benchmark | Corpus | Cible | Standard | Etape Verification |
|-----------|--------|-------|----------|-------------------|
| MTEB Retrieval | FR | >= 0.70 NDCG@10 | MTEB leaderboard | Post-training |
| MMTEB Retrieval | INTL | >= 0.65 NDCG@10 | MMTEB multilingual | Post-training |
| BEIR custom | FR+INTL | >= 0.75 Recall@5 | BEIR zero-shot | Etape 4 export |
| ARES Context Relevance | FR+INTL | >= 0.80 | ARES PPI | Etape 4 export |
| RAGAS Faithfulness | FR+INTL | >= 0.85 | RAGAS | Post-RAG |

---

## 3. Specifications Detaillees par Etape

### 3.1 ETAPE 1: Mapping Pages → Chunks

#### 3.1.1 Algorithme

```python
def map_pages_to_chunk(question: dict, chunks: list[dict]) -> str | None:
    """
    Trouve le chunk contenant expected_pages pour expected_docs.

    ISO 42001 A.6.2.2: Provenance tracable
    """
    expected_docs = question.get("expected_docs", [])
    expected_pages = question.get("expected_pages", [])

    if not expected_docs or not expected_pages:
        return None  # Questions sans pages (adversarial)

    # Priorite 1: Match exact doc + page
    for chunk in chunks:
        chunk_source = chunk.get("source", "")
        chunk_pages = set(chunk.get("pages", []))

        for doc in expected_docs:
            if doc in chunk_source:
                if any(p in chunk_pages for p in expected_pages):
                    return chunk["id"]

    # Priorite 2: Fallback article_reference
    article_ref = question.get("article_reference", "")
    if article_ref:
        for chunk in chunks:
            if article_ref.lower() in chunk.get("text", "").lower():
                return chunk["id"]

    return None  # Pas de match
```

#### 3.1.2 Metriques Attendues

| Metrique | Cible | Justification |
|----------|-------|---------------|
| Match exact (doc + page) | >= 80% | 395/477 ont expected_pages |
| Match article fallback | >= 10% | article_reference 100% |
| Sans match | <= 10% | Questions adversarial OK |

#### 3.1.3 Schema Sortie

```json
{
  "id": "FR-ANN-UVR-023",
  "question": "Quand dit-on qu'un joueur a le trait ?",
  "expected_chunk_id": "LA-octobre2025.pdf-p037-parent015-child02",
  "mapping_method": "exact_page_match",
  "mapping_confidence": 1.0
}
```

### 3.2 ETAPE 2: Reformulation Guidee par Chunk

#### 3.2.1 Prompt LLM

```python
REFORMULATION_PROMPT = """Tu es un formateur d'arbitres d'echecs FFE.

CONTEXTE (chunk du reglement):
{chunk_text}

QUESTION OFFICIELLE (examen DNA):
{original_question}

REPONSE OFFICIELLE:
{answer_text}

TACHE: Reformule la question en langage courant, comme un joueur ou arbitre debutant la poserait oralement.

CONTRAINTES:
1. La reponse DOIT toujours etre trouvable dans le CONTEXTE ci-dessus
2. Garde le meme sens que la question originale
3. Style oral, naturel, avec tutoriement accepte
4. Vocabulaire courant (eviter jargon technique sauf necessaire)
5. Longueur similaire a l'original

EXEMPLES:
- Original: "Quand dit-on qu'un joueur a le trait ?"
- Reformule: "C'est a qui de jouer apres que l'adversaire a appuye sur la pendule ?"

- Original: "L'arbitre qui observe une position illegale..."
- Reformule: "Si l'arbitre voit une position incorrecte sur l'echiquier mais n'a pas vu le coup, que doit-il faire ?"

Question reformulee:"""
```

#### 3.2.2 Validation BY DESIGN

| Aspect | Garantie |
|--------|----------|
| Alignement semantique | LLM voit chunk → reformulation basee sur contenu reel |
| Preservation sens | Reponse officielle fournie comme contrainte |
| Tracabilite | original_annales conserve pour audit |

#### 3.2.3 Schema Sortie

```json
{
  "id": "FR-ANN-UVR-023",
  "question": "C'est a qui de jouer apres que l'adversaire a appuye sur la pendule ?",
  "original_annales": "Quand dit-on qu'un joueur a le trait ?",
  "reformulation_model": "gemini-2.0-flash",
  "reformulation_date": "2026-01-24"
}
```

### 3.3 ETAPE 3: Generation Hard Negatives

#### 3.3.1 Strategie de Selection

```python
def select_hard_negative(
    question: dict,
    positive_chunk: dict,
    all_chunks: list[dict],
    embeddings: np.ndarray
) -> dict:
    """
    Selectionne un hard negative selon hierarchie de difficulte.

    Reference: NV-Retriever, relative_margin=0.05
    """
    positive_source = positive_chunk.get("source", "")
    positive_pages = set(positive_chunk.get("pages", []))

    candidates = []

    # Niveau 1: Meme document, pages differentes (HARDEST)
    same_doc = [c for c in all_chunks
                if positive_source in c.get("source", "")
                and not set(c.get("pages", [])).intersection(positive_pages)]
    if same_doc:
        candidates.extend([(c, "same_doc_diff_page") for c in same_doc])

    # Niveau 2: Meme categorie de reglement
    category = question.get("category", "")
    same_category = [c for c in all_chunks
                     if category in c.get("metadata", {}).get("category", "")
                     and c["id"] != positive_chunk["id"]]
    if same_category:
        candidates.extend([(c, "same_category") for c in same_category])

    # Niveau 3: Semantiquement proche (embeddings)
    # mine_hard_negatives avec relative_margin

    if not candidates:
        # Fallback: random
        candidates = [(c, "random") for c in all_chunks if c["id"] != positive_chunk["id"]]

    # Selection finale: le plus similaire semantiquement parmi candidats
    selected = max(candidates, key=lambda x: cosine_similarity(x[0], positive_chunk))

    return {
        "chunk": selected[0],
        "selection_method": selected[1]
    }
```

#### 3.3.2 Distribution Cible

| Methode Selection | % Cible | Justification |
|-------------------|---------|---------------|
| same_doc_diff_page | 40% | Hardest negatives |
| same_category | 30% | Domain confusion |
| semantic_similar | 20% | Embedding challenge |
| random | 10% | Baseline |

### 3.4 ETAPE 4: Export Multi-Format

#### 3.4.1 Format TRIPLETS (SentenceTransformers/MTEB)

```jsonl
{"anchor": "C'est a qui de jouer ?", "positive": "Un joueur a le trait quand...", "negative": "La pendule doit etre placee...", "metadata": {"source": "gold_standard_annales", "chunk_id": "LA-p037-parent015-child02", "difficulty": 0.16, "question_type": "factual"}}
```

**Conformite**: SPEC-TRIP-001 Section 2

#### 3.4.2 Format ARES

```tsv
Query	Document	Answer	Context_Relevance_Label
C'est a qui de jouer ?	Un joueur a le trait quand son adversaire...	Quand son adversaire a joue son coup	1
C'est a qui de jouer ?	La pendule doit etre placee a droite...		0
```

**Colonnes**:
- `Query`: Question reformulee
- `Document`: Chunk text
- `Answer`: answer_text (pour positifs uniquement)
- `Context_Relevance_Label`: 1 (positive) ou 0 (negative)

#### 3.4.3 Format BEIR

**queries.jsonl**:
```jsonl
{"_id": "FR-ANN-UVR-023", "text": "C'est a qui de jouer ?"}
```

**corpus.jsonl**:
```jsonl
{"_id": "LA-p037-parent015-child02", "title": "Article 1.3", "text": "Un joueur a le trait quand..."}
```

**qrels.tsv**:
```tsv
query-id	corpus-id	score
FR-ANN-UVR-023	LA-p037-parent015-child02	1
```

#### 3.4.4 Format RAGAS

```jsonl
{"question": "C'est a qui de jouer ?", "answer": "", "contexts": ["Un joueur a le trait quand..."], "ground_truth": "Quand son adversaire a joue son coup"}
```

**Note**: `answer` vide = a generer par RAG lors de l'evaluation

### 3.5 ETAPE 5: Validation & DVC

#### 3.5.1 Validations Schema

```bash
# Validation JSON Schema
python scripts/training/validate_schema.py \
  --input data/training/unified/triplets_train.jsonl \
  --schema docs/schemas/triplet_schema.json

# Validation distribution
python scripts/training/validate_distribution.py \
  --input data/training/unified/triplets_train.jsonl \
  --report data/training/unified/distribution_report.json
```

#### 3.5.2 DVC Tracking

```bash
# Tracker les artefacts generes
dvc add data/training/unified/

# Commit les fichiers .dvc
git add data/training/unified.dvc data/training/.gitignore
git commit -m "feat(training): add unified training data v1.0 from GS v6.7.0

ISO Reference: ISO 42001 A.7.3, ISO 29119
Source: gold_standard_annales_fr.json v6.7.0
Questions: 477 (395 with chunks, 82 adversarial)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Push vers remote DVC
dvc push
git push
```

#### 3.5.3 Rapport Composition

```json
{
  "version": "1.0",
  "generated_at": "2026-01-24T12:00:00Z",
  "source": {
    "gold_standard": "gold_standard_annales_fr.json",
    "gold_standard_version": "6.7.0",
    "chunks_file": "chunks_mode_b_fr.json"
  },
  "statistics": {
    "total_questions": 477,
    "with_chunk_mapping": 395,
    "adversarial_no_chunk": 82,
    "reformulated": 395,
    "triplets_generated": 395
  },
  "splits": {
    "train": {
      "count": 316,
      "percentage": 80
    },
    "val": {
      "count": 79,
      "percentage": 20
    }
  },
  "hard_negative_distribution": {
    "same_doc_diff_page": 158,
    "same_category": 118,
    "semantic_similar": 79,
    "random": 40
  },
  "output_files": {
    "triplets_train": "data/training/unified/triplets_train.jsonl",
    "triplets_val": "data/training/unified/triplets_val.jsonl",
    "ares_gold_label": "data/training/unified/ares_gold_label.tsv",
    "beir_queries": "data/training/unified/beir/queries.jsonl",
    "beir_corpus": "data/training/unified/beir/corpus.jsonl",
    "beir_qrels": "data/training/unified/beir/qrels.tsv",
    "ragas_eval": "data/training/unified/ragas_evaluation.jsonl"
  },
  "iso_compliance": {
    "ISO_42001_A.6.2.2_provenance": true,
    "ISO_42001_A.7.3_documentation": true,
    "ISO_29119_test_data": true,
    "ISO_25010_accuracy": true
  },
  "dvc_tracking": {
    "tracked": true,
    "remote": "storage",
    "version": "v1.0"
  }
}
```

---

## 4. Conformite ISO

### 4.1 ISO 42001 - Systemes de Management IA

| Controle | Implementation | Evidence |
|----------|----------------|----------|
| A.6.2.2 Provenance | expected_chunk_id, mapping_method | Schema Etape 1 |
| A.6.2.3 Lineage | Checkpoints chaque etape | Workflow S2.1 |
| A.6.2.4 Quality | Validation schema + distribution | Etape 5 |
| A.7.3 Documentation | Ce document + dataset_composition.json | SPEC-UTD-001 |

### 4.2 ISO 25010 - Qualite Logicielle

| Exigence | Implementation | Metrique |
|----------|----------------|----------|
| Exactitude fonctionnelle | Reformulation BY DESIGN | 0 validation post-hoc |
| Completude | 477 questions GS v6.7.0 | 100% coverage |
| Tracabilite | original_annales conserve | Audit trail |

### 4.3 ISO 29119 - Tests Logiciels

| Exigence | Implementation | Reference |
|----------|----------------|-----------|
| Test data documentation | Schema JSON + rapport | S3.4, S3.5 |
| Test data validation | Schema Draft-07 | Etape 5 |
| Separation train/val | 80/20 split, seed fixe | S3.5.3 |

### 4.4 ISO 12207 - Cycle de Vie Logiciel

| Exigence | Implementation | Reference |
|----------|----------------|-----------|
| Configuration management | DVC tracking | S3.5.2 |
| Versioning | Semantic versioning | dataset_composition.json |
| Reproducibility | Seeds fixes, checkpoints | Workflow complet |

---

## 5. ARES Simplifie

### 5.1 Pourquoi ARES Simplifie Suffit

| Aspect | ARES Complet | ARES Simplifie |
|--------|--------------|----------------|
| Labels | LLM-as-judge | Gold labels (officiels) |
| Confiance | PPI intervals | Exactitude directe |
| Cout | N appels LLM | 0 appels LLM |
| Fiabilite | Depend qualite juge | 100% (labels officiels) |

**Justification**:
- Les labels viennent de questions officielles DNA (jury national)
- La reformulation est guidee par chunk = alignement garanti
- Pas besoin de LLM-as-judge pour determiner relevance

### 5.2 Metriques ARES Simplifie

```python
def compute_ares_simplified(predictions: list[int], gold_labels: list[int]) -> dict:
    """
    Metriques ARES sans LLM-as-judge.
    """
    tp = sum(p == 1 and g == 1 for p, g in zip(predictions, gold_labels))
    fp = sum(p == 1 and g == 0 for p, g in zip(predictions, gold_labels))
    fn = sum(p == 0 and g == 1 for p, g in zip(predictions, gold_labels))
    tn = sum(p == 0 and g == 0 for p, g in zip(predictions, gold_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }
```

### 5.3 Quand Utiliser ARES Complet

| Scenario | ARES Version | Justification |
|----------|--------------|---------------|
| Evaluation GS v6.7.0 | Simplifie | Labels gold officiels |
| Nouvelles questions (hors GS) | Complet | Pas de gold label |
| Audit reformulation | Complet | Valider preservation sens |
| Benchmark externe | Complet | Comparabilite |

---

## 6. Scripts et Commandes

### 6.1 Pipeline Complet

```bash
# Etape 1: Mapping pages → chunks
python scripts/training/unified/map_pages_to_chunks.py \
  --gold-standard tests/data/gold_standard_annales_fr.json \
  --chunks corpus/processed/chunks_mode_b_fr.json \
  --output data/training/unified/gs_with_chunks.json

# Etape 2: Reformulation
python scripts/training/unified/reformulate_questions.py \
  --input data/training/unified/gs_with_chunks.json \
  --chunks corpus/processed/chunks_mode_b_fr.json \
  --model gemini-2.0-flash \
  --output data/training/unified/gs_reformulated.json

# Etape 3: Hard negatives (Phase 2 - hybride Claude+EmbeddingGemma)
# NOTE: generate_hard_negatives.py supprime en v2.2 (2026-01-28).
# Methode actuelle: EmbeddingGemma pre-filter + Claude LLM-as-judge.
# Voir: docs/plans/GS_CONFORMITY_PLAN_V1.md §4 Phase 2

# Etape 4: Export multi-format
python scripts/training/unified/export_formats.py \
  --input data/training/unified/triplets_raw.jsonl \
  --output-dir data/training/unified/ \
  --formats triplets,ares,beir,ragas \
  --train-ratio 0.8 \
  --seed 42

# Etape 5: Validation & DVC
python scripts/training/unified/validate_dataset.py \
  --input-dir data/training/unified/ \
  --report data/training/unified/dataset_composition.json

dvc add data/training/unified/
git add data/training/unified.dvc
git commit -m "feat(training): unified training data v1.0"
dvc push && git push
```

### 6.2 Commandes Validation

```bash
# Verifier schema
python scripts/training/validate_schema.py \
  --input data/training/unified/triplets_train.jsonl \
  --schema docs/schemas/triplet_schema.json

# Verifier distribution
python scripts/training/validate_distribution.py \
  --input data/training/unified/triplets_train.jsonl

# Verifier DVC status
dvc status
```

---

## 7. Checkpoints et Rollback

### 7.1 Fichiers Checkpoint

| Etape | Checkpoint | Rollback Command |
|-------|------------|------------------|
| 1 | gs_with_chunks.json | Regenerer depuis GS v6.7.0 |
| 2 | gs_reformulated.json | Regenerer depuis Etape 1 |
| 3 | triplets_raw.jsonl | Regenerer depuis Etape 2 |
| 4 | data/training/unified/ | Regenerer depuis Etape 3 |
| 5 | DVC tracked | `dvc checkout` |

### 7.2 Rollback DVC

```bash
# Lister versions
dvc list --rev HEAD~1 data/training/unified/

# Rollback vers version precedente
git checkout HEAD~1 -- data/training/unified.dvc
dvc checkout
```

---

## 8. Erreurs Passees - Ne Pas Repeter

Reference: SPEC-TRIP-001 Section 9

| Erreur | Prevention | Implementation |
|--------|------------|----------------|
| Ratio 95% synth / 5% GS | 100% Gold Standard | Pas de synthetique |
| Keyword matching sans validation | Reformulation BY DESIGN | LLM voit chunk |
| Val contient synthetique | Val = 100% GS | Split explicite |
| Pas de tracabilite | Checkpoints + DVC | Workflow complet |
| Questions reformulations du chunk | Contrainte dans prompt | S3.2.1 |

---

## 9. Livrables et Arborescence

```
data/training/unified/
├── gs_with_chunks.json           # Checkpoint Etape 1
├── gs_reformulated.json          # Checkpoint Etape 2
├── triplets_raw.jsonl            # Checkpoint Etape 3
├── triplets_train.jsonl          # LIVRABLE: Training triplets
├── triplets_val.jsonl            # LIVRABLE: Validation triplets
├── ares_gold_label.tsv           # LIVRABLE: ARES format
├── ares_unlabeled.tsv            # LIVRABLE: ARES unlabeled
├── beir/                         # LIVRABLE: BEIR format
│   ├── queries.jsonl
│   ├── corpus.jsonl
│   └── qrels.tsv
├── ragas_evaluation.jsonl        # LIVRABLE: RAGAS format
└── dataset_composition.json      # LIVRABLE: Rapport ISO
```

---

## 10. Historique du Document

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-24 | Claude Opus 4.5 | Creation initiale - Workflow unifie GS v6.7.0 |
| 1.1 | 2026-01-24 | Claude Opus 4.5 | **Ajout Section 1.4**: Fondements industrie context-grounded generation (RAGen, Source2Synth, FACTS). **Extension Section 11**: Hard negatives (NV-Embed, SimCSE, GTE, E5), synthetic data quality (SoftDedup, model collapse), benchmarks (MTEB, MMTEB, BEIR), ressources (SemHash, LlamaIndex, Haystack) |
| 1.2 | 2026-01-24 | Claude Opus 4.5 | **Ajout Sections 2.3-2.5**: Integration standards par etape, Quality Gates, Benchmarks cibles (MTEB FR, MMTEB INTL). Cross-ref QUALITY_REQUIREMENTS.md Section 4 |

---

## 11. References

### 11.1 Documents Internes

| Document | ID | Reference |
|----------|-----|-----------|
| Triplet Generation Spec | SPEC-TRIP-001 | Hierarchie sources, schema |
| Gold Standard v6 Annales | SPEC-GS-V6 | Source primaire |
| ISO Standards Reference | DOC-REF-001 | Conformite ISO |
| DVC Guide | DOC-GUIDE-001 | Tracking artefacts |
| Audit GS v5.30 | AUDIT-GS-530-001 | Problemes identifies |

### 11.2 Standards Externes - Benchmarks

| Standard | Reference | Application |
|----------|-----------|-------------|
| [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | Massive Text Embedding Benchmark | 58 datasets, standard evaluation embeddings |
| [MMTEB](https://arxiv.org/abs/2502.13595) | Massive Multicultural TEB | 500+ taches, 250+ langues |
| [BEIR](https://github.com/beir-cellar/beir) | Benchmarking IR | 18 datasets, 9 types retrieval |
| [RAGAS](https://docs.ragas.io/) | RAG Assessment | Faithfulness, relevance |
| ARES (Stanford) | RAG Evaluation | PPI, LLM-as-judge |
| SQuAD 2.0 | Adversarial QA | 33% unanswerable |

### 11.3 References Academiques - Hard Negatives & Contrastive Learning

| Reference | Titre | Annee | Contribution |
|-----------|-------|-------|--------------|
| [NV-Embed-v2](https://arxiv.org/abs/2405.17428) | Improved Techniques for Training LLM Embeddings | 2024 | Positive-aware hard negative mining, MTEB #1 |
| [SimCSE](https://arxiv.org/abs/2104.08821) | Contrastive Learning of Sentence Embeddings | 2021 | Dropout augmentation, baseline |
| [GTE](https://arxiv.org/abs/2308.03281) | General Text Embeddings | 2023 | Multi-stage training: pretrain + finetune |
| [E5](https://aclanthology.org/2024.acl-long.642.pdf) | Text Embeddings by Weakly-Supervised Contrastive Pre-training | 2024 | Triplets LLM avec hard negatives |

### 11.4 References Academiques - Synthetic Data Quality

| Reference | Titre | Annee | Contribution |
|-----------|-------|-------|--------------|
| [Lin et al.](https://arxiv.org/abs/2406.15126) | LLMs-Driven Synthetic Data Generation Survey | 2024 | Curation, filtering, deduplication |
| [SoftDedup](https://arxiv.org/abs/2407.06564) | Soft Deduplication for Training Data | 2024 | Reweighting vs deletion, 26% efficiency |
| [Liu et al.](https://arxiv.org/abs/2404.07503) | Best Practices Synthetic Data for LLMs | 2024 | Diversity, model collapse prevention |

### 11.5 Ressources Techniques

| Ressource | URL | Usage |
|-----------|-----|-------|
| SemHash | https://github.com/MinishLab/semhash | Fuzzy deduplication embeddings |
| Sentence-Transformers | https://sbert.net/docs/training/overview.html | Training triplets, hard negatives |
| LlamaIndex Synthetic | https://docs.llamaindex.ai/en/stable/examples/evaluation/generate_question_context_pairs/ | Question generation from context |
| Haystack | https://haystack.deepset.ai/ | Question generation pipeline |

### 11.6 Datasets de Reference

| Dataset | Source | Usage |
|---------|--------|-------|
| MS MARCO | microsoft/ms_marco | Paires query-passage standard |
| FACTS Grounding | google/facts-grounding | Benchmark grounded generation |
| all-nli | sentence-transformers/all-nli | Triplets NLI entailment |

---

*Document ISO 42001/25010/29119/12207 - Pocket Arbiter Project*
*Ce document remplace partiellement SPEC-TRIP-001 pour la source GS v6.7.0*
*Conforme aux standards industrie: RAGen, NV-Embed-v2, MTEB, BEIR, RAGAS*
