# Plan de Remediation Phase 1B - ISO Conforme

> **Document ID**: PLAN-REM-001
> **ISO Reference**: ISO/IEC 25010, 29119, 42001
> **Date**: 2026-01-15
> **Statut**: En cours de validation

---

## 1. Analyse Root Cause

### 1.1 Recall 27% vs 80% cible

| Cause | Impact | Evidence |
|-------|--------|----------|
| **Mismatch semantique** | CRITIQUE | Question "toucher-jouer" vs chunks "toucher une piece" |
| **Encodage UTF-8** | ELEVE | Caracteres speciaux corrompus dans chunks |
| **Modele non-FR** | ELEVE | BGE-base optimise anglais, pas francais |
| **Chunk size 256** | MOYEN | Peut fragmenter contexte necessaire |

### 1.2 Violations ISO detectees

| Violation | Norme | Severite |
|-----------|-------|----------|
| Test adversarial falsifie (`passed = True`) | ISO 42001 A.3 | BLOQUANTE |
| Seuil recall abaisse 80%->20% | ISO 25010 4.2 | BLOQUANTE |
| Precision non mesuree | ISO 25010 FA-02 | MAJEURE |
| Schema SDK non valide | ISO 12207 7.3.3 | MAJEURE |

---

## 2. Actions Correctives

### 2.1 Phase immediate (P0) - Violations bloquantes

#### A. Restaurer test recall bloquant

```python
# test_recall.py - AVANT (FAUX)
if embedding_dim >= 768:
    threshold = 0.80
else:
    threshold = 0.20  # BYPASS INTERDIT

# APRES (CONFORME)
threshold = 0.80  # ISO 25010 - BLOQUANT SANS EXCEPTION
```

#### B. Corriger test adversarial

```python
# AVANT (FALSIFIE)
passed = True  # Toujours vrai

# APRES (CONFORME ISO 42001)
# Score seuil base sur distribution empirique
HIGH_CONFIDENCE_THRESHOLD = 0.85
passed = top_score < HIGH_CONFIDENCE_THRESHOLD
```

### 2.2 Phase technique (P1) - Amelioration recall

#### A. Correction encodage UTF-8

**Source du probleme**: Extraction PDF avec encodage incorrect
**Action**: Re-extraire les PDFs avec encodage UTF-8 force

```python
# extract_pdf.py - Ajout
text = page.extract_text()
if text:
    text = text.encode('utf-8', errors='replace').decode('utf-8')
```

#### B. Modele embedding adapte francais

Selon [MTEB-French Benchmark](https://arxiv.org/html/2405.20468v2):

| Modele | NDCG@10 | Dim | Recommandation |
|--------|---------|-----|----------------|
| sentence-camembert-large | 0.72 | 1024 | **RECOMMANDE FR** |
| multilingual-e5-large | 0.79 | 1024 | Multilingual |
| mistral-embed | 0.80 | 1024 | API payante |
| intfloat/multilingual-e5-base | 0.75 | 768 | **Alternative gratuite** |

**Decision**: `intfloat/multilingual-e5-base` (768D, gratuit, multilingual)

#### C. Query expansion

Selon [Amazon Bedrock RAG](https://aws-samples.github.io/amazon-bedrock-samples/rag/knowledge-bases/features-examples/02-optimizing-accuracy-retrieved-results/query_reformulation/):

```python
def expand_query(question: str) -> list[str]:
    """Expande une question en variantes semantiques."""
    # Exemple: "toucher-jouer" -> ["toucher", "piece touchee", "obligation jouer"]
    return [question] + extract_keywords(question) + generate_synonyms(question)
```

#### D. Hybrid retrieval (optionnel)

Selon recherche: +15-25% recall avec keyword + vector search combine.

### 2.3 Phase validation (P2) - Conformite finale

#### A. Metriques ISO 25010 completes

| Metrique | Formule | Cible |
|----------|---------|-------|
| Recall@5 | pages_trouvees / pages_attendues | >= 80% |
| Precision@5 | pages_pertinentes / pages_retournees | >= 70% |
| MRR | 1 / rang_premiere_bonne | >= 0.6 |
| F1 | 2 * (P*R) / (P+R) | >= 75% |

#### B. Test adversarial conforme ISO 42001

30 questions reparties:
- 10 hors-sujet (recette cuisine, meteo)
- 10 fausses regles (inventions)
- 10 ambigues (reponse multiple)

Critere: Le systeme doit reconnaitre l'absence de source fiable.

---

## 3. Schema SDK Google AI Edge

Selon [documentation officielle](https://ai.google.dev/edge/mediapipe/solutions/genai/rag):

```kotlin
// Configuration officielle
SqliteVectorStore(768)  // Dimension fixe a 768
```

**Validation schema**:
- Dimension embeddings: 768 (OBLIGATOIRE)
- Format BLOB: float32 array
- Tables: metadata, chunks (indices sur source, page)

---

## 4. Plan d'execution

### Etape 1: Corrections bloquantes (2h)

1. Restaurer seuil 80% dans test_recall.py
2. Implementer vrai test adversarial
3. Marquer Phase 1B comme INCOMPLETE

### Etape 2: Re-generation corpus (4h)

1. Re-extraire PDFs avec UTF-8 correct
2. Re-chunker avec meme parametres
3. Generer embeddings avec `multilingual-e5-base` (768D)
4. Exporter vers SqliteVectorStore

### Etape 3: Validation recall (2h)

1. Executer benchmark recall
2. Si < 80%: analyser questions echouees
3. Implementer query expansion si necessaire
4. Re-tester jusqu'a >= 80%

### Etape 4: Validation finale (1h)

1. Tous tests passent (pytest)
2. Coverage >= 80%
3. Lint = 0 erreurs
4. Recall >= 80%
5. Adversarial = 100% pass

---

## 5. Definition of Done (DoD) Phase 1B

| Critere | Cible | Bloquant |
|---------|-------|----------|
| Recall@5 FR | >= 80% | **OUI** |
| Recall@5 INTL | >= 70% | NON |
| Adversarial pass rate | 100% (30/30) | **OUI** |
| Precision@5 | >= 70% | NON |
| Coverage tests | >= 80% | **OUI** |
| Lint errors | 0 | **OUI** |
| Schema SDK valide | 768D | **OUI** |

---

## Sources

- [Google AI Edge RAG SDK](https://ai.google.dev/edge/mediapipe/solutions/genai/rag)
- [MTEB-French Benchmark](https://arxiv.org/html/2405.20468v2)
- [Amazon Bedrock Query Reformulation](https://aws-samples.github.io/amazon-bedrock-samples/rag/knowledge-bases/features-examples/02-optimizing-accuracy-retrieved-results/query_reformulation/)
- [Chunking Strategies for RAG 2025](https://medium.com/@adnanmasood/chunking-strategies-for-retrieval-augmented-generation-rag-a-comprehensive-guide-5522c4ea2a90)
- [Chroma Research - Chunking Evaluation](https://research.trychroma.com/evaluating-chunking)
- [Pinecone - IR Evaluation Metrics](https://www.pinecone.io/learn/offline-evaluation/)
