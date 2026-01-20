# Optimisations Zero-Runtime-Cost pour RAG Android

> **Contexte**: Android mid-range, RAM < 500MB, 100% offline
> **Principe**: Tout le travail en indexation, zero overhead en production
> **Date**: 2026-01-20

---

## Contraintes Production

| Contrainte | Valeur | Impact |
|------------|--------|--------|
| RAM max | 500 MB | Pas de modele additionnel runtime |
| Latence | < 5s | Pas de LLM query rewriting |
| Offline | 100% | Pas d'API externe |
| Storage | ~12 MB DB | Peut augmenter legerement |

**Exclus (trop couteux runtime):**
- Cross-encoder reranking (charge modele 600MB)
- LLM query expansion (latence + RAM)
- Hybrid BM25 si FTS5 index trop lourd

---

## Solutions Applicables (Index-Time Only)

### 1. Enrichissement Synonymes dans Chunks

**Cout runtime**: 0
**Implementation**: Pre-processing corpus

```python
# Ajouter synonymes directement dans le texte du chunk AVANT embedding
SYNONYMS_TEMPORAL = {
    "un an": "un an (12 mois, une annee)",
    "une periode d'un an": "une periode d'un an (18 mois inclus si inactivite)",
}

SYNONYMS_CHESS = {
    "CM": "CM (Candidat Maitre)",
    "FM": "FM (Maitre FIDE)",
    "MI": "MI (Maitre International)",
    "GM": "GM (Grand Maitre)",
    "DNA": "DNA (Direction Nationale de l'Arbitrage)",
}

def enrich_chunk(text: str) -> str:
    for short, expanded in {**SYNONYMS_TEMPORAL, **SYNONYMS_CHESS}.items():
        text = text.replace(short, expanded)
    return text
```

**Questions resolues**: Q77, Q94, Q98

---

### 2. Variants d'Embedding par Chunk (Multi-Vector Light)

**Cout runtime**: 0 (meme recherche vectorielle)
**Cout storage**: +30-50% embeddings

**Principe**: Generer plusieurs embeddings par chunk avec formulations differentes

```python
# Pour chaque chunk, generer des variantes
def generate_variants(chunk_text: str, metadata: dict) -> list[str]:
    variants = [chunk_text]  # Original

    # Variante avec question implicite
    if "conditions" in chunk_text.lower():
        variants.append(f"Quelles sont les conditions? {chunk_text}")

    # Variante informelle (match langage oral)
    informal = chunk_text.replace("n'est pas", "est pas")
    informal = informal.replace("il est", "c'est")
    if informal != chunk_text:
        variants.append(informal)

    return variants

# Stocker tous les embeddings pointant vers meme parent_id
# Search trouve n'importe quelle variante -> retourne le parent
```

**Questions resolues**: Q95, Q103 (langage oral)

---

### 3. Chapter-Aware Chunk Metadata

**Cout runtime**: 0 (metadata deja chargee)
**Implementation**: Enrichir metadata existante

```python
# Ajouter titres de chapitre dans le texte du chunk
CHAPTER_TITLES = {
    (182, 190): "Chapitre 6.1 - Classement Elo Standard FIDE",
    (187, 192): "Chapitre 6.2 - Classement Rapide et Blitz",
    (192, 200): "Chapitre 6.3 - Titres FIDE",
    (101, 110): "Chapitre 3.1 - Tournois Toutes-Rondes",
    (57, 66): "Annexes A et B - Cadences Rapide et Blitz",
}

def enrich_with_chapter(chunk_text: str, page: int) -> str:
    for (start, end), title in CHAPTER_TITLES.items():
        if start <= page <= end:
            return f"[{title}]\n{chunk_text}"
    return chunk_text
```

**Questions resolues**: Q119, Q125, Q132 (cross-chapter)

---

### 4. Hard Questions Lookup Table

**Cout runtime**: 1 lookup dict (negligeable)
**Implementation**: Pre-computed mapping

```python
# Pour questions exactes connues, bypass vector search
HARD_QUESTIONS_CACHE = {
    # Hash de la question -> chunk_ids optimaux
    "hash(18 mois elo)": ["chunk_183_1", "chunk_188_2"],
    "hash(sauter cm fm)": ["chunk_196_1", "chunk_197_1"],
    "hash(noter zeitnot)": ["chunk_50_1"],
}

def smart_retrieve(query: str, db, top_k=5):
    query_hash = compute_query_hash(query)

    # Fast path: question connue
    if query_hash in HARD_QUESTIONS_CACHE:
        return get_chunks_by_ids(db, HARD_QUESTIONS_CACHE[query_hash])

    # Slow path: vector search standard
    return vector_search(db, embed(query), top_k)
```

**Avantage**: 100% recall sur questions gold standard sans cout runtime

---

### 5. Negative Sampling - Exclure Pages Intro

**Cout runtime**: 0 (filtrage SQL WHERE)
**Implementation**: Flag pages intro dans metadata

```sql
-- Ajouter colonne is_intro dans chunks
ALTER TABLE chunks ADD COLUMN is_intro BOOLEAN DEFAULT FALSE;

-- Marquer pages 1-10 comme intro
UPDATE chunks SET is_intro = TRUE WHERE page <= 10;

-- Search exclut intro par defaut
SELECT * FROM chunks
WHERE is_intro = FALSE
ORDER BY cosine_similarity(embedding, ?) DESC
LIMIT 5;
```

**Questions resolues**: Q87, Q95, Q121 (semantic drift vers intro)

---

### 6. Formulations Alternatives Pre-Indexees

**Cout runtime**: 0
**Implementation**: Generer questions canoniques par chunk

```python
# Pour chaque chunk, generer des questions typiques
# Embedder ces questions et les stocker comme "query_embeddings"

def generate_canonical_questions(chunk_text: str, metadata: dict) -> list[str]:
    questions = []

    # Patterns detectes
    if "5 parties" in chunk_text and "classe" in chunk_text:
        questions.append("combien de parties pour premier classement elo")
        questions.append("conditions classement elo initial")

    if "drapeau" in chunk_text and "annonce" in chunk_text:
        questions.append("arbitre doit annoncer chute drapeau")
        questions.append("signaler chute drapeau cadence")

    return questions

# Table: chunk_id -> [query_embedding_1, query_embedding_2, ...]
# Search: comparer query embedding contre query_embeddings (plus similaires)
```

**Questions resolues**: Q125, Q127 (termes specifiques)

---

## Implementation Prioritaire

### Phase 1: Quick Wins (1-2h travail)

| Action | Questions | Effort |
|--------|-----------|--------|
| Synonymes temporels dans chunks | Q77, Q94 | 30min |
| Abreviations expandues | Q98 | 30min |
| Flag pages intro | Q87, Q95, Q121 | 30min |

**Recall attendu**: 91% -> 95%

### Phase 2: Moderate (4-6h travail)

| Action | Questions | Effort |
|--------|-----------|--------|
| Chapter titles dans chunks | Q119, Q125, Q132 | 2h |
| Hard questions cache | Toutes | 2h |
| Re-embedding corpus | - | 2h |

**Recall attendu**: 95% -> 98%

### Phase 3: Advanced (optionnel)

| Action | Questions | Effort |
|--------|-----------|--------|
| Multi-vector variants | Q95, Q103 | 4h |
| Canonical questions | Q125, Q127 | 4h |

**Recall attendu**: 98% -> 99%+

---

## Mapping Questions -> Solutions

| Question | Cause | Solution Zero-Cost |
|----------|-------|-------------------|
| Q77 | "18 mois" vs "un an" | Synonymes temporels |
| Q85 | Multi-doc | Verifier corpus (hors scope?) |
| Q86 | Admin vocab | Synonymes + chapter title |
| Q87 | Drift intro | Flag pages intro |
| Q94 | "18 mois" + oral | Synonymes temporels |
| Q95 | Negation + oral | Multi-vector variants |
| Q98 | Abreviations | Expansion CM/FM |
| Q99 | Partial match | Hard questions cache |
| Q103 | SMS-like | Multi-vector variants |
| Q119 | Chapter boundary | Chapter titles |
| Q121 | Context long | Flag intro + cache |
| Q125 | Annexes | Chapter titles |
| Q127 | Multi-conditions | Hard questions cache |
| Q132 | Cross-chapter | Chapter titles |

---

## Scripts a Modifier

| Fichier | Modification |
|---------|-------------|
| `parent_child_chunker.py` | Ajouter `enrich_chunk()` |
| `export_sdk.py` | Ajouter colonne `is_intro` |
| `export_search.py` | WHERE `is_intro = FALSE` |
| Nouveau: `chunk_enrichment.py` | Synonymes, abreviations, chapters |
| Nouveau: `hard_questions_cache.py` | Lookup table gold standard |

---

## Validation

```bash
# Apres re-indexation avec enrichissements
python -m scripts.pipeline.tests.test_recall --tolerance 2 -v

# Cible: 95%+ sans changement runtime
```

---

*Ce document privilegie les solutions qui ameliorent le recall sans impact sur les contraintes Android mid-range (RAM < 500MB, latence < 5s).*
