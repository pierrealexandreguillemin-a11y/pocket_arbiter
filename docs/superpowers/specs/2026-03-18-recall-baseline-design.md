# Chantier 3 : Baseline Recall Measurement — Design Spec

> **Date**: 2026-03-18
> **Statut**: Approuve
> **Objectif**: Mesurer recall@1/3/5/10 et MRR sur 298 questions GS testables avec le pipeline actuel (reglages de base). Etablir un baseline fiable pour la decision fine-tuning vs prompt engineering.

---

## Contexte

Le pipeline v2 est complet (chantier 2) : extract, chunk, index, search hybrid (cosine+BM25+RRF, adaptive-k largest-gap). Le recall n'a jamais ete mesure sur le pipeline v2 avec les 298 questions testables propres.

L'ancien baseline (ere 3) donnait 36.6% recall@5 chunk / 66.8% recall@5 page sur 358 questions avec l'ancien pipeline (chunks trop petits, pas de parents, pas de table summaries). Ces resultats ne sont plus pertinents — archives dans `data/archive/benchmarks_v8/`.

### Contraintes

- **Reglages de base** : aucune optimisation, parametres par defaut du pipeline
- **Modele actuel** : `google/embeddinggemma-300m-qat-q4_0-unquantized` (tel quel)
- **Match level** : page-level (chunk_ids GS incompatibles avec DB v2)
- **Pas de calibration** : adaptive_k avec defaults (min_score=0.005, max_k=10)

---

## Architecture

```
recall.py
  |
  ├─ load_gs(path) → list[dict]  (298 questions testables)
  ├─ build_page_index(conn) → dict[(source, page), list[child_id]]
  ├─ evaluate(question, search_result, page_index) → EvalResult
  ├─ compute_metrics(results) → Metrics
  ├─ error_analysis(results, n=20) → list[ErrorCase]
  └─ main(db_path, gs_path) → JSON + Markdown
```

### Flow

1. Charger GS, filtrer `is_impossible=False` → 298 questions
2. Charger modele embedding une fois
3. Ouvrir DB, construire page_index une fois
4. Pour chaque question :
   - `search(db, question.text, model=model)` → SearchResult
   - Collecter tous `(source, page)` des children retournes (via children_matched → DB lookup)
   - Comparer avec `(provenance.docs[i], provenance.pages[i])` du GS
   - Enregistrer : hit@1, hit@3, hit@5, hit@10, rank du premier hit, scores
5. Agreger metriques globales + segmentees
6. Error analysis sur les echecs recall@10
7. Ecrire JSON + Markdown

---

## Page-level matching

Le GS a `provenance.docs` et `provenance.pages`. La DB a `children.source` et `children.page`.

Pour chaque Context retourne par search() :
- `context.children_matched` → liste de child_ids
- Pour chaque child_id → `SELECT source, page FROM children WHERE id = ?`
- Si `(source, page)` matche un `(provenance.docs[i], provenance.pages[i])` → HIT

**Pourquoi page-level** : les chunk_ids GS (format `source-p{page}-parent{pid}-child{cid}`) ne correspondent pas aux chunk_ids DB v2 (format `source-c{index}`). Le corpus a ete re-chunke, les frontieres de chunks ont change. La page est l'invariant stable.

**Ordering** : les contexts sont tries par score desc. Le rang d'un hit est sa position (1-indexed) dans cette liste.

---

## Metriques

| Metrique | Definition |
|----------|-----------|
| recall@k | % questions ayant >= 1 hit page-level dans les k premiers contexts |
| MRR | Moyenne de 1/rank du premier hit (0 si absent dans top-10) |

### Segmentation

| Dimension | Buckets |
|-----------|---------|
| reasoning_class | summary, fact_single, arithmetic, reasoning |
| difficulty | easy (0-0.33), medium (0.33-0.66), hard (0.66-1.0) |

---

## Seuils de decision

| recall@5 | Action |
|----------|--------|
| >= 80% | Prompt engineering suffisant, passer au proto Android |
| 60-80% | Optimisations retrieval (calibration adaptive_k, synonymes, reranker) |
| < 60% | Fine-tuning embeddings justifie (QLoRA sur EmbeddingGemma base) |

---

## Outputs

### JSON (`data/benchmarks/recall_baseline.json`)

```json
{
  "metadata": {
    "generated": "2026-03-18T...",
    "pipeline": "hybrid cosine+BM25 RRF, adaptive-k largest-gap",
    "model": "google/embeddinggemma-300m-qat-q4_0-unquantized",
    "db": "corpus_v2_fr.db",
    "gs_version": "9.0.0",
    "match_level": "page",
    "settings": {"min_score": 0.005, "max_k": 10, "rrf_k": 60},
    "questions_total": 298,
    "questions_with_context": 37
  },
  "global": {
    "recall@1": 0.0,
    "recall@3": 0.0,
    "recall@5": 0.0,
    "recall@10": 0.0,
    "mrr": 0.0
  },
  "segments": {
    "reasoning_class": {
      "summary": {"count": 0, "recall@1": 0.0, "...": "..."},
      "...": "..."
    },
    "difficulty": {
      "easy": {"count": 0, "recall@1": 0.0, "...": "..."},
      "...": "..."
    }
  },
  "errors": [
    {
      "question": "...",
      "expected_source": "...",
      "expected_pages": [],
      "retrieved_pages": [],
      "top_scores": [],
      "reasoning_class": "...",
      "difficulty": 0.0
    }
  ],
  "per_question": [
    {
      "id": "FR-Q-001",
      "hit@1": false,
      "hit@3": false,
      "hit@5": true,
      "rank": 4,
      "rrf_score": 0.016
    }
  ]
}
```

### Markdown (`data/benchmarks/recall_baseline.md`)

Header YAML parseable, auto-genere par le script. Jamais edite a la main.

```markdown
---
generated: 2026-03-18T14:30:00Z
pipeline: hybrid cosine+BM25 RRF, adaptive-k largest-gap
model: google/embeddinggemma-300m-qat-q4_0-unquantized
db: corpus_v2_fr.db (1253 children, 332 parents, 111 tables)
gs: gold_standard_annales_fr_v8_adversarial.json v9.0.0
questions_total: 298
match_level: page
settings: {min_score: 0.005, max_k: 10, rrf_k: 60}
decision_threshold: ">=80% prompt eng | 60-80% retrieval optim | <60% fine-tune"
---

# Recall Baseline — Pipeline v2

## Global

| Metrique | Score |
|----------|-------|
| recall@1 | X.X% |
| recall@3 | X.X% |
| recall@5 | X.X% |
| recall@10 | X.X% |
| MRR | X.XXX |

## Par reasoning_class

| Class | Count | R@1 | R@3 | R@5 | R@10 |
|-------|-------|-----|-----|-----|------|
| ... | ... | ... | ... | ... | ... |

## Par difficulte

| Bucket | Count | R@1 | R@3 | R@5 | R@10 |
|--------|-------|-----|-----|-----|------|
| ... | ... | ... | ... | ... | ... |

## Top 20 echecs

| # | Question (50 chars) | Expected | Retrieved | Score |
|---|---------------------|----------|-----------|-------|
| ... | ... | ... | ... | ... |

## Decision

recall@5 = X.X% → [DECISION]
```

---

## Fichiers

| Fichier | Action | Lignes estimees |
|---------|--------|-----------------|
| `scripts/pipeline/recall.py` | CREATE | ~200 |
| `scripts/pipeline/tests/test_recall.py` | CREATE | ~150 |
| `data/benchmarks/recall_baseline.json` | OUTPUT | auto |
| `data/benchmarks/recall_baseline.md` | OUTPUT | auto |

---

## Ce qu'on ne fait PAS

- Pas de calibration adaptive_k (run separe si recall 60-80%)
- Pas de comparaison cosine-only vs hybrid (run separe)
- Pas de re-embedding (modele actuel tel quel)
- Pas de modification du pipeline search
- Pas de chunk-level recall (IDs incompatibles)
- Pas de segmentation par source PDF (trop granulaire)

---

## Standards

- **BEIR/MTEB** : recall@k et MRR sont les metriques standard
- **ISO 29119** : TDD, tests unitaires avec mock search
- **ISO 25010** : fichiers <= 300 lignes
- **ISO 12207** : commits conventionnels
- **ISO 42001** : tracabilite complete (metadata dans JSON + Markdown header)

---

## Sources

- [BEIR Benchmark](https://github.com/beir-cellar/beir) — recall@k, NDCG@k standard
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — evaluation standard
- Baseline v8 spec: `docs/plans/2026-02-27-baseline-recall-v8-design.md` (archive)
- EMNLP 2025 Adaptive-k: "No Tuning, No Iteration, Just Adaptive-k"
