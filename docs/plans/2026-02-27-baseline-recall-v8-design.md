# Design: Baseline Recall v8

> **Date**: 2026-02-27
> **Statut**: Approuvé
> **Objectif**: Établir une baseline recall fiable du retrieval sur le GS v8 avant toute optimisation

## Contexte

Le pipeline RAG est complet (29 PDFs → 1857 chunks → EmbeddingGemma 768D) et le GS v8 est validé (525Q, 100% chunk traçabilité). Le retrieval est identifié comme point faible, mais aucune mesure n'existe sur le v8 actuel.

**Principe**: Mesurer d'abord, optimiser ensuite.

## Périmètre

- **Questions**: 328 testables du v8 (exclut 92 requires_context + 105 unanswerable)
- **Composant mesuré**: Retrieval uniquement (pas de génération)
- **Variantes**: Vector-only + Hybrid (BM25 0.5 + Vector 0.5, RRF k=60)

## Métriques

| Métrique | Description |
|----------|-------------|
| recall@1, @3, @5, @10 (chunk) | Chunk exact attendu dans le top-K |
| recall@5 (page±2) | Page attendue dans le top-K (rétrocompat) |
| MRR | Reciprocal Rank du premier chunk correct |

### Segmentation

- `reasoning_class`: summary, fact_single, arithmetic, reasoning
- `difficulty`: facile, moyen, difficile
- `source_uuid`: par document (28 documents)
- `source_session`: par session d'examen (10 sessions)

## Analyse d'erreurs

Pour chaque question échouée (recall@5 chunk < 1.0):
- Question text
- Chunk attendu (id + extrait texte 200 chars)
- Top-5 chunks récupérés (id + score cosine + extrait)
- Distance cosine query ↔ chunk attendu
- Diagnostic probable (vocabulaire mismatch, mauvais doc filtré, chunk trop long...)

## Implémentation

**Approche**: Étendre `scripts/pipeline/tests/test_recall.py` existant.

### Ajouts:
1. `compute_recall_at_k_chunk()` — recall par chunk_id (pas page)
2. Multi-K en un pass — top-10 récupéré, tronqué à 1/3/5/10
3. MRR calculation
4. Segmentation par metadata GS
5. Error analysis détaillée
6. Export JSON dans `data/benchmarks/baseline_v8_YYYY-MM-DD.json`

### Réutilisation:
- `smart_retrieve()` — retrieval vector
- `retrieve_hybrid()` — retrieval hybrid
- `_parse_question()` — parsing Schema V2
- `load_embedding_model()` — chargement modèle

## Sortie

### Console
```
=== BASELINE RECALL v8 — 2026-02-27 ===
328 questions testables | EmbeddingGemma-300M-QAT | 1857 chunks

                    Vector-Only    Hybrid
recall@1  (chunk)      ??%         ??%
recall@3  (chunk)      ??%         ??%
recall@5  (chunk)      ??%         ??%
recall@10 (chunk)      ??%         ??%
recall@5  (page±2)     ??%         ??%
MRR                    ??          ??

--- Par reasoning_class ---
...
--- Par difficulty ---
...
--- Top-10 pires documents ---
...
--- Erreurs détaillées (top-20) ---
...
```

### JSON
Rapport complet sauvegardé pour analyse post-hoc.

## Critères de décision

| recall@5 chunk | Action |
|----------------|--------|
| >= 80% | RAG pur + prompt engineering suffit |
| 60-80% | Optimisations retrieval (query expansion, re-ranking) |
| < 60% | Fine-tuning embeddings justifié |

## Hors périmètre

- Optimisation du chunking
- Query expansion / re-ranking
- Fine-tuning embeddings
- Évaluation de la génération (Gemma)
- Nettoyage du projet (à traiter séparément)
