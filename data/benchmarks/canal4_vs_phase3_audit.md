# Canal 4 vs Phase 3 Audit — 2026-04-01

## Context

Canal 4 (narrative rows cosine, weight 0.5 in RRF) and Phase 3 (table injection
page±1 in build_context) both serve table retrieval. This audit checks overlap
and identifies optimization opportunities.

## Overlap Analysis

| Measure | Value |
|---------|-------|
| Total table_summaries | 117 |
| Tables reachable by Phase 3 (page±1 of a child) | 96/117 (82%) |
| Tables reachable by Canal 4 (narrative rows cosine) | 117/117 (100%) |
| Overlap (both channels) | 96/117 (82%) |
| Canal 4 exclusive (isolated tables, not adjacent to children) | 21 |
| Phase 3 exclusive | 0 |

**Verdict**: 82% overlap. Canal 4 is the ONLY path to 21 isolated tables.
Do NOT remove Canal 4. Weight reduction (0.5 → 0.3) is worth testing.

## Recall Misses — Tabular Questions

| Category | Count | Root cause | Fix |
|----------|-------|------------|-----|
| rank 6-10 (found but outside top-5) | 5 | Displaced by other results | Weight tuning |
| rank=0, table on page±1 | 26 | Embedding mismatch (query vs table) | FTS5 cells, priority boost |
| rank=0, no table nearby | 16 | Content not in DB at all | Coverage gap (out of scope) |
| **Total tab misses** | **47/122** | | |

Tabular recall@5 = 61.5% (75/122).

## Documented Offline Optimizations — Status

### Applied (chantier 3)
- OPT-1 DONE: Contextual retrieval (Anthropic 2024) — +3.7pp R@1
- OPT-2 DONE: Abbreviation expansion (12 terms)
- OPT-4 DONE: Chapter overrides (85 chunks LA)
- OPT-6 DONE: Config tuning 450/50

### Skipped (tested, negative)
- OPT-7: Score calibration — 4 formulas, all degrade R@1
- OPT-8: Query decomposition — 3/110 matches

### Not yet implemented (positive expected value)

| # | Item | Source | Effort | Target |
|---|------|--------|--------|--------|
| 1 | FTS5 on structured_cells | structured-tables-design A.1 | +30 lines | 26 rank-0 tab misses |
| 2 | Priority boost 6 critical tables | structured-tables-design A.2 | +10 lines | Top tables (cadences, categories, Elo) |
| 3 | Flag intro pages (OPT-5) | recall-optimization OPT-5 | +20 lines | Remove sommaire/preface from top-5 |
| 4 | Canal 4 weight sweep | This audit | 1 script | Test w=0.3 vs 0.5 |
| 5 | Intent detection scoring | structured-tables-design B.4 | +40 lines | Gradient vs binary triggers |

### Excluded (confirmed)
- Cross-encoder reranker: RAM 600MB runtime
- SPLADE: no Android offline port
- Late chunking: EmbeddingGemma 2048 token complexity
- Multi-granularity indexing: complexity vs uncertain gain

### Deferred (requires rebuild)
- Row-as-chunk targeted (6 priority tables, ~200 rows): structured-tables-design C.10
- Column-name normalization: structured-tables-design B.5
- Multi-granularity chunking: 200/450/900 tok parallel index

## Recommendation

Items 1-4 are implementable in ~1h without rebuild. Expected impact:
- 5 rank 6-10 → top-5 (weight tuning)
- Subset of 26 rank-0 "search problems" → hits (FTS cells + priority boost)
- Fewer false-positive intro pages polluting top-5

## Sources

- docs/superpowers/specs/2026-03-19-recall-optimization-design.md
- docs/superpowers/specs/2026-03-19-structured-tables-design.md
- data/benchmarks/recall_baseline.json (63.4% R@5, 189/298)
- data/benchmarks/table_recall_baseline.json (122 tabular questions)
