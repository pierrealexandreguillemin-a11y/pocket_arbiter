---
date: 2026-03-19
experiment: row-as-chunk (table level 2)
result: REVERTED — regression -6pp R@5
---

# Row-as-chunk Experiment — REVERTED

## Setup

- 111 raw tables parsed into 1355 row-chunks (header + data row)
- Dot-padding cleaned, abbreviations expanded
- Embedded with EmbeddingGemma-300M base, CCH title with section
- Stored in `table_rows` table, searchable via cosine + BM25 FTS5
- Standard: Ragie table chunking (fix-pipeline-task3 level 2)

## Results

| Metrique | Base-only | Enrichi seul | Enrichi + rows | Delta rows |
|----------|-----------|-------------|---------------|------------|
| recall@1 | 35.2% | 38.9% | 36.6% | -2.3pp |
| recall@3 | 52.7% | 55.7% | 51.0% | -4.7pp |
| recall@5 | 59.1% | 60.4% | 54.4% | -6.0pp |
| recall@10 | 63.1% | 63.8% | 58.7% | -5.1pp |
| MRR | 0.448 | 0.479 | 0.444 | -0.035 |
| R3 regression | — | +12/-10 | +6/-19 | -13 net |

## Root cause

1. **1355 row-chunks noient les children** — short 2-line chunks with generic embeddings
   take top-k slots from more relevant prose children
2. **Header repetition degrades BM25** — Ragie explicitly warns: "repeating key names
   frequently negatively impacts hybrid search results". Same header indexed 1355 times.
3. **Row-chunks too short for meaningful cosine** — header + 1 data row = ~10-20 tokens,
   embedding is dominated by header terms, not data values

## Decision

Row-as-chunk reverted from search. Code kept but disabled. Table retrieval will use
level 3 (SQLite structured lookup) instead — deterministic, no embedding pollution.

## Sources

- [Ragie: Our Approach to Table Chunking](https://www.ragie.ai/blog/our-approach-to-table-chunking)
- [TableRAG (NeurIPS 2024)](https://arxiv.org/abs/2410.04739) — column:value pairs, not row embedding
