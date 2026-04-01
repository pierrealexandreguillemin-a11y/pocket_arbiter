# Page Interpolation Fix — 2026-03-31

## Finding

**46 children pages** were incorrectly assigned across 3 sources (LA, A02, Interclubs).
The content existed in the DB with correct text and embeddings, but the `page` column
was off-by-1 to off-by-3, making 15 GS questions unreachable by page-level recall.

## Root Cause

`chunker_utils._build_text_to_page()` assigns pages by walking the markdown and tracking
`current_page` via headings in `heading_pages`. Between two headings on different pages,
ALL content inherits the previous heading's page. If heading A is on page 5 and heading B
is on page 7, everything on page 6 is assigned to page 5.

The repeated page headers (e.g., "Partie 2:Les regles du jeu Chapitre 2.1 :Les regles du
jeu d'echecs" on every LA page) could serve as page markers, but `_strip_page_headers()`
in extract.py removes them (headings appearing 3+ times -> stripped as noise). After stripping,
the interpolator has no reference points between sparse content headings.

## Impact

| Metric | Before | After |
|--------|--------|-------|
| Missing pages (children only) | 46 | 22 (20 TABLE_ONLY, 2 genuinely empty) |
| Pages covered (children + tables) | 95% | 98.8% (222/227 LA) |
| GS questions impacted | 15/298 (5.0%) | 0/298 (0%) |
| GS recall ceiling | 95.0% | 100.0% |
| GS chunk_ids v2-aligned | 0/298 | 298/298 |
| Table summaries | 111/117 | 117/117 (100%) |
| Recall@5 GS | 63.1% (Phase 2) | 63.4% (+0.3pp, +1 hit net) |

## Fixes Applied

### 1. Page corrections (120 children)
Pdfplumber ground truth: for each child in LA, A02, and Interclubs, extracted text
from the PDF, matched child content against PDF pages (+/-10 page window, accent-normalized
matching), and updated the `page` column.

### 2. GS chunk_id realignment (298/298)
Updated all chunk_ids from v1 format (`source.pdf-p010-parent024-child00`) to v2 format
(`source.pdf-c0042`) by matching (source, page) -> children/table_summaries in DB.

### 3. 6 missing table_summaries injected
6 tables extracted by docling had no summaries in `table_summaries_claude.json`:
- LA-octobre2025-table0 (p.2): Table des matieres LA
- LA-octobre2025-table47 (p.136): Fiche resultats individuelle (template)
- LA-octobre2025-table69 (p.178): Bareme frais deplacement arbitres
- LA-octobre2025-table82 (p.198): Conditions normes titres FIDE
- LA-octobre2025-table83 (p.198): Exigences normes titres FIDE
- 2022_Reglement_medical-table3 (p.11): Formulaire declaration medicaments (template vierge)

For each: summary generated, embedding (EmbeddingGemma-300M base), narrative rows,
structured cells, FTS stemmed.

### 4. 1 child injected (Interclubs p.7)
Extraction gap docling: page 7 content (592 chars) never chunked. Child injected with
embedding, FTS, parent linkage.

### 5. FTS stemming fix (73 entries)
Bug: manual inserts used raw text instead of stemmed text for FTS. Fixed: 6 table_summaries
+ 67 table_rows FTS entries re-stemmed via `stem_text()`.

### 6. Root cause code fix
- `extract.py`: `_extract_text_pages(doc)` extracts ordered (text[:80], page_no) for ALL
  docling text items (not just section_headers). Added to extraction output as `text_pages`.
- `chunker_utils.py`: `_build_text_to_page()` accepts `text_pages` for dense page tracking.
  For repeated text (page headers), maintains an ordered consumption queue per key.
- `chunker.py`: `chunk_document()` passes `text_pages` through to `interpolate_pages()`.

### 7. Phase 3 context injection
- `context.py`: `_inject_neighbor_tables()` — when a prose child on page X is retrieved,
  table_summaries on pages X-1, X, X+1 from the same source are injected into context.
- Dedup against already-present tables, 500-word budget cap, lower score than retrieved.
- 5 unit tests in `test_context.py`.

## Remaining (not fixable without rebuild)

- 22 LA pages without children: 20 are TABLE_ONLY (covered by table_summaries), 2 genuinely
  empty (p.1 cover page, p.100 section title). No GS impact.
- Root cause code fix requires next full rebuild to take effect on existing pages.

## Regression Analysis (25 regressions, 26 improvements, net +1)

Regressions are dispersed — not caused by page corrections. The 72 new vectors (6 summaries
+ 66 narrative rows) add competition in cosine top-k. Several LA p.56 regressions verified:
GS expects p.56 but content is physically on p.57 in PDF (GS page inherited from v1 pipeline).

## Files Modified

- `corpus/processed/corpus_v2_fr.db`: 120 page fixes, 1 child, 6 summaries, 66 rows, 154 cells, 73 FTS fixes
- `corpus/processed/table_summaries_claude.json`: 6 summaries added (111->117)
- `tests/data/gold_standard_annales_fr_v8_adversarial.json`: 298 chunk_ids v2, page corrections
- `scripts/pipeline/context.py`: _inject_neighbor_tables() + integration
- `scripts/pipeline/chunker_utils.py`: text_pages dense page tracking
- `scripts/pipeline/chunker.py`: text_pages passthrough
- `scripts/pipeline/extract.py`: _extract_text_pages()
- `scripts/pipeline/tests/test_context.py`: 5 injection tests
- `data/benchmarks/page_fixes.json`: fix manifest
- `data/benchmarks/recall_baseline.json`: post-fix recall
- `corpus/processed/corpus_v2_fr.db.bak_pre_pagefix`: backup
- `CLAUDE.md`: updated counts, recall, findings
