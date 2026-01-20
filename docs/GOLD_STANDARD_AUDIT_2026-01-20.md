# Gold Standard Audit Report - 2026-01-20

## Summary

| Metric | v5.21 | v5.22 (final) |
|--------|-------|---------------|
| Total Questions | 134 | 134 |
| Hard Cases | 45 | 45 |
| Recall@5 (tol=2) | 90.17% | **91.17%** |
| Failed Questions | 20/134 | **14/134** |
| ISO 25010 (>=80%) | PASS | **PASS** |
| Target (>=90%) | PASS | **PASS** |

## Changes Applied

### File Renames (ISO Compliance)

Renamed for ISO-compliant naming:
- `questions_fr.json` -> `gold_standard_fr.json`
- `questions_intl.json` -> `gold_standard_intl.json`

Deleted obsolete files:
- `questions_fr_v1_backup.json`
- `questions_fr_v2.json`
- `questions_fr_v5_arbiter.json`
- `scripts/create_gold_standard_v3.py`
- `scripts/create_gold_standard_v4.py`
- `scripts/create_gold_standard_v4_1.py`
- `scripts/validate_gold_standard_v5.py`

### New Questions Added (22 hard cases from Annales)

**From Annales-Juin-2025 (FR-Q113 to FR-Q124):**
- Appariement suisse, Buchholz, Sonneborn-Berger
- Toutes rondes, flotteur, classement initial FIDE
- Calcul Elo, barème indemnisation

**From Annales-Decembre-2024 (FR-Q125 to FR-Q134):**
- Chute drapeau par cadence
- Position illégale, notation zeitnot
- 75 coups, 5x répétition
- Matériel insuffisant, classement rapide

### Page Corrections (6 questions)

| Question | Before | After | Recall | Status |
|----------|--------|-------|--------|--------|
| FR-Q119 (toutes rondes) | [73,74] | [101,102,103,104] | 25% | PENDING |
| FR-Q121 (classement initial) | [182,183,184] | [183,185] | 50% | PENDING |
| FR-Q125 (drapeau cadence) | [47,58] | [57,58] | 50% | PENDING |
| FR-Q127 (noter zeitnot) | [50,51] | [50] | 0% | **FAILED_RETRIEVAL** |
| FR-Q131 (classement rapide) | [192,193] | [187,188,189] | **100%** | **VALIDATED** |
| FR-Q132 (passerelle rapide/std) | [182,183,192] | [183,185,187,188] | 25% | PENDING |

## Methodology

### Page Verification Process

Each question's `expected_pages` was verified against `chunks_parent_child_fr.json`:

1. **FR-Q119**: Searched "toutes rondes" -> found Chapitre 3.1 pages 101-104
2. **FR-Q121**: Searched "classement initial" -> found Art. 7.1.4 (p183) + Art. 8.2 (p185)
3. **FR-Q125**: Searched "drapeau" -> found Annexe A.5.5 (p57) + Annexe B (p58)
4. **FR-Q127**: Searched "notation zeitnot" -> found Art. 8.4 (p50)
5. **FR-Q131**: Searched "classement rapide" -> found Chapitre 6.2 (p187-189)
6. **FR-Q132**: Searched cross-chapter -> Ch6.1 (183,185) + Ch6.2 (187,188)

### Validation Status Definitions

- **VALIDATED**: Recall 100% after page correction
- **PENDING_VALIDATION**: Partial recall (>0%), pages correct but retrieval limited
- **FAILED_RETRIEVAL**: 0% recall despite correct pages (hard case for embedding model)

## Remaining Failed Questions (14)

| ID | Recall | Expected | Retrieved | Issue |
|----|--------|----------|-----------|-------|
| FR-Q77 | 0% | [183,188] | [4,3,227,2,225] | Semantic drift |
| FR-Q85 | 0% | [165] | [25,174,172] | Document mismatch |
| FR-Q86 | 0% | [169-171] | [74,124,84] | Multi-page spread |
| FR-Q87 | 0% | [182,183] | [4,197,194] | Wrong chapter |
| FR-Q94 | 0% | [183] | [4,3,186] | Semantic drift |
| FR-Q95 | 0% | [182,183] | [4,3,194] | Wrong section |
| FR-Q98 | 0% | [196-198] | [193,192] | Adjacent pages |
| FR-Q99 | 67% | [200-202] | [167,197,199] | Partial match |
| FR-Q103 | 0% | [169-171] | [10,11,36] | Wrong document |
| FR-Q119 | 25% | [101-104] | [107,118,106] | Chapter boundary |
| FR-Q121 | 50% | [183,185] | [9,123,186] | Cross-article |
| FR-Q125 | 50% | [57,58] | [66,50,55] | Annexe confusion |
| FR-Q127 | 0% | [50] | [3,56,66] | Zeitnot terminology |
| FR-Q132 | 25% | [183,185,187,188] | [1,4,168,190] | Cross-chapter |

## ISO Compliance

- **ISO 25010**: Recall 91.17% PASS (threshold >=80%)
- **ISO 29119**: Test methodology documented, validation status tracked
- **ISO 42001**: Full traceability of corrections with audit timestamps

## Files Modified

- `tests/data/gold_standard_fr.json` - v5.22 (134 questions, 6 page corrections)
- `tests/data/gold_standard_intl.json` - renamed from questions_intl.json
- `scripts/iso/phases.py` - updated file reference
- `scripts/iso/tests/conftest.py` - updated test fixture

## Recommendations

1. **Hard cases**: FR-Q127 (zeitnot) requires query reformulation or hybrid search
2. **Cross-chapter**: FR-Q132 spans Ch6.1+6.2, may need multi-vector approach
3. **Reranker**: Consider cross-encoder reranking for remaining 14 questions
