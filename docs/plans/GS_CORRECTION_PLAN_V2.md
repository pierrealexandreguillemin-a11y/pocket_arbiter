# Plan de Correction GS Scratch v2 - Generation Supplementaire

> **Document ID**: PLAN-GS-CORR-002
> **ISO Reference**: ISO 29119-3 (Test Data), ISO 25010 (Quality), ISO 42001 (AI Traceability)
> **Version**: 1.0
> **Date**: 2026-02-18
> **Statut**: Draft
> **Parent**: PLAN-GS-SCRATCH-001, SPEC-GS-METH-001
> **Input**: Audit gs_scratch_v1.json (614Q) contre standards industrie

---

## 1. Etat des Lieux

### 1.1 Audit gs_scratch_v1.json (v1.1, 614Q)

| Finding | Metrique | Actuel | Cible | Severite |
|---------|----------|--------|-------|----------|
| **F1** | Corpus coverage | 25.3% (470/1857) | >= 80% (1485) | **CRITIQUE** |
| **F2** | Hard difficulty (>=0.7) | 0% (0/397) | >= 10% | **BLOQUANT** (G5-4) |
| **F3** | Cognitive levels | 2/4 (Remember, Understand) | 4/4 Bloom | MAJEUR |
| **F4** | Reasoning single-hop | ~96% fact_single+summary | Multi-hop >= 15% | MAJEUR |
| **F5** | Question types | 3/4 (manque comparative) | 4/4 | MINEUR |
| **F6** | answer_type | 100% extractive | Diversifier | MINEUR |
| **F7** | Schema fields | 41-42/46 | 46/46 | **BLOQUANT** (G4-1 = 42, OK mais < spec) |
| **F8** | Unanswerable ratio | 35.3% | 25-40% | OK (dans la plage) |
| **F9** | Summary ratio | 30.2% | 15-25% | WARNING (G5-3 passe, hors cible interne) |
| **F10** | Difficulty variance | 0% hard, 0% easy <0.4 (*) | 3 niveaux | MAJEUR |

(*) Correction: le script montre 27.5% easy, 72.5% medium, 0% hard. Le finding F2 est le vrai bloquant.

### 1.2 Gate Status Post-Audit

| Gate | Seuil spec | Valeur reelle | Status reel |
|------|-----------|---------------|-------------|
| G0-2 | >= 80% coverage | 25.3% | **FAIL** |
| G5-4 | hard >= 10% | 0% | **FAIL** |
| G4-1 | >= 42 champs | 41-42 | **BORDERLINE** |
| G1-3 | fact_single < 60% | 53.9% | PASS |
| G5-5 | unanswerable 25-40% | 35.3% | PASS |

### 1.3 Distributions Actuelles Detaillees

```
reasoning_class (answerable 397):
  fact_single:  214 (53.9%)
  summary:      120 (30.2%)  ← au-dessus cible 15-25%
  reasoning:     63 (15.9%)

cognitive_level (answerable 397):
  Understand:   263 (66.2%)
  Remember:     134 (33.8%)
  Apply:          0 (0.0%)   ← ABSENT
  Analyze:        0 (0.0%)   ← ABSENT

question_type (answerable 397):
  factual:      207 (52.1%)
  procedural:   154 (38.8%)
  scenario:      36 (9.1%)
  comparative:    0 (0.0%)   ← ABSENT

difficulty (answerable 397):
  easy (<0.4):  109 (27.5%)
  medium:       288 (72.5%)
  hard (>=0.7):   0 (0.0%)   ← ABSENT

answer_type (answerable 397):
  extractive:   397 (100.0%)
  inferential:    0 (0.0%)   ← ABSENT

hard_type (unanswerable 217):
  OUT_OF_DATABASE:        79 (36.4%)
  FALSE_PRESUPPOSITION:   50 (23.0%)
  UNDERSPECIFIED:          45 (20.7%)
  MODALITY_LIMITED:        15 (6.9%)
  SAFETY_CONCERNED:       15 (6.9%)
  NONSENSICAL:            13 (6.0%)  ← OK, 6/6 categories
```

---

## 2. Standards Industrie de Reference

### 2.1 Sources

| Standard | Reference | Application |
|----------|-----------|-------------|
| SQuAD 2.0 | arXiv:1806.03822 | Unanswerable ratio train ~33.4%, difficulty via adversarial crowdworkers |
| UAEval4RAG | arXiv:2412.12300 | 6 categories unanswerable, Unanswered Ratio + Acceptable Ratio |
| Know Your RAG | COLING 2025 | reasoning_class taxonomy (fact_single/summary/reasoning), label-targeted generation |
| Bloom's Taxonomy | Anderson & Krathwohl 2001 | 6 niveaux cognitifs (Remember → Create), minimum 4 recommande (arXiv:2601.20253) |
| BEIR/MTEB | NeurIPS 2021 / 2024 | Corpus coverage benchmark, evaluation diversite |
| NV-Embed v2 | arXiv:2024 | Anchor independence < 0.90 pour triplet training |
| SemHash/SoftDedup | 2024 | Deduplication semantique seuil 0.95 |

### 2.2 Cibles de Distribution (post-correction)

Les cibles ci-dessous sont des **choix projet** inspires des standards (cf. Section 10 SPEC-GS-METH-001).

| Dimension | Cible | Justification |
|-----------|-------|---------------|
| **Corpus coverage** | >= 80% (1485+ chunks) | BEIR/MTEB: representativite corpus |
| **Unanswerable ratio** | 25-33% | SQuAD 2.0 train split adaptation |
| **fact_single** | 40-55% | Know Your RAG: eviter dominance |
| **summary** | 15-25% | Equilibre reasoning_class |
| **reasoning** | 15-25% | Know Your RAG: label-targeted |
| **Cognitive: Apply** | >= 10% | Bloom: 4 niveaux minimum |
| **Cognitive: Analyze** | >= 10% | Bloom: 4 niveaux minimum |
| **Hard difficulty** | >= 10% | G5-4 bloquant |
| **Comparative questions** | >= 5% | Diversite question_type |
| **Inferential answers** | >= 10% | Diversite answer_type |

---

## 3. Strategie de Correction

### 3.1 Approche Generale

La correction se fait en **4 phases sequentielles**, chacune avec ses gates de validation. L'objectif est d'amener le GS de 614Q (25.3% coverage) a ~1600-1800Q (>=80% coverage) avec des distributions conformes.

```
Phase A: Fix metadata (614Q existantes)
  → Corriger champs manquants, difficulty, cognitive_level
  |
Phase B: Generation massive (1015+ nouvelles Q answerable)
  → Couvrir les 1387 chunks non couverts
  |
Phase C: Generation unanswerable supplementaire (~250-350 nouvelles Q)
  → Maintenir ratio 25-33% sur le total elargi
  |
Phase D: Equilibrage final + deduplication
  → Verifier toutes les gates sur le GS complet
  |
OUTPUT: gs_scratch_v2.json (~1600-1800Q)
```

### 3.2 Estimation Volumes

| Phase | Input | Output estime | Methode |
|-------|-------|---------------|---------|
| A | 614Q existantes | 614Q corrigees | Script Python (metadata fix) |
| B | 1387 chunks non couverts | ~1100-1300 Q answerable | BY DESIGN generation |
| C | GS elargi | ~250-350 Q unanswerable | BY DESIGN generation |
| D | ~1950-2250 Q brut | ~1600-1800 Q final | Dedup + balance |

---

## 4. Phase A - Fix Metadata (existantes)

### 4.1 Objectif

Corriger les 614 questions existantes sans en generer de nouvelles.

### 4.2 Corrections

| Correction | Champs | Methode |
|-----------|--------|---------|
| **A1: Difficulty hard** | `classification.difficulty` | Re-evaluer: questions multi-articles, exceptions, sanctions → difficulty >= 0.7 |
| **A2: Cognitive Apply** | `classification.cognitive_level` | Questions procedurales avec scenario concret → "Apply" |
| **A3: Cognitive Analyze** | `classification.cognitive_level` | Questions comparatives ou multi-concept → "Analyze" |
| **A4: Schema 46 champs** | Tous groupes | Ajouter champs manquants (identifier lesquels avec diff schema spec vs reel) |
| **A5: answer_type diversite** | `classification.answer_type` | Questions summary/reasoning dont la reponse necessite synthese → "inferential" |

### 4.3 Script

**Fichier**: `scripts/evaluation/annales/fix_gs_v2_metadata.py`

```python
# Pseudocode Phase A
def fix_metadata(gs_data, chunks_index):
    for q in gs_data["questions"]:
        if q["content"].get("is_impossible"):
            continue

        # A1: Re-evaluer difficulty
        q["classification"]["difficulty"] = reassess_difficulty(q, chunks_index)

        # A2+A3: Re-classifier cognitive_level
        q["classification"]["cognitive_level"] = reassess_cognitive_level(q)

        # A5: Diversifier answer_type
        if q["classification"]["reasoning_class"] in ("summary", "reasoning"):
            if not is_verbatim_in_chunk(q["content"]["expected_answer"], chunks_index[q["provenance"]["chunk_id"]]):
                q["classification"]["answer_type"] = "inferential"

        # A4: Completer schema 46 champs
        ensure_46_fields(q)

    return gs_data
```

**Regles de reclassification difficulty**:
- `>= 0.7` (hard): question impliquant 2+ articles, exceptions a des regles, sanctions, calcul Elo, conditions multiples
- `0.4-0.7` (medium): question standard a comprehension directe
- `< 0.4` (easy): definition simple, fait unique

**Regles de reclassification cognitive_level**:
- `Apply`: question_type == "scenario" OU contient "que doit faire", "comment appliquer"
- `Analyze`: reasoning_class == "reasoning" OU question implique comparaison/deduction
- `Understand`: reasoning_class == "summary" (synthese)
- `Remember`: reasoning_class == "fact_single" (rappel simple)

### 4.4 Gates Phase A

| Gate | Seuil | Verification |
|------|-------|--------------|
| A-G1 | 46/46 champs | `count_schema_fields(q) == 46` pour 100% |
| A-G2 | hard >= 10% | `sum(d >= 0.7) / len(answerable) >= 0.10` |
| A-G3 | 4 cognitive levels | `len(set(cognitive_levels)) >= 4` |
| A-G4 | chunk_match_score inchange | `all(score == 100)` |

---

## 5. Phase B - Generation Answerable (nouveaux chunks)

### 5.1 Objectif

Couvrir les 1387 chunks actuellement sans question pour atteindre >= 80% de couverture corpus (1485+ chunks).

### 5.2 Strategie de Couverture

Pas tous les chunks meritent une question. Certains sont des tables des matieres, en-tetes, listes vides. Le script de generation retourne `[]` pour ces cas (cf. PLAN-GS-SCRATCH-001 Phase 1).

**Taux de generation estime**: ~0.8 Q/chunk (base: 397 Q / 470 chunks = 0.84 dans v1).

**Priorite de couverture**:

| Priorite | Chunks | Description | Raison |
|----------|--------|-------------|--------|
| 1 (haute) | LA-octobre2025 (806 uncov) | Lois de l'arbitrage | Corpus principal |
| 2 | C01-C04 Coupes (118 uncov) | Competitions coupes | 0% couverture |
| 3 | A02, F01, J03 (103 uncov) | Championnats | 0% couverture |
| 4 | 2025_RI, Contrat, Statuts (161 uncov) | Admin/statutaire | < 50% couverture |
| 5 | Reste | Autres documents | Complement |

### 5.3 Prompts de Generation Conformes

Le prompt de generation Phase 1 doit etre **enrichi** pour forcer la diversite des dimensions manquantes. Modification du prompt `generate_real_questions.py`:

```python
GENERATION_PROMPT_V2 = """
CHUNK (ID: {chunk_id}, Source: {source}, Page: {page}):
---
{chunk_text}
---

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES OBLIGATOIRES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. chunk_match_score = 100 (BY DESIGN)

DIVERSITE REQUISE (varier entre les questions):
- question_type: factual | procedural | scenario | comparative
- cognitive_level: Remember | Understand | Apply | Analyze
- reasoning_class: fact_single | summary | reasoning
- difficulty: 0.0-1.0 (inclure >= 0.7 pour questions complexes)
- answer_type: extractive | inferential

GUIDE PAR COGNITIVE LEVEL:
- Remember: "Quel est...?", "Combien...?", fait unique
- Understand: "Expliquez...", "Que signifie...?", synthese d'un passage
- Apply: "Que doit faire l'arbitre si...?", application concrete d'une regle
- Analyze: "Quelle difference entre X et Y?", "Pourquoi cette regle plutot que...?"

GUIDE PAR DIFFICULTY:
- easy (<0.4): definition simple, fait isole, chiffre unique
- medium (0.4-0.7): comprehension regle standard, procedure normale
- hard (>=0.7): exception a une regle, conditions multiples, interaction entre articles, sanction avec circonstances

OUTPUT FORMAT (JSON array):
[
  {{
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0,
    "answer_type": "extractive|inferential"
  }}
]

Si le chunk n'est pas propice (table des matieres, en-tete, liste vide): retourner [].
"""
```

### 5.4 Batching

| Parametre | Valeur | Justification |
|-----------|--------|---------------|
| Batch size | 30 chunks | Token budget ~150K IN / 50K OUT par batch |
| Distribution check | Tous les 5 batches | Verifier distributions courantes |
| Adaptive steering | Oui | Si fact_single > 55%, forcer reasoning/Apply pour batch suivant |

### 5.5 Gates Phase B

| Gate | Seuil | Verification |
|------|-------|--------------|
| G1-1 | chunk_match_score = 100 | Invariant BY DESIGN |
| G1-2 | answer in chunk | Verbatim OR semantic >= 0.95 |
| G1-4 | finit par "?" | Format |
| B-G1 | coverage >= 80% | `len(covered_chunks) / 1857 >= 0.80` |
| B-G2 | Apply >= 10% | Sur ensemble total |
| B-G3 | Analyze >= 10% | Sur ensemble total |
| B-G4 | hard >= 10% | Sur ensemble total |
| B-G5 | comparative >= 5% | Sur ensemble total |

---

## 6. Phase C - Generation Unanswerable Supplementaire

### 6.1 Objectif

Apres Phase B, le total answerable sera ~1500+. Pour maintenir le ratio unanswerable dans [25%, 33%], il faut generer des questions unanswerable supplementaires.

### 6.2 Calcul du Volume

```
Hypothese: ~1500 answerable total (397 existant + ~1100 nouveau)
Cible unanswerable: 30% du total
Total cible: 1500 / 0.70 = ~2143
Unanswerable cible: 2143 - 1500 = ~643
Existant unanswerable: 217
A generer: ~426 nouvelles questions unanswerable
```

### 6.3 Distribution UAEval4RAG Cible

Les 6 categories doivent etre representees. Distribution cible pour les ~426 nouvelles:

| Categorie | Ratio cible | Nouvelles Q | Total (avec 217 existantes) |
|-----------|-------------|-------------|----------------------------|
| OUT_OF_DATABASE | 25-30% | ~115 | ~194 |
| FALSE_PRESUPPOSITION | 20-25% | ~90 | ~140 |
| UNDERSPECIFIED | 15-20% | ~75 | ~120 |
| NONSENSICAL | 10-15% | ~55 | ~68 |
| MODALITY_LIMITED | 10-15% | ~50 | ~65 |
| SAFETY_CONCERNED | 8-12% | ~41 | ~56 |

### 6.4 Gates Phase C

| Gate | Seuil | Verification |
|------|-------|--------------|
| G2-1 | is_impossible = true | 100% questions Phase C |
| G2-2 | unanswerable 25-33% | Sur total GS |
| G2-3 | >= 4 categories | Distribution hard_type |

---

## 7. Phase D - Equilibrage Final

### 7.1 Pipeline Post-Generation

```
Input: GS brut (~2100-2200 Q)
  |
  1. Deduplication semantique (seuil 0.95)
  |
  2. Anchor independence check (seuil 0.90)
  |
  3. Verification distributions cibles
  |
  4. Trimming si necessaire (retirer surplus fact_single/summary)
  |
  5. Validation anti-hallucination (3 niveaux)
  |
  6. Enrichissement Schema v2.0 (46 champs)
  |
  7. Toutes 21 quality gates
  |
OUTPUT: gs_scratch_v2.json
```

### 7.2 Distributions Cibles Finales

| Dimension | Cible | Gate |
|-----------|-------|------|
| Total questions | 1600-1800 | - |
| Corpus coverage | >= 80% | G0-2 |
| Unanswerable | 25-33% | G2-2, G5-5 |
| fact_single | 40-55% | G1-3, G5-3 |
| summary | 15-25% | - |
| reasoning | 15-25% | - |
| Remember | 20-35% | - |
| Understand | 25-40% | - |
| Apply | >= 10% | - |
| Analyze | >= 10% | - |
| Hard (>= 0.7) | >= 10% | G5-4 |
| Comparative | >= 5% | - |
| Inferential | >= 10% | - |
| Schema fields | 46/46 | G4-1 |
| Duplicates | 0 | G5-1 |
| Anchor independence | < 0.90 | G5-2 |

### 7.3 Scripts a Reutiliser

| Script existant | Phase D usage |
|----------------|---------------|
| `balance_distribution.py` | Dedup + equilibrage (Phase D etape 1-4) |
| `validate_anti_hallucination.py` | Validation (Phase D etape 5) |
| `enrich_schema_v2.py` | Schema v2 (Phase D etape 6) |
| `quality_gates.py` | 21 gates (Phase D etape 7) |

### 7.4 Gates Phase D

Toutes les 21 gates de SPEC-GS-METH-001 Section 4 doivent passer, plus:

| Gate | Seuil | Severite |
|------|-------|----------|
| D-G1 | coverage >= 80% | BLOQUANT |
| D-G2 | 4/4 cognitive levels | BLOQUANT |
| D-G3 | hard >= 10% | BLOQUANT |
| D-G4 | inferential >= 10% | WARNING |
| D-G5 | comparative >= 5% | WARNING |

---

## 8. Scripts a Creer / Modifier

### 8.1 Nouveaux Scripts

| Script | Role | Phase |
|--------|------|-------|
| `fix_gs_v2_metadata.py` | Corriger metadata 614Q existantes | A |
| `generate_v2_coverage.py` | Orchestrateur generation massive | B+C |

### 8.2 Scripts a Modifier

| Script | Modification | Phase |
|--------|-------------|-------|
| `generate_real_questions.py` | Prompt V2 (cognitive_level, difficulty hard, comparative, inferential) | B |
| `quality_gates.py` | Ajouter gates D-G1 a D-G5 | D |
| `test_gs_data_integrity.py` | Tests pour nouvelles distributions | D |

### 8.3 Scripts Existants Reutilises Sans Modification

| Script | Usage |
|--------|-------|
| `stratify_corpus.py` | Phase 0 (deja execute, restratifier pour chunks non couverts) |
| `validate_anti_hallucination.py` | Phase D validation |
| `enrich_schema_v2.py` | Phase D enrichissement |
| `balance_distribution.py` | Phase D equilibrage |
| `reformulate_by_design.py` | Eventuellement pour reformuler questions trop proches |

---

## 9. Ordre d'Execution

```
ETAPE 1: Phase A - Fix metadata (614Q)
  Script: fix_gs_v2_metadata.py
  Input:  gs_scratch_v1.json
  Output: gs_scratch_v1_fixed.json
  Duree:  ~30min (script seul, pas de LLM)
  Gate:   A-G1 a A-G4

ETAPE 2: Phase B - Stratification chunks non couverts
  Script: stratify_corpus.py (re-exec sur 1387 chunks)
  Input:  chunks_mode_b_fr.json - chunks deja couverts
  Output: uncovered_chunk_strata.json
  Duree:  ~5min

ETAPE 3: Phase B - Generation answerable par batches
  Script: generate_v2_coverage.py → generate_real_questions.py
  Input:  uncovered_chunk_strata.json
  Output: new_answerable_questions.json (~1100-1300 Q)
  Duree:  ~40 batches x 30 chunks = estimatif selon token budget
  Gate:   G1-1, G1-2, G1-4, B-G1 a B-G5

ETAPE 4: Phase C - Generation unanswerable
  Script: generate_v2_coverage.py (mode unanswerable)
  Input:  GS elargi
  Output: new_unanswerable_questions.json (~426 Q)
  Gate:   G2-1, G2-2, G2-3

ETAPE 5: Phase D - Merge + equilibrage + validation
  Scripts: balance_distribution.py, validate_anti_hallucination.py,
           enrich_schema_v2.py, quality_gates.py
  Input:  614 fixees + ~1300 answerable + ~426 unanswerable
  Output: gs_scratch_v2.json (~1600-1800 Q)
  Gate:   21 gates SPEC-GS-METH-001 + D-G1 a D-G5

ETAPE 6: Test integration
  Script: test_gs_data_integrity.py (mis a jour)
  Verification: pytest pass sur nouveau GS
```

---

## 10. Risques et Mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Chunks non propices (TdM, listes) | Coverage < 80% malgre generation | Accepter ~75% si chunks filtres > 20% du total |
| Token budget excessif (40 batches) | Cout/temps | Batches paralleles, generation selective |
| Qualite degradee a grande echelle | Questions repetitives | Dedup Phase D + distribution steering |
| Hard questions artificielles | Questions forcees sans naturalite | Review echantillon + LLM-as-Judge (DOC-VAL-001) |
| Anchor violation (sim >= 0.90) | Triplet training invalide | Phase D etape 2 filtre systematique |

---

## 11. Criteres de Succes

Le GS v2 est considere conforme quand:

1. **21/21 gates PASS** (SPEC-GS-METH-001 Section 4)
2. **Coverage >= 80%** (1485+ chunks couverts)
3. **4/4 cognitive levels** presents avec >= 10% Apply et >= 10% Analyze
4. **Hard >= 10%** (difficulty >= 0.7)
5. **0 duplicates semantiques** (seuil 0.95)
6. **100% chunk_match_score = 100** (invariant BY DESIGN)
7. **0% hallucination** (validation 3 niveaux)
8. **46/46 champs schema** pour chaque question

---

## 12. Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-02-18 | Creation - audit findings F1-F10, plan 4 phases, cibles distribution |

---

*Plan ISO 42001/25010/29119 - Pocket Arbiter Project*
*Methode: BY DESIGN (chunk = INPUT, pas OUTPUT)*
