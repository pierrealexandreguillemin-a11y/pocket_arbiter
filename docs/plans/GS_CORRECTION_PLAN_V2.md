# Plan de Correction GS Scratch v2 - Generation Supplementaire

> **Document ID**: PLAN-GS-CORR-002
> **ISO Reference**: ISO 29119-3 (Test Data), ISO 25010 (Quality), ISO 42001 (AI Traceability)
> **Version**: 2.0
> **Date**: 2026-02-19
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
| **F7** | Schema fields | 40 (answerable) / 36 (unanswerable) | 46/46 | **BLOQUANT** (G4-1 spec = 42) |
| **F8** | Unanswerable ratio | 35.3% | 25-40% | OK (dans la plage) |
| **F9** | Summary ratio | 30.2% | 15-25% | WARNING (G5-3 passe, hors cible interne) |
| **F10** | Difficulty variance | 27.5% easy, 72.5% medium, 0% hard | 3 niveaux | MAJEUR |

### 1.2 Gate Status Post-Audit (24 gates, cf. quality_gates.py)

| Gate | Seuil spec | Valeur reelle | Status reel |
|------|-----------|---------------|-------------|
| G0-2 | >= 80% coverage doc | 60.7% (17/28) | **FAIL** |
| G5-4 | hard answerable >= 10% | 0% (0/397) | **FAIL** |
| G5-6 | >= 4 cognitive levels | 2 (Remember, Understand) | **FAIL** |
| G5-7 | 4 question_types | 3/4 (missing: comparative) | **FAIL** |
| G5-8 | chunk coverage >= 80% | 25.3% (470/1857) | **FAIL** |
| G4-1 | >= 42 champs | 40 (answerable) | **BORDERLINE** |
| G1-3 | fact_single < 60% | 53.9% | PASS |
| G5-5 | unanswerable 25-40% | 35.3% | PASS |

### 1.3 Distributions Actuelles Detaillees

```
reasoning_class (answerable 397):
  fact_single:  214 (53.9%)
  summary:      120 (30.2%)  <- au-dessus cible 15-25%
  reasoning:     63 (15.9%)

cognitive_level (answerable 397):
  Understand:   263 (66.2%)
  Remember:     134 (33.8%)
  Apply:          0 (0.0%)   <- ABSENT
  Analyze:        0 (0.0%)   <- ABSENT

question_type (answerable 397):
  factual:      207 (52.1%)
  procedural:   154 (38.8%)
  scenario:      36 (9.1%)
  comparative:    0 (0.0%)   <- ABSENT

difficulty (answerable 397):
  easy (<0.4):  109 (27.5%)
  medium:       288 (72.5%)
  hard (>=0.7):   0 (0.0%)   <- ABSENT

answer_type (answerable 397):
  extractive:   397 (100.0%)
  inferential:    0 (0.0%)   <- ABSENT

hard_type (unanswerable 217):
  OUT_OF_DATABASE:        79 (36.4%)
  FALSE_PRESUPPOSITION:   50 (23.0%)
  UNDERSPECIFIED:          45 (20.7%)
  MODALITY_LIMITED:        15 (6.9%)
  SAFETY_CONCERNED:       15 (6.9%)
  NONSENSICAL:            13 (6.0%)  <- OK, 6/6 categories
```

---

## 2. Standards Industrie de Reference

### 2.1 Sources

| Standard | Reference | Ce que dit le standard | Applicable au projet |
|----------|-----------|----------------------|---------------------|
| SQuAD 2.0 | arXiv:1806.03822 | Train: 33.4% unanswerable, Dev: 50%. Difficulty via adversarial crowdworkers. | Oui — ratio unanswerable |
| UAEval4RAG | arXiv:2412.12300 | 6 categories unanswerable. Evaluation via Unanswered Ratio + Acceptable Ratio. | Oui — categories hard_type |
| Know Your RAG | COLING 2025 | Taxonomie 3 labels: fact_single/summary/reasoning. fact_single varie de 15.8% a 83.0% selon datasets. **Aucun seuil prescrit.** | Taxonomie adoptee |
| Bloom's Taxonomy | Anderson & Krathwohl 2001 | 6 niveaux cognitifs (Remember, Understand, Apply, Analyze, Evaluate, Create). **Framework descriptif, pas de minimum prescrit.** | Framework adopte |
| BEIR/MTEB | NeurIPS 2021 / 2024 | Benchmark multi-dataset. Corpus coverage implicite (100% des documents contribuent). | Inspiration coverage |
| NV-Embed v2 | arXiv:2405.17428 | Hard negative mining via similarity margins. **Pas de seuil 0.90 specifie.** | Seuil 0.90 = choix projet |
| SemHash/SoftDedup | 2024 | Seuil par defaut 0.90, range recommande [0.75-0.95]. | Seuil 0.95 = choix projet |

### 2.2 Cibles de Distribution (post-correction)

> **Note d'honnetete**: Les cibles ci-dessous sont des **choix projet** inspires de la
> litterature. Aucun standard cite ne prescrit ces valeurs exactes. La colonne
> "Justification" distingue ce qui vient du standard et ce qui est un choix d'equipe.

| Dimension | Cible | Justification |
|-----------|-------|---------------|
| **Corpus coverage** | >= 80% (1485+ chunks) | **Choix projet** inspire BEIR (representativite implicite) |
| **Unanswerable ratio** | 25-33% | **Choix projet** adapte de SQuAD 2.0 (train 33.4%, dev 50%) |
| **fact_single** | 40-55% | **Choix projet** — Know Your RAG observe 15.8-83.0%, aucun seuil prescrit |
| **summary** | 15-25% | **Choix projet** — equilibre reasoning_class, aucun standard |
| **reasoning** | 15-25% | **Choix projet** — equilibre reasoning_class, aucun standard |
| **Cognitive: Apply** | >= 10% | **Choix projet** — Bloom ne prescrit pas de minimum |
| **Cognitive: Analyze** | >= 10% | **Choix projet** — Bloom ne prescrit pas de minimum |
| **Hard difficulty** | >= 10% | **Gate G5-4** (bloquant, code quality_gates.py) |
| **Comparative questions** | >= 5% | **Choix projet** — diversite question_type |
| **Inferential answers** | >= 10% | **Choix projet** — diversite answer_type |

---

## 3. Strategie de Correction

### 3.1 Approche Generale

La correction se fait en **4 phases sequentielles**, chacune avec ses gates de validation et un **point de decision go/no-go** (ISO 12207).

```
Phase A: Re-generation ciblee (614Q existantes)
  -> Re-generer les questions necessitant Apply/Analyze/hard/comparative
  -> GO/NO-GO: A-G1 a A-G5 + regression test
  |
Phase B: Generation massive (1015+ nouvelles Q answerable)
  -> Couvrir les 1387 chunks non couverts
  -> GO/NO-GO: B-G1 a B-G5 + stop gate qualite
  |
Phase C: Generation unanswerable supplementaire (~250-350 nouvelles Q)
  -> Maintenir ratio 25-33% sur le total elargi
  -> GO/NO-GO: G2-1, G2-2, G2-3
  |
Phase D: Equilibrage final + deduplication
  -> Verifier toutes les 24 gates sur le GS complet
  |
OUTPUT: gs_scratch_v2.json (~1600-1800Q)
```

### 3.2 Go/No-Go Inter-Phases (ISO 12207)

| Transition | Condition GO | Action si NO-GO |
|------------|-------------|-----------------|
| A -> B | A-G1 a A-G5 PASS + regression 37/37 tests pass + 5 xfail | Corriger Phase A, ne pas avancer |
| B -> C | B-G1 a B-G5 PASS + coverage >= 70% (seuil minimal) | Reduire cible coverage, documenter deviation |
| C -> D | G2-2 unanswerable dans [25%, 33%] | Ajuster volume Phase C |
| D -> Release | 24/24 gates PASS (dont 11 bloquantes) | Iterer Phase D, documenter gates WARNING |

### 3.3 Estimation Volumes

| Phase | Input | Output estime | Intervalle | Hypotheses |
|-------|-------|---------------|------------|------------|
| A | 614Q existantes | 614Q corrigees | N/A | Script + re-generation partielle |
| B | 1387 chunks non couverts | ~830-1100 Q | [pessimiste: 0.6 Q/chunk, optimiste: 0.8 Q/chunk] | v1 avait 0.84 sur chunks faciles ; chunks restants contiennent ~20% TdM/en-tetes |
| C | GS elargi | ~250-430 Q | Depend output Phase B | Ajuste pour ratio 30% cible |
| D | ~1700-2150 Q brut | ~1500-1800 Q final | -10% a -15% via dedup | Taux dedup v1 = 0.16% (1 dupe/614) |

**Scenario pessimiste** : 0.6 Q/chunk sur 1387 = 832 nouvelles. Total answerable: 1229. Unanswerable cible 30%: 527. A generer: 310. Final post-dedup: ~1600.

**Scenario optimiste** : 0.8 Q/chunk sur 1387 = 1110 nouvelles. Total answerable: 1507. Unanswerable cible 30%: 645. A generer: 428. Final post-dedup: ~1850.

### 3.4 Budget de Ressources

| Phase | Tokens IN (estime) | Tokens OUT (estime) | Cout estime (Claude Sonnet) | Duree |
|-------|-------------------|--------------------|-----------------------------|-------|
| A | ~500K (614 Q x ~800 tok/Q) | ~300K | ~$2 | ~1h |
| B | ~6M (40 batches x 150K) | ~2M (40 x 50K) | ~$30 | ~4-6h |
| C | ~2M (15 batches x 130K) | ~800K | ~$10 | ~2h |
| D | ~1M (embeddings + validation) | ~200K | ~$5 | ~1h |
| **Total** | **~9.5M** | **~3.3M** | **~$47** | **~8-10h** |

Note: couts estimes sur base Claude Sonnet 4. Ajuster si modele different.

---

## 4. Phase A - Re-generation Ciblee (existantes)

### 4.1 Objectif

Corriger les 614 questions existantes. **Principe** : les reclassifications de metadata
ne sont autorisees que si elles sont verifiables programmatiquement sans ambiguite.
Les cas ambigus necessitent une **re-generation BY DESIGN** de la question.

### 4.2 Ce qui est autorise vs interdit en Phase A

> **Principe BY DESIGN** : modifier une metadata sans modifier la question/reponse
> n'est autorise que si la nouvelle valeur est prouvable a partir du contenu existant.
> Sinon, c'est du maquillage de metriques (gate Potemkine).

| Action | Autorise ? | Raison |
|--------|-----------|--------|
| Reclassifier `cognitive_level: Understand -> Apply` si question contient "que doit faire X si Y" | Oui | Verifiable par pattern matching sur le texte |
| Reclassifier `difficulty: 0.5 -> 0.8` sans changer la question | **Non** | Invente une difficulte non demontree |
| Reclassifier `answer_type: extractive -> inferential` si reponse est verbatim dans chunk | **Non** | Mensonge — violation ISO 42001 A.6.2.2 |
| Re-generer une question Apply/Analyze depuis le meme chunk | Oui | BY DESIGN conforme |
| Ajouter champs schema manquants avec valeurs par defaut | Oui | Completion technique |

### 4.3 Corrections autorisees (script)

| Correction | Champs | Methode | Verifiable ? |
|-----------|--------|---------|-------------|
| **A1: Schema completion** | Champs manquants | Ajouter valeurs par defaut pour les 6 champs absents (voir Section 4.5) | Oui — programmatique |
| **A2: Cognitive reclassification safe** | `cognitive_level` | Pattern matching : "que doit faire" + scenario -> Apply. Conserver seulement si verifiable | Oui — regex sur question |
| **A3: audit.history update** | `audit.history` | Ajouter trace "[PHASE A] reclassified 2026-02-19" | Oui — piste d'audit |

### 4.4 Corrections necessitant re-generation (LLM)

| Correction | Pourquoi re-generation ? | Volume estime |
|-----------|------------------------|---------------|
| **A4: Questions hard** (difficulty >= 0.7) | Changer la difficulte sans changer la question = Potemkine | ~40-50 Q (10% de 397) |
| **A5: Questions Apply** | Necessite une question d'application concrete, pas juste un label | ~40-50 Q (10% de 397) |
| **A6: Questions Analyze** | Necessite une question de comparaison/deduction | ~40-50 Q (10% de 397) |
| **A7: Questions comparative** | Type absent, doit etre genere | ~20-30 Q (5% de 397) |
| **A8: Reponses inferential** | answer_type doit correspondre au contenu reel de la reponse | ~40-50 Q (10% de 397) |

**Methode re-generation** : Pour chaque question a re-generer, utiliser le chunk source
(provenance.chunk_id) et un prompt specialise demandant le type cible. La nouvelle
question **remplace** l'ancienne (meme chunk, nouvel ID, audit trail).

### 4.5 Schema v2.0 : les 46 champs

Les 46 champs du Schema v2.0 (GS_SCHEMA_V2.md) repartis en 8 groupes :

| Groupe | Champs | Count |
|--------|--------|-------|
| Root | `id`, `legacy_id` | 2 |
| content | `question`, `expected_answer`, `is_impossible` | 3 |
| mcq | `original_question`, `choices`, `mcq_answer`, `correct_answer`, `original_answer` | 5 |
| provenance | `chunk_id`, `docs`, `pages`, `article_reference`, `answer_explanation`, `annales_source` | 6 |
| provenance.annales_source | `session`, `uv`, `question_num`, `success_rate` | 4 |
| classification | `category`, `keywords`, `difficulty`, `question_type`, `cognitive_level`, `reasoning_type`, `reasoning_class`, `answer_type`, `hard_type` | 9 |
| validation | `status`, `method`, `reviewer`, `answer_current`, `verified_date`, `pages_verified`, `batch` | 7 |
| processing | `chunk_match_score`, `chunk_match_method`, `reasoning_class_method`, `triplet_ready`, `extraction_flags`, `answer_source`, `quality_score` | 7 |
| audit | `history`, `qat_revalidation`, `requires_inference` | 3 |
| **Total** | | **46** |

**Champs actuellement manquants** (diff spec vs reel) :
- `provenance.annales_source` : `null` pour questions BY DESIGN (pas d'annale source) → 4 sous-champs = 0
- `mcq.choices` : souvent `{}` pour questions non-MCQ → compte comme 0
- Total effectif : 40 (answerable) / 36 (unanswerable)

**Decision** : `annales_source = null` est correct pour questions BY DESIGN (pas de source annale).
Le threshold G4-1 reste a 42 dans quality_gates.py. Les questions MCQ sans choices sont un
ecart documente (Phase B generera des questions avec MCQ complet).

### 4.6 Gates Phase A

| Gate | Seuil | Verification |
|------|-------|--------------|
| A-G1 | Schema >= 40 (answerable) | `count_schema_fields(q) >= 40` pour 100% |
| A-G2 | hard >= 10% | `sum(d >= 0.7) / len(answerable) >= 0.10` |
| A-G3 | 4 cognitive levels | `len(set(cognitive_levels)) >= 4` |
| A-G4 | chunk_match_score inchange | `all(score == 100)` — invariant BY DESIGN |
| A-G5 | **Regression test** | 37/37 tests passed + 5 xfail dans test_gs_data_integrity.py |

### 4.7 Validation Phase A : echantillon de controle

**ISO 29119-3** exige des criteres d'acceptation verifiables pour les reclassifications.

| Verification | Methode | Seuil |
|-------------|---------|-------|
| Reclassification cognitive_level | Review manuel sur echantillon aleatoire 30 Q | >= 90% accord (27/30) |
| Re-generation hard/Apply/Analyze | LLM-as-Judge (second modele) verifie la classification | Cohen's kappa >= 0.6 vs annotateur primaire |
| answer_type inferential | Verification programmatique : reponse PAS verbatim dans chunk | 100% coherence (automatise) |

---

## 5. Phase B - Generation Answerable (nouveaux chunks)

### 5.1 Objectif

Couvrir les 1387 chunks actuellement sans question pour atteindre >= 80% de couverture corpus (1485+ chunks).

### 5.2 Strategie de Couverture

Pas tous les chunks meritent une question. Certains sont des tables des matieres, en-tetes, listes vides. Le script de generation retourne `[]` pour ces cas (cf. PLAN-GS-SCRATCH-001 Phase 1).

**Taux de generation estime** : 0.6-0.8 Q/chunk.
- Base v1 : 397 Q / 470 chunks = 0.84. Mais les 470 chunks couverts etaient les plus riches en contenu.
- Les 1387 restants incluent des TdM, en-tetes, chunks administratifs. Estimation conservatrice : ~20% seront filtres (0 Q).
- **Intervalle realiste** : 0.6 Q/chunk (pessimiste, chunks pauvres) a 0.8 Q/chunk (optimiste).

**Priorite de couverture** :

| Priorite | Chunks | Description | Raison |
|----------|--------|-------------|--------|
| 1 (haute) | LA-octobre2025 (806 uncov) | Lois de l'arbitrage | Corpus principal |
| 2 | C01-C04 Coupes (118 uncov) | Competitions coupes | 0% couverture |
| 3 | A02, F01, J03 (103 uncov) | Championnats | 0% couverture |
| 4 | 2025_RI, Contrat, Statuts (161 uncov) | Admin/statutaire | < 50% couverture |
| 5 | Reste | Autres documents | Complement |

### 5.3 Prompts de Generation Conformes

Le prompt de generation Phase 1 doit etre **enrichi** pour forcer la diversite des dimensions manquantes. Modification du prompt `generate_real_questions.py` :

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
- hard (>=0.7): exception a une regle, conditions multiples, interaction entre
  articles, sanction avec circonstances

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

### 5.4 Batching et Stop Gate

| Parametre | Valeur | Justification |
|-----------|--------|---------------|
| Batch size | 30 chunks | Token budget ~150K IN / 50K OUT par batch |
| Distribution check | Tous les 5 batches | Verifier distributions courantes |
| Adaptive steering | Oui | Si fact_single > 55%, forcer reasoning/Apply pour batch suivant |
| **Stop gate qualite** | Keyword coverage < 0.3 sur > 20% du batch | Arret si qualite se degrade |
| **Stop gate budget** | Tokens cumules > 10M | Alerte, revue avant continuation |

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

### 5.6 Inter-Annotator Agreement Phase B

Toutes les classifications (difficulty, cognitive_level, question_type) sont generees
par un LLM unique. Pour valider la fiabilite des annotations :

| Verification | Methode | Seuil d'acceptation |
|-------------|---------|---------------------|
| Classification IAA | Second modele (ex: GPT-4o) annote echantillon 50 Q | Cohen's kappa >= 0.6 |
| Difficulty IAA | LLM-as-Judge evalue la difficulte sur 50 Q | Correlation Spearman >= 0.7 |
| Si kappa < 0.6 | Adjudication manuelle + revision regles prompt | Pas de release tant que kappa < 0.6 |

---

## 6. Phase C - Generation Unanswerable Supplementaire

### 6.1 Objectif

Apres Phase B, le total answerable sera ~1230-1510. Pour maintenir le ratio unanswerable dans [25%, 33%], il faut generer des questions unanswerable supplementaires.

### 6.2 Calcul du Volume

```
Scenario pessimiste:
  Answerable total: 397 + 832 = 1229
  Cible unanswerable 30%: 1229 / 0.70 = 1756 total
  Unanswerable cible: 1756 - 1229 = 527
  Existant: 217. A generer: ~310

Scenario optimiste:
  Answerable total: 397 + 1110 = 1507
  Cible unanswerable 30%: 1507 / 0.70 = 2153 total
  Unanswerable cible: 2153 - 1507 = 646
  Existant: 217. A generer: ~429

Intervalle: 310-429 nouvelles questions unanswerable.
```

### 6.3 Distribution UAEval4RAG Cible

Les 6 categories doivent etre representees. Distribution cible pour les nouvelles :

| Categorie | Ratio cible | Nouvelles Q (min-max) | Total (avec 217 existantes) |
|-----------|-------------|----------------------|----------------------------|
| OUT_OF_DATABASE | 25-30% | 78-129 | 157-208 |
| FALSE_PRESUPPOSITION | 20-25% | 62-107 | 112-157 |
| UNDERSPECIFIED | 15-20% | 47-86 | 92-131 |
| NONSENSICAL | 10-15% | 31-64 | 44-77 |
| MODALITY_LIMITED | 10-15% | 31-64 | 46-79 |
| SAFETY_CONCERNED | 8-12% | 25-51 | 40-66 |

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
Input: GS brut (~1700-2150 Q)
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
  7. Toutes 24 quality gates
  |
OUTPUT: gs_scratch_v2.json
```

### 7.2 Distributions Cibles Finales

| Dimension | Cible | Gate | Source du seuil |
|-----------|-------|------|-----------------|
| Total questions | 1500-1800 | - | Choix projet |
| Corpus coverage | >= 80% | G0-2, G5-8 | Choix projet |
| Unanswerable | 25-33% | G2-2, G5-5 | Adapte SQuAD 2.0 |
| fact_single | 40-55% | G1-3, G5-3 | Choix projet |
| summary | 15-25% | - | Choix projet |
| reasoning | 15-25% | - | Choix projet |
| Remember | 20-35% | - | Choix projet |
| Understand | 25-40% | - | Choix projet |
| Apply | >= 10% | G5-6 (partiel) | Choix projet |
| Analyze | >= 10% | G5-6 (partiel) | Choix projet |
| Hard (>= 0.7) | >= 10% | G5-4 | Gate bloquante |
| Comparative | >= 5% | G5-7 (partiel) | Choix projet |
| Inferential | >= 10% | - | Choix projet |
| Schema fields | >= 42 | G4-1 | Gate bloquante |
| Duplicates | 0 | G5-1 | Gate WARNING |
| Anchor independence | < 0.90 | G5-2 | Choix projet |

### 7.3 Scripts a Reutiliser

| Script existant | Phase D usage |
|----------------|---------------|
| `balance_distribution.py` | Dedup + equilibrage (Phase D etape 1-4) |
| `validate_anti_hallucination.py` | Validation (Phase D etape 5) |
| `enrich_schema_v2.py` | Schema v2 (Phase D etape 6) |
| `quality_gates.py` | 24 gates (Phase D etape 7) |

### 7.4 Gates Phase D

Toutes les 24 gates de quality_gates.py (SPEC-GS-METH-001 Section 4) doivent passer, plus :

| Gate | Seuil | Severite |
|------|-------|----------|
| D-G1 | coverage >= 80% | BLOQUANT |
| D-G2 | 4/4 cognitive levels | BLOQUANT |
| D-G3 | hard >= 10% | BLOQUANT |
| D-G4 | inferential >= 10% | WARNING |
| D-G5 | comparative >= 5% | WARNING |

---

## 8. Tests de Regression (ISO 29119)

### 8.1 Principe

Chaque phase doit garantir que les questions existantes ne sont **pas degradees**.
Un snapshot des tests est pris avant Phase A et doit passer apres chaque phase.

### 8.2 Suite de regression

| Test | Verification | Attendu apres chaque phase |
|------|-------------|---------------------------|
| test_gs_data_integrity.py (37 tests) | Structure, schema, chunks, coherence | 37 PASS |
| test_gs_data_integrity.py (xfail) | Gates sinceres qui echouent | xfail convertis en PASS progressivement |
| test_quality_gates.py (62 tests) | Logique des gates | 62 PASS (unitaire, pas de regression) |
| Snapshot answerable count | Pas de perte de questions existantes | >= 397 answerable apres chaque phase |

### 8.3 Snapshot avant/apres Phase A

```python
# Script de verification regression Phase A
def verify_regression(before_path, after_path):
    before = load_json(before_path)
    after = load_json(after_path)

    # Pas de perte de questions
    assert len(after["questions"]) >= len(before["questions"])

    # chunk_match_score inchange pour toutes les questions conservees
    before_scores = {q["id"]: q["processing"]["chunk_match_score"] for q in before["questions"]}
    for q in after["questions"]:
        if q["id"] in before_scores:
            assert q["processing"]["chunk_match_score"] == before_scores[q["id"]]

    # Pas de question existante supprimee
    before_ids = {q["id"] for q in before["questions"]}
    after_ids = {q["id"] for q in after["questions"]}
    assert before_ids.issubset(after_ids), f"Questions perdues: {before_ids - after_ids}"
```

---

## 9. Plan de Rollback (ISO 12207)

### 9.1 Principe

Chaque phase produit un fichier intermediaire. En cas d'echec, on revient au fichier
precedent sans perte.

| Phase | Input | Output | Rollback = |
|-------|-------|--------|-----------|
| A | gs_scratch_v1.json | gs_scratch_v1_fixed.json | Garder gs_scratch_v1.json |
| B | gs_scratch_v1_fixed.json | gs_scratch_v1_with_new_answerable.json | Garder gs_scratch_v1_fixed.json |
| C | ...with_new_answerable.json | gs_scratch_v1_with_unanswerable.json | Garder ...with_new_answerable.json |
| D | ...with_unanswerable.json | gs_scratch_v2.json | Garder ...with_unanswerable.json |

### 9.2 Criteres de rollback automatique

| Condition | Action |
|-----------|--------|
| Dedup retire > 15% des questions | STOP — investigation qualite avant continuation |
| keyword_score < 0.3 sur > 30% d'un batch Phase B | STOP batch — reviser prompt |
| Coverage progresse < 5% sur 5 batches consecutifs | STOP — chunks restants probablement non-generables |
| Budget tokens > 15M cumule | ALERTE — revue ROI avant continuation |

---

## 10. Scripts a Creer / Modifier

### 10.1 Nouveaux Scripts

| Script | Role | Phase |
|--------|------|-------|
| `fix_gs_v2_metadata.py` | Corrections safe (schema, cognitive reclassification verifiable) | A |
| `regenerate_targeted.py` | Re-generation BY DESIGN ciblee (hard, Apply, Analyze, comparative) | A |
| `generate_v2_coverage.py` | Orchestrateur generation massive avec stop gates | B+C |
| `verify_regression.py` | Tests de regression snapshot avant/apres | A-D |

### 10.2 Scripts a Modifier

| Script | Modification | Phase |
|--------|-------------|-------|
| `generate_real_questions.py` | Prompt V2 (cognitive_level, difficulty hard, comparative, inferential) | B |
| `quality_gates.py` | Ajouter gates D-G1 a D-G5 (si non deja couvertes par G5-6/G5-7/G5-8) | D |
| `test_gs_data_integrity.py` | Tests pour nouvelles distributions, retirer xfail quand corriges | D |

### 10.3 Scripts Existants Reutilises Sans Modification

| Script | Usage |
|--------|-------|
| `stratify_corpus.py` | Phase 0 (restratifier pour chunks non couverts) |
| `validate_anti_hallucination.py` | Phase D validation |
| `enrich_schema_v2.py` | Phase D enrichissement |
| `balance_distribution.py` | Phase D equilibrage |
| `reformulate_by_design.py` | Eventuellement pour reformuler questions trop proches |

---

## 11. Ordre d'Execution

```
ETAPE 1: Snapshot regression
  Script: verify_regression.py --snapshot gs_scratch_v1.json
  Output: snapshots/gs_v1_baseline.json
  Duree:  ~1min

ETAPE 2: Phase A - Fix metadata (614Q)
  Script: fix_gs_v2_metadata.py + regenerate_targeted.py
  Input:  gs_scratch_v1.json
  Output: gs_scratch_v1_fixed.json
  Duree:  ~1h (script + re-generation LLM ~100Q)
  Gate:   A-G1 a A-G5
  GO/NO-GO: regression test PASS → continuer

ETAPE 3: Phase B - Stratification chunks non couverts
  Script: stratify_corpus.py (re-exec sur 1387 chunks)
  Input:  chunks_mode_b_fr.json - chunks deja couverts
  Output: uncovered_chunk_strata.json
  Duree:  ~5min

ETAPE 4: Phase B - Generation answerable par batches
  Script: generate_v2_coverage.py -> generate_real_questions.py
  Input:  uncovered_chunk_strata.json
  Output: new_answerable_questions.json (~830-1110 Q)
  Duree:  ~28-37 batches x 30 chunks (4-6h)
  Gate:   G1-1, G1-2, G1-4, B-G1 a B-G5
  Stop:   Si keyword_coverage < 0.3 sur > 30% batch
  GO/NO-GO: coverage >= 70% → continuer

ETAPE 5: IAA validation Phase B
  Script: Echantillon 50 Q → second modele
  Verification: Cohen's kappa >= 0.6
  GO/NO-GO: kappa >= 0.6 → continuer

ETAPE 6: Phase C - Generation unanswerable
  Script: generate_v2_coverage.py (mode unanswerable)
  Input:  GS elargi
  Output: new_unanswerable_questions.json (~310-429 Q)
  Gate:   G2-1, G2-2, G2-3
  GO/NO-GO: ratio unanswerable dans [25%, 33%] → continuer

ETAPE 7: Phase D - Merge + equilibrage + validation
  Scripts: balance_distribution.py, validate_anti_hallucination.py,
           enrich_schema_v2.py, quality_gates.py
  Input:  614 fixees + ~830-1110 answerable + ~310-429 unanswerable
  Output: gs_scratch_v2.json (~1500-1800 Q)
  Gate:   24 gates quality_gates.py + D-G1 a D-G5

ETAPE 8: Regression finale + test integration
  Script: verify_regression.py + test_gs_data_integrity.py (mis a jour)
  Verification: pytest pass sur nouveau GS, xfail convertis en PASS
```

---

## 12. Risques et Mitigations

| Risque | Impact | Probabilite | Mitigation | Critere declenchement |
|--------|--------|-------------|------------|----------------------|
| Chunks non propices (TdM, listes) | Coverage < 80% malgre generation | Elevee | Documenter deviation formelle, accepter >= 75% avec justification | > 20% chunks retournent [] |
| Token budget excessif | Cout > $100 | Moyenne | Stop gate a $50 cumule, revoir strategie | Budget > 10M tokens |
| Qualite degradee a grande echelle | Questions repetitives / generiques | Moyenne | Dedup Phase D + distribution steering + stop gate keyword_score | keyword_score < 0.3 sur > 30% batch |
| Hard questions artificielles | Questions forcees sans naturalite | Elevee | IAA avec second modele (kappa >= 0.6) | kappa < 0.6 sur echantillon |
| Anchor violation (sim >= 0.90) | Triplet training invalide | Faible | Phase D etape 2 filtre systematique | > 5% questions violant seuil |
| Regression sur existantes | Questions v1 degradees/perdues | Faible | Snapshot regression + tests automatiques | Toute question v1 modifiee sans trace |
| IAA insuffisant | Annotations non fiables | Moyenne | Revision prompt + adjudication manuelle | kappa < 0.6 |

---

## 13. Criteres de Succes

Le GS v2 est considere conforme quand :

1. **24/24 gates PASS** dans quality_gates.py (11 bloquantes, 13 warning)
2. **Coverage >= 80%** (1485+ chunks couverts) — ou deviation documentee si >= 75%
3. **4/4 cognitive levels** presents avec >= 10% Apply et >= 10% Analyze
4. **Hard >= 10%** (difficulty >= 0.7, answerable seulement)
5. **0 duplicates semantiques** (seuil 0.95)
6. **100% chunk_match_score = 100** (invariant BY DESIGN)
7. **0% hallucination** (validation 3 niveaux)
8. **Schema >= 42 champs** pour chaque question (G4-1)
9. **IAA kappa >= 0.6** sur echantillon de validation
10. **Regression test PASS** : toutes les questions v1 conservees

---

## 14. Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-02-18 | Creation - audit findings F1-F10, plan 4 phases, cibles distribution |
| 2.0 | 2026-02-19 | Corrections 10 findings rigueur : (1) criteres acceptation Phase A mesurables, (2) distinction seuils projet vs normatifs, (3) go/no-go inter-phases + rollback, (4) intervalles de confiance volumes, (5) Phase A BY DESIGN (pas de reclassification Potemkine), (6) IAA kappa >= 0.6, (7) schema 46 champs liste complete, (8) budget tokens/cout, (9) 21 -> 24 gates, (10) tests regression |

---

*Plan ISO 42001/25010/29119 - Pocket Arbiter Project*
*Methode: BY DESIGN (chunk = INPUT, pas OUTPUT)*
