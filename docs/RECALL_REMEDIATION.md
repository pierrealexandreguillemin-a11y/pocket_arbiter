# Plan de Remédiation Recall - ISO 25010 FA-01

> **Document ID**: DOC-REM-001
> **Date**: 2026-01-15
> **Statut**: ÉCHEC BLOQUANT
> **Priorité**: CRITIQUE

---

## 1. Constat

### 1.1 Situation actuelle

| Métrique | Cible ISO | Actuel | Statut |
|----------|-----------|--------|--------|
| Recall@5 FR | ≥ 80% | **48.89%** | ❌ ÉCHEC |

### 1.2 Méthodologie d'évaluation

- **Gold standard**: v4.0 (non-circulaire, ISO 42001 compliant)
- **Base**: expected_pages créées manuellement (v1)
- **Validation**: Keywords présents dans corpus extrait
- **Corpus**: 28 PDFs FFE, 1076 chunks sentence

### 1.3 Questions faibles (recall < 50%)

| ID | Question | Expected | Retrieved | Gap |
|----|----------|----------|-----------|-----|
| FR-Q03 | Retard joueur | [46, 68] | [1, 54, 64] | Hors sujet |
| FR-Q04 | Temps dépassé | [45, 46, 50] | [4, 52, 3] | Hors sujet |
| FR-Q08 | Règle 50 coups | [55] | [56, 68, 70] | Adjacent |
| FR-Q22 | Départage | [149-154] | [146, 147] | Adjacent |
| FR-Q23 | Anti-triche | [71-73] | [36, 30, 34] | Hors sujet |
| FR-Q24 | Système suisse | [120-122] | [138, 120, 148] | Partiel |

---

## 2. Analyse des causes

### 2.1 Causes identifiées

1. **Chunking sentence trop fragmenté**
   - Moyenne: 131 tokens/chunk (vs 400-512 recommandé)
   - 63% chunks < 100 tokens
   - Perte de contexte sémantique

2. **Mismatch pages v1 vs chunks**
   - Pages v1 basées sur PDF original
   - Chunks créés par sentence splitter
   - Décalage possible des numéros de page

3. **EmbeddingGemma QAT + français**
   - Modèle optimisé pour anglais
   - Performance réduite sur terminologie échecs FR
   - Pas de fine-tuning domain-specific

### 2.2 Causes NON identifiées (éliminées)

- ❌ Biais circulaire gold standard → Corrigé en v4
- ❌ Coverage tests insuffisant → 76% > 60% requis
- ❌ Tests cassés → Corrigés

---

## 3. Plan de remédiation

### 3.1 Actions court terme (Sprint actuel)

| # | Action | Impact estimé | Effort |
|---|--------|---------------|--------|
| 1 | Augmenter chunk size (512 tokens) | +10-15% recall | Faible |
| 2 | Activer recherche hybride (vector + BM25) | +10-20% recall | Moyen |
| 3 | Tolérance pages ±2 dans gold standard | +5-10% recall | Faible |

### 3.2 Actions moyen terme (Phase 2)

| # | Action | Impact estimé | Effort |
|---|--------|---------------|--------|
| 4 | Chunking sémantique par Article/Section | +15-25% recall | Moyen |
| 5 | Re-ranker cross-encoder | +5-10% recall | Moyen |
| 6 | Augmenter top-k à 10 | +10% recall | Trivial |

### 3.3 Actions long terme (Phase 3)

| # | Action | Impact estimé | Effort |
|---|--------|---------------|--------|
| 7 | Fine-tuning embeddings sur corpus FFE | +10-20% recall | Élevé |
| 8 | Gold standard 50+ questions (ISO 42001) | Meilleure éval | Moyen |
| 9 | Évaluation humaine FA-03 | Validation réelle | Élevé |

---

## 4. Critères de succès

### 4.1 Gate Phase 1B

- [ ] Recall@5 ≥ 70% (objectif intermédiaire)
- [ ] Recall@10 ≥ 85%
- [ ] Recherche hybride activée

### 4.2 Gate Phase 2

- [ ] Recall@5 ≥ 80% (ISO 25010 FA-01)
- [ ] Gold standard 50 questions
- [ ] Chunking sémantique validé

---

## 5. Suivi

| Date | Recall@5 | Action | Résultat |
|------|----------|--------|----------|
| 2026-01-15 | 48.89% | Gold standard v4 (non-circulaire) | Baseline établi |
| - | - | - | - |

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-15 | Création - Constat échec recall |
