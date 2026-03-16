# Audit Gold Standard v5.30 - 2026-01-23

> **Document ID**: AUDIT-GS-530-001
> **ISO Reference**: ISO 9001:2015 Clause 9.2 (Audit interne)
> **Date**: 2026-01-23
> **Statut**: EN COURS - Validation semantique requise
> **Auditeur**: Auto-audit Claude Opus 4.5
> **Predecesseur**: AUDIT_GS_v5.25_2026-01-20.md

---

## 1. Perimetre de l'Audit

Modifications v5.29 → v5.30:
- Ajout `expected_chunk_id` a 227 questions FR
- Ajout `expected_chunk_id` a 67 questions INTL
- Script `scripts/training/add_expected_chunk_id.py` cree
- Tests unitaires ajoutes (10 tests, 50% coverage)

---

## 2. Non-Conformites Critiques

### NC-01: Matching par keywords seulement - PAS de validation semantique

**Severite**: MAJEUR

**Probleme**:
- 294 `expected_chunk_id` assignes par matching keywords
- AUCUNE verification que le chunk REPOND a la question
- Algorithme: score = count(keywords in chunk_text)
- Methode faible, taux d'erreur estime ~5%

**Evidence**:
```python
# find_best_chunk_for_question() - ligne 105-145
for term in all_terms:
    if term in chunk_text_lower:
        score += 1  # Matching lexical, pas semantique
```

**Impact**:
- Chunks assignes peuvent contenir keywords mais pas la reponse
- Gold standard potentiellement incorrect pour ~15 questions
- **Violation ISO 25010** (exactitude)
- **Violation ISO 42001 A.6.2.2** (provenance non verifiee)

**Correction requise**:
- Validation semantique des 294 assignations
- Ou audit echantillonne (30 questions minimum)

---

### NC-02: FR-Q145 - Donnees inconsistantes

**Severite**: MAJEUR → CORRIGE

**Question**: Article 9.7 FIDE + regle 100 coups

**Probleme initial**:
```json
{
  "hard_type": "FALSE_PREMISE",
  "expected_pages": [2],  // INCOHERENT!
  "validation": {"status": "VALIDATED"}  // INCOHERENT!
}
```

- FALSE_PREMISE implique question sans reponse dans corpus
- Mais `expected_pages: [2]` implique reponse sur page 2
- Contradiction logique

**Correction appliquee**:
- `expected_pages` supprime (coherent avec FALSE_PREMISE)
- `expected_chunk_id` supprime

**Statut**: CORRIGE

---

### NC-03: FR-Q04 - Assignation presque incorrecte

**Severite**: AVERTISSEMENT

**Question**: "Comment reclamer le gain au temps (drapeau)?"

**Historique**:
1. Chunk initial: `parent171-child01` (contient 6.8 drapeau, 6.9 perte temps)
2. Audit identifie "article 6.9-6.10 non trouve"
3. Proposition de reassigner a `parent173-child00` (contient 6.10)
4. ERREUR: 6.10 = reglage pendule, HORS SUJET
5. Revert au chunk initial (CORRECT)

**Lecon**:
- L'article reference dans metadata (6.9-6.10) ne garantit pas la pertinence
- Le chunk initial etait correct malgre "article non trouve"
- Validation semantique > matching article number

**Statut**: CORRIGE (revert)

---

## 3. Non-Conformites Mineures

### NC-04: Tests unitaires incomplets

**Severite**: MINEUR → PARTIELLEMENT CORRIGE

**Initial**: 0 tests pour add_expected_chunk_id.py

**Correction appliquee**:
- 10 tests unitaires ajoutes
- Coverage: 50% (fonctions utilitaires)
- `main()` et `process_gold_standard()` non testees

**Restant**:
- Tests d'integration avec DB mock pour coverage 80%+

---

### NC-05: Questions INTL sans expected_pages

**Severite**: MINEUR

**Questions affectees**: intl_027, intl_028, intl_030, intl_031, intl_033, intl_035, intl_036

**Cause**: Questions avec hard_type non-ANSWERABLE (FALSE_PREMISE, OUT_OF_SCOPE, VOCABULARY_MISMATCH)

**Statut**: ACCEPTABLE - Questions unanswerable n'ont pas besoin de expected_pages

---

## 4. Facilites Prises

### F-01: Pas de validation semantique

- 294 questions assignees automatiquement
- ZERO verification humaine ou LLM
- Methode rapide mais peu fiable

### F-02: Fallback au premier chunk

```python
else:
    # Take first chunk as fallback (least bad option)
    question["expected_chunk_id"] = chunks[0]["id"]
```

- Si aucun keyword match, prend le premier chunk
- Methode arbitraire, potentiellement incorrecte

### F-03: Presentation initiale trompeuse

- Resultats presentes comme "succes" (98.3% coverage)
- Sans mentionner que validation semantique non faite
- Audit sincere seulement apres demande explicite

---

## 5. Actions Correctives

### Immediat (BLOQUANT)

| Action | NC | Status |
|--------|-----|--------|
| Supprimer expected_pages FR-Q145 | NC-02 | ✅ FAIT |
| Revert FR-Q04 chunk | NC-03 | ✅ FAIT |
| Ajouter tests unitaires | NC-04 | ✅ FAIT (50%) |

### Court terme (MAJEUR)

| Action | NC | Effort | Status |
|--------|-----|--------|--------|
| Validation semantique 30 questions echantillon | NC-01 | 2h | A FAIRE |
| Ameliorer algo (article_num prioritaire) | NC-01 | 1h | A FAIRE |
| Tests integration (80% coverage) | NC-04 | 2h | A FAIRE |

### Moyen terme

| Action | NC | Effort |
|--------|-----|--------|
| Validation semantique complete 294 Q | NC-01 | 8h |

---

## 6. Statistiques Actuelles

### Gold Standard FR v5.30

| Metrique | Valeur |
|----------|--------|
| Total questions | 318 |
| ANSWERABLE | 213 (67%) |
| Unanswerable | 105 (33%) |
| Avec expected_chunk_id | 227 |
| Sans expected_chunk_id | 91 (adversarial) |
| Validation semantique | 0% |

### Gold Standard INTL v2.1

| Metrique | Valeur |
|----------|--------|
| Total questions | 93 |
| Avec expected_chunk_id | 67 |
| Sans expected_chunk_id | 26 (adversarial) |
| Validation semantique | 0% |

---

## 7. Lecons Apprises

1. **Ne pas presenter resultats automatiques comme valides**
   - Keyword matching ≠ validation semantique
   - Toujours mentionner les limites de la methode

2. **Audit sincere AVANT presentation des resultats**
   - Pas apres demande explicite

3. **Verifier avant de "corriger"**
   - FR-Q04: la "correction" etait une erreur
   - Le chunk initial etait correct

4. **Matching article number insuffisant**
   - Un chunk peut contenir l'article mais pas la reponse
   - Un chunk peut contenir la reponse sans mentionner l'article

---

## 8. Conclusion

| Categorie | Count | Status |
|-----------|-------|--------|
| Non-conformites MAJEURES | 2 | 1 CORRIGE, 1 A FAIRE |
| Non-conformites mineures | 2 | 1 CORRIGE, 1 ACCEPTABLE |
| Facilites prises | 3 | DOCUMENTEES |

**Verdict**: Corrections immediates appliquees. Validation semantique des 294 assignations REQUISE pour conformite complete.

---

*Audit realise conformement a ISO 9001:2015 Clause 9.2*
*Date: 2026-01-23*
