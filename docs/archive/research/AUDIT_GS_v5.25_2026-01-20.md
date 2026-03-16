# Audit Gold Standard v5.25 - 2026-01-20

> **Document ID**: AUDIT-GS-525-001
> **ISO Reference**: ISO 9001:2015 Clause 9.2 (Audit interne)
> **Date**: 2026-01-20
> **Statut**: CRITIQUE - Corrections requises
> **Auditeur**: Auto-audit Claude Opus 4.5

---

## 1. Perimetre de l'Audit

Questions ajoutees dans v5.25:
- FR-Q140 a FR-Q149 (10 questions)
- 5 questions A02/R01 (complexes/edge)
- 5 questions LA-octobre (unanswerable)

---

## 2. Non-Conformites Critiques

### NC-01: FR-Q145 - PREMISSE FAUSSE INVENTEE

**Severite**: BLOQUANT

**Question**:
> "Selon l'article 12.9 des Lois du Jeu FIDE, quelle est la procedure exacte pour reclamer une nulle apres 75 coups..."

**Affirmation**:
> `hard_type: "FALSE_PREMISE"`
> `corpus_truth: "FAUX - Art 12.9 n'existe pas"`

**REALITE**:
- **L'article 12.9 EXISTE** dans LA-octobre2025.pdf
- Mentionne **16 fois** dans le corpus
- Contenu: Sanctions pour comportement incorrect (p44, p47, etc.)
- Page 56: "12.9 L'arbitre peut infliger des penalites..."

**Impact**:
- Question basee sur une premisse que J'AI INVENTEE comme fausse
- **Violation ISO 42001 A.6.2** (qualite des donnees)
- Gold standard contient maintenant une question INVALIDE

**Correction requise**:
- Supprimer FR-Q145 OU
- La transformer en question ANSWERABLE sur article 12.9

---

### NC-02: FR-Q147 - Pages incorrectes

**Severite**: MAJEUR

**Question**: Apple Watch / montre connectee

**Pages declarees**: `[54, 55]`

**Pages reelles avec 11.3.x**:
- Page 53: 3 chunks
- Page 54: 1 chunk
- Page 55: 0 chunks (!)

**Correction**: `expected_pages: [53, 54]`

---

## 3. Non-Conformites Mineures

### NC-03: FR-Q148 - Pages a verifier

**Question**: Coups illegaux blitz

**Pages declarees**: `[48, 57]`

**Verification**:
- Page 48: OK (7.5.x)
- Page 57: OK (A.5.x)
- Page 58 aussi pertinent

**Statut**: ACCEPTABLE mais pourrait inclure p58

---

### NC-04: Distribution non conforme

**Objectif**:
| Categorie | Cible |
|-----------|-------|
| ANSWERABLE | 40-50% |
| Complex | 20-30% |
| Edge | 15-20% |
| OOS | 10-15% |

**Actuel apres v5.25** (149 questions):
- ANSWERABLE: ~88% (encore trop eleve)
- Edge/Complex/OOS: ~12%

**Impact**: Distribution toujours non conforme aux best practices UAEval4RAG

---

## 4. Facilites Prises

### F-01: Pas de verification avant affirmation FALSE_PREMISE

J'ai invente des articles "inexistants" sans verifier:
- Article 12.9: **EXISTE** (erreur critique)
- Article 4.5 A02: Verifie OK - n'existe pas
- Article 7.3 A01: Non verifie

### F-02: Pages non verifiees systematiquement

Certaines `expected_pages` assignees par supposition, pas verification corpus:
- FR-Q147: pages 54-55 au lieu de 53-54

### F-03: corpus_truth parfois approximatif

Certains `corpus_truth` bases sur memoire, pas lecture directe du chunk.

---

## 5. Actions Correctives

### Immediat (BLOQUANT)

| Action | NC | Effort |
|--------|-----|--------|
| Corriger FR-Q145 (supprimer ou transformer) | NC-01 | 5min |
| Corriger FR-Q147 expected_pages [53,54] | NC-02 | 2min |

### Court terme

| Action | NC | Effort |
|--------|-----|--------|
| Verifier article 7.3 A01 (FR-Q137) | F-01 | 5min |
| Ajouter plus de questions edge/OOS | NC-04 | 30min |

---

## 6. Lecons Apprises

1. **TOUJOURS verifier les "faux" avant de les declarer faux**
   - Chercher l'article dans le corpus AVANT d'affirmer qu'il n'existe pas

2. **Verifier les pages exactes dans la DB**
   - Ne pas supposer les numeros de page

3. **FALSE_PREMISE est dangereux**
   - Risque d'inventer des faussetes qui sont en fait vraies

4. **Audit systematique avant commit**
   - Script de verification automatique recommande

---

## 7. Conclusion

| Categorie | Count |
|-----------|-------|
| Non-conformites BLOQUANTES | 1 |
| Non-conformites MAJEURES | 1 |
| Non-conformites mineures | 2 |
| Facilites prises | 3 |

**Verdict**: ~~Corrections BLOQUANTES requises avant commit.~~ **CORRIGE**

---

## 8. Corrections Effectuees

| NC | Action | Status |
|----|--------|--------|
| NC-01 | FR-Q145: Article 12.9 -> 9.7 (verifie inexistant) + regle 100 coups (fausse) | CORRIGE |
| NC-02 | FR-Q147: expected_pages [54,55] -> [53,54] | CORRIGE |
| NC-03 | FR-Q147: corpus_truth p54-55 -> p53-54 | CORRIGE |

### Verifications Finales Passees

- **3/3 questions FALSE_PREMISE** verifiees: articles 7.3 A01, 4.5 A02, 9.7 FIDE tous inexistants dans scope
- **Toutes expected_pages** verifiees contre corpus_fr.db
- **Tous corpus_truth** coherents avec expected_pages

---

*Audit realise conformement a ISO 9001:2015 Clause 9.2*
*Corrections appliquees: 2026-01-20*
