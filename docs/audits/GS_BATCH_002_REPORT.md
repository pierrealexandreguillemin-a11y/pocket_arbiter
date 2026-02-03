# GS Batch 002 Audit Report

> **Batch**: 002
> **Questions**: Q11-Q20 (indices 10-19)
> **Date**: 2026-02-03
> **Auditor**: Claude Opus 4.5 (LLM-as-judge)
> **ISO Reference**: ISO 42001 A.6.2.2, ISO 29119-3

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Questions Processed | 10 |
| Mapping Corrections | 1 (Q14) |
| MCQ→Direct Reformulations | 10 |
| reasoning_class Fixes | 2 (Q11, Q12: arithmetic) |
| requires_inference Added | 9 |
| Quality Gates PASS | 11/11 (100%) |
| Status | **PASS** |

### Lessons Applied from Batch 001

- Questions negatives: omission explicite + requires_inference=true
- Pas d'acronymes inventes
- Questions arithmetiques flaggees

---

## 2. Question-by-Question Audit

### Q11: ffe:annales:clubs:011:0ed8268f

**Topic**: Situations de forfait

| Issue | Resolution |
|-------|------------|
| Question negative | requires_inference=true |
| MCQ style | Reformulee |
| Arithmetique (4/8 = moitie exacte) | reasoning_class=arithmetic |

**Analyse**: Le chunk liste les cas de forfait incluant "moins de la moitie des membres". 4/8 = exactement la moitie, donc ce n'est PAS un forfait. Inference logique.

---

### Q12: ffe:annales:clubs:012:f348e858

**Topic**: Homologation resultat

| Issue | Resolution |
|-------|------------|
| Calcul de jours | reasoning_class=arithmetic |
| requires_inference | true (dimanche + 4 = jeudi) |

---

### Q13: ffe:annales:clubs:013:a7e8c494

**Topic**: Droits inscription handicape

**Analyse**: Le chunk dit "La majoration ne s'applique pas pour carte mobilite inclusion". La personne handicapee paie le tarif normal (30€), pas le majore (40€).

---

### Q14: ffe:annales:clubs:014:cd3052cc — MAPPING CORRIGE

**Topic**: Licence A/B pour tournoi bi-phase

| Issue | Resolution |
|-------|------------|
| **Mapping incorrect** | Corrige: parent009 → parent010 |
| Chunk original | Parlait de nationalite/FIDE |
| Nouveau chunk | Explique type A (licence A) vs type B (licence B) |

**Nouveau chunk**: `R03_2025_26_Competitions_homologuees.pdf-p002-parent010-child00`

---

### Q15: ffe:annales:clubs:015:9da1351e

**Topic**: Inscription gratuite champions (tournoi rapide)

**Analyse**: La gratuité pour champions de France jeunes s'applique aux "tournois de type A" (cadence lente). Un tournoi rapide est type B, donc la règle ne s'applique pas.

---

### Q16: ffe:annales:clubs:016:223bd59d

**Topic**: Championnat clubs - situations autorisees

| Issue | Resolution |
|-------|------------|
| Question negative | requires_inference=true |
| MCQ style | Reformulee |

**Analyse**: Le chunk dit "Un club ne peut engager qu'une équipe en N1". Donc 2 équipes en N1 = non autorisé.

---

### Q17: ffe:annales:clubs:017:5a48507f

**Topic**: Nomination directions de groupe N3

**Analyse**: Direct du chunk: "la personne chargée de la direction de Nationale nomme [...] les directions de groupe" pour N1-N3.

---

### Q18: ffe:annales:clubs:018:d470292e

**Topic**: Niveau arbitre minimum N4

| Issue | Resolution |
|-------|------------|
| Inference sur "minimum" | requires_inference=true |

**Analyse**: Le chunk liste "Elite, d'Open, de Club, ou Jeune" pour N4. L'inference que "Jeune" est le minimum n'est pas explicite dans le chunk.

---

### Q19: ffe:annales:clubs:019:4d3a9bf8

**Topic**: Regle des 3 matchs equipe plus forte

**Analyse**: Le chunk contient la règle 3.7.c. La joueuse a joué 1 fois en N2, donc n'a pas atteint la limite de 3 mais ne peut plus descendre.

---

### Q20: ffe:annales:clubs:020:c41e0ed8

**Topic**: Forfait retard

| Issue | Resolution |
|-------|------------|
| Calcul heure | requires_inference=true |

**Analyse**: Retard 60 min calculé depuis heure PREVUE (14:15), pas heure réelle (14:30). Donc forfait à 15:15.

---

## 3. Statistics

| Categorie | Count |
|-----------|-------|
| Mapping corrections | 1 (Q14) |
| Questions negatives | 2 (Q11, Q16) |
| Questions arithmetiques | 2 (Q11, Q12) |
| requires_inference | 9/10 |
| Total gates validated | 110/110 |

---

## 4. Lessons Learned (Batch 002)

1. **Mapping verification critique**: Q14 avait un chunk completement hors-sujet
2. **Questions bi-phase**: Tournois avec cadences différentes = types différents = licences différentes
3. **Calculs temporels**: Jours calendaires, heures de forfait = requires_inference

---

## 5. Files Modified

1. `tests/data/gold_standard_annales_fr_v7.json` - Questions Q11-Q20 updated
2. `docs/audits/GS_MANUAL_AUDIT_CHECKLIST.md` - Updated with batch 002 results
3. `docs/audits/GS_BATCH_002_REPORT.md` - This report

---

## 6. Next Steps

- [ ] Batch 003: Questions Q21-Q30
- [ ] Continue until all 420 questions audited (42 batches total)
- [ ] Update agent lessons learned with batch 002 findings
