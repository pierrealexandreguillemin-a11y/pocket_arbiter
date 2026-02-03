# GS Batch 001 Audit Report

> **Batch**: 001
> **Questions**: Q1-Q10 (indices 0-9)
> **Date**: 2026-02-03
> **Auditor**: Claude Opus 4.5 (LLM-as-judge)
> **ISO Reference**: ISO 42001 A.6.2.2, ISO 29119-3

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Questions Processed | 10 |
| Mapping Corrections | 0 |
| MCQ→Direct Reformulations | 10 |
| reasoning_class Fixes | 1 |
| Quality Gates PASS | 11/11 (100%) |
| Status | **PASS** |

---

## 2. Question-by-Question Audit

### Q1: ffe:annales:clubs:001:55d409b5

**Topic**: Missions de l'arbitre

| Gate | Status | Notes |
|------|--------|-------|
| G1 | PASS | Question finit par `?` |
| G2 | PASS | 54 chars > 10 |
| G3 | PASS | 231 chars > 5 |
| G4 | PASS | LA-octobre2025.pdf-p010-parent024-child00 existe |
| G5 | PASS | Chunk contient "mise en place des jeux, pendules, cavaliers..." |
| G6 | PASS | Mapping cohérent |
| G7 | PASS | difficulty=0.15 ∈ [0,1] |
| G8 | PASS | reasoning_class=summary |
| G9 | PASS | requires_context=false (réponse dans chunk) |
| G10 | PASS | validation.status=VALIDATED |
| G11 | PASS | Reformulée: MCQ→directe |

**Reformulation**:
- AVANT: "Quelle proposition parmi les suivantes ne correspond pas à une des missions de l'arbitre ?"
- APRÈS: "Quelle tâche ne fait pas partie des missions de l'arbitre ?"

---

### Q2: ffe:annales:clubs:002:79da8e92

**Topic**: Composition jury d'appel

| Gate | Status | Notes |
|------|--------|-------|
| G1-G11 | PASS | Tous gates validés |

**Reformulation**:
- AVANT: "Pour la formation d'un jury d'appel lors d'une compétition, quelle personne parmi les suivantes ne peut siéger ?"
- APRÈS: "Qui ne peut pas siéger dans un jury d'appel lors d'une compétition d'échecs ?"

---

### Q3: ffe:annales:clubs:003:06c34715

**Topic**: Sanctions capitaines

| Gate | Status | Notes |
|------|--------|-------|
| G1-G11 | PASS | Tous gates validés |

**Reformulation**:
- AVANT: "L'arbitre d'un match peut sanctionner les capitaines qui ne respectent pas les règles. Quelle sanction parmi les suivantes n'est pas applicable ?"
- APRÈS: "Quelle sanction l'arbitre ne peut-il pas appliquer à un capitaine d'équipe qui ne respecte pas les règles ?"

---

### Q4: ffe:annales:clubs:004:d22e1892

**Topic**: Entente de clubs

| Gate | Status | Notes |
|------|--------|-------|
| G1-G11 | PASS | Tous gates validés |

**Analyse Mapping**:
Le chunk liste les compétitions AUTORISÉES pour une entente:
- Championnats de France des clubs
- Interclubs Jeunes
- Championnats de France des clubs féminins

La Coupe de France N'EST PAS dans la liste → réponse B correcte par omission.

**Reformulation**:
- AVANT: "Deux clubs souhaitent former une entente afin de participer à certaines compétitions dans la plus petite division locale car ils n'ont pas suffisamment de licenciés. Pour quelle compétition parmi les suivantes n'est-ce pas possible ?"
- APRÈS: "À quelle compétition une entente entre clubs ne peut-elle pas participer ?"

---

### Q5: ffe:annales:clubs:005:b740870c

**Topic**: Missions DRA

| Gate | Status | Notes |
|------|--------|-------|
| G1-G11 | PASS | Tous gates validés |

**Analyse Mapping**:
Le chunk indique clairement: "Le montant est déterminé par le comité directeur de la Ligue, qui contrôle l'utilisation de ces fonds."
→ Ce n'est PAS le DRA qui contrôle le budget.

**Reformulation**:
- AVANT: "Claude est Directeur Régional de l'Arbitrage dans sa Ligue. Quelle proposition parmi les suivantes ne correspond pas à une de ses missions ?"
- APRÈS: "Quelle mission ne relève pas du Directeur Régional de l'Arbitrage (DRA) ?"

---

### Q6: ffe:annales:clubs:006:1bcec3d3

**Topic**: Arbitre inactif 10 ans

| Gate | Status | Notes |
|------|--------|-------|
| G1-G11 | PASS | Tous gates validés |

**Analyse Mapping**:
Chunk clair: "Un arbitre inactif n'ayant fait aucune formation durant une période de dix ans sera en plus dans l'obligation de valider à nouveau l'UVR."
+ "sera dans l'obligation de suivre un stage SC"

---

### Q7: ffe:annales:clubs:007:3431b54c

**Topic**: Critères administratifs titre arbitre

| Gate | Status | Notes |
|------|--------|-------|
| G1-G11 | PASS | Tous gates validés |

**Analyse Mapping**:
Chunk: "Avoir possédé, à un moment donné, un classement Elo (lent ou rapide, national ou F.I.D.E.)."
→ Pas besoin de classement ACTUEL, juste historique.

---

### Q8: ffe:annales:clubs:008:5e3f7d24

**Topic**: Arbitre international étranger

| Gate | Status | Notes |
|------|--------|-------|
| G1-G11 | PASS | Tous gates validés |

**Analyse Mapping**:
Chunk: "Les licenciés A dont le code F.I.D.E. n'est pas FRA et possédant un titre d'arbitre de la F.I.D.E. (AF ou AI) [...] peuvent obtenir un titre fédéral leur permettant d'officier en tant qu'arbitre, en validant l'UVC."

---

### Q9: ffe:annales:clubs:009:fd51d9be

**Topic**: Instance qualification joueur

| Gate | Status | Notes |
|------|--------|-------|
| G1-G11 | PASS | Tous gates validés |

**Analyse Mapping**:
Chunk: "Ceux et celles qui n'ont pas obtenu l'accord de la Commission Technique fédérale..."

---

### Q10: ffe:annales:clubs:010:4c1cad22

**Topic**: Catégorie d'âge

| Gate | Status | Notes |
|------|--------|-------|
| G1-G10 | PASS | |
| G11 | PASS | Reformulée |
| **reasoning_class** | **FIX** | summary → arithmetic |

**Fix Applied**:
- Cette question nécessite un calcul d'âge (né 03/01/2011, saison 2024-2025)
- Classification corrigée: `reasoning_class: arithmetic`

**Reformulation**:
- AVANT: "Pour la saison 2024-2025, un joueur a pris sa licence le 30/09/2024, il est né le 03/01/2011. Quelle est sa catégorie d'âge ?"
- APRÈS: "Quelle est la catégorie d'âge d'un joueur né le 03/01/2011 pour la saison 2024-2025 ?"

---

## 3. Statistics

| Catégorie | Count |
|-----------|-------|
| Mapping OK | 10/10 |
| Reformulations | 10/10 |
| reasoning_class fixes | 1 (Q10) |
| Total gates validated | 110/110 |

---

## 4. Files Modified

1. `tests/data/gold_standard_annales_fr_v7.json` - Questions Q1-Q10 updated
2. `docs/audits/GS_MANUAL_AUDIT_CHECKLIST.md` - Updated with batch 001 results
3. `docs/audits/GS_BATCH_001_REPORT.md` - This report

---

## 5. Next Steps

- [ ] Batch 002: Questions Q11-Q20
- [ ] Continue until all 420 questions audited (42 batches total)
