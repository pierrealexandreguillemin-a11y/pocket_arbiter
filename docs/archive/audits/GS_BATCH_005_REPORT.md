# GS Batch 005 Audit Report

> **Batch**: 005
> **Questions**: Q41-Q50 (indices 40-49)
> **Date**: 2026-02-03
> **Auditor**: Claude Opus 4.5 (LLM-as-judge)
> **ISO Reference**: ISO 42001 A.6.2.2, ISO 29119-3

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Questions Processed | 10 |
| Questions VALIDATED | 9 |
| Questions FAILED | 1 (Q47) |
| Mapping Corrections | 1 (Q42: forfait double) |
| MCQ->Direct Reformulations | 4 (Q44, Q46, Q48, Q50) |
| reasoning_class Fixes | 2 (Q42, Q45: arithmetic) |
| requires_inference Added | 2 (Q42, Q45) |
| Quality Gates PASS | 99/110 (90%) |
| Status | **PASS avec 1 FAIL** |

---

## 2. Question-by-Question Audit

### Q41: Transmission resultats
- Chunk confirme: "3.11 Transmission des resultats"
- Mapping OK

### Q42: Forfait derniere ronde N3 (CORRIGE)
- Question: "Quel est le montant de l'amende pour forfait d'equipe en derniere ronde N3?"
- Ancien mapping: chunk sans doublement
- NOUVEAU MAPPING: `A02...parent020-child01` (article 3.9)
- Chunk confirme: "300 € en N3. Cette amende sera doublee si le forfait a lieu a la derniere ronde"
- Calcul: 300€ x 2 = 600€
- reasoning_class=arithmetic, requires_inference=true

### Q43: Delai appel
- Chunk confirme: "Sous peine d'irrecevabilite, l'appel doit etre forme..."
- expected_answer corrige (texte complet)
- Mapping OK

### Q44: Responsable rencontre (REFORMULE)
- Suppression balise `<!-- image -->` dans question
- Chunk confirme: "La personne responsable de la rencontre est tenue..."
- Mapping OK

### Q45: 17 equipes Loubatiere (ARITHMETIQUE)
- Regle: "1 equipe pour 1-4, 2 pour 5-8, etc."
- Calcul: 17 equipes = 5 qualifiees (1+2+3+4+5 = progression)
- reasoning_class=arithmetic
- Mapping OK

### Q46: Arbitre phase 2 (REFORMULE)
- Question MCQ reformulee vers question directe
- Chunk confirme: "Des la phase 2, l'arbitre ne peut pas etre joueur"
- Mapping OK

### Q47: Indemnites championnat feminin (FAILED)
- **PROBLEME**: Question demande "montant total des indemnites" (euros)
- **PROBLEME**: Choices MCQ sont "2, 3, 4, 5" (nombres)
- **PROBLEME**: original_answer etait "85 €"
- **STATUS**: FAILED - Question/choices corrompus, besoin revue PDF source
- **ACTION**: Marque FAILED, a revoir manuellement

### Q48: Phases championnat feminin (NUMERIC)
- Question: "Combien de phases..."
- Chunk confirme: "2 phases: zone interdepartementale + finale"
- expected_answer inclut "2 phases"
- Mapping OK

### Q49: Titre arbitre regional
- Chunk confirme: "7.1. Criteres administratifs..."
- Mapping OK

### Q50: Suivi DNA (REFORMULE)
- Question negative reformulee
- Chunk confirme: "Le suivi des mesures administratives de la D.N.A."
- Mapping OK

---

## 3. Self-Audit Findings

| Q# | Probleme detecte | Action |
|----|------------------|--------|
| Q42 | Doublement forfait derniere ronde non pris en compte | Mapping corrige vers chunk avec regle doublement |
| Q47 | Question/choices completement incoherents | Marque FAILED |

---

## 4. Lessons Learned (Batch 005)

1. **Forfait double derniere ronde**: Toujours verifier si la question mentionne "derniere ronde" - l'amende est doublee
2. **Questions corrompues**: Certaines questions ont des choices MCQ incoherents avec la question - marquer FAILED plutot que deviner
3. **Verification numerique**: Si question demande "combien", verifier que expected_answer contient le nombre

---

## 5. Files Modified

1. `tests/data/gold_standard_annales_fr_v7.json` - Questions Q41-Q50
2. `docs/audits/GS_MANUAL_AUDIT_CHECKLIST.md` - 50 questions audited
3. `docs/audits/GS_BATCH_005_REPORT.md` - This report

---

## 6. Progression

```
Batches completes: 5/42
Questions auditees: 50/420 (11.9%)
Questions PASS: 49
Questions FAIL: 1 (Q47)
Restant: 370 questions
```
