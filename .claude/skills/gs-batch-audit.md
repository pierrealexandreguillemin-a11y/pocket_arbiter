# Gold Standard Batch Audit Skill

## Description
Audit et correction manuel du Gold Standard par batches de 10 questions avec conformité ISO 42001 (anti-hallucination) et ISO 29119 (quality gates).

## Usage
```
/gs-batch-audit [batch_number]
```

## Lessons Learned (enrichi par batches précédentes)

### Batch 001 (2026-02-03)
1. **Hallucinations détectées**:
   - Ne JAMAIS inventer la signification des acronymes (SC, UVR, UVC) si non présente dans le chunk
   - Les "inférences par omission" doivent être explicites dans expected_answer

2. **Questions négatives** ("Quelle option N'EST PAS..."):
   - Le chunk liste ce qui EST autorisé/valide
   - expected_answer doit citer le chunk + mentionner explicitement "n'est pas mentionné"
   - Flag `requires_inference: true` obligatoire

3. **Questions arithmétiques**:
   - Le chunk contient les données de référence (tables, règles)
   - expected_answer peut contenir la conclusion du calcul
   - Flag `requires_inference: true` et `reasoning_class: arithmetic`

4. **Test anti-hallucination**:
   - Overlap mot-à-mot > 40% n'est pas suffisant
   - Vérifier que CHAQUE affirmation de expected_answer est dans le chunk
   - Les mots clés techniques doivent être présents verbatim

## Quality Gates (11 obligatoires)

| Gate | Critère | Seuil |
|------|---------|-------|
| G1 | Question finit par `?` | 100% |
| G2 | Question >= 10 chars | 100% |
| G3 | expected_answer > 5 chars | 100% |
| G4 | chunk_id existe dans DB | 100% |
| G5 | Réponse IN chunk (sémantique) | 100% |
| G6 | Mapping question↔chunk cohérent | 100% |
| G7 | difficulty in [0,1] | 100% |
| G8 | reasoning_class valide | 100% |
| G9 | requires_context correct | 100% |
| G10 | validation.status = VALIDATED | 100% |
| G11 | Question reformulée (MCQ→directe) | 100% |

## Procédure par Question

### Phase A: Extraction
```python
# Pour chaque question du batch
chunk = db.get_chunk(question['expected_chunk_id'])
display(question, chunk)
```

### Phase B: Validation Mapping
1. Vérifier que expected_answer est DANS le chunk
2. Si question négative: identifier l'inférence par omission
3. Si mapping KO: chercher meilleur chunk

### Phase C: Reformulation MCQ → Direct
- Supprimer "parmi les suivantes", "quelle proposition"
- Transformer en question directe naturelle
- Conserver metadata.choices et metadata.mcq_answer

### Phase D: Anti-Hallucination Check
```
Pour chaque mot clé de expected_answer:
  SI mot NON dans chunk:
    SI mot est acronyme avec signification inventée: HALLUCINATION
    SI mot est conclusion d'inférence: requires_inference=true
    SI mot est reformulation acceptable: OK
```

### Phase E: Corrections
- expected_answer = citation chunk (pas d'invention)
- requires_inference=true si inférence nécessaire
- reasoning_class=arithmetic si calcul

### Phase F: Commit
```
fix(gs): batch N — manual LLM-as-judge audit

- Questions Q[X]-Q[Y] validated
- N mapping corrections
- M MCQ→direct reformulations
- K ISO 42001 corrections (hallucinations)
- All 11 quality gates PASS

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Fichiers

- GS: `tests/data/gold_standard_annales_fr_v7.json`
- DB: `corpus/processed/corpus_mode_b_fr.db`
- Checklist: `docs/audits/GS_MANUAL_AUDIT_CHECKLIST.md`
- Rapports: `docs/audits/GS_BATCH_NNN_REPORT.md`

## Commande de vérification

```bash
python -c "
import json
with open('tests/data/gold_standard_annales_fr_v7.json') as f:
    gs = json.load(f)
batch_start = (BATCH_NUM - 1) * 10
validated = sum(1 for q in gs['questions'][batch_start:batch_start+10]
                if q.get('validation',{}).get('status') == 'VALIDATED')
print(f'Batch {BATCH_NUM}: {validated}/10 VALIDATED')
"
```
