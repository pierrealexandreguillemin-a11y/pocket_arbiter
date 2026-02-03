# GS Batch Corrector Agent

## Agent Type
`gs-batch-corrector`

## Description
Agent spécialisé pour la correction du Gold Standard par batches de 10 questions.
Incorpore les apprentissages des batches précédentes pour éviter les erreurs récurrentes.

## Invocation
```
Task tool avec subagent_type="gs-batch-corrector"
prompt="Corriger batch N (questions X-Y)"
```

## Outils disponibles
- Read, Edit, Write (fichiers)
- Bash (DB SQLite, Git)
- Grep, Glob (recherche)

## Contexte automatique
L'agent charge automatiquement:
1. Le skill `gs-batch-audit.md` avec les lessons learned
2. La checklist `GS_MANUAL_AUDIT_CHECKLIST.md`
3. Les rapports des batches précédentes

## Workflow

### 1. Initialisation
```python
# Charger GS et DB
gs = load_json('tests/data/gold_standard_annales_fr_v7.json')
db = connect('corpus/processed/corpus_mode_b_fr.db')
batch_questions = gs['questions'][start:end]
```

### 2. Pour chaque question
```
a) Extraire question + chunk
b) Vérifier mapping (G4, G5, G6)
c) Vérifier reformulation (G11)
d) Vérifier anti-hallucination (ISO 42001)
e) Appliquer corrections si nécessaire
f) Mettre à jour validation.status
```

### 3. Post-traitement
```
a) Vérifier tous les quality gates (G1-G11)
b) Générer rapport batch
c) Mettre à jour checklist
d) Commit avec message normalisé
```

## Erreurs à éviter (Lessons Learned)

### Batch 001
| Erreur | Description | Solution |
|--------|-------------|----------|
| Acronymes inventés | SC, UVR, UVC avec significations non présentes dans chunk | Ne JAMAIS expliciter un acronyme si non dans chunk |
| Inférence implicite | "Coupe de France n'est pas autorisée" sans mention dans chunk | Citer le chunk + "n'est pas mentionné" explicite |
| Calcul dans answer | "13 ans donc U14" avec le raisonnement | requires_inference=true, garder les faits du chunk |
| Overlap insuffisant | Test mot-à-mot à 30% rate les hallucinations | Vérification sémantique manuelle obligatoire |

### Batch 002
| Erreur | Description | Solution |
|--------|-------------|----------|
| Mapping hors-sujet | Q14: chunk parlait de nationalité, pas de licences A/B | Toujours vérifier que le chunk répond à la question |
| Inference "minimum" | Q18: "Jeune est le minimum" non explicite | requires_inference=true pour conclusions |
| Calculs temporels | Jours calendaires, heures de forfait | reasoning_class=arithmetic + requires_inference |
| Tournois bi-phase | Cadences différentes = types différents | Vérifier règles type A vs type B |

## Métriques de qualité

Après chaque batch, vérifier:
- 10/10 questions VALIDATED
- 0 hallucinations (ISO 42001)
- 11/11 quality gates PASS par question
- Commit passant tous les hooks

## Évolution

Ce fichier est enrichi après chaque batch avec les nouvelles lessons learned.
Format: `### Batch NNN` suivi des erreurs détectées et solutions.
