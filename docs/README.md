# docs/ - Documentation ISO

> **ISO 9001**: Controle documentaire | **ISO 82045**: Metadonnees

## Conformite obligatoire

| Regle | Verification |
|-------|--------------|
| En-tete ISO 82045 | Chaque .md doit avoir Document ID, Version, Date, Mots-cles |
| Indexation ISO 999 | Chaque doc doit etre dans INDEX.md |
| Liens valides | Pas de liens casses entre docs |

## Documents obligatoires

| Fichier | ID | Statut requis |
|---------|-----|---------------|
| INDEX.md | DOC-IDX-001 | Approuve |
| DOC_CONTROL.md | DOC-CTRL-001 | Approuve |
| VISION.md | SPEC-VIS-001 | Draft min |
| AI_POLICY.md | DOC-POL-001 | Draft min |
| QUALITY_REQUIREMENTS.md | SPEC-REQ-001 | Draft min |
| TEST_PLAN.md | TEST-PLAN-001 | Draft min |

## Avant commit

Verifier que chaque document a l'en-tete:
```markdown
> **Document ID**: [CAT]-[TYPE]-[NUM]
> **Version**: X.Y
> **Date**: AAAA-MM-JJ
> **Mots-cles**: terme1, terme2
```

**Non-conformite = PR bloquee**
