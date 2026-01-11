# prompts/ - Prompts IA versiones

> **ISO 42001**: Gouvernance IA | **Tolerance zero hallucination**

## Conformite obligatoire

| Regle | Verification |
|-------|--------------|
| Chaque prompt versionne | Fichier nomme `*_v{N}.txt` |
| CHANGELOG.md a jour | Chaque modif documentee |
| Instructions citation | Prompt DOIT exiger source |
| Pas de generation libre | Grounding obligatoire |

## Structure requise

```
prompts/
├── CHANGELOG.md           # Historique (obligatoire)
├── interpretation_v1.txt  # Prompt synthese (Phase 3)
├── README.md              # Ce fichier
└── CLAUDE_CODE_PHASE1.md  # Instructions dev
```

## Regles anti-hallucination

Chaque prompt de synthese DOIT contenir:
- [ ] Instruction de citer la source
- [ ] Instruction de refuser si hors corpus
- [ ] Format de citation defini

Exemple obligatoire dans prompt:
```
TOUJOURS citer la source (document + page).
Si l'information n'est pas dans le contexte, repondre "Non trouve".
NE JAMAIS inventer d'information.
```

## Versionning

1. Ne JAMAIS modifier un prompt existant
2. Creer nouvelle version: `prompt_v2.txt`
3. Documenter dans CHANGELOG.md

**Prompt sans grounding = BLOQUANT**
