# corpus/ - Sources documentaires

> **ISO 12207**: Gestion de configuration | **DVC**: Tracabilite

## Conformite obligatoire

| Regle | Verification |
|-------|--------------|
| INVENTORY.md a jour | Chaque PDF documente |
| DVC tracking | `*.dvc` pour chaque dossier |
| Sources officielles | PDFs de FFE/FIDE uniquement |
| Pas de modif manuelle | PDFs originaux preserves |

## Structure requise

```
corpus/
├── fr/           # Reglements FFE (DVC tracked)
├── intl/         # Reglements FIDE (DVC tracked)
├── fr.dvc        # Fichier tracking DVC
├── intl.dvc      # Fichier tracking DVC
├── INVENTORY.md  # Inventaire obligatoire
└── README.md     # Ce fichier
```

## Format INVENTORY.md

Chaque PDF doit avoir:
```markdown
| Fichier | Source | Version | Pages | Hash MD5 |
```

## Ajout de nouveau PDF

```bash
# 1. Ajouter le PDF dans corpus/fr/ ou corpus/intl/
# 2. Mettre a jour INVENTORY.md
# 3. Tracker avec DVC
dvc add corpus/fr corpus/intl
git add corpus/*.dvc corpus/INVENTORY.md
```

**PDF non documente = BLOQUANT**
