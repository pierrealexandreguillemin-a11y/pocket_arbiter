# scripts/ - Pipeline Python

> **ISO 12207**: Processus technique | **ISO 25010**: Maintenabilite

## Conformite obligatoire

| Regle | Seuil | Verification |
|-------|-------|--------------|
| Couverture tests | >= 60% | `pytest --cov` |
| Lint critique | 0 erreur | `flake8 --select=E9,F63,F7,F82` |
| Fichier > 300 lignes | Refactor requis | Split en modules |
| Fonction > 50 lignes | Refactor requis | Extraire helpers |

## Structure requise

```
scripts/
├── iso/              # Validation ISO (ce module)
├── pipeline/         # Extraction, chunking, embedding
├── requirements.txt  # Dependances (obligatoire)
└── README.md         # Ce fichier
```

## Avant commit

```bash
flake8 scripts/ --select=E9,F63,F7,F82
pytest scripts/ --cov --cov-fail-under=60
```

**Non-conformite = PR bloquee**
