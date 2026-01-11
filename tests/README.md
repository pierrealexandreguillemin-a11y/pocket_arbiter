# tests/ - Tests ISO 29119

> **ISO 29119**: Documentation de test | **ISO 25010**: Fiabilite

## Conformite obligatoire

| Regle | Seuil | Verification |
|-------|-------|--------------|
| Tests unitaires | 100% pass | `pytest` |
| Donnees de test | JSON valides | Validation CI |
| Tests hallucination | 0% failure | adversarial.json |
| Recall retrieval | >= 80% | questions_*.json |

## Structure requise

```
tests/
├── data/                    # Donnees de test (obligatoire)
│   ├── questions_fr.json    # Questions test FR
│   ├── questions_intl.json  # Questions test INTL
│   └── adversarial.json     # Tests anti-hallucination
├── reports/                 # Rapports generes par CI
└── README.md                # Ce fichier
```

## Format questions_*.json

```json
[
  {
    "id": "FR-Q01",
    "question": "...",
    "expected_docs": ["doc.pdf"],
    "expected_pages": [1, 2]
  }
]
```

## Avant commit

```bash
pytest tests/ -v
python -c "import json; json.load(open('tests/data/adversarial.json'))"
```

**Non-conformite = PR bloquee**
