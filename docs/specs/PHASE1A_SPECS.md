# Specifications Phase 1A - Pipeline Extract + Chunk

> **Document ID**: SPEC-PH1A-001
> **ISO Reference**: ISO/IEC 12207:2017 - Processus du cycle de vie logiciel
> **Version**: 2.0
> **Date**: 2026-01-19
> **Statut**: Approuve
> **Phase**: 1A
> **Effort estime**: 20h

---

## 1. Objectif

Extraire le contenu textuel des PDF du corpus (FFE + FIDE) et le segmenter en chunks semantiques de 256 tokens maximum, avec preservation complete des metadonnees source.

---

## 2. Normes ISO Applicables

| Norme | Clause | Application |
|-------|--------|-------------|
| **ISO 12207** | 7.3 Development | Scripts Python, tracabilite code |
| **ISO 25010** | 4.2 Performance efficiency | Temps extraction, taille output |
| **ISO 29119** | 8.3 Test execution | Tests unitaires obligatoires |
| **ISO 82045** | Metadonnees | Schema JSON documente |
| **ISO 9001** | 7.5 Documented info | Specs dans docs/ |

---

## 3. Entrees (ISO 12207 §7.3.1)

### 3.1 Corpus PDF

| Corpus | Dossier | Fichiers | Pages estimees |
|--------|---------|----------|----------------|
| FR (FFE) | `corpus/fr/` | 29 PDF | ~300 pages |
| INTL (FIDE) | `corpus/intl/` | 1 PDF | ~70 pages |

**Inventaire**: Voir `corpus/INVENTORY.md`

### 3.2 Documents de reference

- `docs/QUALITY_REQUIREMENTS.md` - Contraintes performance
- `docs/ISO_STANDARDS_REFERENCE.md` - Normes applicables
- `.iso/config.json` - Configuration gates Phase 1

### 3.3 Contraintes techniques

| Contrainte | Valeur | Source |
|------------|--------|--------|
| Taille chunk | 256 tokens max | QUALITY_REQUIREMENTS.md |
| Overlap | 50 tokens (20%) | Best practice RAG |
| Encodage | UTF-8 | Standard |
| Tokenizer | tiktoken (cl100k_base) | Compatibilite OpenAI/LLM |

---

## 4. Sorties attendues (ISO 12207 §7.3.3)

### 4.1 Fichiers produits

| Fichier | Format | Description |
|---------|--------|-------------|
| `corpus/processed/chunks_fr.json` | JSON | Chunks corpus FR |
| `corpus/processed/chunks_intl.json` | JSON | Chunks corpus INTL |
| `corpus/processed/extraction_report.json` | JSON | Rapport extraction |

### 4.2 Schema JSON des chunks

Voir `docs/CHUNK_SCHEMA.md` pour le schema complet.

```json
{
  "id": "FR-001-015",
  "text": "Article 4.1 - Le toucher-jouer...",
  "source": "LA-octobre2025.pdf",
  "page": 15,
  "tokens": 256,
  "metadata": {
    "section": "Regles du jeu",
    "corpus": "fr",
    "extraction_date": "2026-01-XX",
    "version": "1.0"
  }
}
```

### 4.3 Metriques attendues

| Metrique | Cible | Tolerance |
|----------|-------|-----------|
| Chunks FR | ~500 | ±100 |
| Chunks INTL | ~100 | ±30 |
| Tokens/chunk moyen | 200-256 | - |
| Erreurs extraction | 0 | 0 |

---

## 5. Architecture technique

### 5.1 Structure modules

```
scripts/
├── pipeline/
│   ├── __init__.py
│   ├── extract_docling.py      # Extraction PDF (Docling ML)
│   ├── parent_child_chunker.py # Chunking hierarchique
│   ├── table_multivector.py    # Tables + LLM summaries
│   ├── token_utils.py          # Tokenization cl100k_base
│   ├── embeddings.py           # Generation embeddings
│   ├── utils.py                # Utilitaires communs
│   └── tests/
│       ├── __init__.py
│       ├── test_parent_child_chunker.py
│       ├── test_table_multivector.py
│       └── conftest.py         # Fixtures pytest
├── requirements.txt            # Dependencies Python
└── README.md                   # Instructions execution
```

### 5.2 Diagramme de flux

```
┌─────────────────────────────────────────────────────────────┐
│              PIPELINE UNIQUE ISO CONFORME                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  corpus/fr/*.pdf, corpus/intl/*.pdf                         │
│        │                                                    │
│        ▼                                                    │
│  ┌──────────────────┐                                       │
│  │ extract_docling  │  Docling (ML-based)                   │
│  │     .py          │  - Extraction texte                   │
│  │                  │  - Extraction tables                  │
│  │                  │  - Detection sections                 │
│  └────────┬─────────┘                                       │
│           │                                                 │
│     ┌─────┴─────┐                                           │
│     │           │                                           │
│     ▼           ▼                                           │
│  [texte]     [tables]                                       │
│     │           │                                           │
│     ▼           ▼                                           │
│  ┌────────────────────┐  ┌────────────────────┐             │
│  │parent_child_chunker│  │table_multivector.py│             │
│  │        .py         │  │  - LLM summaries   │             │
│  │ Parents: 1024 tok  │  │  - Multi-vector    │             │
│  │ Children: 450 tok  │  │                    │             │
│  │ Overlap: 15%       │  │                    │             │
│  └─────────┬──────────┘  └─────────┬──────────┘             │
│            │                       │                        │
│            ▼                       ▼                        │
│     chunks_parent_child.json  tables_multivector.json       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Dependencies Python

```
# requirements.txt
pymupdf>=1.23.0         # PDF extraction
tiktoken>=0.5.0         # Tokenization
tqdm>=4.66.0            # Progress bars
pytest>=7.4.0           # Testing
pytest-cov>=4.1.0       # Coverage
```

---

## 6. Specifications fonctionnelles

### 6.1 extract_pdf.py

**Responsabilite**: Extraction texte brut depuis PDF avec metadonnees

**Interface**:
```python
def extract_pdf(pdf_path: Path) -> dict:
    """
    Extrait le texte d'un PDF avec metadonnees.

    Args:
        pdf_path: Chemin vers le fichier PDF

    Returns:
        dict avec:
        - "filename": nom du fichier
        - "pages": list de dict {"page_num", "text", "section"}
        - "total_pages": nombre de pages
        - "extraction_date": timestamp ISO

    Raises:
        FileNotFoundError: si PDF n'existe pas
        PyMuPDFError: si PDF corrompu
    """
```

**Regles metier**:
- Ignorer les pages vides (< 50 caracteres)
- Detecter les titres de section (pattern: "Article X" ou "Chapitre X")
- Conserver la numerotation des pages originale
- Encoder en UTF-8, normaliser les caracteres speciaux

### 6.2 chunker.py

**Responsabilite**: Segmenter le texte en chunks de taille optimale

**Interface**:
```python
def chunk_text(
    text: str,
    max_tokens: int = 256,
    overlap_tokens: int = 50,
    metadata: dict = None
) -> list[dict]:
    """
    Segmente le texte en chunks avec overlap.

    Args:
        text: Texte brut a segmenter
        max_tokens: Taille max par chunk (default 256)
        overlap_tokens: Chevauchement entre chunks (default 50)
        metadata: Metadonnees a propager (source, page, etc.)

    Returns:
        Liste de chunks conformes au schema CHUNK_SCHEMA.md

    Raises:
        ValueError: si max_tokens <= overlap_tokens
    """
```

**Regles metier**:
- Respecter les limites de phrases (ne pas couper au milieu)
- Propager les metadonnees source vers chaque chunk
- Generer un ID unique par chunk (format: `{corpus}-{doc}-{page}-{seq}`)
- Calculer le nombre exact de tokens avec tiktoken

---

## 7. Specifications non-fonctionnelles (ISO 25010)

### 7.1 Performance (§4.2)

| Metrique | Cible | Methode mesure |
|----------|-------|----------------|
| Temps extraction total | < 60s | time script |
| Memoire pic | < 500 MB | memory_profiler |

### 7.2 Fiabilite (§4.5)

| Metrique | Cible | Methode mesure |
|----------|-------|----------------|
| Taux erreur extraction | 0% | Comptage exceptions |
| Coherence output | 100% | Validation JSON schema |

### 7.3 Maintenabilite (§4.6)

| Metrique | Cible | Methode mesure |
|----------|-------|----------------|
| Complexite cyclomatique | < 10 | radon |
| Documentation | 100% fonctions | interrogation docstrings |

---

## 8. Plan de test (ISO 29119)

### 8.1 Tests unitaires

| Test | Fichier | Couverture |
|------|---------|------------|
| Extraction PDF valide | test_extract.py | extract_pdf() |
| Extraction PDF corrompu | test_extract.py | gestion erreur |
| Chunking simple | test_chunker.py | chunk_text() |
| Chunking overlap | test_chunker.py | verification overlap |
| Chunking metadonnees | test_chunker.py | propagation metadata |

### 8.2 Tests integration

| Test | Description | Validation |
|------|-------------|------------|
| Pipeline complet FR | Extraire + chunker tous PDF FR | ~500 chunks generes |
| Pipeline complet INTL | Extraire + chunker PDF INTL | ~100 chunks generes |
| Validation schema | Tous chunks conformes schema | 0 erreur validation |

### 8.3 Criteres de couverture

| Metrique | Cible | Bloquant |
|----------|-------|----------|
| Line coverage | ≥ 80% | OUI |
| Branch coverage | ≥ 70% | NON |

---

## 9. Definition of Done (ISO 12207 §7.6)

### 9.1 Criteres obligatoires

| Critere | Norme | Validation |
|---------|-------|------------|
| 100% PDF extraits sans erreur | ISO 25010 Reliability | Script execute sur corpus complet |
| ~500 chunks FR generes | ISO 12207 | `wc -l chunks_fr.json` |
| ~100 chunks INTL generes | ISO 12207 | `wc -l chunks_intl.json` |
| Tests passent (≥80% coverage) | ISO 29119 | `pytest --cov-fail-under=80` |
| Schema JSON documente | ISO 82045 | `docs/CHUNK_SCHEMA.md` existe |
| Tracabilite corpus | ISO 12207 | `corpus/INVENTORY.md` a jour |

### 9.2 Gate Phase 1A

```bash
python scripts/iso/validate_project.py --phase 1 --gates
# Gate: corpus_processed = True
```

### 9.3 Checklist avant cloture

- [ ] `scripts/pipeline/extract_pdf.py` implemente et teste
- [ ] `scripts/pipeline/chunker.py` implemente et teste
- [ ] `corpus/processed/chunks_fr.json` genere
- [ ] `corpus/processed/chunks_intl.json` genere
- [ ] Coverage ≥ 80%
- [ ] Documentation a jour (ce document)
- [ ] Commit avec message `feat(pipeline): implement PDF extraction and chunking`

---

## 10. Risques et mitigations

| Risque | Probabilite | Impact | Mitigation |
|--------|-------------|--------|------------|
| PDF mal formates | MOYEN | MOYEN | Test sur echantillon avant full run |
| Encoding issues | FAIBLE | FAIBLE | Force UTF-8, normalisation |
| Section detection fail | MOYEN | FAIBLE | Fallback: pas de section |

---

## 11. Historique

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-14 | Claude Code | Creation initiale |

---

## 12. Approbations

| Role | Date | Signature |
|------|------|-----------|
| Dev Lead | | |
| QA | | |
