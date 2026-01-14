# Schema JSON des Chunks - Pocket Arbiter

> **Document ID**: SPEC-SCH-001
> **ISO Reference**: ISO 82045 - Document management
> **Version**: 1.0
> **Date**: 2026-01-14

---

## 1. Vue d'ensemble

Ce document definit le schema JSON pour les chunks de texte utilises dans le pipeline RAG de Pocket Arbiter. Le schema garantit l'interoperabilite entre les phases 1 (extraction), 2 (retrieval) et 3 (synthese).

---

## 2. Schema JSON (Draft-07)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://pocket-arbiter/schemas/chunk.json",
  "title": "RAG Chunk Schema",
  "description": "Schema pour les chunks de texte du corpus d'arbitrage echecs",
  "type": "object",
  "required": ["id", "text", "source", "page", "tokens", "metadata"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Identifiant unique du chunk",
      "pattern": "^(FR|INTL)-[0-9]{3}-[0-9]{3}-[0-9]{2}$",
      "examples": ["FR-001-015-01", "INTL-001-042-03"]
    },
    "text": {
      "type": "string",
      "description": "Contenu textuel du chunk",
      "minLength": 50,
      "maxLength": 2000
    },
    "source": {
      "type": "string",
      "description": "Nom du fichier PDF source",
      "pattern": "^.+\\.pdf$",
      "examples": ["LA-octobre2025.pdf", "FIDE_Laws_2024.pdf"]
    },
    "page": {
      "type": "integer",
      "description": "Numero de page dans le PDF source",
      "minimum": 1
    },
    "tokens": {
      "type": "integer",
      "description": "Nombre de tokens (tiktoken cl100k_base)",
      "minimum": 1,
      "maximum": 512
    },
    "metadata": {
      "type": "object",
      "description": "Metadonnees supplementaires",
      "required": ["corpus", "extraction_date", "version"],
      "properties": {
        "section": {
          "type": "string",
          "description": "Section ou chapitre du document",
          "examples": ["Regles du jeu", "Article 4", "Chapitre 3"]
        },
        "corpus": {
          "type": "string",
          "description": "Corpus d'appartenance",
          "enum": ["fr", "intl"]
        },
        "extraction_date": {
          "type": "string",
          "description": "Date d'extraction ISO 8601",
          "format": "date",
          "examples": ["2026-01-14"]
        },
        "version": {
          "type": "string",
          "description": "Version du schema",
          "pattern": "^[0-9]+\\.[0-9]+$",
          "examples": ["1.0", "1.1"]
        },
        "article": {
          "type": "string",
          "description": "Reference article si applicable",
          "examples": ["4.1", "6.2.1", "A.3"]
        },
        "prev_chunk_id": {
          "type": "string",
          "description": "ID du chunk precedent (pour overlap)",
          "examples": ["FR-001-015-00"]
        },
        "next_chunk_id": {
          "type": "string",
          "description": "ID du chunk suivant (pour overlap)",
          "examples": ["FR-001-015-02"]
        }
      }
    }
  }
}
```

---

## 3. Format ID des chunks

### 3.1 Structure

```
{CORPUS}-{DOC_NUM}-{PAGE}-{SEQ}
   │        │       │      │
   │        │       │      └── Sequence dans la page (00-99)
   │        │       └───────── Numero de page (001-999)
   │        └───────────────── Numero document (001-999)
   └────────────────────────── Corpus: FR ou INTL
```

### 3.2 Exemples

| ID | Description |
|----|-------------|
| `FR-001-015-01` | Corpus FR, doc 1, page 15, chunk 1 |
| `FR-001-015-02` | Corpus FR, doc 1, page 15, chunk 2 |
| `INTL-001-042-03` | Corpus INTL, doc 1, page 42, chunk 3 |

---

## 4. Exemple complet

```json
{
  "id": "FR-001-015-01",
  "text": "Article 4.1 - Le toucher-jouer\n\nLorsqu'un joueur ayant le trait touche deliberement sur l'echiquier, avec l'intention de jouer ou de prendre:\n- une ou plusieurs de ses propres pieces, il doit jouer la premiere piece touchee qui peut etre jouee, ou\n- une ou plusieurs pieces de son adversaire, il doit prendre la premiere piece touchee qui peut etre prise.",
  "source": "LA-octobre2025.pdf",
  "page": 15,
  "tokens": 78,
  "metadata": {
    "section": "Regles du jeu - Toucher-jouer",
    "corpus": "fr",
    "extraction_date": "2026-01-14",
    "version": "1.0",
    "article": "4.1",
    "prev_chunk_id": null,
    "next_chunk_id": "FR-001-015-02"
  }
}
```

---

## 5. Fichiers de corpus

### 5.1 Structure fichier

Chaque fichier de corpus est un JSON array de chunks:

```json
{
  "metadata": {
    "corpus": "fr",
    "generated": "2026-01-14T10:30:00Z",
    "total_chunks": 512,
    "schema_version": "1.0"
  },
  "chunks": [
    { "id": "FR-001-001-01", ... },
    { "id": "FR-001-001-02", ... },
    ...
  ]
}
```

### 5.2 Fichiers attendus

| Fichier | Description | Chunks estimes |
|---------|-------------|----------------|
| `corpus/processed/chunks_fr.json` | Chunks corpus FR (FFE) | ~500 |
| `corpus/processed/chunks_intl.json` | Chunks corpus INTL (FIDE) | ~100 |

---

## 6. Export Android (Google AI Edge SDK)

### 6.1 Format `<chunk_splitter>`

Pour compatibilite avec le SDK Google AI Edge RAG, les chunks peuvent etre exportes au format texte simple:

```
Article 4.1 - Le toucher-jouer...
[Source: LA-octobre2025.pdf, Page 15]
<chunk_splitter>
Article 4.2 - Ajustement des pieces...
[Source: LA-octobre2025.pdf, Page 16]
<chunk_splitter>
...
```

### 6.2 Script export

```python
def export_for_android(chunks_json: str, output_txt: str):
    """Exporte les chunks au format SDK Google AI Edge."""
    with open(chunks_json) as f:
        data = json.load(f)

    with open(output_txt, 'w', encoding='utf-8') as f:
        for chunk in data['chunks']:
            f.write(chunk['text'])
            f.write(f"\n[Source: {chunk['source']}, Page {chunk['page']}]\n")
            f.write("<chunk_splitter>\n")
```

---

## 7. Validation

### 7.1 Script de validation

```python
import jsonschema
import json

def validate_chunks(chunks_file: str, schema_file: str) -> bool:
    """Valide un fichier de chunks contre le schema."""
    with open(schema_file) as f:
        schema = json.load(f)

    with open(chunks_file) as f:
        data = json.load(f)

    for chunk in data['chunks']:
        jsonschema.validate(chunk, schema)

    return True
```

### 7.2 Regles de validation

| Regle | Validation |
|-------|------------|
| ID unique | Pas de doublons dans le fichier |
| Text non vide | text.length >= 50 |
| Tokens coherent | tokens <= 512 |
| Source existe | Pattern .pdf$ |
| Page positive | page >= 1 |

---

## 8. Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-14 | Creation initiale |

---

## 9. References

- [JSON Schema Draft-07](https://json-schema.org/draft-07/json-schema-release-notes.html)
- [ISO 82045 - Document management](https://www.iso.org/standard/55691.html)
- [Google AI Edge RAG SDK](https://github.com/google-ai-edge/ai-edge-apis)
