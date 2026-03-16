# Schema JSON des Chunks - Pocket Arbiter

> **Document ID**: SPEC-SCH-001
> **ISO Reference**: ISO 82045 - Document management
> **Version**: 2.1
> **Date**: 2026-01-22

---

## 1. Vue d'ensemble

Ce document definit le schema JSON pour les chunks de texte utilises dans le pipeline RAG de Pocket Arbiter. Le schema garantit l'interoperabilite entre les phases 1 (extraction), 2 (retrieval) et 3 (synthese).

---

## 2. Schema JSON (Draft-07) - Parent-Child + Table Summary

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://pocket-arbiter/schemas/chunk.json",
  "title": "RAG Chunk Schema v3.0",
  "description": "Schema hierarchique Parent-Child pour RAG (NVIDIA 2025 + EmbeddingGemma)",
  "type": "object",
  "required": ["id", "text", "source", "tokens", "corpus", "chunk_type", "page"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Identifiant unique du chunk (format: source-pXXX-parent/childYYY-ZZ)",
      "examples": ["LA-octobre2025.pdf-p015-parent001", "LA-octobre2025.pdf-p015-child001-00"]
    },
    "text": {
      "type": "string",
      "description": "Contenu textuel du chunk",
      "minLength": 30
    },
    "source": {
      "type": "string",
      "description": "Nom du fichier PDF source",
      "pattern": "^.+\\.pdf$"
    },
    "tokens": {
      "type": "integer",
      "description": "Nombre de tokens (EmbeddingGemma tokenizer)",
      "minimum": 1
    },
    "corpus": {
      "type": "string",
      "description": "Corpus d'appartenance",
      "enum": ["fr", "intl"]
    },
    "chunk_type": {
      "type": "string",
      "description": "Type de chunk (hierarchie NVIDIA 2025)",
      "enum": ["parent", "child", "table_summary"]
    },
    "section": {
      "type": ["string", "null"],
      "description": "Section extraite par MarkdownHeaderTextSplitter (h1/h2)",
      "examples": ["CHAMPIONNAT DE FRANCE", "Article 4.1", "Chapitre 3"]
    },
    "article_num": {
      "type": ["string", "null"],
      "description": "Numero d'article extrait (h3/h4)",
      "examples": ["4.1", "6.2.1", "1.2.3"]
    },
    "parent_id": {
      "type": ["string", "null"],
      "description": "ID du parent (pour child chunks)",
      "examples": ["LA-octobre2025.pdf-p001"]
    },
    "table_type": {
      "type": ["string", "null"],
      "description": "Type de table (pour table_summary)",
      "examples": ["schedule", "scoring", "other"]
    },
    "page": {
      "type": "integer",
      "description": "Numero de page principal (OBLIGATOIRE - ISO 42001 A.6.2.2)",
      "minimum": 1
    },
    "pages": {
      "type": "array",
      "description": "Liste des pages couvertes (multi-page chunks)",
      "items": { "type": "integer", "minimum": 1 },
      "examples": [[15], [15, 16], [42, 43, 44]]
    }
  }
}
```

### 2.1 Types de chunks

| Type | Taille | Usage | Overlap |
|------|--------|-------|---------|
| `parent` | 1024 tokens | Contexte LLM | 15% (154t) |
| `child` | 450 tokens | Embedding/Search | 15% (68t) |
| `table_summary` | ~50 tokens | Tables LLM summaries | - |

---

## 3. Format ID des chunks

### 3.1 Structure (Mode B — Production)

```
{SOURCE_PDF}-p{PAGE:03d}-parent{ID}-child{IDX:02d}
     │           │          │           │
     │           │          │           └── Index enfant dans le parent (00-99)
     │           │          └────────────── ID parent séquentiel (000-999)
     │           └───────────────────────── Page PDF (zero-padded 3 digits)
     └───────────────────────────────────── Nom fichier PDF source
```

### 3.2 Exemples

| ID | Description |
|----|-------------|
| `LA-octobre2025.pdf-p041-parent162-child01` | LA doc, page 41, parent 162, child 1 |
| `R01_2025_26_Regles_generales.pdf-p005-parent027-child01` | Regles generales, page 5 |
| `A02_2025_26_Championnat_de_France_des_Clubs.pdf-p004-parent011-child00` | Champ. clubs, page 4 |

### 3.3 Hiérarchie Parent-Child

```
Parent (1024 tokens, contexte riche pour LLM)
├── Child 00 (450 tokens, pour embedding/search)
├── Child 01 (450 tokens, pour embedding/search)
└── Child 02 (450 tokens, optionnel)
```

Avg ~1.3 children per parent. 1857 chunks FR (Mode B production).

---

## 4. Exemple complet (Mode B)

```json
{
  "id": "LA-octobre2025.pdf-p041-parent162-child01",
  "text": "## Article 4 L'exécution du déplacement\n\n4.1 Chaque coup doit être exécuté d'une seule main...",
  "source": "LA-octobre2025.pdf",
  "page": 41,
  "pages": [41],
  "section": "Article 4 L'exécution du déplacement",
  "article_num": "4.1",
  "parent_id": "LA-octobre2025.pdf-p041-parent162",
  "tokens": 333,
  "corpus": "fr",
  "chunk_type": "child"
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

| Fichier | Description | Chunks | Status |
|---------|-------------|--------|--------|
| `corpus/processed/chunks_mode_b_fr.json` | Chunks corpus FR Mode B (prod) | **1857** | ACTIF |
| `corpus/processed/chunks_mode_b_intl.json` | Chunks corpus INTL Mode B | 866 | ACTIF |
| `corpus/processed/chunks_fr.json` | Legacy chunks FR | 2558 | DEPRECATED |
| `corpus/processed/chunks_intl.json` | Legacy chunks INTL | 1020 | DEPRECATED |

---

## 6. Export Android (Google AI Edge SDK)

### 6.1 Format `<chunk_splitter>`

Pour compatibilite avec le SDK Google AI Edge RAG, les chunks DOIVENT etre exportes
avec `<chunk_splitter>` AU DEBUT de chaque chunk (pas a la fin).

**Format requis par le SDK** (source: [RagPipeline.kt](https://github.com/google-ai-edge/ai-edge-apis/blob/main/examples/rag/android/app/src/main/java/com/google/ai/edge/samples/rag/RagPipeline.kt)):

```
<chunk_splitter> Article 4.1 - Le toucher-jouer. Lorsqu'un joueur ayant le trait touche deliberement sur l'echiquier... [Source: LA-octobre2025.pdf, Page 41]
<chunk_splitter> Article 4.2 - Ajustement des pieces. Si un joueur ayant le trait ajuste une ou plusieurs pieces... [Source: LA-octobre2025.pdf, Page 42]
<chunk_splitter> Article 4.3 - Roque. Si le joueur touche une tour puis son roi... [Source: LA-octobre2025.pdf, Page 42]
```

**Regles** :
- `<chunk_splitter>` au DEBUT de chaque chunk (pas a la fin)
- Le contenu suit IMMEDIATEMENT apres un espace
- Un chunk peut s'etendre sur plusieurs lignes
- La citation source est INTEGREE dans le texte du chunk

### 6.2 Script export

```python
def export_for_android(chunks_json: str, output_txt: str):
    """Exporte les chunks au format SDK Google AI Edge."""
    with open(chunks_json) as f:
        data = json.load(f)

    with open(output_txt, 'w', encoding='utf-8') as f:
        for chunk in data['chunks']:
            # Format SDK: <chunk_splitter> suivi du contenu sur la meme ligne
            content = chunk['text'].replace('\n', ' ')  # Une seule ligne
            citation = f"[Source: {chunk['source']}, Page {chunk['page']}]"
            f.write(f"<chunk_splitter> {content} {citation}\n")
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
| 2.2 | 2026-01-30 | ID format: source-based (not FR-001-015-01), exemples Mode B, fichiers prod vs deprecated, 1857 chunks FR |
| 2.1 | 2026-01-22 | Tokenizer: tiktoken → EmbeddingGemma, page coverage 100% |
| 2.0 | 2026-01-20 | Parent-Child architecture, table_summary type |
| 1.0 | 2026-01-14 | Creation initiale |

---

## 9. References

- [JSON Schema Draft-07](https://json-schema.org/draft-07/json-schema-release-notes.html)
- [ISO 82045 - Document management](https://www.iso.org/standard/55691.html)
- [Google AI Edge RAG SDK](https://github.com/google-ai-edge/ai-edge-apis)
