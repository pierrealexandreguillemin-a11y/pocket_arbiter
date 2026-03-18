# Chunker Rewrite — Design Spec

> **Date**: 2026-03-18
> **Statut**: Approuve
> **Objectif**: Remplacer le chunker maison (350 lignes, bugs integrite) par le pattern industrie LangChain deux etapes, avec quality gates relationnelles.

---

## Contexte

Le chunker maison (`chunker.py`) a 3 bugs d'integrite relationnelle decouverts lors du recall baseline :

1. **7 parents avec texte mais 0 children** (8020 tokens invisibles au search, 7 questions GS impactees)
2. **28 parents root vides** (`p000`) avec 46 children orphelins (jetes par `build_context`)
3. **Parents geants** (jusqu'a 39K tokens, 55 children) — heading mal detecte

Ces bugs sont structurels : le chunker maison ne garantit pas la couverture totale du texte source. `LangChain.split_documents()` garantit cette propriete par construction.

### Deps installees et ignorees

- `langchain-text-splitters>=0.3.0` dans `requirements.txt`
- `semchunk` installe
- `ARCHITECTURE.md` prevoit `RecursiveCharacterTextSplitter`

---

## Architecture

**Ordre des stages critique** : les tables doivent etre extraites AVANT le header split,
sinon MarkdownHeaderTextSplitter fragmente les tableaux en morceaux non-detectables.

Verification empirique : 98 blocs tableaux dans LA-octobre2025, seulement 22 detectes
apres header split (ratio `|` dilue par le texte environnant).

```
chunk_document(markdown, source, heading_pages)
  |
  +-- Stage 1: Table extraction (AVANT header split)
  |     Detecter blocs de lignes consecutives avec | -> tables list
  |     Remplacer chaque table par <!-- TABLE_N --> placeholder
  |     -> markdown_clean (sans tables) + tables list
  |
  +-- Stage 2: MarkdownHeaderTextSplitter
  |     headers_to_split_on = [(#,h1), (##,h2), (###,h3), (####,h4)]
  |     strip_headers = False
  |     -> list[Document] avec metadata {h1, h2, h3, h4}
  |
  +-- Stage 3: RecursiveCharacterTextSplitter.from_tiktoken_encoder
  |     encoding_name = "cl100k_base"
  |     chunk_size = 512
  |     chunk_overlap = 100
  |     separators = ["\n\n", "\n", ".\n", " ", ""]
  |     -> list[Document] children, metadata heritees
  |     Note: ".\n" au lieu de ". " (evite de couper "Art. 3", "L. 131")
  |
  +-- Stage 4: Parent assembly
  |     Grouper children par metadata (h1, h2) -> parent text
  |     Cap parent a 2048 tokens (split en sous-parents si depasse)
  |
  +-- Stage 5: Page interpolation
  |     heading_pages mapping -> page par child
  |
  +-- Stage 6: CCH title
  |     metadata {h1, h2, h3} -> " > ".join() -> section field
  |
  +-- Stage 7: Table linkage
  |     Pour chaque table extraite, trouver la section englobante
  |     via position du placeholder -> metadata heading -> parent_id
  |
  +-- return {children, parents, tables}
```

---

## Parametres (standards industrie 2026)

| Parametre | Valeur | Source |
|-----------|--------|--------|
| `chunk_size` | **512 tokens** | FloTorch 2026 benchmark: 69% accuracy (best) |
| `chunk_overlap` | **100 tokens** (20%) | Microsoft Azure + PremAI 2026 standard |
| `parent_max_tokens` | **2048** | EmbeddingGemma max seq + budget LLM mobile |
| `parent_ratio` | ~3-5x child | Industry (LangChain, GraphRAG) |
| `separators` | `["\n\n", "\n", ".\n", " ", ""]` | ".\n" au lieu de ". " (evite "Art. 3", "L. 131") |
| `encoding` | `cl100k_base` | Tiktoken, proxy Gemma tokenizer |
| `headers` | h1-h4 | Couverture hierarchie FFE (Partie > Chapitre > Section > Article) |

---

## Stages detail

### Stage 1: Table extraction (AVANT header split)

Detecter et extraire les tables du markdown brut AVANT le header split.
Sinon MarkdownHeaderTextSplitter fragmente les tableaux (verifie : 98 tables
dans LA, seulement 22 detectees apres header split).

```python
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")

def extract_tables(markdown: str) -> tuple[str, list[dict]]:
    """Extract table blocks, replace with placeholders."""
    lines = markdown.split("\n")
    tables = []
    clean_lines = []
    i = 0
    while i < len(lines):
        if TABLE_LINE_RE.match(lines[i]):
            # Collect consecutive table lines
            table_lines = []
            while i < len(lines) and TABLE_LINE_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 3:  # header + separator + data
                tables.append({"raw_text": "\n".join(table_lines)})
                clean_lines.append(f"<!-- TABLE_{len(tables) - 1} -->")
            else:
                clean_lines.extend(table_lines)  # too short, keep as text
        else:
            clean_lines.append(lines[i])
            i += 1
    return "\n".join(clean_lines), tables
```

### Stage 2: MarkdownHeaderTextSplitter

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,
)
header_splits = md_splitter.split_text(markdown_clean)
```

Chaque `Document` a :
- `page_content` : texte de la section (heading inclus, tables remplacees par placeholders)
- `metadata` : `{"h1": "Regles generales", "h2": "Forfaits", "h3": "Definitions"}`

### Stage 3: RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".\n", " ", ""],
)
children_docs = text_splitter.split_documents(header_splits)
```

**Separateur ".\n"** au lieu de ". " : evite de couper les abreviations francaises
("Art. 3", "L. 131-8", "M. Dupont"). Verifie empiriquement : 0 sections denses
sans `\n` dans le corpus (920 sections testees), donc `\n` est toujours disponible.

**Garantie couverture totale** : `split_documents` ne perd aucun texte.

### Stage 4: Parent assembly

Grouper les children par `(h1, h2)` metadata (niveau section). Le texte parent = concatenation de tous les children du groupe.

Si le parent depasse `PARENT_MAX_TOKENS` (2048), le decouper en sous-parents sequentiels.

```python
PARENT_MAX_TOKENS = 2048

def build_parents(children, source):
    groups = defaultdict(list)
    for child in children:
        key = (child.metadata.get("h1", ""), child.metadata.get("h2", ""))
        groups[key].append(child)

    parents = []
    for (h1, h2), group_children in groups.items():
        parent_text = "\n\n".join(c.page_content for c in group_children)
        tokens = count_tokens(parent_text)
        if tokens <= PARENT_MAX_TOKENS:
            parents.append(make_parent(parent_text, h1, h2, source))
        else:
            # Split into sub-parents
            for sub_text in split_parent(parent_text, PARENT_MAX_TOKENS):
                parents.append(make_parent(sub_text, h1, h2, source))
    return parents
```

### Stage 5: Page interpolation

Meme logique que l'actuel : `heading_pages` mapping fournit la page de chaque heading. Les children heritent la page du heading le plus proche dans leur metadata.

Pour les children splites d'une section multi-page, interpolation lineaire sur le page span.

### Stage 6: CCH title

```python
def build_cch_title(metadata: dict) -> str:
    parts = [metadata.get(f"h{i}", "") for i in range(1, 5)]
    return " > ".join(p for p in parts if p)
```

Le `section` field de chaque child = CCH title hierarchique. Utilise par l'indexer pour `format_document(cch_title, text)`.

---

### Stage 7: Table linkage

Pour chaque table extraite en Stage 1, retrouver sa section englobante :
- Le placeholder `<!-- TABLE_N -->` se retrouve dans un Document apres Stage 2
- Ce Document a des metadata {h1, h2, ...} -> la table herite ces metadata
- Le `parent_id` = le parent de la section ou le placeholder est apparu
- La page = `heading_pages` du heading le plus proche

```python
def link_tables(tables, header_splits, parents):
    for i, table in enumerate(tables):
        placeholder = f"<!-- TABLE_{i} -->"
        for doc in header_splits:
            if placeholder in doc.page_content:
                table["section"] = build_cch_title(doc.metadata)
                table["parent_id"] = find_parent_id(doc.metadata, parents)
                break
```

## Tables : integration multi-vector

Les 111 table summaries pre-generees (`table_summaries_claude.json`) restent le mecanisme de retrieval.

### Validation croisee tables (nouveau)

Le chunker detecte les tables dans le markdown (Stage 1). L'indexer doit verifier que
chaque table detectee a une summary correspondante dans `table_summaries_claude.json`.

| Verification | Source | Action si mismatch |
|-------------|--------|-------------------|
| Table detectee sans summary | Chunker vs JSON | WARNING (table non-searchable, raw text seulement) |
| Summary sans table detectee | JSON vs chunker | WARNING (summary orpheline, peut-etre table trop petite) |
| Count match | 98 brut vs 111 DB | Tolerer ecart (merges, tables courtes) |

Le matching se fait par `(source, position_dans_document)` ou par recherche du `raw_table_text` dans le markdown source.

---

## Format de sortie (inchange + tables)

```python
{
    "children": [
        {
            "id": "source.pdf-c0000",
            "text": "...",
            "parent_id": "source.pdf-p000",
            "source": "source.pdf",
            "page": 5,
            "article_num": "3.1",
            "section": "Regles generales > Forfaits > Definitions",
            "tokens": 480,
        }
    ],
    "parents": [
        {
            "id": "source.pdf-p000",
            "text": "...",
            "source": "source.pdf",
            "section": "Regles generales > Forfaits",
            "tokens": 1800,
            "page": 5,
        }
    ],
    "tables": [
        {
            "raw_text": "| col1 | col2 |\n|---|---|\n...",
            "source": "source.pdf",
            "page": 3,
            "section": "Regles generales > Cadences",
            "parent_id": "source.pdf-p002",
        }
    ]
}
```

---

## Quality gates (indexer, post-build)

| Gate | Verification | Action si echec |
|------|-------------|-----------------|
| I1 | Aucun parent avec texte et 0 children | FAIL build |
| I2 | Aucun child avec parent_id invalide | FAIL build |
| I3 | Aucun child avec page NULL | FAIL build |
| I4 | Aucun child avec embedding NULL | FAIL build |
| I5 | FTS5 counts = source table counts | FAIL build |
| I6 | Aucun parent > 2048 tokens | FAIL build |
| I7 | Couverture totale : sum(child tokens) >= 90% sum(section tokens) | FAIL build |
| I8 | Aucun child contient un placeholder `<!-- TABLE_N -->` non-resolu | FAIL build |
| I9 | Chaque table_summary a un source+page valide dans la DB | FAIL build |

Gate I7 est nouveau : il verifie que le chunker n'a pas perdu de contenu. La couverture n'est pas 100% a cause de l'overlap (tokens comptes en double) et des tables extraites, mais doit etre >= 90%.

---

## Fichiers impactes

| Fichier | Action | Lignes estimees |
|---------|--------|-----------------|
| `scripts/pipeline/chunker.py` | REWRITE | ~120 |
| `scripts/pipeline/tests/test_chunker.py` | REWRITE | ~200 |
| `scripts/pipeline/indexer.py` | MODIFY (quality gates I1-I7, table parent_id) | +30 |
| `scripts/pipeline/tests/test_indexer.py` | MODIFY (tests gates) | +50 |

---

## Ce qui ne change PAS

- Interface `chunk_document(markdown, source, heading_pages)` — meme signature
- Format de sortie children/parents (meme schema, + tables)
- Table summaries pre-generees (fichier `table_summaries_claude.json`)
- Indexer : embedding, SQLite, FTS5
- Search : cosine, BM25, RRF, adaptive_k
- Page interpolation logique

---

## Standards appliques

| Standard | Application |
|----------|-------------|
| FloTorch 2026 | chunk_size=512, recursive splitting |
| Microsoft Azure 2026 | 20% overlap (100 tokens) |
| NAACL 2025 | Recursive > semantic pour prod |
| LangChain Multi-Vector | Tables : embed summary, return raw |
| LangChain Parent-Child | split_documents couverture totale |
| NVIDIA 2024 | Header-based first, recursive within |
| ISO 29119 | TDD, quality gates |
| ISO 25010 | Fichiers <= 300 lignes |
| ISO 42001 | Tracabilite metadata heading hierarchy |

---

## Sources

- [FloTorch 2026 Benchmark](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/)
- [Microsoft Azure RAG Chunking](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)
- [LangChain MarkdownHeaderTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter)
- [LangChain Multi-Vector Retriever](https://blog.langchain.com/semi-structured-multi-modal-rag/)
- [NVIDIA Chunking Benchmark](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/)
- [NAACL 2025 Chunking Study](https://ragaboutit.com/the-2026-rag-performance-paradox-why-simpler-chunking-strategies-are-outperforming-complex-ai-driven-methods/)
- [Parent-Child Chunking LangChain](https://medium.com/@seahorse.technologies.sl/parent-child-chunking-in-langchain-for-advanced-rag-e7c37171995a)
- [GraphRAG Parent-Child Retriever](https://graphrag.com/reference/graphrag/parent-child-retriever/)
