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

```
chunk_document(markdown, source, heading_pages)
  |
  +-- Stage 1: MarkdownHeaderTextSplitter
  |     headers_to_split_on = [(#,h1), (##,h2), (###,h3), (####,h4)]
  |     strip_headers = False
  |     -> list[Document] avec metadata {h1, h2, h3, h4}
  |
  +-- Stage 2: Table extraction
  |     Sections with >50% pipe lines -> tables list (not split)
  |     Remaining sections -> text pipeline
  |
  +-- Stage 3: RecursiveCharacterTextSplitter.from_tiktoken_encoder
  |     encoding_name = "cl100k_base"
  |     chunk_size = 512
  |     chunk_overlap = 100
  |     separators = ["\n\n", "\n", ". ", " ", ""]
  |     -> list[Document] children, metadata heritees
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
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | LangChain default, optimal pour prose FR |
| `encoding` | `cl100k_base` | Tiktoken, proxy Gemma tokenizer |
| `headers` | h1-h4 | Couverture hierarchie FFE (Partie > Chapitre > Section > Article) |

---

## Stages detail

### Stage 1: MarkdownHeaderTextSplitter

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
header_splits = md_splitter.split_text(markdown)
```

Chaque `Document` a :
- `page_content` : texte de la section (heading inclus)
- `metadata` : `{"h1": "Regles generales", "h2": "Forfaits", "h3": "Definitions"}`

### Stage 2: Table extraction

Avant le split par tokens, extraire les sections qui sont des tableaux :

```python
def _is_table(doc: Document) -> bool:
    lines = doc.page_content.strip().split("\n")
    pipe_lines = sum(1 for l in lines if "|" in l)
    return len(lines) > 2 and pipe_lines > len(lines) * 0.5
```

Les tables ne passent PAS par le RecursiveCharacterTextSplitter. Elles sont retournees telles quelles dans la liste `tables`, avec leur metadata heading pour le `parent_id`.

### Stage 3: RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)
children_docs = text_splitter.split_documents(text_sections)
```

**Garantie couverture totale** : `split_documents` ne perd aucun texte. Chaque token du source est dans au moins un child.

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

## Tables : integration multi-vector

Les 111 table summaries pre-generees (`table_summaries_claude.json`) restent le mecanisme de retrieval pour les tableaux.

Changement : chaque table summary recoit un `parent_id` lie a la section englobante (via metadata heading du stage 2). L'indexer stocke ce lien.

Le chunker retourne les sections-tables detectees dans le champ `tables` du format de sortie. L'indexer les matche avec les summaries pre-generees par `(source, page)` ou texte.

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
