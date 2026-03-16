# Chantier 2 : Fix Pipeline — Design Spec

> **Date**: 2026-03-16
> **Statut**: Approuve
> **Objectif**: Reconstruire le pipeline retrieval pour integrer parents, table summaries, et corriger la taille des chunks

---

## Contexte

Le pipeline actuel produit 1857 chunks de 109 tokens median (sur-chunke pour 390 pages). Les parents (1394) et table summaries (111) existent sur disque mais ne sont pas dans l'index de recherche. Le recall baseline est a 36.6% chunk@5.

Les extractions docling (`corpus/processed/docling_fr/*.json`) sont verifiees fideles aux PDFs source (audit mars 2026 sur R01 et LA-octobre2025).

### Hypotheses documentees
- Tous les headings docling sont de niveau `##` (1582 headings, zero `#` ou `###`). Si une future extraction produit d'autres niveaux, le chunker devra etre adapte.
- Les tokens sont comptes avec tiktoken cl100k_base (proxy pour le tokenizer Gemma, ecarts negligeables en francais). Le budget 400-512 est en tokens tiktoken.

---

## Architecture

```
corpus/processed/docling_fr/*.json  (29 fichiers markdown structure)
        |
   chunker.py (structure-aware, 400-512 tokens)
        |
   ~600-800 children + ~200-400 parents
        |
   table_summaries_claude.json (111 resumes existants)
        |
   indexer.py (EmbeddingGemma-300M, 768D)
        |
   SQLite DB:
     - children: id, text, embedding, parent_id, metadata
     - parents: id, text (pas d'embedding)
     - table_summaries: id, text, embedding, source, metadata
        |
   search.py
     query -> embed -> cosine brute-force sur children + table_summaries
     -> top-k children -> retourner parents dedupliques
```

---

## 1. Chunker (`scripts/pipeline/chunker.py`)

### Source
Les 29 fichiers `corpus/processed/docling_fr/*.json`, champ `markdown`.

### Strategie structure-aware

Les PDFs FFE sont structures en headings markdown (tous `##`). La hierarchie parent-child se deduit de la **numerotation des articles** (ex: `2.` est parent de `2.1`, `2.2`, etc.), pas des niveaux markdown.

Le chunker :

1. Parse le markdown en sections basees sur les headings `##`
2. Deduit la hierarchie parent-child depuis la numerotation des articles
3. Applique les regles suivantes :

### Regles de chunking

| Cas | Regle |
|-----|-------|
| Section vide (heading sans body) | N'est PAS un child. Sert de label parent pour les sections qui suivent. ~198 headings concernes. |
| Section <= 512 tokens | 1 child = la section entiere |
| Section > 512 tokens (prose) | Split en sous-chunks de ~450 tokens avec overlap 50 tokens |
| Section > 512 tokens (tableau) | PAS de split. Le tableau brut est stocke tel quel. La table summary correspondante couvre le retrieval. |
| Sections consecutives < 100 tokens sous meme parent | Merge en un seul child (jusqu'a 512 tokens max) |
| Page markers (`R01-1/6`, etc.) | Supprimes du texte avant chunking |

### Parent = regroupement par numerotation

Exemple pour R01 :
```
Parent: "2. Statut d'un Joueur" (texte complet articles 2 a 2.4)
  Child: "2.1. Nationalite" (texte de l'article)
  Child: "2.2. La Mutation" (texte)
  Child: "2.3+2.4" (merge si petits)
```

Pour les sections sans numerotation (rares), le parent = le heading precedent de "niveau semantique" superieur.

### Pas d'overlap entre articles
Les articles reglementaires sont des unites semantiques autonomes. Le contexte inter-articles est donne par le parent au LLM.

### Cible
~600-800 chunks (estimation revisee). L'estimation initiale de ~500 etait irrealiste : 1384 sections avec texte, dont beaucoup sont deja dans la plage 200-512 tokens. Le merge des petites sections et l'exclusion des headings vides ramenent le compte, mais pas a 500.

### Metadata par chunk
- `id` : format `{source}-art{article_num}-c{child_index}` (source = nom du PDF)
- `source` : nom du PDF
- `pages` : JSON array des pages couvertes
- `article_num` : numero d'article si present
- `section` : heading parent
- `parent_id` : reference vers le parent
- `tokens` : nombre de tokens (tiktoken cl100k_base)

---

## 2. Indexer (`scripts/pipeline/indexer.py`)

### Embeddings
- Modele : EmbeddingGemma-300M (768D)
- Embed : tous les children + tous les table summaries (111)
- Ne PAS embed les parents (trop longs, servent de contexte pas de retrieval)

### Table summaries
Les 111 resumes dans `corpus/processed/table_summaries_claude.json` sont integres dans l'index. Chaque summary est embeddee et stockee avec un lien vers le tableau brut source (pour retourner le tableau complet au LLM).

### SQLite DB
Schema :

```sql
CREATE TABLE children (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- 768 floats
    parent_id TEXT NOT NULL REFERENCES parents(id),
    source TEXT NOT NULL,
    pages TEXT,  -- JSON array
    article_num TEXT,
    section TEXT,
    tokens INTEGER
);

CREATE TABLE parents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    section TEXT,
    tokens INTEGER
);

CREATE TABLE table_summaries (
    id TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL,
    raw_table_text TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- 768 floats
    source TEXT NOT NULL,
    page INTEGER,
    tokens INTEGER
);
```

---

## 3. Search (`scripts/pipeline/search.py`)

### Retrieval
1. Embed la query avec EmbeddingGemma-300M
2. Cosine similarity brute-force sur `children.embedding` + `table_summaries.embedding`
3. Top-k results (k=10 pour mesure, ajustable)
4. Pour chaque child dans top-k : recuperer le `parent.text` via `parent_id`
5. Dedupliquer les parents (plusieurs children du meme parent possible)
6. Retourner les parents dedupliques + les table summaries matchees (raw_table_text)

### Recherche globale
Pas de filtre par competition/document. La hierarchie pyramidale des reglements FFE (LA → R01 → competition specifique) impose de chercher dans tout le corpus. Le metadata sert a la citation, pas au filtrage.

### Cosine brute-force
~700 children + 111 summaries = ~811 vecteurs x 768D. Sub-10ms sur CPU. Pas besoin d'index ANN.

---

## 4. Contraintes

### EmbeddingGemma-300M
- Max sequence : 2048 tokens
- Google evalue a 512 tokens principalement
- "Larger inputs often result in information loss" (HuggingFace)
- Children a 400-512 tokens = dans le sweet spot

### Corpus specifique
- 30 PDFs de reglements d'echecs francais — scope etroit, confusions possibles entre competitions similaires
- Cascade reglementaire : LA → regles generales → competition specifique
- Metadata (source, article, section) dans chaque chunk pour citation

### Standards industrie
- 400-512 tokens/chunk optimal pour corpus reglementaire (NVIDIA, Vecta, Milvus)
- ~1.5-3 chunks/page pour ~390 pages (vs 4.8 actuels)
- Structure-aware chunking bat fixed-size pour documents reglementaires (PMC study: 87% vs 13%)

---

## 5. Fichiers a creer

| Fichier | Responsabilite |
|---------|---------------|
| `scripts/pipeline/chunker.py` | Markdown → children + parents |
| `scripts/pipeline/indexer.py` | Children + summaries → SQLite DB |
| `scripts/pipeline/search.py` | Query → top-k parents |
| `scripts/pipeline/tests/test_chunker.py` | Tests chunker |
| `scripts/pipeline/tests/test_indexer.py` | Tests indexer |
| `scripts/pipeline/tests/test_search.py` | Tests search |

3 scripts, 3 fichiers de tests. Pas plus.

---

## 6. Compatibilite GS (chantier 3)

Les nouveaux chunk IDs (`{source}-art{N}-c{M}`) sont **incompatibles** avec les IDs actuels du GS (`{source}-p{page}-parent{N}-child{NN}`).

Pour la mesure recall (chantier 3) :
- L'ancien `chunks_mode_b_fr.json` est **preserve** (pas supprime)
- Le mapping se fera par **matching textuel** : pour chaque question GS, comparer le texte du chunk GS attendu avec les nouveaux chunks
- Pas de re-mapping d'IDs dans ce chantier

---

## 7. Ce qu'on ne fait PAS

- Re-extraction PDF (docling OK)
- Re-mapping GS (on mesure d'abord, matching textuel en chantier 3)
- Reclassification GS (apres mesure)
- Kotlin / Android (apres)
- Fine-tuning embeddings (apres mesure recall)
- Overlap entre articles (contexte donne par parent)

---

## 8. Livrable

Une SQLite DB (`corpus/processed/corpus_v2_fr.db`) contenant :
- ~600-800 children embeddes (400-512 tokens, structure-aware)
- ~200-400 parents (texte brut, sections)
- 111 table summaries embeddees
- Metadata complete (source, article, section, pages)

Prete pour mesurer le recall (chantier 3).
