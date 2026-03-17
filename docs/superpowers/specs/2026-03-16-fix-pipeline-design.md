# Chantier 2 : Fix Pipeline — Design Spec

> **Date**: 2026-03-16
> **Statut**: Approuve
> **Objectif**: Reconstruire le pipeline retrieval pour integrer parents, table summaries, et corriger la taille des chunks

---

## Contexte

Le pipeline actuel produit 1857 chunks de 109 tokens median (sur-chunke pour 390 pages). Les parents (1394) et table summaries (111) existent sur disque mais ne sont pas dans l'index de recherche. Le recall baseline est a 36.6% chunk@5.

Les extractions docling existantes (`corpus/processed/docling_fr/*.json`) sont fideles au texte des PDFs (audit mars 2026) mais ont un defaut : tous les headings sont aplatis en `##` (limitation connue de docling pour les PDFs). La hierarchie reelle (Partie > Chapitre > Section > Article) est perdue.

### Re-extraction avec heading levels
Le package `docling-hierarchical-pdf` (github.com/krrome/docling-hierarchical-pdf) post-traite les resultats docling pour inferer les niveaux de heading depuis les bookmarks PDF, la numerotation des articles, et la taille de police. Teste sur 60+ PDFs texte.

**Approche** : re-extraire les 29 PDFs avec docling + post-processeur hierarchique. Verifier les heading levels contre les PDFs sources (audit echantillon). Si resultats insatisfaisants, fallback sur les extractions existantes avec deduction de hierarchie par numerotation.

### Hypotheses documentees
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
Re-extraction des 29 PDFs FR avec docling + `docling-hierarchical-pdf`. Produit du markdown avec vrais niveaux (`#`, `##`, `###`, etc.) correspondant a la hierarchie reelle du document.

Fallback si post-processeur insatisfaisant : extractions existantes (`corpus/processed/docling_fr/*.json`) avec deduction de hierarchie par numerotation.

### Verification obligatoire
Audit des heading levels extraits contre les PDFs sources sur un echantillon (R01 6 pages, LA echantillon, 2-3 petits PDFs). Les niveaux doivent correspondre a la structure visuelle du PDF.

### Strategie structure-aware

La hierarchie parent-child vient directement des **niveaux de heading** du markdown re-extrait (`#` = niveau 1, `##` = niveau 2, etc.).

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

## 3. Indexer — Contextual Chunk Headers (CCH)

### Principe (Anthropic Contextual Retrieval, mars 2025)
Avant l'embedding, prependre le contexte documentaire au chunk :
```
[Document: Regles generales competitions FFE 2025-26 | Section: 3.2. Forfait isole]
3.2.1. Est consideree comme etant forfait : ...
```

Le modele d'embedding voit le contexte (quel document, quelle section) en plus du contenu. Cela desambiguise les chunks similaires entre competitions differentes.

- **Impact mesure** : +35% reduction d'echecs de retrieval (Anthropic), +49% avec BM25 hybride
- **Cout** : zero runtime — le header est prepend au build-time uniquement
- Stocke dans la DB : `text` = chunk brut, `embedding` = embed du chunk contextualise

### Implementation
L'indexer genere le texte contextualise pour chaque child :
```python
contextualized = f"[Document: {source_title} | Section: {parent_section}]\n{child_text}"
embedding = embed(contextualized)  # embed le contextualise
```
Mais stocke le `text` original (sans header) dans la DB pour le LLM.

---

## 4. Search (`scripts/pipeline/search.py`)

### Retrieval
1. Embed la query avec EmbeddingGemma-300M
2. Cosine similarity brute-force sur `children.embedding` + `table_summaries.embedding`
3. Top-k results (k=10 pour mesure, ajustable)
4. Score-based adaptive k : exclure resultats sous seuil 0.72 ou gap > 0.10 avec precedent
5. Pour chaque child dans top-k : recuperer le `parent.text` via `parent_id`
6. Dedupliquer les parents (plusieurs children du meme parent possible)
7. Retourner les parents dedupliques + les table summaries matchees (raw_table_text)

### Recherche globale
Pas de filtre par competition/document. La hierarchie pyramidale des reglements FFE (LA → R01 → competition specifique) impose de chercher dans tout le corpus. Le metadata sert a la citation, pas au filtrage.

### Cosine brute-force
~1200 children + 111 summaries = ~1311 vecteurs x 768D. Sub-10ms sur CPU. Pas besoin d'index ANN.

---

## 5. Optimisations retrieval — Roadmap par priorite

Recherche web mars 2026 : techniques RAG pour petit corpus specialise (<500 pages, offline mobile).

### P0 — Integre dans ce chantier

| Technique | Impact | Effort | Mobile |
|-----------|--------|--------|--------|
| **Contextual Chunk Headers (CCH)** | +35% recall | Faible (build-time) | Oui |
| **Score-based adaptive k** (seuil 0.72, gap 0.10) | -10% hallucination | Faible | Oui |

### P1 — Chantier suivant (post re-mesure recall)

| Technique | Impact | Effort | Mobile |
|-----------|--------|--------|--------|
| **BM25 hybride** (alpha=0.6 semantic + 0.4 BM25) | +10-15% recall | Moyen (inverted index <10MB) | Oui |
| **Fine-tuning EmbeddingGemma** sur triplets GS | +15-20% recall | Moyen (depend decision chantier 3) | Oui |

### P2 — Si P0+P1 insuffisants

| Technique | Impact | Effort | Mobile |
|-----------|--------|--------|--------|
| **LLM context summaries** par chunk (batch offline) | +15% additionnel | Moyen (one-time batch) | Oui (build-time) |
| **Cross-encoder reranker** (MiniLM 80MB) | +20-35% accuracy | Eleve (+80-120MB app) | Conditionnel |

### Skip — Pas adapte

| Technique | Raison |
|-----------|--------|
| HyDE | Latence mobile (3-7s), marginal apres fine-tuning |
| ColBERT | 200x stockage, complexite deploiement mobile |
| Metadata filtering | Incompatible avec cascade reglementaire FFE |

Sources : Anthropic Contextual Retrieval (2025), arXiv 2502.16767 (hybrid regulatory), Pocket RAG (arXiv 2602.13229), MobileRAG (arXiv 2507.01079).

---

## 6. Contraintes

### EmbeddingGemma-300M
- Max sequence : 2048 tokens
- Google evalue a 512 tokens principalement
- "Larger inputs often result in information loss" (HuggingFace)
- Children a 400-512 tokens = dans le sweet spot
- Matryoshka : 768D par defaut, 256D possible (index 1311 x 256 x 4 = ~1.3MB)

### Corpus specifique
- 28 PDFs de reglements d'echecs francais — scope etroit, confusions possibles entre competitions similaires
- Cascade reglementaire : LA → regles generales → competition specifique
- Metadata (source, article, section) dans chaque chunk pour citation

### Standards industrie
- 400-512 tokens/chunk optimal pour corpus reglementaire (NVIDIA, Vecta, Milvus)
- Structure-aware chunking bat fixed-size pour documents reglementaires (PMC study: 87% vs 13%)
- CCH = technique la plus impactante pour petit corpus specialise (Anthropic: +35-49%)

---

## 5. Fichiers a creer

| Fichier | Responsabilite |
|---------|---------------|
| `scripts/pipeline/extract.py` | Re-extraction PDFs avec docling + hierarchical post-processor |
| `scripts/pipeline/chunker.py` | Markdown hierarchique → children + parents |
| `scripts/pipeline/indexer.py` | Children + summaries → SQLite DB |
| `scripts/pipeline/search.py` | Query → top-k parents |
| `scripts/pipeline/tests/test_extract.py` | Tests extraction + verification heading levels |
| `scripts/pipeline/tests/test_chunker.py` | Tests chunker |
| `scripts/pipeline/tests/test_indexer.py` | Tests indexer |
| `scripts/pipeline/tests/test_search.py` | Tests search |

4 scripts, 4 fichiers de tests.

---

## 6. Compatibilite GS (chantier 3)

Les nouveaux chunk IDs (`{source}-art{N}-c{M}`) sont **incompatibles** avec les IDs actuels du GS (`{source}-p{page}-parent{N}-child{NN}`).

Pour la mesure recall (chantier 3) :
- L'ancien `chunks_mode_b_fr.json` est **preserve** (pas supprime)
- Le mapping se fera par **matching textuel** : pour chaque question GS, comparer le texte du chunk GS attendu avec les nouveaux chunks
- Pas de re-mapping d'IDs dans ce chantier

---

## 7. Ce qu'on ne fait PAS

- Re-extraction depuis zero (on reutilise docling avec post-processeur)
- Re-mapping GS (on mesure d'abord, matching textuel en chantier 3)
- Reclassification GS (apres mesure)
- Kotlin / Android (apres)
- Fine-tuning embeddings (apres mesure recall)
- Overlap entre articles (contexte donne par parent)

---

## 8. Livrable

Une SQLite DB (`corpus/processed/corpus_v2_fr.db`) contenant :
- ~1150 children embeddes (median 311 tokens, structure-aware) avec CCH
- ~330 parents (texte brut, median 863 tokens)
- 111 table summaries embeddees avec CCH
- Metadata complete (source, article, section, pages)
- Score-based adaptive k dans le search

Prete pour mesurer le recall (chantier 3).

Techniques P1/P2 documentees pour implementation post re-mesure.

---

## 9. Mise a jour annuelle du corpus

Les reglements FFE sont mis a jour chaque saison (septembre). Process de rebuild :

1. **Remplacer les PDFs** dans `corpus/fr/` (nouveaux reglements saison N+1)
2. **Re-extraire** : `python scripts/pipeline/extract.py` (~1h avec LA 222 pages)
3. **Verifier** : audit heading levels sur echantillon, edge cases H01/H02/R02
4. **Re-chunker** : automatique via `build_index()` (chunker integre)
5. **Re-embedder** : automatique via `build_index()` (~18 min CPU, ou GPU si dispo)
6. **Quality gates** : les 11 gates existantes + G8c pages GS (si GS mis a jour)
7. **Livrer** : remplacer `corpus_v2_fr.db` dans les assets Android

Pas de delta update — full rebuild. Le corpus est petit (<500 pages), le cout est acceptable (~20 min).

Si le GS est mis a jour (nouvelles questions annales), re-verifier G8 (text match).

---

## 10. Decisions architecture Android (recherche web mars 2026)

### Vector search : Pure Kotlin brute-force

| Option | Latence 1364 vecteurs | Dependance | Decision |
|--------|----------------------|------------|----------|
| **Pure Kotlin dot product** | ~5ms (extrapole de 500ms/100K) | Aucune | **RETENU** |
| sqlite-vec extension | ~2ms (SIMD natif) | JNI C library | Overkill |
| ObjectBox vector DB | Non benchmarke | ~3MB + JNI | Overkill, pas evalue |

Justification : 1364 vecteurs x 768D x float32 = ~4MB en memoire. Brute-force dot product
sur ARM64 NEON est sub-10ms. Pas besoin d'index ANN (HNSW/IVF) sous 10K vecteurs.

Source : [On-Device RAG (Denisov 2026)](https://medium.com/google-developer-experts/on-device-rag-for-app-developers-embeddings-vector-search-and-beyond-47127e954c24), benchmarks ente.io SQLite vs ObjectBox vs Isar.
