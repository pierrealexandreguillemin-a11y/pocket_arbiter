# Recall Optimization — Design Spec

> **Date**: 2026-03-19
> **Statut**: En revue
> **Baseline**: recall@5 = 56.7%, recall@10 = 63.1% (LangChain chunker, EmbeddingGemma-300M QAT)
> **Target**: recall@5 >= 80% (seuil prompt engineering suffisant)
> **Contrainte**: un seul rebuild (12 min embedding)

---

## Contexte

### Root cause analysis (session 2026-03-18)

- 110 failures sur 298 questions testables
- 14 failures (13%) = pages manquantes (chunker/page problem)
- **96 failures (87%) = embedding ne trouve pas le bon chunk** (retrieval problem)
- Parmi les 96 : 8 dans top-20 cosine, 12 dans top-50, 13 au-dela de rank 100

### Standards industrie consultes

| Source | Finding | Impact |
|--------|---------|--------|
| NAACL 2025 Vectara | Chunking config = autant d'influence que embedding model | Optimiser chunking AVANT fine-tune |
| Anthropic 2024 | Contextual Retrieval : -35% a -49% failures | Technique #1 |
| Firecrawl 2026 | Fixed 400 tok + 20% overlap = 82% recall | Tester chunk_size=400 |
| Haystack | Synonym/abbreviation expansion in chunks = +5-10% | Enrichissement texte |
| Google EmbeddingGemma | `title: {title} | text: {content}` format confirme | CCH dans title: |

### Docs projet existants integres

- `docs/research/OFFLINE_OPTIMIZATIONS_2026-01-20.md` : 7 optimisations zero-cost
- `docs/ISO_VECTOR_SOLUTIONS.md` : Contextual Retrieval, MRL, fine-tuning specs
- `docs/research/OFFLINE_OPTIMIZATIONS` §5 : benchmark jan 2026 (86.94% baseline avec 450t)

---

## Architecture

### Pipeline d'enrichissement (build-time)

```
chunk_document(markdown, source, heading_pages)
  -> {children, parents, tables}               # Chunker existant

enrich_chunks(children, parents)               # NOUVEAU MODULE
  -> Pour chaque child:
     1. Abbreviation expansion (dict in-place replace)
     2. CCH text injection ("[h1 > h2 > h3]\n\n")
     3. Contextual retrieval (1-2 phrases situant le chunk)
  -> children enrichis (text modifie)
  -> chunk_contexts.json (auditable)

indexer.build_index()
  -> Embed: format_document(cch_title, enriched_text)
  -> Prompt final:
     "title: {Doc Title > Section} | text: {contexte}. [Heading hierarchy]\n\n{texte enrichi abreviations}"
```

### Pipeline de recherche (query-time, modifie)

```
search(query)
  -> score_calibration(cosine_results)         # NOUVEAU: normaliser par source
  -> query_decomposition(query)                # NOUVEAU: split queries complexes
  -> bm25_search(sub_queries)                  # Existing (expand_query + FTS5)
  -> rrf_fusion + adaptive_k                   # Existing
  -> build_context                             # Existing
```

---

## Optimisations detaillees

### OPT-1 : Contextual Retrieval (build-time)

**Standard** : Anthropic 2024, -35% a -49% failures.

Pour chaque chunk, generer 1-2 phrases de contexte factuel situant le chunk dans le document.
Le contexte est genere par le LLM (Claude Code) pendant l'implementation et stocke dans un
fichier JSON intermediaire auditable.

**Input** : chunk text + parent text + metadata (source, section, page)
**Output** : 1-2 phrases factuelles

**Regles de generation** :
- Factuel uniquement (pas de paraphrase, pas d'interpretation)
- Mentionner : le document source, la section, le sujet traite
- Max 120 tokens par contexte (median observee : 54 tokens)
- Langue : francais sans accents (compatible FTS5 stemming)

**Fichier** : `corpus/processed/chunk_contexts.json`
```json
{
  "R01_2025_26_Regles_generales.pdf-c0005": "Extrait du Reglement General FFE, section Forfaits, sous-section Principes generaux. Definit le forfait et ses consequences sur le score.",
  "LA-octobre2025.pdf-c0042": "Extrait du Livre de l'Arbitre FFE, chapitre Missions de l'arbitre. Decrit la mission sur les lieux d'un tournoi."
}
```

**Quality gate** : chaque contexte verifie — pas de hallucination, pas d'info absente du chunk/parent.

### OPT-2 : Abbreviation expansion (build-time)

**Standard** : Haystack query expansion, doc OFFLINE_OPTIMIZATIONS §1.

Remplacer les abbreviations dans le texte des chunks par leur forme expandue.
Dictionnaire hardcode, verifie contre le corpus.

```python
ABBREVIATIONS = {
    "CM": "CM (Candidat Maitre)",
    "FM": "FM (Maitre FIDE)",
    "MI": "MI (Maitre International)",
    "GMI": "GMI (Grand Maitre International)",
    "DNA": "DNA (Direction Nationale de l'Arbitrage)",
    "AFJ": "AFJ (Arbitre Federal Jeune)",
    "AFC": "AFC (Arbitre Federal de Club)",
    "AF3": "AF3 (Arbitre Federal 3)",
    "AF2": "AF2 (Arbitre Federal 2)",
    "AF1": "AF1 (Arbitre Federal 1)",
    "AI": "AI (Arbitre International)",
    "FFE": "FFE (Federation Francaise des Echecs)",
    "FIDE": "FIDE (Federation Internationale des Echecs)",
    "UV": "UV (Unite de Valeur)",
    "CDJE": "CDJE (Comite Departemental du Jeu d'Echecs)",
}
```

**Regles** :
- Match mot entier uniquement (regex `\bCM\b`)
- Ne PAS expander si deja expande ("CM (Candidat" → skip)
- Verifie : chaque cle du dict existe dans >= 1 chunk du corpus

### OPT-3 : CCH text injection (build-time) — RETIRE

**Statut** : RETIRE apres self-audit (mars 2026)

**Spec originale** : injecter la hierarchie heading dans `text:` en plus de `title:`.

**Pourquoi retire** : duplication CCH dans les deux canaux = gaspillage de tokens sans
standard industriel pour le supporter. L'architecture retenue separe les roles :

| Canal | Contenu | Standard |
|-------|---------|----------|
| `title:` | Position structurelle — CCH heading hierarchy | Google EmbeddingGemma model card |
| `text:` | Sens semantique — contexte OPT-1 + contenu chunk | Anthropic Contextual Retrieval 2024 |

Ce design combine le meilleur des deux approches :
- Google prescrit `title:` pour le metadata structurel (ou, dans quel chapitre)
- Anthropic prescrit le prepend dans `text:` pour le contexte semantique (quoi, quel sujet)
- Pas de duplication = budget tokens optimal (EmbeddingGemma seq 256)

**Sources consultees** :
- Google EmbeddingGemma : format `title: | text:` prescrit, `title:` pour metadata, pas de double-injection
- Anthropic 2024 : contexte prepende dans le texte (`text_to_embed = f"{context}\n\n{content}"`)
- LlamaIndex, Haystack, CCH (NirDiamant) : metadata prependee au texte, un seul canal
- arXiv:2601.11863 (jan 2026) : meta-prefix +22pp recall, mais section titles = "modest drop" vs identifiants globaux
- arXiv:2404.12283 (avr 2024) : enrichissement texte = resultats mixtes selon le domaine (-3.45pp a +8.21pp)

**Resultat final** : le prompt embedde ressemble a :
```
title: Lois des Echecs FFE > Forfaits > Principes generaux
text: Extrait du chapitre Forfaits du Reglement General FFE, section Principes generaux.
Definit le forfait et ses consequences sur le score. Le forfait d'un joueur...
```

### OPT-4 : Chapter titles hardcodes (build-time)

**Standard** : doc OFFLINE_OPTIMIZATIONS §4, arXiv 2501.07391.

Overrides pour les chapitres specifiques du Livre de l'Arbitre ou la hierarchie
heading ne suffit pas (chapitres 6.x FIDE, annexes A/B).

```python
CHAPTER_OVERRIDES = {
    (182, 186): "Classement Elo Standard FIDE",
    (187, 191): "Classement Rapide et Blitz FIDE",
    (192, 205): "Titres FIDE",
    (56, 57): "Annexe A - Cadence Rapide",
    (58, 66): "Annexe B - Cadence Blitz",
}
```

Applique apres le CCH automatique — override le champ `title:` du prompt si match par page.

### OPT-5 : Flag pages intro (query-time)

**Standard** : arXiv FILCO filtering, doc OFFLINE_OPTIMIZATIONS §6.

Exclure les pages d'introduction (sommaire, preface) du search par defaut.
Colonne `is_intro` dans la table children, filtree dans `cosine_search` et `bm25_search`.

```sql
-- Marquer pages intro (pages 1-3 de LA = sommaire/preface)
-- Specifique par document, pas un seuil global
```

### OPT-6 : Chunk size / overlap tuning (config)

**Standard** : Firecrawl 2026 (400 tok = 82% recall), NAACL 2025 Vectara.

Tester chunk_size=400 et overlap=50 (au lieu de 512/100 actuels).
Le benchmark janvier (doc OFFLINE_OPTIMIZATIONS §5) montrait 86.94% avec 450t single-size.

**Decision** : chunk_size=450, overlap=50 (compromis entre Firecrawl 400 et FloTorch 512,
aligne avec le benchmark janvier qui avait le meilleur recall a 450t).

### OPT-7 : Score calibration par source (query-time)

**Standard** : non documente (technique avancee).

Normaliser les scores cosine par nombre de chunks dans le document source.
LA-octobre2025 = 53% des chunks → bias probabiliste vers LA.

```python
def calibrate_scores(results, conn):
    source_counts = {}  # cache {source: n_chunks}
    calibrated = []
    for doc_id, score in results:
        source = get_source(conn, doc_id)
        n = source_counts.setdefault(source, count_source_chunks(conn, source))
        calibrated.append((doc_id, score * (1 / math.log2(n + 1))))
    return sorted(calibrated, key=lambda x: -x[1])
```

**Testable sans rebuild** sur la DB actuelle.

### OPT-8 : Query decomposition (query-time)

**Standard** : non documente (technique avancee), compatible offline.

Decomposer les queries complexes en sous-queries pour le BM25.
Heuristiques regex, pas de LLM.

```python
SPLIT_PATTERNS = [
    r"\bsi\b",           # "... si la ronde ..."
    r"\blorsque\b",      # "... lorsque le joueur ..."
    r"\bet\b(?!.*mat)",  # "... et ..." (pas "echec et mat")
    r"\bquand\b",        # "... quand ..."
]
```

Chaque sous-query est envoyee separement a `bm25_search`, resultats fusionnes.
Le cosine search garde la query complete (pas de decomposition pour les embeddings).

**Testable sans rebuild** sur la DB actuelle.

---

## Tables et table summaries

Les 111 table summaries existantes restent inchangees. Elles beneficient de :
- OPT-1 : pas de contextual retrieval (summaries deja concises)
- OPT-2 : abbreviation expansion dans `summary_text` avant embedding
- ~~OPT-3~~ : RETIRE — CCH reste dans `title:` uniquement (pas de double-injection, voir OPT-3 ci-dessus)
- Le champ `title:` du prompt utilise deja le CCH du chunker (session precedente)

Le raw_table_text retourne au LLM n'est PAS modifie (Anthropic multi-vector pattern).

---

## SRP / fichiers

| Fichier | Responsabilite | Action | Lignes max |
|---------|---------------|--------|------------|
| `scripts/pipeline/enrichment.py` | CREATE — OPT 1-2-4 (context loader, abbreviations, chapter overrides) | Nouveau | 150 |
| `scripts/pipeline/search.py` | MODIFY — OPT 7-8 (score_calibration, query_decomposition) | Ajouter | 300 |
| `scripts/pipeline/indexer.py` | MODIFY — appeler enrich_chunks avant embedding | Modifier | 250 |
| `scripts/pipeline/chunker.py` | MODIFY — OPT 6 (chunk_size=450, overlap=50) | Config | 140 |
| `scripts/pipeline/tests/test_enrichment.py` | CREATE — tests enrichment | Nouveau | 200 |
| `scripts/pipeline/tests/test_search.py` | MODIFY — tests calibration + decomposition | Ajouter | 300 |
| `corpus/processed/chunk_contexts.json` | OUTPUT — contextes auditables | Genere | ~1073 entries |

---

## Quality gates

### Avant rebuild (sur les donnees enrichies)

| Gate | Verification |
|------|-------------|
| E1 | chunk_contexts.json : 1116 entries, aucune vide, max 120 tokens (median 54) |
| E2 | Abbreviations : chaque cle du dict matche >= 1 chunk (regex `\b`) |
| E3 | ~~CCH injection~~ RETIRE (OPT-3 retire, CCH dans `title:` uniquement) |
| E4 | Chapter overrides : pages specifiees existent dans la DB |
| E5 | Echantillon 20 chunks enrichis : verification visuelle (pas de hallucination) |
| E6 | Token distribution post-enrichment : median 350-500, max < 2048 |

### Apres rebuild (integrite DB)

I1-I9 existants (session precedente) PASS.

### Apres recall measurement

| Gate | Verification |
|------|-------------|
| R1 | recall@5 >= 70% (amelioration significative vs 56.7%) |
| R2 | recall@10 >= 75% |
| R3 | Aucune regression sur les questions qui passaient avant |
| R4 | MRR >= 0.50 (vs 0.441 actuel) |

---

## Ordre d'execution

1. **Query-time first** (OPT 7-8) — testable sans rebuild, mesure le gain immediat
2. **Enrichissement** (OPT 1-4) — generer chunk_contexts.json, enrichir texte
3. **Config tuning** (OPT 6) — chunk_size=450, overlap=50
4. **Full corpus audit** — verifier AVANT rebuild
5. **Un seul rebuild** — 12 min
6. **Recall measurement** — comparer avec baseline
7. **Flag intro** (OPT 5) — query-time, apres rebuild, testable independamment

---

## Ce qu'on ne fait PAS

- Pas de formulations alternatives generees (texte non verifiable)
- Pas de late chunking (complexite EmbeddingGemma 2048 tokens)
- Pas de SPLADE (pas de port Android offline)
- Pas de fine-tuning (si >= 80% apres optimisations)
- Pas de multi-granularity indexing (complexite vs gain incertain)
- Pas de cross-encoder reranker (RAM runtime 600MB)

---

## Standards appliques

| Standard | Application |
|----------|-------------|
| Anthropic 2024 Contextual Retrieval | OPT-1 : -35% a -49% failures |
| NAACL 2025 Vectara | Chunking config = embedding model influence |
| Firecrawl 2026 | chunk_size=400-450 optimal |
| Haystack | Abbreviation/synonym expansion |
| Google EmbeddingGemma | `title: {title} \| text: {content}` format |
| arXiv FILCO | Flag intro pages |
| arXiv 2501.07391 | Chapter metadata in chunks |
| ISO 29119 | TDD, quality gates |
| ISO 25010 | Fichiers <= 300 lignes |
| ISO 12207 | Commits conventionnels |
| ISO 42001 | Tracabilite (chunk_contexts.json auditable) |

---

## Sources

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [NAACL 2025 Vectara — chunking vs embedding](https://aclanthology.org/2025.findings-naacl.114.pdf)
- [Firecrawl 2026 Chunking Benchmark](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Google EmbeddingGemma Model Card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)
- [Haystack Query Expansion](https://haystack.deepset.ai/blog/query-expansion)
- [PremAI Production RAG 2026](https://blog.premai.io/building-production-rag-architecture-chunking-evaluation-monitoring-2026-guide/)
- docs/research/OFFLINE_OPTIMIZATIONS_2026-01-20.md (projet)
- docs/ISO_VECTOR_SOLUTIONS.md (projet)
