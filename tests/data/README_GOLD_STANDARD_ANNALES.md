# Gold Standard Annales - Documentation

> Version: 7.4.3
> Date: 2026-01-26
> Source: Annales examens arbitres FFE

## Vue d'ensemble

- **420 questions** issues des annales d'examen d'arbitre FFE
- **28 documents** dans le corpus (PDF FFE)
- **100% chunk_ids valides** (tous pointent vers un chunk existant)
- **Format actuel**: Questions au format QCM original (reformulation en attente)

## État de la Reformulation

| Version | État | Notes |
|---------|------|-------|
| v7.4.x | **QCM original** | Format examen FFE avec choix A/B/C/D |
| v7.5.0 | REVERTÉE | Reformulation échouée (21.2% answerability) |
| v7.6.0+ | À faire | Reformulation naturelle prévue après correction chunk_ids |

**Champs QCM présents** (seront retirés après reformulation):
- `metadata.choices`: Options A/B/C/D
- `metadata.mcq_answer`: Lettre de la bonne réponse
- `metadata.answer_type`: "multiple_choice" pour 334/420 questions

## Structure JSON d'une Question

```json
{
  "id": "string",                    // Format: "ffe:annales:<category>:<num>:<hash>"
  "question": "string",              // Texte de la question (format QCM actuellement)
  "expected_answer": "string",       // Réponse attendue (texte complet, pas la lettre)
  "is_impossible": false,            // Si réponse impossible (SQuAD 2.0 style)
  "expected_chunk_id": "string",     // ID du chunk contenant la réponse
  "expected_docs": ["string"],       // Liste des documents sources
  "expected_pages": [int],           // Pages sources dans le PDF
  "category": "string",              // Catégorie thématique
  "keywords": ["string"],            // Mots-clés pour recherche
  "validation": {
    "status": "VALIDATED",           // VALIDATED | PENDING | REJECTED
    "method": "annales_official",    // Méthode de validation
    "reviewer": "human",             // human | auto
    "answer_current": true,          // Réponse toujours valide
    "verified_date": "YYYY-MM-DD",
    "pages_verified": true
  },
  "audit": "string",                 // Trace des modifications
  "metadata": {
    "answer_type": "string",         // multiple_choice | extractive | abstractive | list | yes_no
    "reasoning_type": "string",      // single-hop | multi-hop
    "cognitive_level": "string",     // Remember | Understand | Apply | Analyze
    "article_reference": "string",   // Référence article source (ex: "C01 - 3.8")
    "difficulty": float,             // 0.0 - 1.0
    "annales_source": {
      "session": "string",           // dec2023 | dec2024 | ...
      "uv": "string",                // UVC | UVR | UVT | UVO
      "question_num": int,
      "success_rate": float          // Taux de réussite historique
    },
    "question_type": "string",       // factual | scenario | procedural | comparative
    "choices": {                     // [QCM] Options de réponse
      "A": "string",
      "B": "string",
      "C": "string",
      "D": "string"
    },
    "mcq_answer": "string",          // [QCM] Lettre correcte: A | B | C | D
    "quality_score": float,          // Score qualité 0-1
    "chunk_match_score": int,        // Score matching chunk 0-100
    "chunk_match_method": "string",  // article_direct | page_keyword | ...
    "reasoning_class": "string",     // fact_single | summary | reasoning | arithmetic
    "reasoning_class_method": "string",
    "triplet_ready": true            // Prêt pour format triplet RAG
  },
  "legacy_id": "string"              // Ancien ID pour compatibilité
}
```

**Note**: Les champs `[QCM]` seront migrés vers `original_annales` lors de la reformulation.

### Valeurs Possibles par Champ

| Champ | Valeurs | Description |
|-------|---------|-------------|
| `answer_type` | multiple_choice, extractive, abstractive, list, yes_no | Type de réponse |
| `question_type` | factual, scenario, procedural, comparative | Type de question |
| `reasoning_class` | fact_single, summary, reasoning, arithmetic | Classification raisonnement |
| `cognitive_level` | Remember, Understand, Apply, Analyze | Taxonomie de Bloom |
| `validation.status` | VALIDATED, PENDING, REJECTED | État validation |

## Taxonomie Metadata (Standards Industrie)

### reasoning_class
| Valeur | Count | Description |
|--------|-------|-------------|
| `fact_single` | 199 | Réponse dans 1 chunk (single-hop) |
| `summary` | 215 | Synthèse/résumé d'information |
| `reasoning` | 6 | Raisonnement explicite requis |

### question_type
| Valeur | Count | Description |
|--------|-------|-------------|
| `factual` | 238 | Question factuelle directe |
| `scenario` | 161 | Question basée sur un scénario |
| `procedural` | 15 | Procédure/étapes |
| `comparative` | 6 | Comparaison entre options |

### answer_type
| Valeur | Count | Description |
|--------|-------|-------------|
| `multiple_choice` | 334 | QCM (origine annales) |
| `extractive` | 58 | Extraction directe |
| `abstractive` | 19 | Synthèse requise |
| `list` | 8 | Liste d'éléments |
| `yes_no` | 1 | Oui/Non |

## Problèmes Identifiés

### 1. Chunk_ids incorrects (70 questions)

Questions `fact_single` + `factual` dont le chunk actuel ne contient pas la réponse:

```
fact_single + factual: 184 questions
├── Chunk OK (score >= 0.3): 114 ✓
└── Chunk INCORRECT: 70 ✗
```

**Cause**: Le `expected_chunk_id` pointe vers un chunk qui ne contient pas l'information.

**Solution**: Utiliser `article_reference` pour trouver le bon chunk.

**Exemple Q25**:
- Réponse: "200 €"
- Article: "C01 - 3.8. Forfaits sportifs"
- Chunk actuel: INCORRECT
- Chunk correct: `C01_2025_26_Coupe_de_France.pdf-p004-parent019-child00`
  - Contient: "200 € en 16e de finale"

### 2. Classifications potentiellement incorrectes

Certaines questions marquées `fact_single` nécessitent en réalité du calcul:

| Question | Réponse | Classification actuelle | Classification correcte |
|----------|---------|------------------------|------------------------|
| clubs:034 | "4" (transpositions) | fact_single | arithmetic |
| clubs:040 | "4" (flotteurs) | fact_single | arithmetic |
| clubs:042 | "9" (niveaux) | fact_single | arithmetic |
| clubs:012 | "Le jeudi" | summary | temporal_reasoning |

### 3. Questions nécessitant contexte examen

28 questions référencent un contexte spécifique (image, tableau d'appariement):
- Contiennent `<!-- image -->` dans la question
- Mentionnent des noms spécifiques (Daniela, Albert, etc.)

Ces questions sont VALIDES mais nécessitent que le chunk fournisse le contexte de résolution.

## Structure article_reference

Chaque question a un `metadata.article_reference` pointant vers la source:

```
Format: <DOCUMENT> - <SECTION>.<ARTICLE>

Exemples:
- "C01 - 3.8. Forfaits sportifs" → C01_2025_26_Coupe_de_France.pdf
- "Article 3.2 du RIDNA" → LA-octobre2025.pdf (Règlement Intérieur DNA)
- "R01 - 4. Homologation" → R01_2025_26_Regles_generales.pdf
- "LA - Chapitre 1.3" → LA-octobre2025.pdf
```

## Documents du Corpus (28)

| Préfixe | Document | Chunks |
|---------|----------|--------|
| LA | Livre de l'Arbitre (octobre 2025) | 992 |
| R01 | Règles générales | 39 |
| R02 | Règles générales - Annexes | 9 |
| R03 | Compétitions homologuées | 17 |
| A01 | Championnat de France | 22 |
| A02 | Championnat de France des Clubs | 37 |
| A03 | Championnat de France des Clubs rapides | 18 |
| C01 | Coupe de France | 29 |
| C03 | Coupe Jean-Claude Loubatière | 39 |
| C04 | Coupe de la Parité | 40 |
| F01 | Championnat de France des clubs Féminin | 38 |
| F02 | Championnat individuel Féminin rapides | 10 |
| J01 | Championnat de France Jeunes | 29 |
| J02 | Championnat de France Interclubs Jeunes | 34 |
| J03 | Championnat de France scolaire | 28 |
| H01 | Conduite joueurs handicapés | 2 |
| H02 | Joueurs à mobilité réduite | 3 |
| E02 | Classement rapide | 9 |
| - | Règlement Disciplinaire 2018 | 29 |
| - | Règlement médical 2022 | 29 |
| - | Règlement Financier 2023 | 26 |
| - | Statuts 2024 | 62 |
| - | Règlement Intérieur 2025 | 77 |
| - | Contrat de délégation | 82 |
| - | Règlement régional/départemental | ~36 |

**Total**: 1857 chunks

## Scripts de Validation

```bash
# Audit des chunk_ids
python scripts/evaluation/annales/audit_chunk_ids.py

# Correction multi-passes
python scripts/evaluation/annales/fix_chunk_ids.py

# Correction intelligente via article_reference
python scripts/evaluation/annales/smart_chunk_fix.py

# Validation qualité complète
python scripts/evaluation/annales/validate_gs_quality.py
```

## Métriques de Qualité

| Métrique | Valeur | Seuil | Status |
|----------|--------|-------|--------|
| Chunk IDs valides | 100% | >= 95% | ✓ PASS |
| fact_single+factual OK | 62% | >= 80% | ✗ FAIL |

## Actions Requises

### Priorité 1: Corriger les 70 chunk_ids
Pour chaque question `fact_single` + `factual` avec score < 0.3:
1. Parser `article_reference` pour identifier le document et l'article
2. Chercher le chunk contenant cet article
3. Vérifier que la réponse est présente
4. Mettre à jour `expected_chunk_id`

### Priorité 2: Reclassifier les questions arithmetic
Questions avec réponses numériques courtes issues de calculs:
- Changer `reasoning_class` de `fact_single` à `arithmetic`
- Ces questions sont valides mais testent le raisonnement, pas l'extraction

### Priorité 3: Documenter les questions à contexte
Pour les 28 questions nécessitant contexte examen:
- Ajouter `metadata.requires_context: true`
- Ou fournir le contexte dans un champ dédié

## Historique des Versions

| Version | Date | Changements |
|---------|------|-------------|
| 7.4.0 | 2026-01-24 | reasoning_class + question_type complets |
| 7.4.1 | 2026-01-25 | 152 corrections chunk_id (multi-passes) |
| 7.4.2 | 2026-01-26 | 2 chunk_ids invalides corrigés |
| 7.4.3 | 2026-01-26 | 101 corrections via article_reference |

## Références

- HotpotQA: single-hop, multi-hop taxonomy
- SQuAD 2.0: answerable/unanswerable
- Natural Questions: short/long answer
- MS MARCO: extractive/abstractive
