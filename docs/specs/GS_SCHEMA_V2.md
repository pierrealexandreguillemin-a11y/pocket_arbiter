# Gold Standard Schema v2.0

> **Document ID**: SPEC-GS-SCH-002
> **ISO Reference**: ISO 42001 A.6.2.2 (Provenance), ISO 29119-3 (Test Data)
> **Version**: 2.0
> **Date**: 2026-02-03
> **Statut**: Approuve
> **Auteur**: Claude Opus 4.5

---

## 1. Objet

Ce document definit le schema JSON restructure pour les questions du Gold Standard.
L'objectif est d'organiser logiquement les 46+ champs en groupes fonctionnels tout en conservant 100% des informations pour le traitement futur et le suivi.

---

## 2. Justification de la Restructuration

### 2.1 Problemes du Schema v1

| Probleme | Impact |
|----------|--------|
| `metadata` = fourre-tout de 22+ champs | Difficile a maintenir |
| Champs melanges (contenu, process, audit) | Confusion semantique |
| Redondances non documentees | Risque d'incoherence |

### 2.2 Principes v2

| Principe | Implementation |
|----------|----------------|
| Separation des responsabilites | 8 groupes fonctionnels |
| Conservation 100% des donnees | 0 champ supprime |
| Coherence des redondances | Contraintes documentees |

---

## 3. Schema JSON v2.0

### 3.1 Structure Generale

```json
{
  "id": "string",
  "legacy_id": "string",
  "content": { },
  "mcq": { },
  "provenance": { },
  "classification": { },
  "validation": { },
  "processing": { },
  "audit": { }
}
```

### 3.2 Groupe: Racine (2 champs)

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `id` | string | oui | Format `{corpus}:{source}:{category}:{seq}:{hash}` |
| `legacy_id` | string | oui | Format `FR-ANN-UV{X}-{N}` pour retrocompatibilite |

### 3.3 Groupe: content (3 champs)

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `question` | string | oui | Question reformulee (finit par `?`, >= 10 chars) |
| `expected_answer` | string | oui | Reponse attendue (> 5 chars, derivable du chunk) |
| `is_impossible` | boolean | oui | `false` pour questions answerable |

### 3.4 Groupe: mcq (5 champs)

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `original_question` | string | oui | Question MCQ originale des annales |
| `choices` | object | oui | `{"A": "...", "B": "...", "C": "...", "D": "..."}` |
| `mcq_answer` | string | oui | Lettre correcte (A/B/C/D) |
| `correct_answer` | string | oui | Texte de `choices[mcq_answer]` |
| `original_answer` | string | oui | Reponse originale des annales |

**Contrainte de coherence**: `correct_answer` == `choices[mcq_answer]`

### 3.5 Groupe: provenance (7 champs) - ISO 42001 A.6.2.2

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `chunk_id` | string | oui | ID du chunk source dans corpus_mode_b_fr.db |
| `docs` | array | oui | Documents sources (ex: `["LA-octobre2025.pdf"]`) |
| `pages` | array | oui | Pages sources (ex: `[10]`) |
| `article_reference` | string | oui | Reference article (ex: "LA Art. 2.1") |
| `answer_explanation` | string | oui | Corrige detaille expliquant POURQUOI |
| `annales_source` | object | oui | Voir 3.5.1 |

**Contraintes de coherence**:
- `docs[0]` == partie document de `chunk_id`
- `pages[0]` == partie page de `chunk_id`

#### 3.5.1 Sous-groupe: annales_source (4 champs)

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `session` | string | oui | dec2024/dec2023/jun2024/jun2023 |
| `uv` | string | oui | clubs/regles/organisation/travaux |
| `question_num` | integer | oui | Numero dans l'UV |
| `success_rate` | float | oui | Taux de reussite [0, 1] |

### 3.6 Groupe: classification (8 champs)

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `category` | string | oui | Categorie thematique |
| `keywords` | array | oui | Mots-cles pertinents |
| `difficulty` | float | oui | Difficulte [0, 1] |
| `question_type` | string | oui | factual/procedural/scenario/comparative |
| `cognitive_level` | string | oui | Remember/Understand/Apply/Analyze (Bloom) |
| `reasoning_type` | string | oui | single-hop/multi-hop/temporal |
| `reasoning_class` | string | oui | fact_single/summary/arithmetic/reasoning |
| `answer_type` | string | oui | multiple_choice/extractive/etc. |

### 3.7 Groupe: validation (7 champs) - ISO 29119

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `status` | string | oui | VALIDATED/PENDING/NEEDS_REAUDIT |
| `method` | string | oui | manual_llm_as_judge/annales_official |
| `reviewer` | string | oui | Identifiant du revieweur |
| `answer_current` | boolean | oui | Reponse toujours valide |
| `verified_date` | string | oui | Date ISO de verification |
| `pages_verified` | boolean | oui | Pages verifiees contre PDF |
| `batch` | string | oui | Identifiant du batch d'audit |

### 3.8 Groupe: processing (7 champs)

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `chunk_match_score` | integer | oui | Score de matching (100 si manuel) |
| `chunk_match_method` | string | oui | manual_by_design/auto |
| `reasoning_class_method` | string | oui | inferred/explicit |
| `triplet_ready` | boolean | oui | Pret pour generation triplets |
| `extraction_flags` | array | oui | Flags d'extraction ([] si OK) |
| `answer_source` | string | oui | choice/existing |
| `quality_score` | float | oui | Score qualite [0, 1] |

### 3.9 Groupe: audit (3 champs)

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `history` | string | oui | Historique des modifications |
| `qat_revalidation` | object | non | Donnees de revalidation QAT |
| `requires_inference` | boolean | non | Necessite inference/calcul |

---

## 4. Mapping v1 -> v2

| Champ v1 | Chemin v2 |
|----------|-----------|
| `id` | `id` |
| `legacy_id` | `legacy_id` |
| `question` | `content.question` |
| `expected_answer` | `content.expected_answer` |
| `is_impossible` | `content.is_impossible` |
| `expected_chunk_id` | `provenance.chunk_id` |
| `expected_docs` | `provenance.docs` |
| `expected_pages` | `provenance.pages` |
| `category` | `classification.category` |
| `keywords` | `classification.keywords` |
| `validation.*` | `validation.*` |
| `audit` | `audit.history` |
| `metadata.answer_type` | `classification.answer_type` |
| `metadata.reasoning_type` | `classification.reasoning_type` |
| `metadata.cognitive_level` | `classification.cognitive_level` |
| `metadata.article_reference` | `provenance.article_reference` |
| `metadata.difficulty` | `classification.difficulty` |
| `metadata.annales_source` | `provenance.annales_source` |
| `metadata.question_type` | `classification.question_type` |
| `metadata.choices` | `mcq.choices` |
| `metadata.mcq_answer` | `mcq.mcq_answer` |
| `metadata.quality_score` | `processing.quality_score` |
| `metadata.chunk_match_score` | `processing.chunk_match_score` |
| `metadata.chunk_match_method` | `processing.chunk_match_method` |
| `metadata.reasoning_class` | `classification.reasoning_class` |
| `metadata.reasoning_class_method` | `processing.reasoning_class_method` |
| `metadata.triplet_ready` | `processing.triplet_ready` |
| `metadata.extraction_flags` | `processing.extraction_flags` |
| `metadata.answer_explanation` | `provenance.answer_explanation` |
| `metadata.answer_source` | `processing.answer_source` |
| `metadata.original_answer` | `mcq.original_answer` |
| `metadata.correct_answer` | `mcq.correct_answer` |
| `metadata.original_question` | `mcq.original_question` |
| `metadata.qat_revalidation` | `audit.qat_revalidation` |
| `metadata.requires_inference` | `audit.requires_inference` |

---

## 5. Contraintes de Coherence

| Contrainte | Expression | Verification |
|------------|------------|--------------|
| C1 | `mcq.correct_answer == mcq.choices[mcq.mcq_answer]` | Obligatoire |
| C2 | `provenance.docs[0]` dans `provenance.chunk_id` | Obligatoire |
| C3 | `provenance.pages[0]` dans `provenance.chunk_id` | Obligatoire |
| C4 | `mcq.original_question` == annales.text | Obligatoire |
| C5 | `mcq.choices` == annales.choices | Obligatoire |
| C6 | `content.question` finit par `?` | Obligatoire |
| C7 | `content.expected_answer` > 5 chars | Obligatoire |
| C8 | `classification.difficulty` in [0, 1] | Obligatoire |

---

## 6. Exemple Complet

```json
{
  "id": "ffe:annales:clubs:001:55d409b5",
  "legacy_id": "FR-ANN-UVC-001",

  "content": {
    "question": "Quelle tache ne fait pas partie des missions de l'arbitre ?",
    "expected_answer": "La mise en place des jeux, pendules, cavaliers, numeros de tables et feuilles de parties est faite par l'equipe d'organisation en suivant les consignes de l'arbitre, donc mettre en place les numeros de table et cavaliers n'est pas une mission de l'arbitre.",
    "is_impossible": false
  },

  "mcq": {
    "original_question": "Quelle proposition parmi les suivantes ne correspond pas a une des missions de l'arbitre ?",
    "choices": {
      "A": "S'assurer du confort des joueurs.",
      "B": "Valider les resultats d'une ronde.",
      "C": "Mettre en place les numeros de table et cavaliers.",
      "D": "Verifier le bon fonctionnement des pendules."
    },
    "mcq_answer": "C",
    "correct_answer": "Mettre en place les numeros de table et cavaliers.",
    "original_answer": "Mettre en place les numeros de table et cavaliers."
  },

  "provenance": {
    "chunk_id": "LA-octobre2025.pdf-p010-parent024-child00",
    "docs": ["LA-octobre2025.pdf"],
    "pages": [10],
    "article_reference": "LA - Chapitre 1.2 : Missions de l'arbitre - A : Aupres de l'echiquier : Art. 2.1 et 2.2",
    "answer_explanation": "LA Chapitre 8.1 Materiel: La mise en place des jeux, pendules, cavaliers, numeros de tables est faite par l'equipe d'organisation, pas par l'arbitre. Seule la verification du bon fonctionnement des pendules est une mission de l'arbitre.",
    "annales_source": {
      "session": "dec2024",
      "uv": "clubs",
      "question_num": 1,
      "success_rate": 0.85
    }
  },

  "classification": {
    "category": "arbitrage",
    "keywords": ["arbitre", "missions", "organisation", "materiel"],
    "difficulty": 0.15,
    "question_type": "factual",
    "cognitive_level": "Remember",
    "reasoning_type": "multi-hop",
    "reasoning_class": "summary",
    "answer_type": "multiple_choice"
  },

  "validation": {
    "status": "VALIDATED",
    "method": "manual_llm_as_judge",
    "reviewer": "claude_opus_4.5",
    "answer_current": true,
    "verified_date": "2026-02-03",
    "pages_verified": true,
    "batch": "batch_001"
  },

  "processing": {
    "chunk_match_score": 100,
    "chunk_match_method": "manual_by_design",
    "reasoning_class_method": "inferred",
    "triplet_ready": true,
    "extraction_flags": [],
    "answer_source": "choice",
    "quality_score": 1.0
  },

  "audit": {
    "history": "[BATCH_001] Schema v2 migration 2026-02-03",
    "qat_revalidation": null,
    "requires_inference": false
  }
}
```

---

## 7. Migration

### 7.1 Script de Migration

Fichier: `scripts/evaluation/migrate_gs_schema_v2.py`

### 7.2 Validation Post-Migration

- [ ] 420 questions migrees
- [ ] 0 champ perdu
- [ ] 8 contraintes de coherence verifiees
- [ ] Tests unitaires passent

---

## 8. Historique

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 2.0 | 2026-02-03 | Claude Opus 4.5 | Creation - restructuration schema en 8 groupes |
