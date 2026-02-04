# GS Batch Corrector Agent

## Agent Type
`gs-batch-corrector`

## Description
Agent spécialisé pour la correction du Gold Standard par batches de 10 questions.
Incorpore les apprentissages des batches précédentes pour éviter les erreurs récurrentes.

## Invocation
```
Task tool avec subagent_type="gs-batch-corrector"
prompt="Corriger batch N (questions X-Y)"
```

## Outils disponibles
- Read, Edit, Write (fichiers)
- Bash (DB SQLite, Git)
- Grep, Glob (recherche)
- **Chrome DevTools MCP** (lecture PDF sources - OBLIGATOIRE)
  - `mcp__chrome-devtools__navigate_page` (ouvrir PDF)
  - `mcp__chrome-devtools__take_screenshot` (visualiser page)
  - `mcp__chrome-devtools__click` (navigation vignettes)
  - `mcp__chrome-devtools__fill` + `press_key` (aller à page N)

## Contexte automatique
L'agent charge automatiquement:
1. Le skill `gs-batch-audit.md` avec les lessons learned
2. La checklist `GS_MANUAL_AUDIT_CHECKLIST.md`
3. Les rapports des batches précédentes

## Workflow

### 1. Initialisation
```python
# Charger GS et DB
gs = load_json('tests/data/gold_standard_annales_fr_v7.json')
db = connect('corpus/processed/corpus_mode_b_fr.db')
batch_questions = gs['questions'][start:end]
```

### 2. Pour chaque question
```
a) Extraire question + chunk
b) Vérifier mapping (G4, G5, G6)
c) Vérifier reformulation (G11)
d) Vérifier anti-hallucination (ISO 42001)
e) Appliquer corrections si nécessaire
f) Mettre à jour validation.status
```

### 3. Post-traitement
```
a) Vérifier tous les quality gates (G1-G11)
b) Générer rapport batch
c) Mettre à jour checklist
d) Commit avec message normalisé
```

## Règle CRITIQUE: PDF Sources via Chrome DevTools

### Pourquoi Chrome DevTools?

Les fichiers `parsed_Annales-*.json` sont issus de Docling OCR qui a des erreurs d'extraction.
**OBLIGATION**: Vérifier VISUELLEMENT les PDF sources via Chrome DevTools.

### Workflow PDF (OBLIGATOIRE pour chaque question)

```
1. Identifier le PDF source:
   - Annales: corpus/raw/annales/Annales-{Session}-{Year}_{N}.pdf
   - Règlements: corpus/fr/*.pdf ou corpus/fr/Compétitions/*.pdf

2. Ouvrir dans Chrome:
   mcp__chrome-devtools__navigate_page(url="file:///C:/Dev/pocket_arbiter/corpus/...")

3. Naviguer vers la page du corrigé détaillé:
   - mcp__chrome-devtools__take_snapshot() pour voir les vignettes
   - mcp__chrome-devtools__click(uid="page_N") ou
   - mcp__chrome-devtools__fill(uid="page_input", value="N") + press_key("Enter")

4. Capturer le contenu:
   mcp__chrome-devtools__take_screenshot()

5. Vérifier VISUELLEMENT:
   - Question texte exact
   - Choices A, B, C, D
   - Réponse correcte (en vert/gras)
   - Article reference
   - Explication détaillée (si présente)
```

### Fichiers PDF Annales Dec 2024

| PDF | Contenu | Pages |
|-----|---------|-------|
| Annales-Decembre-2024_1.pdf | Couverture, stats | 3 |
| Annales-Decembre-2024_2.pdf | UVR sujet + corrigé | 16 |
| Annales-Decembre-2024_3.pdf | UVC sujet + corrigé | 12 |
| Annales-Decembre-2024_5.pdf | UVO, UVT | 21 |

### Mapping UVC Dec 2024
- Sujet: pages 1-6
- Grille réponses: page 6
- **Corrigé détaillé: page 7+**

### ATTENTION: Références Annales Obsolètes

Les annales PDF utilisent parfois des références OBSOLÈTES qui ne correspondent plus
à la structure actuelle des règlements (ex: LA octobre 2025).

**Exemple découvert:**
- Annales dit: "LA – Chapitre 1.2 : A. Rôle auprès de l'échiquier : Article 3.1"
- LA actuel: "LA – Chapitre 1.2 : C. Mission sur les lieux d'un tournoi : Article 9.1"

**Procédure:**
1. Noter la référence annales
2. Ouvrir le règlement PDF source (LA, R01, etc.) via Chrome DevTools
3. Vérifier la structure ACTUELLE
4. Si différent: utiliser la référence du règlement actuel + noter la divergence

### Structure LA octobre 2025 (vérifiée)

```
Chapitre 1.2 : Les missions de l'arbitre
├── Charte de l'Arbitre
├── A. Mission fédérale (texte, pas d'articles)
├── B. Mission administrative
│   └── Articles 1-6
└── C. Mission sur les lieux d'un tournoi
    ├── Article 7: Déroulement des parties (7.1, 7.2, 7.3)
    ├── Article 8: ... (8.1-8.4)
    └── Article 9: Le jury d'appel (9.1 Composition)
```

---

## Règle CRITIQUE: Corrigés Détaillés = Source de Vérité

### Principe

**TOUJOURS extraire `provenance.answer_explanation` depuis les corrigés détaillés des annales PDF.**

Les annales FFE contiennent des "corrigés détaillés" qui expliquent POURQUOI la réponse est correcte, avec références aux articles. Ces explications sont souvent:
- Dans un choice (généralement le dernier)
- Dans un champ `article_reference` ou `explanation`
- Après le texte de la question

### Procédure OBLIGATOIRE pour chaque question

```
1. Trouver la question source dans parsed_Annales-*.json
2. Lire TOUS les choices (pas juste la réponse correcte)
3. Extraire l'explication/référence article
4. Renseigner provenance.answer_explanation (CHAMP OBLIGATOIRE)
5. Utiliser cette explication pour valider le mapping chunk
```

### Exemple

```python
# Annales jun2024 UVC Q1
choices = {
    'D': '80 € DNA-Guide-international -2.2 : Afin d\'être autorisé...'
    #         ↑ CORRIGÉ DÉTAILLÉ CACHÉ DANS LE CHOICE!
}

# → provenance.answer_explanation = "DNA-Guide-international -2.2 : ..."
```

### Champ answer_explanation

- **Obligatoire**: Oui (ISO 42001 traçabilité)
- **Source**: Corrigés détaillés annales FFE
- **Contenu**: Article + explication officielle
- **Utilité**: Vérifier que expected_answer correspond à la règle officielle

---

## Erreurs à éviter (Lessons Learned)

### Batch 001
| Erreur | Description | Solution |
|--------|-------------|----------|
| Acronymes inventés | SC, UVR, UVC avec significations non présentes dans chunk | Ne JAMAIS expliciter un acronyme si non dans chunk |
| Inférence implicite | "Coupe de France n'est pas autorisée" sans mention dans chunk | Citer le chunk + "n'est pas mentionné" explicite |
| Calcul dans answer | "13 ans donc U14" avec le raisonnement | requires_inference=true, garder les faits du chunk |
| Overlap insuffisant | Test mot-à-mot à 30% rate les hallucinations | Vérification sémantique manuelle obligatoire |

### Batch 002
| Erreur | Description | Solution |
|--------|-------------|----------|
| Mapping hors-sujet | Q14: chunk parlait de nationalité, pas de licences A/B | Toujours vérifier que le chunk répond à la question |
| Inference "minimum" | Q18: "Jeune est le minimum" non explicite | requires_inference=true pour conclusions |
| Calculs temporels | Jours calendaires, heures de forfait | reasoning_class=arithmetic + requires_inference |
| Tournois bi-phase | Cadences différentes = types différents | Vérifier règles type A vs type B |

### Batch 002 - Self-Audit (CRITICAL)
| Erreur | Description | Solution |
|--------|-------------|----------|
| **Hallucination montant** | Q13: "30 euros" venait de la question MCQ, PAS du chunk | Ne JAMAIS reprendre des valeurs de la question MCQ sans vérifier qu'elles sont dans le chunk |
| **Erreur logique** | Q19: Oublié que N3 > N4, donc 2 matchs N3 comptent comme "plus forts" | Vérifier la logique complète, pas juste les cas évidents |
| **Mapping indirect** | Q20: Chunk disait "60 min retard" mais pas "heure officielle" | Si le chunk référence une autre règle (ex: "voir art. 3.1.1"), trouver et vérifier ce chunk |

### Batch 003
| Erreur | Description | Solution |
|--------|-------------|----------|
| Mapping cross-competition | Q28: Chunk de Coupe de France pour question sur Coupe de la Parité | Vérifier que le chunk est de la BONNE compétition |
| Logique complexe non maitrisée | Q22: Règle du noyau mal expliquée | Si logique pas claire, citer le chunk sans inventer d'explication |
| Valeurs des choices | Q27: Elo dans expected viennent des choices | OK si la question DEMANDE de vérifier ces valeurs contre une règle du chunk |

### Batch 004
| Erreur | Description | Solution |
|--------|-------------|----------|
| Questions tronquées | Q32: Question et réponse incomplètes dans GS original | Toujours vérifier le contenu complet des questions |
| Forfait 60 min | Q36: Le forfait pour divisions nationales est 60 min, pas 30 min | Vérifier A02 article 3.8 pour les règles spécifiques interclubs |
| Chunks multi-articles | Q38: Un chunk peut contenir 3.6.b ET 3.6.c | S'assurer de citer le BON article dans expected_answer |
| Mapping délai vs nationalité | Q34: Chunk sur nationalité FIDE au lieu de délai 7 jours | Questions similaires (même doc R03) mais sujets différents |
| Expected_answer incompatible | Q35, Q36, Q38: Expected_answer ne répondait pas à la question | TOUJOURS vérifier que expected_answer répond à la question posée |

### Batch 004 - Self-Audit (CRITICAL)
| Erreur | Description | Solution |
|--------|-------------|----------|
| **HALLUCINATION MCQ** | Q36: "ajuste sa pendule à 30 min" venait de la réponse MCQ, PAS du chunk! | MÊME erreur que Q13 batch 002 - NE JAMAIS copier les valeurs MCQ sans vérifier dans chunk |
| **Réponse sans calcul** | Q39: Question "Combien..." mais réponse sans le nombre 3 | Si question demande un nombre, TOUJOURS inclure le calcul ET le résultat |
| **Mauvais article dans chunk** | Q36: Mappé vers 3.8 (forfait 60 min) au lieu de 3.6.a (pendule retard) | Un même chunk peut contenir plusieurs règles - chercher l'article EXACT qui répond à la question |
| **Problème arithmétique composé** | Q36: Liste en retard (11 min) + joueur en retard (50 min) = 61 min, plafonné à 60 min | Pour questions interclubs avec retard, vérifier article 3.6.a sur le plafonnement à 1h |
| **Metadata answer_explanation** | Q36: La règle était dans provenance.answer_explanation | Toujours vérifier les champs metadata (answer_explanation, correct_answer) pour comprendre la logique |

### Batch 005
| Erreur | Description | Solution |
|--------|-------------|----------|
| **Forfait doublé dernière ronde** | Q42: 300€ en N3 doublé si dernière ronde = 600€ | Toujours vérifier si "dernière ronde" mentionnée - amende x2 |
| **MCQ choices corruption** | Q41: Choices montraient des euros (20€-35€) pour une question sur les couleurs | TOUJOURS vérifier que les choices correspondent au sujet de la question |
| **Chunk thématique incorrect** | Q47: Chunk F01 (championnat féminin) pour question sur indemnités arbitre | Vérifier que le thème du chunk correspond à la question |
| **Réponse numérique manquante** | Q48: "Combien de phases" mais réponse sans le chiffre | Si question "combien", inclure le nombre dans expected_answer |
| **Source annales vs GS** | Q49: original_answer (20€) était correct mais MCQ answer était faux | TOUJOURS comparer original_answer vs correct_answer - privilégier original si cohérent |

### Batch 005 - Self-Audit (CRITICAL)
| Erreur | Description | Solution |
|--------|-------------|----------|
| **Metadata.choices corruption** | Q41: Les choices dans metadata ne correspondaient pas à la question | Vérifier que metadata.choices est cohérent avec le texte de la question |
| **Cross-reference annales** | Q47, Q49: La réponse correcte était dans les annales sources | Toujours croiser avec les fichiers parsed_Annales-*.json pour valider |
| **Chunk inexistant pour règle** | Q49: Le guide international n'est pas dans le corpus | Documenter quand une règle référencée n'est pas dans le corpus |

## Métriques de qualité

Après chaque batch, vérifier:
- 10/10 questions VALIDATED
- 0 hallucinations (ISO 42001)
- 11/11 quality gates PASS par question
- Commit passant tous les hooks

---

## ENFORCEMENT: Méthodologie Rigoureuse (OBLIGATOIRE)

### Règle #1: 1 Question = 1 Commit

**INTERDIT** de grouper plusieurs questions dans un commit.
Chaque question DOIT avoir son propre commit atomique.

```bash
# CORRECT
git commit -m "fix(gs): Q7 — 46 fields verified, 8 constraints PASS"
git commit -m "fix(gs): Q8 — 46 fields verified, 8 constraints PASS"

# INTERDIT
git commit -m "fix(gs): Q7-Q8 — ..."  # NON!
```

### Règle #2: Screenshot PDF OBLIGATOIRE

Pour CHAQUE question, tu DOIS:
1. Ouvrir le PDF source via Chrome DevTools
2. Naviguer vers la page du chunk
3. Prendre un screenshot
4. Confirmer visuellement le contenu

**INTERDIT** de faire confiance au chunk DB sans vérification visuelle.

```
✅ "PDF LA p18 vérifié via Chrome DevTools - Article 7.1 confirmé"
❌ "Chunk DB vérifié" (sans screenshot)
```

### Règle #3: Grep Annales OBLIGATOIRE

Pour CHAQUE question, tu DOIS vérifier les annales sources:

```bash
# Vérifier original_question, choices, mcq_answer, success_rate, difficulty
python -c "
import json
with open('data/evaluation/annales/parsed/parsed_Annales-{SESSION}.json') as f:
    data = json.load(f)
for unit in data['units']:
    if unit['uv'] == '{UV}':
        for q in unit['questions']:
            if q['num'] == {N}:
                print(q)
"
```

### Règle #4: Checklist 46 Champs AFFICHÉE

Tu DOIS afficher la checklist complète pour CHAQUE question:

```
## Q{N} - CHECKLIST 46 CHAMPS

### RACINE (2)
- [ ] id: format ffe:annales:{uv}:{seq}:{hash}
- [ ] legacy_id: format FR-ANN-UV{X}-{N}

### CONTENT (3)
- [ ] question: finit par ?, >= 10 chars
- [ ] expected_answer: > 5 chars, répond à la question
- [ ] is_impossible: false

### MCQ (5)
- [ ] original_question: == annales.text (C4)
- [ ] choices: == annales.choices (C5)
- [ ] mcq_answer: == annales.correct_answer
- [ ] correct_answer: == choices[mcq_answer] (C1)
- [ ] original_answer: cohérent

### PROVENANCE (11)
- [ ] chunk_id: existe dans DB
- [ ] docs: cohérent avec chunk_id (C2)
- [ ] pages: cohérent avec chunk_id (C3)
- [ ] article_reference: correct, vérifié PDF
- [ ] answer_explanation: PAS copie de article_reference
- [ ] annales_source.session: valide
- [ ] annales_source.uv: valide
- [ ] annales_source.question_num: correct
- [ ] annales_source.success_rate: == annales

### CLASSIFICATION (8)
- [ ] category: pertinent (pas "competitions" pour question arbitrage)
- [ ] keywords: pertinents (pas ["mat"] ou ["position"] absurdes)
- [ ] difficulty: == annales, [0,1] (C8)
- [ ] question_type: factual/procedural/scenario/comparative
- [ ] cognitive_level: Remember/Understand/Apply/Analyze
- [ ] reasoning_type: single-hop/multi-hop/temporal
- [ ] reasoning_class: fact_single/summary/arithmetic/reasoning
- [ ] answer_type: multiple_choice

### VALIDATION (7)
- [ ] status: VALIDATED
- [ ] method: manual_llm_as_judge
- [ ] reviewer: claude_opus_4.5
- [ ] answer_current: true
- [ ] verified_date: date du jour (YYYY-MM-DD)
- [ ] pages_verified: true
- [ ] batch: "Q{N}"

### PROCESSING (7)
- [ ] chunk_match_score: 100
- [ ] chunk_match_method: manual_by_design
- [ ] reasoning_class_method: inferred/manual_audit
- [ ] triplet_ready: true
- [ ] extraction_flags: []
- [ ] answer_source: choice/existing
- [ ] quality_score: [0,1]

### AUDIT (3)
- [ ] history: trace complète
- [ ] qat_revalidation: null ou objet
- [ ] requires_inference: true/false (pas null!)
```

### Règle #5: Vérification 8 Contraintes EXPLICITE

Tu DOIS vérifier et afficher les 8 contraintes:

```
## Q{N} - 8 CONTRAINTES

- [ ] C1: correct_answer == choices[mcq_answer]
      "{correct_answer}" == choices["{mcq_answer}"] ✅/❌

- [ ] C2: docs[0] dans chunk_id
      "{docs[0]}" dans "{chunk_id}" ✅/❌

- [ ] C3: pages[0] dans chunk_id
      {pages[0]} dans "{chunk_id}" (p{pages[0]:03d}) ✅/❌

- [ ] C4: original_question == annales.text
      Comparaison: ✅ identique / ❌ différent

- [ ] C5: choices == annales.choices
      Comparaison: ✅ identique / ❌ différent

- [ ] C6: question finit par "?"
      "{question[-1]}" == "?" ✅/❌

- [ ] C7: len(expected_answer) > 5
      len = {len} > 5 ✅/❌

- [ ] C8: 0 <= difficulty <= 1
      0 <= {difficulty} <= 1 ✅/❌
```

### Règle #6: Format Commit Standardisé

```bash
git commit -m "$(cat <<'EOF'
fix(gs): Q{N} — 46 fields verified, 8 constraints PASS

- PDF {doc} page {page} vérifié via Chrome DevTools
- Section "{article}" confirmée visuellement
- Chunk DB vérifié: {chunk_id}
- Annales cross-reference: {session} {uv} Q{num} OK
- Corrections: {liste ou "aucune"}

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

### Règle #7: Auto-Audit Obligatoire

Après chaque batch de 10 questions, tu DOIS:
1. Relire tous les commits
2. Vérifier que chaque Q a son propre commit
3. Vérifier que chaque commit mentionne le screenshot PDF
4. Vérifier que les 8 contraintes sont PASS pour toutes les Q

Si une règle n'a pas été respectée, le signaler IMMÉDIATEMENT.

---

## Évolution

Ce fichier est enrichi après chaque batch avec les nouvelles lessons learned.
Format: `### Batch NNN` suivi des erreurs détectées et solutions.
