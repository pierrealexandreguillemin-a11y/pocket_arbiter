# Analyse Approfondie des Echecs de Recall - 2026-01-20

## Executive Summary

**14 questions en echec** sur 134 (89.55% de succes sans tolerance, 91.17% avec tolerance ±2)

### Classification des causes racines

| Cause | Questions | % |
|-------|-----------|---|
| **Mismatch terminologique** | Q77, Q94 | 14% |
| **Langage oral/informel** | Q95, Q98, Q103 | 21% |
| **Cross-chapter/multi-doc** | Q85, Q86, Q132 | 21% |
| **Abreviations** | Q98, Q119 | 14% |
| **Semantic drift** | Q87, Q121 | 14% |
| **Combinaison termes specifiques** | Q99, Q125, Q127 | 21% |

---

## Analyse Detaillee par Question

### 1. FR-Q77 (Recall: 0%)

**Question**: "Un joueur qui n'a joue aucune partie pendant **18 mois** voit-il son classement Elo Standard supprime?"

**Expected**: pages [183, 188]
**Retrieved**: pages [4, 3, 227, 2, 225] (table des matieres, intro)

**Cause racine**: **MISMATCH TERMINOLOGIQUE**

Le corpus utilise "**une periode d'un an**" (page 183, Art. 7.2.2):
```
"Un joueur ou une joueuse est considere comme inactif s'il/si elle
ne joue aucune partie prise en compte pour le classement dans
une periode d'un an."
```

La question utilise "18 mois" qui n'existe PAS dans le corpus.

**Solution proposee**:
- Query expansion: `18 mois` → `un an OR 12 mois OR periode d'inactivite`
- Synonyme mapping pre-retrieval

---

### 2. FR-Q85 (Recall: 0%)

**Question**: "Dans quel **delai** un arbitre doit-il transmettre le **rapport** d'un match individuel a la **DNA**?"

**Expected**: pages [165]
**Retrieved**: pages [25, 174, 172] (reglements divers)

**Cause racine**: **CONTENU ABSENT OU MULTI-DOC**

Le terme "DNA" (Direction Nationale de l'Arbitrage) apparait dans l'en-tete de chaque page mais le **delai de transmission** specifique peut etre dans un reglement FFE separe, pas dans le LA-octobre2025.pdf.

**Analyse du chunk p165**:
- Page 165 = Chapitre 5.1 "Gestion des matchs"
- Contient procedures generales mais peut-etre pas le delai exact

**Solution proposee**:
- Verifier si le delai est dans un autre document FFE (R01, R02)
- Si absent du corpus: marquer comme "hors scope corpus"

---

### 3. FR-Q86 (Recall: 0%)

**Question**: "Quelles sont les etapes que l'arbitre doit suivre pour la gestion administrative d'un tournoi en cas d'**absence d'un joueur a la premiere ronde**?"

**Expected**: pages [169, 170, 171]
**Retrieved**: pages [74, 124, 84, 155, 177]

**Cause racine**: **FORMULATION ADMINISTRATIVE + MULTI-PAGE**

Le contenu est reparti sur 3 pages (169-171) pour la "gestion de tournoi". L'embedding de la question ne match pas le vocabulaire administratif du corpus.

**Analyse**:
- "Absence premiere ronde" → corpus utilise "forfait", "non-presentation", "exclusion"
- Le retrieval trouve des pages sur l'appariement suisse (124) qui mentionne les absences

**Solution proposee**:
- Query expansion: `absence → forfait OR non-presentation`
- Hybrid search (BM25 + vector) pour capter "premiere ronde"

---

### 4. FR-Q87 (Recall: 0%)

**Question**: "Pour obtenir un **premier classement Elo Standard FIDE**, quelles sont les conditions minimales?"

**Expected**: pages [182, 183]
**Retrieved**: pages [4, 197, 194, 186, 9]

**Cause racine**: **SEMANTIC DRIFT vers pages populaires**

Les pages 4, 9 sont la table des matieres/introduction - tres "centrales" dans l'embedding space.
Page 197 parle de titres, pas de classement initial.

**Contenu attendu (p183 Art. 7.1.4)**:
```
"Un classement pour un joueur entrant dans la liste sera publie
lorsqu'il sera base sur au moins 5 parties jouees contre des
adversaires classes."
```

**Solution proposee**:
- Re-ranking avec cross-encoder
- Negative examples: exclure pages table des matieres/intro du retrieval
- Source filtering: forcer Chapitre 6.1 si question contient "classement FIDE"

---

### 5. FR-Q94 (Recall: 0%)

**Question**: "Si un joueur joue pas du tout pendant **18 mois**, son Elo finit par **disparaitre** ou pas?"

**Expected**: pages [183]
**Retrieved**: pages [4, 3, 186, 186, 1]

**Cause racine**: **MISMATCH TERMINOLOGIQUE + LANGAGE ORAL**

Memes problemes que Q77:
- "18 mois" → corpus dit "un an"
- "disparaitre" → corpus dit "inactive" ou "non classe"
- Formulation orale ("joue pas", "ou pas")

**Solution proposee**:
- Meme que Q77: query expansion + synonyme mapping
- Pre-processing: normaliser langage oral → formel

---

### 6. FR-Q95 (Recall: 0%)

**Question**: "C'est **pas possible** d'avoir un Elo sans faire au moins **5 parties**, si?"

**Expected**: pages [182, 183]
**Retrieved**: pages [4, 3, 1, 194, 74]

**Cause racine**: **NEGATION + LANGAGE ORAL**

- Negation "pas possible" confuse l'embedding
- Formulation SMS-like ("si?")
- Le chiffre "5" est correct mais noyé dans formulation informelle

**Analyse embedding**:
La negation "pas possible" oriente l'embedding vers du contenu "negatif" (erreurs, exclusions) plutot que vers les conditions d'obtention.

**Solution proposee**:
- Query rewriting: `"pas possible sans 5 parties"` → `"minimum 5 parties requis"`
- Detecter et normaliser les negations avant embedding

---

### 7. FR-Q98 (Recall: 0%)

**Question**: "Est-ce qu'on peut **sauter le CM** pour passer directement **FM**, ou c'est oblige?"

**Expected**: pages [196, 197, 198]
**Retrieved**: pages [193, 2, 192, 2, 10]

**Cause racine**: **ABBREVIATIONS + NEGATION**

- "CM" = Candidat Maitre, "FM" = Maitre FIDE
- Le corpus utilise les noms complets ET les abreviations
- "Sauter" n'est pas un terme technique du reglement

**Analyse corpus (p192-194)**:
```
"Mixtes : Grand-Maitre (GM), Maitre International (MI),
Maitre de la FIDE (MF), Candidat Maitre (CM)"
```
Les pages 192-194 parlent des conditions, pas de "sauter" des etapes.

**Solution proposee**:
- Query expansion: `CM → Candidat Maitre`, `FM → Maitre FIDE`
- Reformulation: `sauter → progression directe OR sans passer par`

---

### 8. FR-Q99 (Recall: 67%)

**Question**: "Une **norme de titre**, elle reste valide **combien de temps** si j'attends pour demander?"

**Expected**: pages [200, 201, 202]
**Retrieved**: pages [167, 197, 199, 18, 205]

**Cause racine**: **PARTIAL MATCH - PAGES ADJACENTES**

Page 199 est dans le top-5 (tolerance ±2 → page 200 matchee).
Le retrieval est proche mais pas exact.

**Analyse**:
- "Validite norme" trouve des pages proches (197, 199)
- La formulation orale ("elle reste", "j'attends") dilue le signal

**Solution proposee**:
- Re-ranking ameliorerait le score
- Query: `norme titre validite duree` (sans mots oraux)

---

### 9. FR-Q103 (Recall: 0%)

**Question**: "Mon joueur **arrive pas** a la premiere ronde, **je fais quoi** comme arbitre?"

**Expected**: pages [169, 170, 171]
**Retrieved**: pages [10, 11, 36, 92, 46]

**Cause racine**: **LANGAGE ULTRA-INFORMEL SMS-LIKE**

- "Arrive pas" au lieu de "est absent"
- "Je fais quoi" au lieu de "quelle procedure"
- L'embedding ne capture pas l'intention technique

**Analyse**:
Meme question que Q86 mais formulation encore plus informelle.
Le retrieval trouve des pages d'introduction (10, 11) car le langage informel matche mieux le contenu general.

**Solution proposee**:
- Query rewriting obligatoire pour langage SMS
- Classifier de formalite: si informal → reformuler avant embed

---

### 10. FR-Q119 (Recall: 25%)

**Question**: "Pour un tournoi **toutes rondes** avec 7 equipes, combien de rondes faut-il programmer?"

**Expected**: pages [101, 102, 103, 104]
**Retrieved**: pages [107, 118, 106, 119, 115]

**Cause racine**: **PAGES ADJACENTES - CHAPITRE BOUNDARY**

Les pages 106, 107, 115, 118, 119 sont dans le meme Chapitre 3 (Systemes d'appariement) mais pas exactement les bonnes.

**Analyse**:
- Pages 101-104 = Chapitre 3.1 "Tournois toutes-rondes"
- Pages 107+ = Sections suivantes du meme chapitre
- L'embedding trouve le bon chapitre mais pas la bonne section

**Solution proposee**:
- Section-level filtering si question mentionne "toutes rondes"
- Re-ranking cross-encoder

---

### 11. FR-Q121 (Recall: 50%)

**Question**: "Vous arbitrez un tournoi FIDE. Un joueur non classe gagne 5 parties contre des joueurs classes. Comment obtient-il son **premier classement Elo**?"

**Expected**: pages [183, 185]
**Retrieved**: pages [9, 123, 4, 186, 4]

**Cause racine**: **SEMANTIC DRIFT + CONTEXT LONG**

- Page 186 est adjacente a 185 (recall partiel)
- Pages 4, 9 sont intro/table des matieres (attracteurs embedding)
- Le contexte long ("arbitrez tournoi FIDE, non classe, gagne 5 parties") dilue le signal cle

**Solution proposee**:
- Query simplification: extraire `premier classement Elo conditions`
- Negative sampling: penaliser pages intro

---

### 12. FR-Q125 (Recall: 50%)

**Question**: "A quelle **cadence** la chute du **drapeau** doit-elle etre signalee par l'arbitre s'il l'observe?"

**Expected**: pages [57, 58]
**Retrieved**: pages [66, 50, 55, 46, 45]

**Cause racine**: **PAGES ADJACENTES - ANNEXES**

- Page 55 est a ±2 de 57 (tolerance matchee)
- Le retrieval trouve des pages sur le temps (45, 46, 50) qui parlent de "drapeau"
- Mais pas les Annexes A/B (rapide/blitz) qui sont a p57-58

**Analyse corpus**:
- p57 = Annexe A.5.5 "L'arbitre doit aussi annoncer la chute du drapeau"
- Le mot "drapeau" apparait dans plusieurs sections (p44-50 pour standard)

**Solution proposee**:
- Query expansion: `cadence → rapide OR blitz`
- Si "cadence" mentionne → forcer Annexes A/B

---

### 13. FR-Q127 (Recall: 0%)

**Question**: "En cadence 1h30+30s, un joueur a moins de 5 minutes. Doit-il continuer a **noter** sa partie?"

**Expected**: pages [50]
**Retrieved**: pages [3, 56, 66, 46, 193]

**Cause racine**: **COMBINAISON TERMES TRES SPECIFIQUE**

La question combine:
- Cadence specifique (1h30+30s)
- Seuil (5 minutes)
- Action (noter)
- Condition (increment)

**Contenu attendu (p50 Art. 8.4)**:
```
"Si un joueur dispose de moins de cinq minutes a sa pendule
et ne beneficie pas d'un temps additionnel de 30 secondes
ou plus a chaque coup, alors il n'est pas dans l'obligation
de respecter les exigences de l'Article 8.1.1 [notation]"
```

Le retrieval trouve p46 (temps), p56 (sanctions) mais pas p50 exactement.

**Solution proposee**:
- Query decomposition: separer les conditions
- Hybrid search obligatoire pour capter "5 minutes" + "noter"

---

### 14. FR-Q132 (Recall: 25%)

**Question**: "Un joueur classe uniquement en Rapide participe a son premier tournoi Standard. Comment obtient-il son **Elo Standard**?"

**Expected**: pages [183, 185, 187, 188]
**Retrieved**: pages [1, 4, 4, 168, 190]

**Cause racine**: **CROSS-CHAPTER CONTENT**

La reponse necessite:
- Chapitre 6.1 (p183, 185): Classement Standard initial
- Chapitre 6.2 (p187, 188): Classement Rapide

L'embedding de la question ne capture pas cette dualite.

**Analyse**:
- Page 190 est proche de 188 (±2)
- Le retrieval trouve des pages intro (1, 4) qui mentionnent les deux classements

**Solution proposee**:
- Multi-vector query: un pour "Standard", un pour "Rapide"
- Merge des resultats

---

## Synthese des Solutions

### 1. Query Pre-processing (7 questions)

| Technique | Questions | Implementation |
|-----------|-----------|----------------|
| Normalisation langage oral | Q94, Q95, Q98, Q103 | Regex + LLM rewriting |
| Expansion synonymes | Q77, Q94 | Dictionary lookup |
| Expansion abreviations | Q98, Q119 | `CM→Candidat Maitre` |
| Negation handling | Q95, Q98 | Detect + invert logic |

### 2. Hybrid Search (5 questions)

| Technique | Questions | Raison |
|-----------|-----------|--------|
| BM25 + Vector | Q86, Q119, Q127 | Termes specifiques |
| Cross-encoder rerank | Q87, Q121, Q99 | Semantic drift |

### 3. Source/Section Filtering (4 questions)

| Technique | Questions | Implementation |
|-----------|-----------|----------------|
| Chapter filter | Q119, Q125 | Si "toutes rondes" → Ch3.1 |
| Exclude intro pages | Q87, Q95, Q121 | pages < 10 penalisees |
| Multi-chapter merge | Q132 | Query split + merge |

### 4. Corpus Enrichment (2 questions)

| Technique | Questions | Implementation |
|-----------|-----------|----------------|
| Verifier multi-doc | Q85 | Contenu peut-etre dans R01 |
| Synonymes dans index | Q77, Q94 | "18 mois" → "un an" au chunk |

---

## Priorites d'Implementation

### Phase 1: Quick Wins (Impact immediat)

1. **Query normalization pipeline**
   - Detecteur langage oral/SMS
   - Expansion abreviations echecs (CM, FM, GM, MI, etc.)
   - Synonymes temporels (18 mois → un an)

2. **Exclude intro pages**
   - Penaliser pages 1-10 dans le scoring
   - Ou filtrer si >80% des resultats sont intro

### Phase 2: Hybrid Search

3. **BM25 fallback**
   - Si vector recall < 0.5, activer BM25
   - Merge avec RRF (Reciprocal Rank Fusion)

4. **Cross-encoder reranking**
   - Pour top-20 candidats, rerank avec cross-encoder
   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` ou equivalent FR

### Phase 3: Advanced

5. **Query decomposition**
   - Pour questions multi-conditions (Q127)
   - Decomposer et agreger resultats

6. **Chapter-aware retrieval**
   - Metadata filtering par chapitre
   - Si question mentionne "classement" → forcer Chapitre 6

---

## Metriques de Succes

| Phase | Questions ciblees | Recall attendu |
|-------|-------------------|----------------|
| Actuel | 120/134 | 91.17% |
| Phase 1 | +5 (Q77, Q94, Q95, Q98, Q103) | ~95% |
| Phase 2 | +4 (Q87, Q99, Q121, Q127) | ~98% |
| Phase 3 | +3 (Q85, Q119, Q132) | ~100% |

---

## Annexe: Mapping Questions → Solutions

| Question | Cause | Solution Prioritaire |
|----------|-------|---------------------|
| FR-Q77 | Mismatch "18 mois" | Synonyme mapping |
| FR-Q85 | Multi-doc | Verifier corpus |
| FR-Q86 | Admin + multi-page | Hybrid search |
| FR-Q87 | Semantic drift | Exclude intro + rerank |
| FR-Q94 | Mismatch + oral | Synonyme + normalization |
| FR-Q95 | Negation + oral | Query rewriting |
| FR-Q98 | Abbreviations | Expansion CM/FM |
| FR-Q99 | Partial match | Cross-encoder rerank |
| FR-Q103 | SMS-like | Query rewriting |
| FR-Q119 | Chapter boundary | Section filtering |
| FR-Q121 | Context long | Query simplification |
| FR-Q125 | Annexes confusion | Chapter filter |
| FR-Q127 | Multi-conditions | Query decomposition |
| FR-Q132 | Cross-chapter | Multi-vector merge |
