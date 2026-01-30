# Methode de Verification Manuelle des chunk_ids

> GATE 6 du plan async-orbiting-mitten.md
> 420 questions (386 annales + 34 human)
> ISO 42001 (tracabilite), ISO 25010 (qualite donnees)

## Principe

Pour chaque question du Gold Standard, verifier MANUELLEMENT que le
`expected_chunk_id` pointe vers un chunk dont le contenu correspond
a l'`article_reference` de la question.

**ZERO script Python.** Toute verification est faite par lecture humaine/IA:
- Outil `Read` pour lire les fichiers JSON
- Outil `Grep` pour chercher des chunk_ids ou du texte dans le corpus
- L'IA LIT le texte du chunk et JUGE l'alignement avec la question

## Donnees d'entree

| Fichier | Contenu | Acces |
|---------|---------|-------|
| `tests/data/gold_standard_annales_fr_v7.json` | 420 questions avec chunk_ids | Read |
| `corpus/processed/chunks_mode_b_fr.json` | 1857 chunks, 28 PDFs sources | Grep par chunk_id |
| Annales Docling (13 JSON) | Corrige detaille avec article_reference | Read |

## Cle de la methode: les annales fournissent tout

Pour chaque question annales, le **Corrige detaille** fournit:
1. **La reponse** (lettre + explication)
2. **L'article de reference** (ex: "R01 - Chapitre 8", "LA - Article 3.1")

L'article de reference = **cle deterministe** pour trouver le bon chunk:
- article_ref contient un code document (LA, R01, A02, C01...)
- ce code = un PDF source dans le corpus
- dans ce PDF, chercher la section/article mentionne
- le chunk qui contient cette section = le bon chunk

## Mapping code document -> PDF source

| Code | PDF source | Chunks |
|------|-----------|--------|
| `LA` | `LA-octobre2025.pdf` | 1085 |
| `R01` | `R01_2025_26_Regles_generales.pdf` | 41 |
| `R02` | `R02_2025_26_Regles_generales_Annexes.pdf` | 9 |
| `R03` | `R03_2025_26_Competitions_homologuees.pdf` | 17 |
| `A01` | `A01_2025_26_Championnat_de_France.pdf` | 22 |
| `A02` | `A02_2025_26_Championnat_de_France_des_Clubs.pdf` | 37 |
| `A03` | `A03_2025_26_Championnat_de_France_des_Clubs_rapides.pdf` | 18 |
| `C01` | `C01_2025_26_Coupe_de_France.pdf` | 29 |
| `C03` | `C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf` | 44 |
| `C04` | `C04_2025_26_Coupe_de_la_parite.pdf` | 45 |
| `F01` | `F01_2025_26_Championnat_de_France_des_clubs_Feminin.pdf` | 38 |
| `F02` | `F02_2025_26_Championnat_individuel_Feminin_parties_rapides.pdf` | 10 |
| `J01` | `J01_2025_26_Championnat_de_France_Jeunes.pdf` | 29 |
| `J02` | `J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf` | 34 |
| `J03` | `J03_2025_26_Championnat_de_France_scolaire.pdf` | 28 |
| `H01` | `H01_2025_26_Conduite_pour_joueur_handicapes.pdf` | 2 |
| `H02` | `H02_2025_26_Joueurs_a_mobilite_reduite.pdf` | 3 |
| `E02` | `E02-Le_classement_rapide.pdf` | 9 |
| `Statuts` | `2024_Statuts20240420.pdf` | 62 |
| `RI` | `2025_Reglement_Interieur_20250503.pdf` | 77 |
| `RF` | `2023_Reglement_Financier20230610.pdf` | 27 |
| `RD` | `2018_Reglement_Disciplinaire20180422.pdf` | 29 |
| `RM` | `2022_Reglement_medical_19082022.pdf` | 32 |
| `N4` | `reglement_n4_2024_2025__1_.pdf` | 10 |

## Procedure manuelle pour 1 question (6 etapes)

### 1. LIRE la question dans le GS
```
Read gold_standard -> questions[idx]
  -> id, question (texte GS), expected_answer, article_reference, expected_chunk_id
```

### 2. COMPARER avec la question COMPLETE des annales Docling
```
Read docling_markdowns -> session -> UV Corrige -> Question N
  -> question_docling = texte COMPLET (enonce + choix + article_ref)
  -> Comparer question GS vs question Docling
```

**Verifications:**
- Le texte GS est-il COMPLET? (pas tronque a 200 chars, pas fusionne avec une autre Q)
- Les choix QCM sont-ils tous presents et corrects?
- L'article_reference GS correspond-il au Corrige Docling?

**Si le texte GS est tronque ou incomplet:**
- Copier le texte COMPLET depuis le Docling
- Mettre a jour le GS avec la question complete

**Si le Docling est merdique** (OCR foireux, texte manquant, artefacts):
- Aller lire le PDF source directement: `corpus/fr/annales/` ou PDF original
- Extraire le texte correct du PDF
- Mettre a jour le GS avec le texte du PDF

### 3. IDENTIFIER le document source attendu
```
article_reference: "R01 - Chapitre 8 : Role des Capitaines"
  -> Code: R01
  -> PDF attendu: R01_2025_26_Regles_generales.pdf
```

### 4. LIRE le chunk assigne
```
Grep chunk_id dans corpus/processed/chunks_mode_b_fr.json
  -> Lire le texte du chunk (200+ chars)
  -> Verifier: le chunk vient-il du bon PDF?
  -> Verifier: le chunk parle-t-il du bon article/section?
```

### 5. JUGER l'alignement
| Verdict | Condition |
|---------|-----------|
| **OK** | Bon document ET bonne section/article |
| **WRONG_SOURCE** | Chunk vient d'un autre PDF que celui attendu |
| **WRONG_SECTION** | Bon PDF mais mauvaise section/article |
| **MISSING** | expected_chunk_id = null |
| **NO_ARTREF** | Pas d'article_reference -> verification par contenu seulement |

### 6. Si WRONG: chercher le bon chunk MANUELLEMENT
```
Grep dans corpus pour: "{PDF_attendu}" + mots-cles de l'article
  -> Lire les chunks candidats
  -> Choisir le meilleur match
  -> Noter le chunk_id de remplacement
```

## Format de presentation interactive (par batch de 20)

```
=== BATCH XX (Q{start}-Q{end}) ===

Q{idx} | {id} | session={session} uv={uv} qnum={qnum}
  --- QUESTION GS ---
  {texte question GS}
  --- QUESTION DOCLING (COMPLETE) ---
  {texte question Docling avec choix}
  --- COMPARAISON ---
  texte_complet: OUI/NON (si NON: texte GS tronque ou fusionne -> corriger)
  choix_ok: OUI/NON
  artref_gs: {article_reference GS}
  artref_docling: {article_reference Docling Corrige}
  artref_match: OUI/NON
  --- CHUNK ---
  chunk_id: {expected_chunk_id}
  chunk_source: {PDF du chunk}
  chunk_texte: {texte du chunk}
  --- VERDICT ---
  question_complete: OUI/NON
  chunk_alignement: OK / WRONG_SOURCE / WRONG_SECTION / MISSING
  [action: {correction a appliquer}]
  [remplacement_chunk: {nouveau_chunk_id}]
```

L'utilisateur valide chaque batch avant de passer au suivant.

## Batches (21 batches de 20)

| Batch | Questions | Plage IDs |
|-------|-----------|-----------|
| 01 | Q000-Q019 | clubs:001 - clubs:020 |
| 02 | Q020-Q039 | clubs:021 - clubs:040 |
| 03 | Q040-Q059 | clubs:041 - clubs:060 |
| 04 | Q060-Q079 | clubs:061 - clubs:080 |
| 05 | Q080-Q099 | clubs:081 - clubs:100 |
| 06 | Q100-Q119 | clubs:101 - clubs:120 |
| 07 | Q120-Q139 | clubs:121 - rules:008 |
| 08 | Q140-Q159 | rules:009 - rules:028 |
| 09 | Q160-Q179 | rules:029 - rules:048 |
| 10 | Q180-Q199 | rules:049 - rules:068 |
| 11 | Q200-Q219 | rules:069 - rules:088 |
| 12 | Q220-Q239 | rules:089 - rules:108 |
| 13 | Q240-Q259 | rules:109 - rules:128 |
| 14 | Q260-Q279 | rules:129 - open:013 |
| 15 | Q280-Q299 | open:014 - open:033 |
| 16 | Q300-Q319 | open:034 - open:053 |
| 17 | Q320-Q339 | open:054 - open:073 |
| 18 | Q340-Q359 | open:074 - tournament:007 |
| 19 | Q360-Q379 | tournament:008 - tournament:027 |
| 20 | Q380-Q399 | tournament:028 - human:014 |
| 21 | Q400-Q419 | human:015 - human:034 |

## Log de verification

Chaque verdict est ecrit dans `chunk_verification_log.jsonl`:
```json
{"idx": 0, "id": "ffe:annales:clubs:001:...", "verdict": "OK", "chunk_id": "...", "note": ""}
{"idx": 54, "id": "ffe:annales:clubs:055:...", "verdict": "WRONG_SOURCE", "chunk_id": "...", "replacement": "...", "note": "chunk etait LA, devrait etre A02"}
```

## IMPORTANT

- **420 questions = 420 verifications manuelles**. Pas de raccourci.
- Meme les 363 "OK source" sont verifiees: bon document ne veut pas dire bonne section.
- Chaque verdict est trace dans le log.
- L'utilisateur valide par batch interactif.
- ZERO script Python de scoring/mapping/automatisation.
- **QUESTION COMPLETE**: Pour chaque question, le texte GS est compare au texte Docling COMPLET. Si tronque/fusionne -> corriger le GS.
- **FALLBACK PDF**: Si le Docling est foireux (OCR, artefacts), aller lire le PDF source directement.
- **DOUBLE VERIFICATION**: question (completude) + chunk (alignement) pour chaque entree.

## Criteres GATE 6

| Controle | Critere |
|----------|---------|
| Couverture | 420/420 questions verifiees |
| Log complet | 420 entrees dans chunk_verification_log.jsonl |
| Chunks valides | Tout chunk rejete a un remplacement ou flag requires_context |
| Pas d'orphelins | 0 question sans chunk_id ET sans flag requires_context |
| Tracabilite | Chaque verdict justifie (article_ref vs chunk content) |
