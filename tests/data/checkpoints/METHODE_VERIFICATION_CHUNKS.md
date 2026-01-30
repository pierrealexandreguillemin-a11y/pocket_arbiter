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

---

## AVANCEMENT

### Batch 01 (Q000-Q019) — clubs:001-020, session dec2024
- **Commit**: `154063a`
- **Resultat**: 13 OK / 7 WRONG_SECTION corriges
- **Findings**: Chunks LA pointaient vers FIDE Laws (Part 2) au lieu de DNA (Part 1). Sections 8.1 Materiel, 6.2 Arbitres inactifs, 7.1 Criteres titres, 9.1 Jury appel, 7.4 Etrangers FIDE identifiees comme bons chunks. R01 3.2 forfait et A02 3.8 forfaits aussi corriges.
- **Pattern**: LA-octobre2025.pdf a 1085 chunks couvrant Part 1 (DNA, p3-p30) et Part 2 (FIDE Laws, p32+). Les chunks originaux etaient assignes a la mauvaise partie du document.

### Batch 02 (Q020-Q039) — clubs:021-040, sessions dec2024 + dec2023
- **Commit initial**: `6cfb6fc` (chunks corriges par section-name matching)
- **Re-verification Docling**: FAITE — chaque question comparee au Corrige detaille
- **Resultat chunks**: 20/20 OK (chunk <-> artref alignement correct)
- **Resultat questions**: 17/20 completes, 3 textes FAUX corriges
- **Findings chunks**: A02 sous-sections, R03 WRONG_SOURCE, RIDNA/FIDE confusion
- **Findings questions**:
  - clubs:030 (idx 29): texte GS="agrement FIDE AFO1" FAUX → corrige: "Qui designe le DRA?"
  - clubs:032 (idx 31): texte GS="Cxg5 coup illegal" FAUX → corrige: "Concernant l'entente entre Clubs"
  - clubs:037 (idx 36): texte GS="roi en echec" FAUX → corrige: "laquelle est vraie?" (arbitre roles)
- **Findings metadata**: 12 UV metadata FAUX (rules/open → clubs), 1 qnum FAUX (6→8)
- **Pattern**: Pipeline parse_annales.py confondait les UV (dec2023 UVC Q1-20 classes comme rules/open). Textes de 3 questions assignes depuis UVR/UVO/UVT au lieu de UVC.
- **Cumul**: 40/420 verifiees (10%), 23 WRONG chunks + 3 textes + 13 metadata corriges.

### Batch 03 (Q040-Q059) — clubs:041-060, sessions dec2023 + jun2024
- **Resultat**: 0/20 OK — 16 WRONG_SECTION + 4 WRONG_SOURCE (catastrophic)
- **Sessions**: dec2023 UVC Q18-29 (8 questions) + jun2024 UVC Q1-12 (12 questions)
- **Findings chunks**:
  - ALL 20 chunks wrong. Pipeline assignment completely broken for this range.
  - 4 WRONG_SOURCE: idx 44-45 (LA→C03), idx 46 (A02→F01), idx 47 (LA→F02)
  - 4 WRONG_SECTION A02/R02: idx 40 (A02 2.6→3.11), idx 41 (A02 3.3→3.8), idx 42 (R02 sect.6→Art 1), idx 43 (R02 sect.3→Art 4)
  - 8 WRONG_SECTION LA RIDNA: idx 48-55 — RIDNA sections confused with FIDE Laws and other LA chapters
  - 4 WRONG_SECTION R01: idx 56 (3.2.1→2.2), idx 57 (2.2→2.4), idx 58 (3.2.1→3.1.3), idx 59 (3.3.3→2.2)
- **Findings metadata**: 16 UV FAUX (open/rules → clubs), 1 qnum FAUX (idx 55: 7→8)
- **Findings questions**: Plusieurs textes GS proviennent d'autres UV (idx 42 delai appel mais choix = noms, idx 47 phases mais reponse = secretariat). Correction requiert re-extraction pipeline (etapes 2-4 du plan).
- **Pattern**: Pipeline chunk assignment completement casse pour cette plage. RIDNA sections confondues avec FIDE Laws et autres chapitres LA.
- **Cumul**: 60/420 verifiees (14%), 43 WRONG chunks + 3 textes + 30 metadata corriges.

### Batch 04 (Q060-Q079) — clubs:061-080, sessions jun2024 + jun2025
- **Resultat**: 1/20 OK — 18 WRONG_SECTION + 1 WRONG_SOURCE
- **Sessions**: jun2024 UVC Q13-30 (16 questions) + jun2025 UVC Q1-5 (4 questions)
- **Findings chunks**:
  - Seul idx 63 (A02 1.2 Deroulement) correct
  - 3 R03 WRONG_SECTION: idx 60-62 (2.2.1/2.2.2 → 2.3/2.5)
  - 6 A02 WRONG_SECTION: idx 64-67, 69 (wrong sub-sections within 3.x)
  - 1 A02 WRONG_SOURCE: idx 68 (LA pairing criteria → A02 3.6.e Elo)
  - 2 J01 WRONG_SECTION: idx 70 (3.2.1→2.), idx 71 (3.3→3.2.2)
  - 4 C01 WRONG_SECTION: idx 72-75 (various wrong sections)
  - 3 LA RIDNA WRONG_SECTION: idx 76-78 (FIDE Laws → RIDNA sections)
  - 1 R01 WRONG_SECTION: idx 79 (3.1.2→2.3.1)
- **Findings metadata**: 7 UV FAUX (rules/open → clubs), 1 artref corrupt (idx 60: texte au lieu d'article)
- **Pattern**: Meme pattern que batch 03 — chunks assignes aux mauvaises sections systematiquement. RIDNA/FIDE confusion continue.
- **Cumul**: 80/420 verifiees (19%), 62 WRONG chunks + 3 textes + 38 metadata corriges.

### Batch 05 (Q080-Q099) — clubs:081-100, session jun2025
- **Resultat**: 4/20 OK — 16 WRONG_SECTION
- **Session**: jun2025 UVC Q6-27 (20 questions, Q3/Q12/Q16 absentes du GS)
- **OK**: idx 82 (R03 2.2.2), idx 90 (A02 2.5), idx 95 (A02 3.11), idx 96 (A02 3.7.d)
- **Findings chunks**:
  - 3 R01 WRONG_SECTION: idx 80 (3.1.2→2.4), idx 81 (3.2.1→3.1.2), idx 84 (3.1.2→4), idx 85 (9→8)
  - 1 R03 WRONG_SECTION: idx 86 (2.2.2→2.5)
  - 8 A02 WRONG_SECTION: idx 83 (2.6→1.1), idx 87 (4.1→2.4), idx 88 (3.6.d→3.6.a), idx 89 (3.3→3.6.d), idx 91 (3.3→2.6), idx 92 (3.6.d→3.7.g), idx 93 (3.7.a→3.7.f), idx 94 (4.1→3.11), idx 97 (3.7.f→3.9)
  - 1 J02 WRONG_SECTION: idx 98 (3.3→2.5)
  - 1 C03 WRONG_SECTION: idx 99 (3.7→2.5)
- **Findings metadata**: 12 UV FAUX (rules → clubs), 1 artref corrupt (idx 83)
- **Cumul**: 100/420 verifiees (24%), 78 WRONG chunks + 3 textes + 51 metadata corriges.

### Batch 06 (Q100-Q119) — clubs:101-120, sessions jun2025 + dec2021 + dec2022
- **Resultat**: 4/20 OK — 11 WRONG_SECTION + 4 WRONG_SOURCE + 1 NO_MATCH
- **Sessions**: jun2025 UVC Q28-29 (2Q) + dec2021 UVC Q1-22 (17Q) + dec2022 UVC Q1 (1Q)
- **OK**: idx 101 (C01 2.5), idx 108 (A02 4.1), idx 114 (LA Art 19), idx 116 (J02 2.5)
- **Findings chunks**:
  - 8 A02 WRONG_SECTION: idx 102-107, 109-110 (A02 sub-sections 3.x confused)
  - 3 A02→J02 WRONG_SOURCE: idx 111-113 (A02 chunks for J02 3.7.b/c artrefs, duplicate qnum=12)
  - 1 LA→C01 WRONG_SOURCE: idx 118 (LA Preambule → C01 3.2 Couleurs)
  - 1 C03 WRONG_SECTION: idx 100 (3.1→1.2)
  - 1 LA WRONG_SECTION: idx 117 (Art 19→Art 18.5)
  - 1 C01 WRONG_SECTION: idx 119 (2.6→2.3.b)
  - 1 NO_MATCH: idx 115 (Guide arbitrage international Art 1.1 — document absent du corpus)
- **Findings metadata**: 11 UV FAUX (rules → clubs)
- **Anomalie**: idx 111-113 ont tous qnum=12 (3 questions identiques — erreur pipeline)
- **Cumul**: 120/420 verifiees (29%), 93 WRONG chunks + 3 textes + 62 metadata corriges.

### Batch 07 (Q120-Q139) — clubs:121-140, sessions jun2022 + jun2023
- **Resultat**: 2/20 OK — 15 WRONG_SECTION + 3 WRONG_SOURCE
- **Sessions**: jun2022 UVC Q1-22 (19Q) + jun2023 UVC Q1 (1Q)
- **OK**: idx 121 (R01 1.4 → 1. Licences), idx 135 (A02 2.5)
- **Findings chunks**:
  - 8 R01 WRONG_SECTION: idx 120-129 spans R01/R03, all wrong sub-sections
  - 3 WRONG_SOURCE: idx 124 (LA→R01), idx 134 (LA→A02), idx 136 (LA→R01)
  - RIDNA art 3.2 (2021) = 3.7 Dir Reglements (2025) — version renumbering
  - RIDNA art 8.3 = AFJ (Arbitre Federal Jeune)
- **Findings metadata**: 7 UV FAUX (rules/open → clubs)
- **Cumul**: 140/420 verifiees (33%), 111 WRONG chunks + 3 textes + 69 metadata corriges.

### Batch 08 (Q140-Q159) — clubs:141-160, session jun2023
- **Resultat**: 3/20 OK — 13 WRONG_SECTION + 4 WRONG_SOURCE
- **Session**: jun2023 UVC Q2-Q20 (19Q) + 1 anomalous qnum=12 duplicate
- **OK**: idx 143 (R01 Art 1.4 → 1. Licences), idx 146 (R01 Art 8 → 8. Capitaines), idx 156 (A02 Art 3.8 → 3.8 Forfaits)
- **Findings chunks**:
  - 2 LA RIDNA WRONG_SECTION: idx 140 (p037 FIDE→p014 RIDNA 3.1), idx 141 (p033 FIDE→p018 RIDNA 8.1)
  - 1 RD WRONG_SECTION: idx 142 (p005 Art 12→p001 Art 2)
  - 2 R01 WRONG_SECTION: idx 144 (3.3.1→2.4), idx 145 (3.2.1→5. Elo)
  - 8 A02 WRONG_SECTION: idx 148 (3.4→3.6.a), idx 149 (3.8→3.7.c), idx 150 (3.8→3.6.a), idx 151 (2.1.b→1.1), idx 152 (2.1.a→1.2), idx 153 (3.2→2.5), idx 154 (2.5→2.6), idx 155 (3.6.d→3.7.g)
  - 1 WRONG_SOURCE LA→R03: idx 147 (LA p215→R03 2.2.2)
  - 1 WRONG_SOURCE LA→J02: idx 157 (LA p086→J02 3.7.b)
  - 2 WRONG_SOURCE LA→C01: idx 158 (LA p037→C01 3.2), idx 159 (LA p028→C01 1.3)
- **Findings metadata**: 15 UV FAUX (14 open + 1 tournament → clubs), 1 qnum FAUX (idx 151: 12→13 per Grille)
- **Anomalies**: idx 159 qnum=12 is 3rd duplicate in this session. Art 1.3 C01 not in jun2023 UVC Grille (20 questions). Possible orphan or mislabeled.
- **Pattern**: A02 sub-sections particularly scrambled — pipeline assigned adjacent/nearby sections instead of correct ones (154↔153 swapped!).
- **Cumul**: 160/420 verifiees (38%), 128 WRONG chunks + 3 textes + 85 metadata corriges.

### Batch 09 (Q160-Q179) — clubs:161-169 + open:001-011, sessions jun2023 + dec2024
- **Resultat**: 1/20 OK — 12 WRONG_SECTION + 7 WRONG_SOURCE
- **Sessions**: jun2023 UVC Q22-Q30 (9Q) + dec2024 UVO Q1-Q11 (11Q)
- **OK**: idx 178 (R03 Art 2.5 parent009)
- **Findings chunks**:
  - Part 1 (jun2023 UVC Q22-Q30): 6 WRONG_SOURCE (LA→H01/C03/C04/F01), 3 WRONG_SECTION (C03/C04/R02 sub-sections)
  - Part 2 (dec2024 UVO Q1-Q11): 5 RIDNA WRONG_SECTION (p026-p055 → p017-p023), 3 LA FIDE WRONG_SECTION (p036/p050 → p167/p182/p185), 1 R03 WRONG_SECTION (parent000→parent007), 1 WRONG_SOURCE (LA→F01)
- **RIDNA version renumbering**: old 8.4 AFO (dec2024 exam) = new 8.5 AFO (2025 LA). AFC inserted at new 8.4.
- **New chunk knowledge**: LA FIDE rating regs at p182-p185, Homologation at p167. H01 has 2 chunks (p001). C04 45 chunks. F01 38 chunks.
- **Findings metadata**: 7 UV FAUX (open → clubs for jun2023 UVC), 3 qnum FAUX (idx 159: 12→21, idx 162: 12→24, idx 163: 12→25)
- **Cumul**: 180/420 verifiees (43%), 147 WRONG chunks + 3 textes + 95 metadata corriges.

### Batch 10 (Q180-Q199) — open:012-031, sessions dec2024 + jun2025
- **Resultat**: 3/20 OK — 15 WRONG_SECTION + 1 WRONG_SOURCE + 1 NO_MATCH
- **Sessions**: dec2024 UVO Q12-Q20 (9Q) + jun2025 UVO Q1-Q14 (11Q)
- **OK**: idx 182 (R01 Art 3.2.1 Forfait ≈ Art 3), idx 197 (LA Annexe A p056), idx 199 (R03 Art 2.3)
- **Findings chunks**:
  - Massive LA WRONG_SECTION: pipeline assigned early pages (p009-p055) instead of correct Part 2 chapters (p146-p187)
  - 9 LA Part 2 chapters: Ch 4.2=p146, 4.3=p157, 5.1=p165, 5.2=p168, 5.3=p010(parent508), 5.4=p172, 5.5=p178-179, 6.2=p187, RIDNA Art 17=p025
  - 2 RIDNA: idx 189 (12.3 AFO p055→p023), idx 191 (6.2 p027→p017)
  - 1 WRONG_SOURCE: idx 198 (LA→J02 Art 3.7.c)
  - 1 NO_MATCH: idx 190 (Guide Intl Art 2 absent corpus, best match RIDNA 6.3)
- **New LA chapter knowledge**: Full mapping of Part 2 chapters to page numbers (4.2-6.2). Annexe A at p056.
- **Findings metadata**: 5 UV FAUX (1 rules→open, 4 clubs→open)
- **Cumul**: 200/420 verifiees (48%), 164 WRONG chunks + 3 textes + 100 metadata corriges.
