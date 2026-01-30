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

### Batch 11 (Q200-Q219) — open:032-051, sessions jun2025 + dec2019 + jun2021
- **Resultat**: 3/20 OK — 14 WRONG_SECTION + 2 WRONG_SOURCE + 1 NO_MATCH
- **Sessions**: jun2025 UVO Q16-Q20 (5Q) + dec2019 UVO (8Q) + jun2021 UVO Q1-Q7 (7Q)
- **OK**: idx 202 (C04 Art 1.2), idx 212 (FIDE Art 9.2.2 parent179-child01), idx 214 (FIDE Art 11.2.1 parent183)
- **Findings chunks**:
  - 5 LA FIDE Laws WRONG_SECTION: idx 205 (Art 7.5 p041→p048-parent175), idx 210 (Art 6.7 parent173→parent171), idx 211 (Art 5 parent185→parent165), idx 213 (Art 9.3 parent181→parent179), idx 215 (Art 7.5 parent174→parent175)
  - 4 LA Part 2 chapters: idx 200 (Ch 6.1 Cadence p015→p182), idx 201 (Ch 6.1 Art 2 p016→p036-parent534), idx 204 (Ch 5.4 Rapport p017→p172), idx 209 (Ch 4.2 Departages p049→p146)
  - 2 LA RIDNA WRONG_SECTION: idx 216 (Art 3.1 p010→p014-parent041), idx 217 (Art 11 Examens p010→p022-parent095)
  - 1 C04 WRONG_SECTION: idx 203 (Art 4.3 parent026=4.4→parent024=4.3.a)
  - 1 R03 WRONG_SECTION: idx 219 (Art 2.3 parent006=2.2.3→parent007=2.3)
  - 2 WRONG_SOURCE: idx 207-208 (LA→J01 Art 2.6 Qualifications d'office, parent012)
  - 1 NO_MATCH: idx 218 (RIDNA Art 18.4 — RIDNA ends at Art 17 in 2025, used FIDE Art 18 as best match p092-parent275)
- **FIDE Laws page mapping**: Art 5=p043-parent165, Art 6.7=p046-parent171, Art 7.5=p048-parent175, Art 9=p051-parent179, Art 11=p053-parent183, Art 18=p092-parent275
- **RIDNA version gap**: Art 18.4 referenced in jun2021 exam does not exist in 2025 LA (RIDNA ends at Art 17). Pages 27-29 contain FIDE Ethics Code. Best approximation: FIDE Laws Art 18 Role Arbitre.
- **Findings metadata**: 10 UV FAUX (clubs/rules → open), 1 qnum FAUX (idx 204: 17→20)
- **Cumul**: 220/420 verifiees (52%), 181 WRONG chunks + 3 textes + 111 metadata corriges.

### Batch 12 (Q220-Q239) — open:052-064 + rules:001-007, sessions jun2021 + dec2024
- **Resultat**: 5/20 OK — 15 WRONG_SECTION
- **Sessions**: jun2021 UVO Q8-Q20 (13Q) + dec2024 UVR Q1-Q7 (7Q)
- **OK**: idx 234 (Art 3.7.3.5 promotion parent159-child01), idx 235 (Art 3.10.3 parent160), idx 236 (Art 4.2.1/4.3.1 parent162-child00), idx 238 (Art 5.1.1 parent165), idx 239 (Art 5.1.2 parent165)
- **Findings chunks**:
  - 6 LA Part 2 chapters WRONG_SECTION: idx 224-225 (Sonneborn-Berger p017/p018→p151), idx 226 (adversaire virtuel p017→p155), idx 229 (Droits homologation p036→p168), idx 230 (Systeme Hort p034→p158), idx 231 (Bareme p036→p178)
  - 3 LA FIDE Laws WRONG_SECTION: idx 222 (Art 12.9 p036→p055), idx 233 (Art 1.3 p028→p036), idx 237 (Art 4.3.3 child00→child01)
  - 2 LA RIDNA WRONG_SECTION: idx 223 (old Art 23.2 = new 17.2 p011→p025), idx 228 (Art 8.24 p015→p185)
  - 2 LA FIDE rating WRONG_SECTION: idx 227 (Art 5.1 p015→p183), idx 232 (Gestion homologation p036→p168)
  - 1 R03 WRONG_SECTION: idx 220 (Art 2.3 parent002→parent007)
  - 1 R01 WRONG_SECTION: idx 221 (Art 3.1.2 parent029→parent011)
- **RIDNA old-to-new renumbering**: Art 23.2 (2021) = 17.2 Mesures administratives (2025). Old Ch 5.1 Departages = new Ch 4.2. Old Ch 5.2 Bareme = new Ch 5.5. Old Ch 5.3 Hort = new Ch 4.3.
- **New chunk knowledge**: Art 12.9 sanctions = p055-parent189-child01/child02. Art 1 = p036-parent155-child00. Adversaire fictif = p155-parent466-child01. Art 5 rating = p183-parent537.
- **Findings metadata**: 5 UV FAUX (clubs → open), 1 artref FAUX (idx 232 broken → fixed)
- **dec2024 UVR**: First batch of rules/FIDE Laws questions. 5/7 chunks correct (71%) — much better than UVC/UVO averages.
- **Cumul**: 240/420 verifiees (57%), 196 WRONG chunks + 3 textes + 117 metadata corriges.

### Batch 13 (Q240-Q259) — rules:008-027, sessions dec2024 + dec2023
- **Resultat**: 8/20 OK — 12 WRONG_SECTION
- **Sessions**: dec2024 UVR Q8-Q27 (16Q, dont 4 duplicates Q5-Q8) + dec2023 UVR Q3-Q6 (4Q)
- **OK**: idx 240 (Art 5.1.2 parent165), idx 245 (Art 5.1.1 dup), idx 246 (Art 5.1.2 dup), idx 247 (Art 5.1.2/7.5.1 dup), idx 249 (Art 9.1.2.3 parent179), idx 250 (Art 9.2.1 parent179), idx 251 (Art 9.5.3 parent181-child00), idx 255 (Commentaire A.5.3 parent191-child01)
- **Findings chunks**:
  - 4 Annexe A: idx 243/253 (A.5.1.1 parent190→parent191), idx 254 (A.5.2 parent190→parent191), idx 256 (A.5.1.2 parent190→parent191)
  - 3 Art 8 notation: idx 242/248 (parent169=Art 6.3→parent177=Art 8), idx 258 (parent142=charte→parent177)
  - 1 Art 6.9: idx 241 (parent170-child01→parent171-child02 = Art 6.9 time forfeit)
  - 1 Art 4.3.3: idx 244 (parent162-child00→child01, duplicate Q5)
  - 1 Art 9.6.2: idx 252 (parent181-child00→child01 = Art 9.6)
  - 1 Art 4.2.1: idx 257 (parent164=4.9→parent162=Art 4, dec2023)
  - 1 Art 5.2.1: idx 259 (parent185=Art 11→parent165-child01=Art 5.2.1, dec2023)
- **New chunk knowledge**: Annexe A split: parent190=A.1-A.4, parent191=A.5+. Art 6.9 at parent171-child02. Art 8 at parent177 (p033, pipeline page bug). Art 4.9 at parent164.
- **Duplicates**: 4 entries (idx 244-247) are duplicates of Q5-Q8 dec2024 UVR (idx 237-240 in batch 12). Pipeline created duplicate GS entries.
- **dec2024 UVR total** (Q1-Q27): 13 OK, 14 WRONG of 27 entries = 48% correct. Better than UVC/UVO.
- **Findings metadata**: 3 UV FAUX (clubs → rules for dec2023), 1 artref FAUX (idx 254: Art 7.5→A.5.2 per Grille)
- **Cumul**: 260/420 verifiees (62%), 208 WRONG chunks + 3 textes + 121 metadata corriges.

### Batch 14 (Q260-Q279) — rules:028-047, sessions dec2023 + jun2024
- **Resultat**: 3/20 OK — 17 WRONG_SECTION
- **Sessions**: dec2023 UVR Q7-Q20 (14Q) + jun2024 UVR Q2-Q7 (6Q)
- **OK**: idx 262 (Art 11.2.3.3 parent185), idx 267 (Art 7.5.5 parent176), idx 270 (Art 4.4.3 parent163)
- **Findings chunks**:
  - 4 Art 5/7 confusions: idx 260/279 (Art 5.2.2 parent163/185→parent165), idx 264 (Art 5.1.2 parent185→parent165), idx 274 (Art 7.2.1 parent176→parent174)
  - 3 Art 7 internal: idx 271 (Art 7.4.1 parent173→parent174), idx 276 (Art 7.3 parent176→parent174), idx 278 (Art 7.5.5 parent174→parent176)
  - 3 Art 6 internal: idx 263 (Art 6.8 p028 RIDNA→parent171), idx 265 (Art 6.11.4 parent171→parent173), idx 266 (Art 6.9 parent174→parent171)
  - 2 Art 9: idx 272/273 (Art 9.2 parent177→parent180)
  - 2 Annexe A: idx 268 (A.3 child00→child02), idx 269 (A.5.2 parent190→parent191)
  - 1 Annexe A.5: idx 275 (parent189→parent191)
  - 1 Annexe D: idx 261 (parent188=Art 12→parent198=Annexe D)
  - 1 Art 4.4: idx 277 (parent162=4.2→parent163=4.4)
- **Comprehensive FIDE Laws parent mapping completed**: parent162=Art4.2, 163=Art4.4-4.7, 164=Art4.9, 165=Art5, 169=Art6.3(page bug), 170=Art6.7 commentary, 171=Art6.7-6.9, 173=Art6.10-6.11, 174=Art7.1-7.4, 175=Art7.5, 176=Art7.5.5, 177=Art8, 179=Art9, 180=Art9.2, 181=Art9.4-9.6, 183=Art11, 185=Art11(span), 188=Art12.2, 189=Art12, 190=Annexe A.1-4, 191=Annexe A.5+, 198=Annexe D
- **Findings metadata**: 12 UV FAUX (clubs → rules, dec2023 pipeline error)
- **Cumul**: 280/420 verifiees (67%), 225 WRONG chunks + 3 textes + 133 metadata corriges.

### Batch 15 (Q280-Q299) — rules:048-067, sessions jun2024 + jun2025
- **Resultat**: 5/20 OK — 15 WRONG_SECTION
- **Sessions**: jun2024 UVR Q8-Q20 (13Q) + jun2025 UVR Q1-Q15 (7Q)
- **OK**: idx 283 (Art 5.1.2 parent165), idx 285 (Art 6.2.3 parent167), idx 292 (Art 8 parent177), idx 293 (Art 9.1.2.2 parent179), idx 299 (Art 3.10.3 parent160)
- **Findings chunks**:
  - 3 Art 5/11 confusions: idx 280/281 (Art 5 parent185→parent165), idx 289 (Art 6.2 parent185→parent167)
  - 3 Art 7.5: idx 284/290/296 (parent174/176→parent175=Art 7.5)
  - 2 Art 6.8 FIDE Ethics vs Laws: idx 286/288 (parent131=FIDE Ethics→parent171=Art 6.8)
  - 2 Annexe A: idx 282/298 (parent190→parent191=A.5)
  - 2 Art 9.6: idx 291/294 (parent179/181-child00→parent181-child01=Art 9.6)
  - 1 Art 6.9: idx 287 (parent173→parent171-child02)
  - 1 Art 7.3: idx 295 (parent174-child01→child00)
  - 1 Art 3: idx 297 (parent159=3.7.3→parent157=Art 3 general)
- **New finding**: parent131 at p029 is FIDE Ethics Code (NOT FIDE Laws Art 6.8). Pipeline confused Ethics section with Laws section.
- **Findings metadata**: 5 UV FAUX (clubs → rules)
- **Cumul**: 300/420 verifiees (71%), 240 WRONG chunks + 3 textes + 138 metadata corriges.

### Batch 16 (Q300-Q319) — rules:068-087, sessions jun2025 + dec2021 + dec2022 + jun2021
- **Resultat**: 7/20 OK — 13 WRONG_SECTION
- **Sessions**: jun2025 UVR Q16,27,30 (3Q) + dec2021 UVR Q15 (1Q) + dec2022 UVR Q2-Q6 (5Q) + jun2021 UVR Q3-Q19 (11Q)
- **OK**: idx 302 (Art 7.5 parent175), idx 303 (Art 9.4 parent181), idx 304 (Art 6.2.3 parent168), idx 305 (Art 4.5 parent163), idx 310 (Art 6.4 parent169-child02), idx 312 (Art 6.4 parent169-child02), idx 313 (Art 11.3.3 parent184)
- **Findings chunks**:
  - 4 Art 7.5 confusions: idx 307/314/317 (parent176=7.5.5→parent175=7.5), idx 309 (parent175-child01→child00)
  - 2 Annexe A: idx 300 (parent191→parent190-child02=A.3), idx 306 (p016 RIDNA→parent190-child02=A.4)
  - 1 Art 7.5 from Art 4.4: idx 301 (parent163=Art4.4→parent175=Art 7.5)
  - 1 Art 3.10: idx 308 (parent161=examples→parent160=Art 3.10)
  - 1 Art 7.2.1 from RIDNA: idx 311 (p025 RIDNA→parent174=Art 7.1-7.4)
  - 1 Art 6.2: idx 315 (parent166=Art 6.1→parent168=Art 6.2)
  - 1 Art 12.1: idx 316 (parent191=Annexe A→parent187=Art 12.1)
  - 1 RIDNA 22.2: idx 318 (parent188=Art 12.2→parent113=RIDNA Mesures admin)
  - 1 Art 8: idx 319 (parent177-child02→child00=Art 8 main)
- **New findings**: parent166=Art 6.1 (p044), parent187=Art 12.1 (p055). RIDNA old 22.2 = parent113 Mesures administratives.
- **Findings metadata**: 0 UV fixes needed (all already correct as rules)
- **Cumul**: 320/420 verifiees (76%), 253 WRONG chunks + 3 textes + 138 metadata corriges.

### Batch 17 (Q320-Q339) — rules:088-106 + tournament:001, sessions jun2021 + dec2022 + jun2025 + dec2021 + dec2024
- **Resultat**: 7/20 OK — 13 WRONG_SECTION
- **Sessions**: jun2021 UVR Q3-21 (11Q) + dec2022 UVR Q2-Q6 (5Q) + jun2025 UVR Q16-30 (3Q) + dec2021 UVR Q15 (1Q)
- **OK**: idx 320 (Art 4.7.3 parent163-child01), idx 321 (Art 1.3 parent155), idx 325 (Art 6.2.3 parent168), idx 327 (Art 1.3 parent155 dup 321), idx 328 (Art 11.3.3 parent184), idx 335 (Art 9.1 parent179), idx 336 (Art 5.1.1 parent165)
- **Findings chunks**:
  - 3 Art 6.9: idx 326/329/333 (parent173=Art 6.10→parent171-child02=Art 6.9)
  - 2 Art 7.5 child: idx 322/331 (parent175-child01→child00)
  - 1 Art 7.5 from Art 7.5.5: idx 324 (parent176→parent175)
  - 1 Art 7.6: idx 330 (parent178=Art 8.5→parent176=Art 7.5.5+7.6)
  - 1 Art 6.4: idx 323 (parent168=Art 6.2→parent169-child02=Art 6.4)
  - 1 Art 5.1.2: idx 334 (parent168=Art 6.2→parent165=Art 5)
  - 1 Art 12.9: idx 337 (parent185=Art 11→parent189-child01=Art 12.9)
  - 1 Art 9.2: idx 338 (parent185=Art 11→parent180=Art 9.2)
  - 1 RIDNA 22.2: idx 332 (parent473=AFO→parent113=RIDNA Mesures admin)
  - 1 Ch 3.5 Swiss: idx 339 (p016 RIDNA→p120 parent351=Ch 3.5 Swiss Haley)
- **New findings**: parent176=Art 7.5.5+7.6 (p049), parent189-child01=Art 12.9 (p055), parent351=Ch 3.5 Systeme suisse (p120).
- **Findings metadata**: 3 UV FAUX (idx 329 open→rules, 332 open→rules, 333 tournament→rules)
- **Duplicate**: idx 321/327 = both jun2021 UVR Q21, same artref, same chunk (pipeline duplication)
- **Cumul**: 340/420 verifiees (81%), 266 WRONG chunks + 3 textes + 141 metadata corriges.

### Batch 18 (Q340-Q359) — tournament:002-021, session dec2024
- **Resultat**: 1/20 OK — 19 WRONG_SECTION
- **Session**: dec2024 UVT Q3-Q29 (20Q) — toutes questions tournament management
- **OK**: idx 347 (Art 6.7+commentaire parent171-child00)
- **Findings chunks**:
  - 3 Ch 3.5 Swiss C.04.3 Neerlandais: idx 342/343/350 (parent190/349/350→parent359=C.04.3)
  - 3 Ch 5.5 Bareme arbitres: idx 357/358/359 (parent177/176/176→parent522=Ch 5.5)
  - 2 R03 Art 2.3: idx 340/341 (parent010/014→parent007=Art 2.3 Droits inscription)
  - 2 Ch 4.2 Departages cumulatif: idx 352/353 (parent118/127→parent436=Art 6.5 Cumulatif)
  - 1 Art 6.9 child: idx 345 (parent171-child00→child02)
  - 1 Art 7.3: idx 346 (parent176=Art 7.5.5→parent174=Art 7.1-7.4)
  - 1 Dutch D. Transpositions: idx 344 (parent348=Exempts→parent380=D. Transpositions)
  - 1 Annexe D.2.10: idx 348 (parent198→parent200=D.2.10)
  - 1 C.04.2 D.8: idx 349 (parent349→parent357=C.04.2 section D rules, D.8 not found explicitly)
  - 1 Ch 4.5 Violence: idx 351 (parent190=FIDE Annexe A→parent484=Ch 4.5 Violence)
  - 1 Ch 4.3 Partage prix: idx 354 (parent161=Art 3.10→parent470=Ch 4.3)
  - 1 Rating K coefficient: idx 355 (parent174=FIDE Art 7.1→parent546=Rating 8.3.3)
  - 1 Rating art 8.3: idx 356 (parent176=FIDE Art 7.5.5→parent545=Rating 8.3)
- **New chapter mappings**: parent351-393=Ch 3.5 Swiss (p120-135), parent426-436=Ch 4.2 Departages (p146-149), parent470=Ch 4.3 Prix (p157), parent484=Ch 4.5 Violence (p163), parent522=Ch 5.5 Bareme (p178), parent545-546=Ch 6.1 Rating 8.3 (p185)
- **Root cause**: Pipeline systematically mapped tournament chapter artrefs to FIDE Laws section chunks
- **Findings metadata**: 10 UV FAUX (clubs→tournament to match ID prefix)
- **Cumul**: 360/420 verifiees (86%), 285 WRONG chunks + 3 textes + 151 metadata corriges.
