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
| **Metadata answer_explanation** | Q36: La règle était dans metadata.answer_explanation | Toujours vérifier les champs metadata (answer_explanation, correct_answer) pour comprendre la logique |

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

## Évolution

Ce fichier est enrichi après chaque batch avec les nouvelles lessons learned.
Format: `### Batch NNN` suivi des erreurs détectées et solutions.
