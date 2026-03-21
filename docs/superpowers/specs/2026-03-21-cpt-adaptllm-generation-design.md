# CPT + AdaptLLM Generation Fine-tuning — Design Spec

> **Date**: 2026-03-21
> **Statut**: En revue
> **Modele**: Gemma 3 270M IT (ADR-001 Option A)
> **Baseline**: Gemma 3 270M IT non fine-tune
> **Target**: qualite generation >= 70% (eval humaine 34 questions)
> **Methode**: TAPT full fine-tuning + AdaptLLM regex-mined SFT
> **Contraintes**: corpus-only (pas de generation de questions), budget 0€, Kaggle T4

---

## Contexte

### Pourquoi fine-tuner la generation

Le retrieval plafonne a recall@5 = 60.1% (chantier 3, 8 optimisations testees, +3.4pp total).
Le fine-tuning retrieval est abandonne (chantier 4a, 2026-03-21 — 3 bugs critiques,
0 precedent litterature a 1116 exemples, precedent echec).

Le levier inexplore est la generation : le modele doit apprendre le vocabulaire et le
style reglementaire FFE pour mieux exploiter les chunks retrouves.

### Pourquoi PAS de generation de questions

10 tentatives de generation de questions ont echoue sur ce projet :
- gs_scratch : 584Q generees, 71.5% garbage, abandonne (Ere 2)
- 6 passes de correction chunk_ids automatiques
- SimCSE+ICT : pseudo-queries incluant les descriptions contextuelles

Le GS est un faux-ami (264/298 = QCM annales ≠ queries terrain).
La contrainte est absolue : aucune question generee par LLM.

### Pourquoi deux phases

1. **TAPT** (Continued Pre-Training) : le modele apprend le vocabulaire et le style FFE
   par next-token prediction sur le corpus brut. Standard : Gururangan et al. ACL 2020.

2. **AdaptLLM** (SFT sur exercices mines) : le modele apprend a raisonner sur le texte
   reglementaire via des exercices de comprehension extraits par regex. Standard : Cheng
   et al. ICLR 2024.

L'ordre CPT → SFT est le pipeline standard (Gururangan ACL 2020 : DAPT puis TAPT).
Note : PIT (Jiang et al. arXiv:2402.12847) propose l'ordre inverse (instruction d'abord,
puis CPT). Pour notre cas, Gemma 270M IT est DEJA instruction-tuned, donc le CPT
vient naturellement en second — ce qui correspond au regime PIT. Le SFT AdaptLLM
est une seconde couche d'instruction-tuning specifique au domaine.

---

## Corpus (mesures verifiees)

| Metrique | Valeur |
|----------|--------|
| Documents | 28 |
| Tokens (tiktoken cl100k) | 381,316 (reference — le tokenizer Gemma SentencePiece peut differer de 20-40%, a mesurer) |
| Chars | 1,373,706 |
| Tables | 117 |
| Document dominant | LA-octobre2025 : 238,529 tok (62.5%) |
| Plus petit document | H02 : 477 tok |
| Format source | JSON docling (`corpus/processed/docling_v2_fr/*.json`) |
| Dependance critique | `docling-hierarchical-pdf>=0.1.5` (heading hierarchy) |

**Risque biais LA** : LA represente 63% des tokens. Le training par documents entiers
surrepresenterait LA. Mitigation : decouper en paragraphes et shuffler.

---

## Phase 1 — TAPT (Continued Pre-Training)

### Methode

Causal Language Modeling (next-token prediction) sur le corpus complet.
Full fine-tuning (pas LoRA) — Biderman et al. 2024 : LoRA sous-performe le FFT
pour CPT, ecart 10-100x en rang de perturbation.

### Preparation donnees

1. Charger les 28 JSON docling, extraire le champ `markdown`
2. Decouper chaque document en paragraphes (split `\n\n`)
3. Shuffler les paragraphes (eviter memorisation sequentielle — LA reste 63% des tokens)
4. Concatener avec `tokenizer.eos_token` (`<eos>`, token ID 1) entre chaque paragraphe
   ATTENTION : Gemma 3 utilise `<eos>` (ID 1), PAS `</s>` (ID 212 = token vocabulaire normal)
5. Tokenizer : `AutoTokenizer.from_pretrained("google/gemma-3-270m-it")`
6. Packing en sequences de 2048 tokens (choix VRAM — context max Gemma 270M = 32,768)

### VRAM (calculee)

| Composant | MB |
|-----------|-----|
| Model fp16 | 515 |
| Gradients fp16 | 515 |
| Optimizer AdamW (fp32 copy + m + v) | 3,090 |
| Activations (batch=4, seq=2048, 18 layers) | ~1,500 (estimee, a verifier empiriquement) |
| **Total** | **~5,620 (5.5 GB)** |
| **T4 headroom** | **~10,764 MB** |

L'estimation d'activations est approximative (depend de l'architecture exacte).
A verifier empiriquement au premier step de training. Meme dans le pire cas,
le T4 a >10 GB de marge. Pas besoin de gradient checkpointing.

### Hyperparametres

| Param | Valeur | Source |
|-------|--------|--------|
| Fine-tuning | Full (PAS LoRA) | Biderman 2024 arXiv:2405.09673 |
| Precision | fp16 | Gemma 270M model card, T4 Turing (pas bf16) |
| Seq length | 2048 | Choix pratique VRAM (context max Gemma 270M = 32K) |
| Epochs | 5 (cap ferme — pas d'extension) | Muennighoff NeurIPS 2023 : au-dela de 4 epochs, rendements decroissants. CLM = meme cible chaque epoch (pas de masking aleatoire). |
| Batch size | 4 | Small corpus, T4 headroom |
| Gradient accumulation | 4 (effective batch 16) | Standard |
| LR | 5e-6 | Conservateur (instruct model deja entraine) |
| LR schedule | Cosine, warmup 10% | Parmar 2024 arXiv:2407.07263 |
| NEFTune | alpha=5 | Jain et al. arXiv:2310.05914 |
| Dropout | 0.1 (verifier que Gemma 3 a attention_dropout dans config) | Xue et al. NeurIPS 2023 arXiv:2305.13230. Si Gemma 3 n'a pas de dropout natif, injecter via `model.config.attention_dropout = 0.1` avant training |
| Weight decay | 0.01 | Standard |
| Max grad norm | 1.0 | Standard, prevent gradient spikes en fp16 FFT |
| Precision | fp16 via `TrainingArguments(fp16=True)` | Utilise AMP + GradScaler internement (pas du pure fp16). Fallback fp32 si NaN. |
| Eval | 2-3 documents entiers holdout (PAS par paragraphe — fuite sinon) | Perplexity tracking |
| Early stopping | patience 3, metric = eval perplexity | Anti-overfitting |
| Seed | 42 | Reproductibilite |
| Logging | steps=1, nan/inf filter | Production standard |

### Validations Phase 1

| Check | Quand | Quoi |
|-------|-------|------|
| GPU validation | Avant training | CUDA available, VRAM >= 14 GB, compute >= 7.0 |
| Architecture validation | Avant training | `num_hidden_layers`, `hidden_size`, `total_params` matchent attendu (~270M). Verifier `attention_dropout` config (injecter 0.1 si absent) |
| Data validation | Avant training | >= 28 docs, >= 300K tokens **Gemma tokenizer** (PAS tiktoken), pas de doc vide |
| Tokenizer validation | Avant training | `eos_token` = `<eos>` (ID 1, PAS `</s>`). Logger ratio tokens Gemma / tiktoken pour reference |
| Train/eval split | Avant training | 2-3 docs holdout entiers (pas split par paragraphe = fuite) |
| Perplexity baseline | Epoch 0 | Base model perplexity sur eval set (reference) |
| Perplexity tracking | Chaque epoch | Doit decroitre, early stop si remonte 3x |
| Loss NaN/Inf | Continu | `logging_nan_inf_filter=True`, assert final_loss < 50 |
| VRAM monitoring | Apres model load + apres step 1 | < 80% T4 |
| Checkpoint save | Chaque epoch | `save_strategy="epoch"` |
| Post-training | Apres training | Charger checkpoint, generer 5 reponses test, verifier coherence |

### Output

Checkpoint `gemma-270m-cpt/` : modele complet (pas d'adapter).

---

## Phase 2 — AdaptLLM (SFT sur exercices mines)

### Methode

Extraction automatique d'exercices de comprehension de lecture depuis le corpus
par regex sur les connecteurs logiques francais. Aucun LLM utilise.

Standard : Cheng et al. "Adapting Large Language Models via Reading Comprehension",
ICLR 2024 (arXiv:2309.09530).

### Regex mining FR

```python
FR_CONNECTORS = {
    "nli_consequent": r"(?:Par conséquent|En conséquence)",
    "nli_contrast": r"(?:Cependant|Toutefois|Néanmoins|En revanche)",
    "causal": r"(?:\bcar\b|parce qu|en raison de|du fait de)",
    "conditional": r"(?:sous réserve|à condition qu|dans le cas où|sauf si)",
    "reference": r"(?:en application de|conformément|au sens de|tel que prévu|en vertu de)",
    "addition": r"(?:De plus|En outre|Par ailleurs)",
}
```

### Yield mesure (corpus reel, regex strict)

| Type tache | Source | Matches | Exercices |
|------------|--------|---------|-----------|
| Connector NLI/causal/ref/etc. | Regex sur 28 docs | 344 | ~688 |
| Summarization | Headings markdown | ~1,500 | ~1,500 |
| Text completion | Phrases >30 chars (7,245) | 7,245 | ~2,000 (cap) |
| **Total** | | | **~4,188** |

Seuil minimum : 500 exercices (gate G2). Le yield reel est 8x au-dessus.

### Format exercices (Gemma IT chat template)

```json
{"messages": [
  {"role": "user", "content": "Résumez le passage suivant du règlement FFE:\n\n{passage}"},
  {"role": "assistant", "content": "{heading_ou_premiere_phrase}"}
]}
```

Le training DOIT utiliser `tokenizer.apply_chat_template()` (format Gemma 3 IT :
`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>`).
Ne PAS formater manuellement les tokens speciaux.

### Types de taches

**Deviation documentee** : le paper AdaptLLM original (Cheng ICLR 2024) utilise
6 types sur des corpus de milliards de tokens en anglais. Notre adaptation FR sur
381K tokens conserve les types connector-based (coeur de la methode) et adapte les
autres. Les types "Summarization" et "Completion" ne sont PAS dans le paper original.

| Type | Dans le paper ? | Prompt template | Source reponse |
|------|----------------|----------------|----------------|
| NLI | OUI | "La phrase suivante est-elle une consequence du passage?" | Oui/Non + justification |
| Causal | OUI | "Quelle est la cause mentionnee dans ce passage?" | Phrase avec connecteur causal |
| Conditional | OUI (adapte) | "Sous quelle(s) condition(s) s'applique cette regle?" | Phrase avec connecteur conditionnel |
| Reference | OUI (adapte) | "A quel texte ce passage fait-il reference?" | Phrase avec connecteur de reference |
| Summarization | NON (ajout) | "Résumez le passage suivant:" | Heading/titre de la section |
| Completion | NON (ajout) | "Completez la phrase suivante:" | Phrase masquee a proximite d'un connecteur (PAS arbitraire) |

**Completion restreint** : seules les phrases dans un rayon de 2 phrases autour d'un
connecteur logique sont eligibles. Cela evite les completions arbitraires (anti-MLM)
et reste dans l'esprit du paper (comprehension logique, pas text infill).

**Mining stats obligatoire** : `mining_stats.json` doit inclure la distribution par type
ET par document source. Si >80% des matches viennent de LA, documenter et evaluer le biais.

### Hyperparametres SFT

| Param | Valeur | Source |
|-------|--------|--------|
| Base model | Checkpoint TAPT (Phase 1) | Pipeline standard : CPT puis SFT (Gemma IT = deja instruction-tuned, cf. regime PIT) |
| Fine-tuning | Full | Coherent avec Phase 1 |
| Precision | fp16 | Coherent avec Phase 1 |
| Format | Chat messages (Gemma IT template) | Google Gemma docs |
| Epochs | 3 | Standard SFT small dataset |
| LR | 2e-5 | Standard SFT (plus haut que CPT) |
| LR schedule | Cosine, warmup 10% | Standard |
| NEFTune | alpha=5 | Coherent avec Phase 1 |
| Batch size | 4, grad_accum=4 | Coherent |
| Eval split | 10% stratifie par type de tache | Standard |
| Seed | 42 | Reproductibilite |

### Validations Phase 2

| Check | Quand | Quoi |
|-------|-------|------|
| Mining yield | Avant SFT | >= 500 exercices (gate G2) |
| Mining stats | Avant SFT | Distribution par type, par doc (mining_stats.json) |
| Data format | Avant SFT | Chat template valide, pas de message vide |
| Checkpoint TAPT | Avant SFT | Existe, chargeable, genere du texte coherent |
| Loss tracking | Pendant SFT | Decroissante, pas NaN/Inf |
| Post-SFT | Apres SFT | Charger checkpoint, generer 5 reponses, verifier format |

### Output

Checkpoint final `gemma-270m-cpt-sft/` : modele complet (pas d'adapter).

---

## Evaluation

### Eval set : 34 questions humaines du GS

Le Gold Standard contient 34 questions `ffe:human:*` ecrites par un humain,
avec chunk_id, docs et pages verifies. Ce ne sont PAS des QCM annales reformules
— ce sont des questions de terrain (calcul Elo, conditions de tournoi, procedures).

Ces questions ne servent PAS au training — elles servent uniquement a l'eval.

### Protocole eval

Pour chaque question humaine :
1. Recuperer les chunks contexte (provenance.chunk_id du GS)
2. Construire le prompt RAG : system prompt + chunks + question
3. Generer la reponse avec le modele (base puis fine-tune)
4. Eval humaine : la reponse est-elle utile, fidele au contexte, bien citee ?

### Gates

| Gate | Condition | Mesure | Action si FAIL |
|------|-----------|--------|----------------|
| G0 | Corpus >= 300K tokens | Avant training | STOP |
| G1 | Perplexity TAPT < perplexity base sur eval set | Apres Phase 1 | Skip Phase 1, SFT seul |
| G2 | Mining yield >= 500 exercices | Avant Phase 2 | Elargir regex ou skip SFT |
| G1b | 34Q eval sur checkpoint TAPT seul (diagnostic, pas de seuil formel) | Apres Phase 1 | Documenter — si degradation, SFT pourrait compenser ou non |
| G3 | Pas de degradation vs base model (34 questions humaines) | Apres Phase 2 | Rollback au checkpoint TAPT ou base model |
| G4a | Qualite generation >= 70% (34Q humaines, eval humaine) | Apres Phase 2 | ADR-001 : passer a Gemma 3 1B IT |
| G4b | Citation automatique >= 80% (264Q annales, regex source/page) | Apres Phase 2 | Investigate — renforce G4a statistiquement |

### Criteres eval humaine (par question, 0 ou 1)

| Critere | Definition | Edge cases |
|---------|------------|------------|
| Utile | La reponse aide un arbitre a prendre une decision | Partiel = 0 si la reponse est trop vague pour agir |
| Fidele | Toutes les affirmations de la reponse sont inferables du chunk contexte | Paraphrase correcte = 1, info absente du chunk = 0 |
| Citee | La reponse mentionne le document source ET l'article/section | Doc sans page = 1, aucune mention = 0, doc invente = 0 |

Score = % de questions ou les 3 criteres sont remplis. Gate G4a = score >= 70%.

**Limite statistique** (n=34) : CI 95% pour 70% = [52%, 84%]. Pouvoir discriminant faible.
Gate G4b (264Q annales, citation auto) compense en apportant un signal quantitatif
sur un echantillon 8x plus grand. Les deux gates ensemble donnent un signal fiable.

**Limite statistique** : avec n=34, l'intervalle de confiance 95% pour 70% est
[52%, 84%] (binomial exact). Le pouvoir discriminant est faible. Pour renforcer :
les 264 questions annales peuvent servir a une eval automatique partielle (critere
"Citee" verifiable par regex sur source/page, sans eval humaine).

---

## Quality Gates

Chaque etape du pipeline produit un artefact versionne. Aucune etape ne detruit
l'artefact precedent. Le rollback est possible a tout point.

### Gate matrix

| Gate | Etape | Condition PASS | Condition FAIL | Action FAIL | Artefact conserve |
|------|-------|---------------|----------------|-------------|-------------------|
| G0 | Pre-training | Corpus >= 300K tokens, 28 docs, aucun vide | Corpus invalide | STOP — ne pas lancer | — |
| G0b | Pre-training | `eos_token` = `<eos>` (ID 1), VRAM < 80% | Token ou VRAM invalide | STOP — fix config | — |
| G1 | Post-TAPT | Perplexity eval < perplexity base | TAPT inutile | Skip TAPT, SFT depuis base model | `gemma-270m-cpt/` conserve |
| G2 | Pre-SFT | Mining yield >= 500 exercices | Pas assez de donnees | Elargir regex ou skip SFT | `reading_tasks.jsonl` conserve |
| G3 | Post-SFT | Pas de degradation vs base (34Q humaines) | Regression | Rollback au checkpoint TAPT ou base | `gemma-270m-cpt-sft/` conserve |
| G4 | Post-SFT | Qualite >= 70% (34Q humaines, eval humaine) | Qualite insuffisante | ADR-001 : passer a Gemma 3 1B IT | `generation_eval.json` conserve |

### Artefacts par etape (tous conserves)

| Etape | Artefact | Chemin | Versionning | Rollback vers |
|-------|----------|--------|-------------|---------------|
| 0. Pre-training | DB existante | `corpus/processed/corpus_v2_fr.db` | DVC | — |
| 1. Data prep TAPT | Corpus concatene tokenise | `data/training/tapt_corpus.bin` | git (petit) ou DVC | Step 0 |
| 2. Data prep AdaptLLM | Exercices mines | `data/training/reading_tasks.jsonl` | git | Step 0 |
| 2b. Stats mining | Distribution par type/doc | `data/training/mining_stats.json` | git | — |
| 3. Post-TAPT | Checkpoint complet | `models/gemma-270m-cpt/` | DVC | Base model HuggingFace |
| 3b. Eval TAPT perplexity | Perplexity base vs TAPT | `data/benchmarks/tapt_perplexity.json` | git | — |
| 3c. Eval TAPT 34Q (G1b) | Reponses TAPT sur 34Q (diagnostic) | `data/benchmarks/generation_eval_tapt.json` | git | — |
| 4. Post-SFT | Checkpoint complet | `models/gemma-270m-cpt-sft/` | DVC | `models/gemma-270m-cpt/` |
| 4b. Eval SFT | Responses 34Q + scores | `data/benchmarks/generation_eval.json` | git | — |
| 5. Eval base (reference) | Responses 34Q base model | `data/benchmarks/generation_eval_base.json` | git | — |

**Regle** : `dvc add` + `dvc push` AVANT chaque etape qui modifie un artefact DVC.
Chaque checkpoint est un modele complet standalone (pas d'adapter, pas de dependance).

### Ordre d'eval obligatoire

1. Generer les reponses avec le **base model** d'abord (reference G3)
2. Generer les reponses avec le **modele TAPT** (gate G1 : perplexity)
3. Generer les reponses avec le **modele TAPT+SFT** (gates G3/G4)
4. Comparer les 3 cote a cote dans `generation_eval.json`

---

## Definition of Done

Le chantier 4 (generation) est **DONE** quand :

- [ ] `data/training/reading_tasks.jsonl` : >= 500 exercices mines, stats documentees
- [ ] `data/training/mining_stats.json` : yield par type, par doc, pas de type a 0
- [ ] `models/gemma-270m-cpt/` : checkpoint TAPT complet, chargeable, genere du texte coherent
- [ ] `data/benchmarks/tapt_perplexity.json` : perplexity base vs TAPT sur eval holdout
- [ ] `data/benchmarks/generation_eval_base.json` : reponses base model sur 34Q humaines
- [ ] `data/benchmarks/generation_eval_tapt.json` : reponses TAPT seul sur 34Q (diagnostic G1b)
- [ ] `models/gemma-270m-cpt-sft/` : checkpoint SFT complet, chargeable, genere du texte coherent
- [ ] `data/benchmarks/generation_eval.json` : reponses TAPT+SFT sur 34Q + scores humains
- [ ] Gate G3 PASS : pas de degradation vs base
- [ ] Gate G4a evaluee : score >= 70% (34Q humaines) ou decision ADR-001
- [ ] Gate G4b evaluee : citation auto >= 80% (264Q annales)
- [ ] `models/model_card.json` : section generation mise a jour avec valeurs mesurees
- [ ] Tous les artefacts DVC pushes
- [ ] Tous les benchmarks dans git
- [ ] Tests mining + training PASS

---

## Pipeline annuel automatisable

### Workflow local → Kaggle → local

```
LOCAL (pre-flight)                    KAGGLE (GPU T4)
─────────────────                     ──────────────
Step 0: DVC version previous model
Step 1: extract.py (docling+hier)
Step 2: mine_reading_tasks.py
Step 3: Upload dataset ──────────>    /kaggle/input/pocket-arbiter-gen-data/
                                        ├── corpus_paragraphs.jsonl
                                        └── reading_tasks.jsonl
                                      Step 4: train_generation.py
                                        ├── Phase 1: TAPT (~3 min)
                                        ├── save gemma-270m-cpt/
                                        ├── Phase 2: SFT (~10 min)
                                        └── save gemma-270m-cpt-sft/
Step 5: Download outputs <────────    /kaggle/working/
Step 6: eval_generation.py (local)      ├── gemma-270m-cpt/       (~565 MB)
Step 7: Human eval gate G3/G4          └── gemma-270m-cpt-sft/   (~565 MB)
Step 8: DVC add + push
```

### Commandes locales

```bash
# Step 0: Versionner le modele precedent
dvc add models/gemma-270m-generation/ && dvc push

# Step 1: Extraire les nouveaux PDFs (docling + hierarchical)
python scripts/pipeline/extract.py corpus/fr corpus/processed/docling_v2_fr

# Step 2: Miner les exercices AdaptLLM (LOCAL, pas Kaggle)
python scripts/training/mine_reading_tasks.py \
  --input corpus/processed/docling_v2_fr \
  --output data/training/reading_tasks.jsonl \
  --stats data/training/mining_stats.json

# Step 3: Preparer et uploader le dataset Kaggle
python scripts/training/prepare_kaggle_dataset.py \
  --corpus corpus/processed/docling_v2_fr \
  --tasks data/training/reading_tasks.jsonl \
  --output kaggle/dataset-generation/
# Puis: kaggle datasets create/version

# Step 4: Push kernel (UI doit avoir T4 configure)
kaggle kernels push -p kaggle/kernel-generation/

# Step 5: Attendre completion, telecharger outputs
kaggle kernels output pguillemin/pocket-arbiter-cpt-generation \
  -p models/

# Step 6: Eval (local, pas besoin de GPU)
python scripts/training/eval_generation.py \
  --model models/gemma-270m-cpt-sft \
  --base-model google/gemma-3-270m-it \
  --gs tests/data/gold_standard_annales_fr_v8_adversarial.json \
  --filter-origin human \
  --output data/benchmarks/generation_eval.json

# Step 7: Human eval gate G3/G4 (Pierre review generation_eval.json)

# Step 8: Versionner
dvc add models/gemma-270m-cpt/ models/gemma-270m-cpt-sft/ && dvc push
```

Steps 0-3, 5-6, 8 : deterministes, reproductibles, sans API externe.
Step 4 : Kaggle T4, ~15-30 min, <1% quota hebdomadaire.
Step 7 : semi-automatise (le script genere les reponses, l'humain evalue).

### Contraintes Kaggle (verifiees)

| Contrainte | Valeur | Notre cas |
|------------|--------|-----------|
| Output disk | 20 GB | ~1.1 GB (2 checkpoints) — OK |
| Runtime | 9h max | ~15-30 min — OK |
| Quota | 30h/semaine | <1% par run — OK |
| GPU | T4 via **UI** (API ignore `accelerator`) | Configurer dans l'UI AVANT push |
| Secrets | Via **UI** (API impossible) | Pas de secrets necessaires (pas de HF push) |
| Dataset path | `/kaggle/input/{slug}/` | Script DOIT tester les deux paths avec fallback |
| Packages | transformers, torch, accelerate pre-installes | pip install tiktoken (stats) |
| Kernel slug | Nouveau slug (pas l'ancien SimCSE+ICT) | `pguillemin/pocket-arbiter-cpt-generation` |

### Pre-flight Kaggle (OBLIGATOIRE avant push)

| Check | Comment verifier | PAS acceptable |
|-------|-----------------|----------------|
| Dataset uploade | `kaggle datasets files pguillemin/pocket-arbiter-gen-data` | "Je l'ai uploade hier" |
| T4 configure | Verifier dans l'UI Kaggle | "C'est dans le metadata" |
| Ancien kernel fini | `kaggle kernels status pguillemin/pocket-arbiter-cpt-generation` | "Il a du finir" |
| Slug incremente si reconfig | Nouveau slug pour nouveau dataset | Meme slug, esperer |

---

## Fichiers et responsabilites

| Fichier | Action | SRP | Lignes max |
|---------|--------|-----|------------|
| `scripts/training/mine_reading_tasks.py` | CREATE | Regex mining FR, 6 types, stats JSON | 200 |
| `scripts/training/prepare_kaggle_dataset.py` | CREATE | Prepare corpus_paragraphs.jsonl + copy tasks | 80 |
| `scripts/training/eval_generation.py` | CREATE | Eval sur 34Q humaines, base vs fine-tune | 150 |
| `scripts/training/tests/test_mine_reading.py` | CREATE | Tests mining sur fixtures corpus | 150 |
| `kaggle/kernel-generation/train_generation.py` | CREATE | Kernel Kaggle self-contained TAPT+SFT | 300 |
| `kaggle/kernel-generation/kernel-metadata.json` | CREATE | Slug, dataset, GPU config | 20 |
| `kaggle/dataset-generation/dataset-metadata.json` | CREATE | Metadata dataset Kaggle | 10 |
| `data/training/reading_tasks.jsonl` | GENERATED | Exercices mines (local) | — |
| `data/training/mining_stats.json` | GENERATED | Yield par type, par doc | — |
| `data/benchmarks/generation_eval_base.json` | GENERATED | Reponses base model 34Q | — |
| `data/benchmarks/generation_eval.json` | GENERATED | Reponses fine-tune 34Q + scores | — |
| `data/benchmarks/tapt_perplexity.json` | GENERATED | Perplexity base vs TAPT | — |
| `models/model_card.json` | UPDATE | Section generation fine-tuning | — |

---

## Risques et mitigations

| Risque | Probabilite | Impact | Mitigation |
|--------|-------------|--------|------------|
| CPT degrade prompting | Moyenne | Generation pire | Gate G3 : eval avant/apres, rollback si degradation |
| Overfitting 381K tokens multi-epoch | Haute | Memorisation | NEFTune + dropout + eval perplexity + early stop patience 3 |
| LA domine le training (63%) | Haute | Biais vers LA | Shuffle paragraphes, pas document order |
| Regex mining FR insuffisant | Basse (yield mesure 3,687) | Pas assez SFT | Gate G2 >= 500, regex elargi si besoin |
| Gemma 270M trop petit pour RAG gen | Moyenne | Qualite insuffisante | Gate G4 → ADR-001 rollback vers Gemma 3 1B |
| Conversion LiteRT post-FFT | Inconnue | Pas de .tflite | Chantier separe, non bloquant pour eval |
| fp16 instabilite Gemma 270M | Basse | NaN/Inf | logging_nan_inf_filter, fallback fp32 (rentre dans T4) |

---

## Avertissement echelle (honnetete intellectuelle)

TAPT a ete concu pour des corpus de 25M+ tokens (Gururangan ACL 2020).
AdaptLLM a ete teste sur des corpus de milliards de tokens (Cheng ICLR 2024).
Notre corpus = 381K tokens.

C'est une reduction de 65x (TAPT) a 2600x+ (AdaptLLM) par rapport aux regimes
originaux des papers. Aucune litterature ne valide ces techniques a cette echelle
sur un modele decoder de 270M params.

Le filet de securite = gates G1-G4, rollback si degradation, eval humaine.
Budget = <1h Kaggle T4, risque = faible (base model preserve, DVC versionne).

---

## Standards

| Standard | Application |
|----------|-------------|
| TAPT (Gururangan ACL 2020) | CPT sur corpus tache, multi-epochs |
| AdaptLLM (Cheng ICLR 2024) | Regex-mined reading comprehension sans LLM |
| NEFTune (Jain et al. arXiv:2310.05914) | Noisy embeddings regularisation |
| LoRA vs FFT (Biderman et al. TMLR 2024) | Full FT > LoRA pour CPT |
| PIT (Jiang et al. arXiv:2402.12847) | Instruct-tuned model + CPT = regime valide |
| Dropout multi-epoch (Xue et al. NeurIPS 2023) | Regularisation efficace pour repetition |
| Scaling data-constrained (Muennighoff NeurIPS 2023) | 4 epochs = seuil rendements decroissants |
| CPT degrade prompting (arXiv:2504.13603) | Risque documente, gate G3 |
| ISO 42001 | Tracabilite (model card, artefacts, rollback) |
| ISO 29119 | Gates, eval, DVC versioning |
| ISO 25010 | Fichiers <= 300 lignes |
| ISO 12207 | Commits conventionnels |

---

## Sources

- [Don't Stop Pretraining (ACL 2020)](https://aclanthology.org/2020.acl-main.740/)
- [AdaptLLM Reading Comprehension (ICLR 2024)](https://arxiv.org/abs/2309.09530)
- [NEFTune (ICLR 2024)](https://arxiv.org/abs/2310.05914)
- [LoRA Learns Less and Forgets Less (2024)](https://arxiv.org/abs/2405.09673)
- [PIT: Pre-Instruction-Tuning (2024)](https://arxiv.org/abs/2402.12847)
- [To Repeat or Not To Repeat (NeurIPS 2023)](https://arxiv.org/abs/2305.13230)
- [Finetune-RAG (2025)](https://arxiv.org/abs/2505.10792)
- [CPT is (not) What You Need (2025)](https://arxiv.org/abs/2504.13603)
- [Scaling Data-Constrained LMs (NeurIPS 2023)](https://arxiv.org/abs/2305.16264)
- [Reuse Don't Retrain (2024)](https://arxiv.org/abs/2407.07263)
- ADR-001: docs/adr/ADR-001-generation-model-selection.md
