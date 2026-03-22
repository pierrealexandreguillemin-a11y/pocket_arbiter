# Eval Generation Kaggle T4 — Design Spec

> **Date**: 2026-03-23
> **Statut**: Approuve
> **Prerequis**: TAPT + SFT checkpoints DONE, eval script existant
> **Target**: 3 JSON eval (base, TAPT, SFT) pour gates G1b/G3/G4a/G4b
> **Plateforme**: Kaggle T4 GPU

---

## Contexte

Le script `scripts/training/eval_generation.py` evalue les modeles generation sur :
- 34 questions humaines (eval humaine : useful/faithful/cited)
- 264 questions annales (auto-citation : regex source/page)

Le script tourne en CPU fp32, ~1-2h par modele, ~3-6h pour 3 modeles.
Sur Kaggle T4 fp16, le meme travail prend ~15-20 min total.

### Checkpoints disponibles

| Modele | Dataset Kaggle | Status |
|--------|---------------|--------|
| Base (Gemma 3 270M IT) | `pguillemin/gemma-3-270m-it` | EXISTE (432 MB) |
| TAPT | `pguillemin/gemma-270m-tapt-checkpoint` | EXISTE (998 MB) |
| SFT (TAPT+SFT) | A creer | ~1055 MB |

### Scripts reutilisables

- `scripts/training/eval_generation.py` — logique eval complete
- `scripts/training/generation_prompt.py` — system prompt RAG

Les deux n'ont aucune dependance locale (seulement stdlib + transformers + torch).
Import sur Kaggle via `sys.path.insert(0, "/kaggle/input/<dataset>/")`.

---

## Architecture

```
Datasets Kaggle (inputs)
+-- pguillemin/gemma-3-270m-it            (base, 432 MB)
+-- pguillemin/gemma-270m-tapt-checkpoint  (TAPT, 998 MB)
+-- pguillemin/gemma-270m-sft-checkpoint   (SFT, ~1 GB)     <-- A CREER
+-- pguillemin/pocket-arbiter-eval-data    (~18 MB)          <-- A CREER
    +-- eval_generation.py
    +-- generation_prompt.py
    +-- gold_standard_annales_fr_v8_adversarial.json
    +-- corpus_v2_fr.db

Kernel: pguillemin/pocket-arbiter-eval-generation
+-- sys.path -> import eval_generation, generation_prompt
+-- Boucle sequentielle: pour chaque modele (base, TAPT, SFT)
|   +-- Load model fp16 GPU
|   +-- 34Q humaines -> responses + scores placeholder
|   +-- 264Q annales -> auto citation check
|   +-- Save JSON -> /kaggle/working/
|   +-- del model; gc.collect(); torch.cuda.empty_cache()
+-- Output: 3 JSON telechargeables
```

### Differences vs script local

| Aspect | Script local | Kernel Kaggle |
|--------|-------------|---------------|
| Device | CPU fp32 | GPU fp32 |
| `device_map` | `"cpu"` | `{"": 0}` |
| `torch_dtype` | `torch.float32` | `torch.float32` |

**NOTE** : Gemma 3 ne supporte PAS fp16 (NaN/empty outputs, issue #36822).
T4 ne supporte pas bf16 nativement. fp32 est la seule option correcte.
270M fp32 = ~1 GB VRAM — tient largement sur T4 15 GB.
| Import | `from scripts.training...` | `sys.path` + import direct |
| Temps | ~1-2h/modele | ~5-7 min/modele |

### VRAM budget (270M inference fp32 sur T4 15 GB)

| Phase | VRAM |
|-------|------|
| Model load fp32 | ~1024 MB |
| Inference (1 seq) | ~1200-1500 MB |
| Entre modeles (apres del) | ~0 MB |
| **Max** | **~1.5 GB (10% T4)** |

Pas de risque OOM — inference seule, pas d'optimizer, pas de gradients.

---

## Datasets a creer

### 1. `pocket-arbiter-eval-data` (~18 MB)

Contenu :
- `eval_generation.py` — copie de `scripts/training/eval_generation.py`
- `generation_prompt.py` — copie de `scripts/training/generation_prompt.py`
- `gold_standard_annales_fr_v8_adversarial.json` — GS
- `corpus_v2_fr.db` — DB SQLite pour lookup contexte
- `dataset-metadata.json`

### 2. `gemma-270m-sft-checkpoint` (~1055 MB)

Contenu (checkpoint final SEULEMENT, pas les epoch intermediaires) :
- `model.safetensors` (1023 MB)
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `generation_config.json`
- `chat_template.jinja`
- `dataset-metadata.json`

Source : `models/kaggle-sft-output/gemma-270m-cpt-sft/` (sans checkpoint-102/204/306, sans training_args.bin)

---

## Kernel (`eval_generation_kaggle.py`, ~200 lignes)

Le kernel :
1. Setup environment (GPU check, sys.path, imports)
2. Resout les paths des 3 modeles et des donnees
3. Boucle sur les 3 modeles sequentiellement :
   - Load model fp16 sur GPU
   - Appelle la logique existante de `eval_generation.py`
   - Save JSON dans `/kaggle/working/`
   - Libere VRAM
4. Log recap final

### Imports et device handling

Le kernel importe SEULEMENT les fonctions pures qui n'ont pas de dependance locale :
- `eval_generation.py` : `load_human_questions`, `load_annales_questions`,
  `load_chunk_context`, `check_citation`
- `generation_prompt.py` : `build_rag_prompt`

Le kernel ne PAS importer `generate_response` car :
1. Elle fait `inputs = tokenizer(text, return_tensors="pt")` sans `.to(model.device)` — crash GPU
2. Elle est dans `main()` qui fait `from scripts.training.generation_prompt import ...` — crash import

Le kernel reimplemente la boucle inference (~15 lignes) avec device handling :
```python
from generation_prompt import build_rag_prompt
from eval_generation import load_human_questions, load_annales_questions, load_chunk_context, check_citation

def generate_response_gpu(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

Note : `from scripts.training.generation_prompt` (ligne 181 de eval_generation.py)
n'est PAS execute car on n'importe pas `main()` ni `generate_response`.
Les fonctions importees (load_*, check_citation) n'ont aucun import local.

### SQLite connexion

Le kernel ouvre la DB UNE fois et passe la connexion, au lieu de 894 open/close.

### Output filenames

Meme noms que le plan (Task 5 Step 10) :
- `generation_eval_base.json`
- `generation_eval_tapt.json`
- `generation_eval.json`

### Deviation du plan principal

Cette spec remplace Task 5 Step 10 du plan principal (eval CPU local → eval Kaggle GPU).
Le script local reste fonctionnel en CPU pour usage futur. Le kernel est un accelerateur,
pas un remplacement permanent.

---

## Outputs

3 fichiers JSON dans `/kaggle/working/`, meme format que le script local :

```json
{
  "model": "path/or/id",
  "questions": [
    {
      "id": "ffe:human:...",
      "question": "...",
      "context": "...",
      "response": "...",
      "scores": {"useful": null, "faithful": null, "cited": null}
    }
  ],
  "auto_citation": {
    "total": 264,
    "cited_count": 42,
    "cited_pct": 15.9
  }
}
```

Telecharges localement via `kaggle kernels output`, puis :
- Pierre remplit `scores` dans `generation_eval.json` (Task 6 du plan principal)
- Script de gate scoring (Task 6 Step 2) evalue G1b/G3/G4a/G4b

---

## Temps estime

| Phase | Temps |
|-------|-------|
| Upload datasets (2) | ~5 min |
| Kernel execution | ~15-20 min |
| Download outputs | ~1 min |
| **Total** | **~25 min** |

vs CPU local : ~3-6h

---

## Commande push

```bash
kaggle kernels push -p kaggle/kernel-eval/ --accelerator NvidiaTeslaT4
```

`--accelerator` est OBLIGATOIRE (sans = P100 par defaut, incompatible cu128+).

## Risques

| Risque | Mitigation |
|--------|------------|
| Dataset path pas monte | Tester les 2 patterns de path avec fallback |
| Model fp16 produit des NaN | Log le count de responses vides |
| GPU pas T4 (P100 par defaut) | `--accelerator NvidiaTeslaT4` dans la commande push |
| Quota GPU | ~20 min << 30h/semaine |
| Import path `from scripts.training...` | Ne PAS importer `generate_response`/`main`, seulement fonctions pures |

---

## Conformite standards industrie ML/AI

### Criteres eval vs standards

| Aspect | Notre eval | Standard industrie | Conformite |
|--------|-----------|-------------------|------------|
| Faithfulness | Critere binaire "faithful" (affirmations inferables du contexte) | RAGAS faithfulness 0-1, cible >= 0.85 (RAGAS 2023, PremAI 2026) | Concept aligne, scoring binaire vs continu |
| Groundedness | "faithful" = toutes affirmations supportees par le contexte | deepset/RAGAS groundedness score | Meme definition |
| Citation checking | Regex source/page, 264Q auto (G4b) | "Citation accuracy 65-70% sans attribution training" (Maxim 2025) | Conforme — large echantillon automatise |
| Human eval criteria | 3 criteres binaires Pass/Fail (useful/faithful/cited) | Pointwise 1-5 ou Pass/Fail (SQuAD) | Acceptable — Pass/Fail est un standard |
| Hybrid auto+human | 264Q auto + 34Q human | Recommande 2025-2026 (PremAI, Maxim, deepset) | Conforme |
| Multi-model comparison | 3 modeles (base, TAPT, SFT) ablation | Standard (ablation study) | Conforme |
| Gates avec rollback | G1b/G3/G4a/G4b, rollback si degradation | Production-grade (MLOps) | Conforme |

### Deviations documentees

| Gap | Standard | Notre choix | Justification |
|-----|----------|-------------|---------------|
| RAGAS faithfulness (TB-04) | Score >= 0.85 (nos QUALITY_REQUIREMENTS) | Non mesure | RAGAS requiert LLM judge cloud — incompatible budget 0€ offline. Critere "faithful" binaire compense partiellement |
| Taille eval humaine | 50 golden questions (PremAI 2026) | 34 questions | GS contient 34 humaines non-QCM. CI 95% = [52%, 84%] pour 70% (documente dans spec generation). G4b (264Q auto) compense la puissance statistique |
| LLM-as-judge | Recommande 2025-2026 (Evidently, Langfuse, Arize) | Non utilise | Budget 0€, modele 270M offline, pas d'API cloud. Le projet a deja utilise LLM-as-judge pour le GS (Gwet's AC1), mais pas applicable ici sans API |
| Score continu | RAGAS/deepset : faithfulness 0-1 | Binaire 0/1 par critere | 270M model = qualite trop variable pour scorer en granulaire. Binaire "ca repond ou pas" est plus actionnable pour la decision gate |

### Standards references

- RAGAS (arXiv:2309.15217) — faithfulness, answer relevance, context precision
- PremAI RAG Evaluation 2026 — "50 golden questions for core metrics, 500 synthetic for regression"
- Maxim RAG Guide 2025 — citation accuracy baseline 65-70%, hybrid auto+human
- deepset Groundedness — fraction claims supportes par contexte
- RAG Evaluation Survey (arXiv:2504.14891) — comprehensive survey methods 2025
- SQuAD 2.0 (Rajpurkar 2018) — Pass/Fail human eval sur questions adversariales
- Nos propres QUALITY_REQUIREMENTS (TB-04 RAGAS >= 0.85, FA-03 fidelite >= 85%)

### Verdict

Eval **pragmatiquement conforme** pour budget 0€, modele 270M offline.
Les gaps (RAGAS, LLM-as-judge, n<50) sont explicitement documentes et
compenses par le volume auto (264Q) et le design multi-gate.
RAGAS et LLM-as-judge sont reportes a la phase Android (si API cloud disponible).

---

## Quality Gates

### Pre-flight (avant push kernel)

| Gate | Verification | Action si FAIL |
|------|-------------|----------------|
| D1 | Dataset `pocket-arbiter-eval-data` uploade : `kaggle datasets files` montre 4 fichiers | STOP — uploader |
| D2 | Dataset `gemma-270m-sft-checkpoint` uploade : `kaggle datasets files` montre model.safetensors | STOP — uploader |
| D3 | Datasets existants confirmes : `kaggle datasets files pguillemin/gemma-3-270m-it` et `pguillemin/gemma-270m-tapt-checkpoint` | STOP — verifier |
| D4 | Aucun kernel precedent en cours : `kaggle kernels status` != "running" | Cancel ou attendre |
| D5 | `--accelerator NvidiaTeslaT4` dans la commande push | STOP — P100 par defaut = incompatible |

### Runtime (dans le kernel)

| Gate | Verification | Action si FAIL |
|------|-------------|----------------|
| K1 | GPU T4 detecte, VRAM >= 14 GB, compute >= 7.5 | STOP — mauvais GPU |
| K2 | 4 datasets montes dans `/kaggle/input/` | STOP — dataset manquant |
| K3 | DB SQLite ouvrable, table `children` existe | STOP — DB corrompue |
| K4 | GS charge, 34 humaines + 264 annales | STOP — GS format incorrect |
| K5 | Pour chaque modele : model.generate produit > 0 tokens sur 1 question test | STOP — model load failed |
| K6 | Responses vides < 10% par modele | WARNING — log mais continuer |

### Post-download (apres recuperation outputs)

| Gate | Verification | Action si FAIL |
|------|-------------|----------------|
| P1 | 3 fichiers JSON presents et parsables | Re-run kernel |
| P2 | Chaque JSON contient 34 `questions` avec `response` non-null | Investiguer model specifique |
| P3 | Chaque JSON contient `auto_citation` avec `total` == 264 | Investiguer GS loading |
| P4 | Responses moyennes > 20 tokens (pas de generations tronquees) | Verifier max_new_tokens |

### Gates generation (du plan principal, evaluees sur les outputs)

| Gate | Condition | Mesure | Action si FAIL |
|------|-----------|--------|----------------|
| G1b | TAPT-only vs base — diagnostic | Auto citation comparison | Documenter |
| G3 | Pas de degradation SFT vs base | Comparaison qualitative 34Q | Rollback TAPT ou base |
| G4a | Qualite >= 70% (34Q humaines) | Score = % ou useful+faithful+cited = 1 | ADR-001 : Gemma 3 1B IT |
| G4b | Citation auto >= 80% (264Q annales) | Regex source/page | Investigate |

---

## Checkpoints et outputs

### Artefacts par etape (tous conserves)

| Etape | Artefact | Chemin | Versionning | Rollback |
|-------|----------|--------|-------------|----------|
| 1. Prep SFT dataset | Checkpoint final clean | `kaggle/sft-checkpoint/` | git (metadata) + Kaggle upload | Re-copier depuis `models/kaggle-sft-output/` |
| 2. Prep eval dataset | Scripts + GS + DB | `kaggle/eval-data/` | git (metadata) + Kaggle upload | Re-copier depuis sources |
| 3. Kernel push | Code kernel | `kaggle/kernel-eval/eval_generation_kaggle.py` | git | Editer et re-push |
| 4. Kernel output | 3 JSON + log | `/kaggle/working/` (Kaggle) | Ephemere — telecharger | Re-run kernel |
| 5. Download | 3 JSON eval | `data/benchmarks/generation_eval_*.json` | git | Re-download ou re-run |
| 6. Human eval | Scores remplis | `data/benchmarks/generation_eval.json` | git | Pierre re-score |
| 7. Model card | Resultats eval | `models/model_card.json` | git | Revert commit |

### Outputs kernel (dans `/kaggle/working/`)

```
/kaggle/working/
├── generation_eval_base.json      # Base model (34Q + 264Q auto citation)
├── generation_eval_tapt.json      # TAPT-only (34Q + 264Q auto citation)
├── generation_eval.json           # TAPT+SFT (34Q + 264Q auto citation)
└── eval_generation.log            # Log complet (timings, warnings, gate checks)
```

Chaque JSON : ~500 KB (34 responses completes + 264 resultats citation bool).
Log : ~50 KB (diagnostics, VRAM, timings par modele).

### Fichiers locaux crees/modifies

| Fichier | Action | Contenu |
|---------|--------|---------|
| `kaggle/sft-checkpoint/` | CREATE | model.safetensors + config + tokenizer (copie clean) |
| `kaggle/sft-checkpoint/dataset-metadata.json` | CREATE | Metadata Kaggle dataset |
| `kaggle/eval-data/` | CREATE | 2 scripts + GS + DB |
| `kaggle/eval-data/dataset-metadata.json` | CREATE | Metadata Kaggle dataset |
| `kaggle/kernel-eval/eval_generation_kaggle.py` | CREATE | Kernel orchestrateur (~100 lignes) |
| `kaggle/kernel-eval/kernel-metadata.json` | CREATE | Metadata kernel Kaggle |
| `data/benchmarks/generation_eval_base.json` | CREATE | Output base model |
| `data/benchmarks/generation_eval_tapt.json` | CREATE | Output TAPT |
| `data/benchmarks/generation_eval.json` | CREATE | Output SFT (+ scores humains apres Task 6) |
| `models/model_card.json` | UPDATE | Section generation eval |

### Regle de non-destruction

Aucune etape ne detruit un artefact precedent. Le kernel produit des fichiers
dans `/kaggle/working/` (espace ephemere Kaggle). Les checkpoints modeles
(`models/kaggle-output/`, `models/kaggle-sft-output/`) ne sont PAS touches.

---

## Definition of Done

Le kernel eval est **DONE** quand :

- [ ] Dataset `pocket-arbiter-eval-data` uploade (4 fichiers : 2 scripts + GS + DB)
- [ ] Dataset `gemma-270m-sft-checkpoint` uploade (checkpoint final ~1 GB)
- [ ] Kernel `pocket-arbiter-eval-generation` cree et pousse avec `--accelerator NvidiaTeslaT4`
- [ ] Kernel termine avec status COMPLETE
- [ ] 3 JSON telecharges dans `data/benchmarks/` :
  - `generation_eval_base.json` (34Q + 264Q auto citation)
  - `generation_eval_tapt.json` (34Q + 264Q auto citation)
  - `generation_eval.json` (34Q + 264Q auto citation)
- [ ] Gates P1-P4 PASS (fichiers valides, responses non-vides)
- [ ] Pierre remplit `scores` dans `generation_eval.json` (eval humaine 34Q)
- [ ] Gates G1b/G3/G4a/G4b evaluees et documentees
- [ ] `models/model_card.json` mis a jour avec resultats eval
- [ ] Artefacts benchmarks commites dans git

---

## Checklist Kaggle Deployment (skill cross-check)

Verification croisee avec le skill `kaggle-deployment` :

| Regle skill | Application ici | Status |
|-------------|----------------|--------|
| `--accelerator NvidiaTeslaT4` obligatoire | Commande push documentee | OK |
| `enable_gpu: true` insuffisant | Pas de dependance au metadata seul | OK |
| Secrets via API impossible | Pas de secrets necessaires (inference only, modeles en datasets) | OK |
| `model_sources` ne monte pas en batch | Tous les modeles en `dataset_sources` | OK |
| Dataset paths : tester 2 patterns avec fallback | Kernel teste `/kaggle/input/{slug}/` et `/kaggle/input/datasets/{user}/{slug}/` | A IMPLEMENTER |
| `device_map={"": 0}` pour single GPU | Spec prescrit `{"": 0}` | OK |
| fp16 model OK pour inference (pas d'AMP/GradScaler) | Spec prescrit `torch.float16` pour load, `torch.no_grad()` pour generate | OK |
| `del model; gc.collect(); torch.cuda.empty_cache()` entre phases | Spec prescrit liberation VRAM entre modeles | OK |
| Slug : verifier apres premier push, update metadata si renomme | Note dans risques | A VERIFIER |
| Failure = diagnostic complet avant retry | Spec renvoie au skill | OK |
| Gated models : upload as dataset, not HF download | Les 3 modeles sont des datasets Kaggle | OK |

---

## Ce qu'on ne fait PAS

- Pas de modification du script eval local (il reste fonctionnel en CPU)
- Pas de GPU inference en local
- Pas de parallelisme (3 modeles sequentiels, plus simple)
- Pas de training (inference only)
- Pas de RAGAS/LLM-as-judge (budget 0€, offline)
- Pas d'augmentation du eval set humain au-dela de 34Q (pas de nouvelles questions disponibles)
