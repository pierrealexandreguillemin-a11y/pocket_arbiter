# SFT v2 Model Assembly — Design Spec

> **Date**: 2026-03-23
> **Statut**: Approuve
> **ISO References**: ISO/IEC 42001:2023 (A.6.2.6 model lifecycle, A.7.2 verification),
>   ISO/IEC 5338:2023 (6.3.2 model training, 6.3.4 evaluation)
> **Prerequis**: TAPT checkpoint-88 uploaded (Kaggle dataset updated)

---

## Contexte

### Probleme identifie

L'analyse du SFT v1 revele **2 couches d'overfit empilees** :

1. **TAPT overfit** : epoch 5 (checkpoint-110) loss remonte (+0.07 vs epoch 4).
   Le TAPT a appris des patterns de repetition du corpus FFE.
2. **SFT overfit** : loss augmente des epoch 2 (+0.20), epoch 3 continue (+0.04).
   76.5% des reponses TAPT sont repetitives (boucles 5-gram).

### Evidence

| Signal | TAPT | SFT v1 |
|--------|------|--------|
| Loss end-of-epoch | 2.69, 2.39, 2.29, **2.22**, ↑2.29 | **1.05**, ↑1.25, ↑1.29 |
| Best epoch | **4** (checkpoint-88) | **1** (checkpoint-102) |
| Overfit signal | Epoch 5 loss +0.07 | Epoch 2 loss +0.20 |

### Standards industrie consultes

| Source | Finding | Impact |
|--------|---------|--------|
| Gururangan ACL 2020 | TAPT original = 100 epochs MLM. CLM overfit plus vite (meme cible a chaque epoch) | 4 epochs CLM = raisonnable |
| arXiv 2508.04117 | Over-memorization starts epoch 3. "1-4 epochs" recommandes. Token accuracy as stopping criterion | 1 epoch SFT + save_steps |
| arXiv 2504.12491 | Perplexity = indicateur trompeur (33% accuracy comme predicteur) | Evaluer sur downstream, pas perplexity seule |
| arXiv 2402.17400 | "Smaller models particularly sensitive to CPT — most significant rates of learning AND forgetting" | LR plus conservateur |
| AdaptLLM ICLR 2024 | LR 1e-5, 10K steps, batch 32. +3-5pp sur 3 domaines (7B) | LR 1e-5 confirme |

---

## Solution

### Architecture

```
Existant (pas de re-training):
  TAPT checkpoint-88 (epoch 4, loss 2.22)  ← Deja sur Kaggle (mis a jour)

Kernel SFT v2 (~20 min Kaggle T4):
  1 epoch, LR 1e-5, save_steps=20
  → 5 checkpoints intermediaires + final
  → sft_v2_metrics.json

Analyse locale (~5 min):
  Choisir best checkpoint (token_accuracy max)
  Upload comme dataset Kaggle

Kernel Eval (~30 min Kaggle T4):
  3 modeles: base + TAPT-ep4 + SFT-v2-best
  → generation_eval_{base,tapt,sft}.json (v2)
```

### Changements SFT v1 → v2

| Parametre | v1 (overfit) | v2 | Justification |
|-----------|-------------|-----|---------------|
| Base checkpoint | TAPT epoch 5 (ckpt-110) | **TAPT epoch 4 (ckpt-88)** | Epoch 5 overfit (+0.07 loss) |
| Epochs | 3 | **1** | Loss monte des epoch 2. arXiv 2508.04117 |
| LR | 2e-5 | **1e-5** | AdaptLLM standard. 2x moins agressif |
| save_strategy | "epoch" | **"steps"** | Checkpoints intra-epoch |
| save_steps | — | **20** | 5 checkpoints dans l'epoch (101 steps total) |
| save_total_limit | 3 | **6** | Garder tous les checkpoints |

**Inchanges** : batch_size=1, grad_accum=16, cosine scheduler, warmup 10%,
NEFTune alpha=5, weight_decay=0.01, max_grad_norm=1.0, seq_length=1024,
eval_strategy="no" (OOM vocab 262K), seed=42, enable_internet=true (pip install trl).

**Precision** : model charge en fp32 (`torch.float32`), training en fp16 AMP
(`fp16=True` dans SFTConfig), inference/eval en fp32 (Gemma 3 fp16 inference
cause NaN — issue #36822, documente dans eval spec).

**Disk** : 6 checkpoints × ~1 GB = ~6-7 GB sur 20 GB output Kaggle. OK.

**Note** : checkpoint-100 et le modele final (step 101) sont quasi-identiques
(1 step d'ecart). Traites comme un seul candidat lors de la selection.

### Inputs Kaggle

| Dataset | Slug | Contenu | Statut |
|---------|------|---------|--------|
| TAPT checkpoint | `pguillemin/gemma-270m-tapt-checkpoint` | checkpoint-88 (epoch 4) | DONE (v2 uploaded) |
| Training data | `pguillemin/pocket-arbiter-gen-data` | reading_tasks.jsonl (1802) | OK |
| Base model | `pguillemin/gemma-3-270m-it` | Gemma 3 270M IT | OK |
| Eval data | `pguillemin/pocket-arbiter-eval-data` | DB + GS + eval scripts | OK |

Paths resolus Kaggle :
- `/kaggle/input/gemma-270m-tapt-checkpoint/model.safetensors`
- `/kaggle/input/pocket-arbiter-gen-data/reading_tasks.jsonl`
- `/kaggle/input/gemma-3-270m-it/model.safetensors` (base, pour eval)
- `/kaggle/input/pocket-arbiter-eval-data/corpus_v2_fr.db`

### Output attendu kernel SFT v2

```
/kaggle/working/
  gemma-270m-cpt-sft-v2/
    model.safetensors          # final (step 101)
    config.json
    tokenizer.json
    tokenizer_config.json
    generation_config.json
    chat_template.jinja
    training_args.bin
    checkpoint-20/             # step 20
      model.safetensors
      trainer_state.json
      ...
    checkpoint-40/             # step 40
    checkpoint-60/             # step 60
    checkpoint-80/             # step 80
    checkpoint-100/            # step 100
  sft_v2_metrics.json          # loss, token_accuracy, overfit check
```

---

## Gates

### Kernel SFT v2

| Gate | Verification | Seuil | Action FAIL |
|------|-------------|-------|-------------|
| G0 | GPU disponible, VRAM >= 14 GB | assert | Abort |
| G1 | TAPT checkpoint charge, params ~270M | assert | Abort |
| G2 | Reading tasks >= 500 | assert | Abort |
| G3 | Loss finale < loss initiale | step 101 < step 1 | Abort |
| G4 | Token accuracy finale > 50% | trainer_state | Warning |
| G5 | Aucun NaN dans la loss | logging_nan_inf_filter | Abort |

### Analyse locale (entre les 2 kernels)

| Check | Methode | Decision |
|-------|---------|----------|
| Best checkpoint | Min loss parmi 5 checkpoints (trainer_state.json) | Upload celui-la |
| Token accuracy | Verifier mean_token_accuracy (TRL log par defaut) | Confirme le choix loss |
| Overfit signal | Loss remonte entre checkpoints consecutifs | Prendre checkpoint avant remontee |
| Degeneration | 5-gram repetition % sur 5 test prompts (meme script que analyse v1) | Si > 0% → checkpoint precedent |

### Kernel Eval

| Gate | Verification | Seuil |
|------|-------------|-------|
| P1 | 3 fichiers, meme nombre de questions | = |
| P2 | Empty responses | SFT v2 <= SFT v1 (0) |
| P3 | 264 annales presentes | 264 |
| P4 | Longueur moyenne reponses | > 15 tokens |

### Comparaison v1 vs v2 (non bloquant)

| Metrique | SFT v1 (ref) | Seuil v2 |
|----------|-------------|----------|
| Repetitions (5-gram) | 0% | 0% |
| Empty responses | 0 | 0 |
| Question echo | 17.6% | < 10% |
| Citations auto | 33.0% | >= 30% |

---

## Fichiers a modifier/creer

| Fichier | Action | Description |
|---------|--------|-------------|
| `kaggle/kernel-sft/train_sft_v2.py` | CREATE | Script SFT v2 (fork de train_sft.py) |
| `kaggle/kernel-sft/kernel-metadata.json` | MODIFY | Pointer vers train_sft_v2.py |
| `kaggle/kernel-eval/eval_generation_kaggle.py` | KEEP | Inchange (base depuis Kaggle dataset) |
| `kaggle/kernel-eval/kernel-metadata.json` | KEEP | Inchange |
| `kaggle/sft-checkpoint/` | UPDATE | Apres analyse locale, best checkpoint v2 |

---

## Ce qu'on ne fait PAS

- Pas de re-training TAPT (checkpoint-88 est bon, loss decroit epochs 1-4)
- Pas de changement d'architecture (270M, ADR-001 gate rollback si qualite < 70%)
- Pas de LoRA (Full FT confirme par AdaptLLM: "Full FT > LoRA" Biderman TMLR 2024)
- Pas de early stopping automatique (1 seul epoch, save_steps suffit)
- Pas de eval_strategy="steps" (OOM vocab 262K, eval manuelle post-training)

---

## Sources

- [Don't Stop Pretraining (Gururangan ACL 2020)](https://aclanthology.org/2020.acl-main.740/)
- [AdaptLLM (Cheng ICLR 2024)](https://arxiv.org/html/2309.09530v2)
- [Over-memorization in LLM Finetuning (arXiv 2508.04117)](https://arxiv.org/html/2508.04117)
- [Revisiting Pre-training Indicators (arXiv 2504.12491)](https://arxiv.org/html/2504.12491v2)
- [Investigating Continual Pretraining (arXiv 2402.17400)](https://arxiv.org/abs/2402.17400)
- [Learning Dynamics in CPT (arXiv 2505.07796)](https://arxiv.org/html/2505.07796v2)
- [NEFTune (Jain ICLR 2024)](https://arxiv.org/abs/2310.05914)
