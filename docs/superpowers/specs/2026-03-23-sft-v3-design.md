# SFT v3 Model Training — Design Spec

> **Date**: 2026-03-23
> **Statut**: En cours
> **ISO References**: ISO/IEC 42001:2023 (A.6.2.6), ISO/IEC 5338:2023 (6.3.2)
> **Prerequis**: TAPT checkpoint-88 uploaded (Kaggle dataset updated)
> **Base**: Adapte de docs/superpowers/specs/2026-03-23-sft-v2-assembly-design.md

---

## Contexte

### Diagnostic v1/v2

| Config | Budget (steps x LR) | Resultat |
|--------|---------------------|----------|
| v1 | 306 x 2e-5 = 0.00612 | Sur-apprend (echo 17.6%, overfit 1.33) |
| v2 step 60 | 60 x 1e-5 = 0.0006 | Sous-apprend (70.6% < 10 mots, median 5w) |
| v2 full 102 | 102 x 1e-5 = 0.00102 | Non evalue (coupe trop tot) |
| **v3** | **202 x 1e-5 = 0.00202** | **~2x v2 full, ~1/3 v1** |

**Erreur v2** : loss step-by-step bruitee interpretee comme overfit. La moyenne mobile
descendait encore (2.13 -> 2.08 steps 51-80). Checkpoint-60 selectionne trop tot.

**Strategie v3** : meme LR que v2 (1e-5), 2x plus de steps. Budget bien centre entre
v2 et v1 → garantie de depasser le sweet spot. ~10 checkpoints pour etudier la transition.

**Context RAG** : revue 17 papers (2020-2026) montre que domain SFT peut nuire a la
faithfulness RAG (post-rationalisation). Under-learning potentiellement meilleur que
sweet spot. v3 donne les checkpoints pour etudier ce compromis.

---

## Changements v2 -> v3

| Parametre | v2 | v3 | Justification |
|-----------|----|----|---------------|
| Epochs | 1 | **2** | Plus de budget, cosine scheduler effectif |
| LR | 1e-5 | **1e-5** | Meme que v2, budget augmente par epochs |
| save_total_limit | 6 | **12** | 10 ckpts + final + marge |
| Total steps | 102 | **202** | 2 epochs x 101 steps |
| Warmup steps | 10 | **20** | 10% de 202 |
| Output dir | gemma-270m-cpt-sft-v2 | **gemma-270m-cpt-sft-v3** | Nommage |
| Metrics file | sft_v2_metrics.json | **sft_v3_metrics.json** | Nommage |
| Script | train_sft_v2.py | **train_sft_v3.py** | Rename |

**Inchanges** : batch_size=1, grad_accum=16, cosine scheduler, warmup 10%,
NEFTune alpha=5, weight_decay=0.01, max_grad_norm=1.0, seq_length=1024,
seed=42, fp32 model load + fp16 AMP, gradient_checkpointing=True,
eval_strategy="no" (OOM vocab 262K), save_only_model=True.

---

## Checkpoints attendus

202 steps, save_steps=20 :

| Checkpoint | Step | Epoch ~% |
|------------|------|----------|
| checkpoint-20 | 20 | 10% |
| checkpoint-40 | 40 | 20% |
| checkpoint-60 | 60 | 30% |
| checkpoint-80 | 80 | 40% |
| checkpoint-100 | 100 | 50% |
| checkpoint-120 | 120 | 59% |
| checkpoint-140 | 140 | 69% |
| checkpoint-160 | 160 | 79% |
| checkpoint-180 | 180 | 89% |
| checkpoint-200 | 200 | 99% |
| final (step 202) | 202 | 100% |

Disk : 11 x 1.1 GB = 12.1 GB / 20 GB limit = OK.

---

## Analyse des checkpoints — REGLE ABSOLUE

**Moyenne mobile window 10** pour selectionner le meilleur checkpoint.
NE JAMAIS interpreter la loss step-by-step (cf. feedback_loss_analysis.md).

Selection : checkpoint ou la moyenne mobile est minimale ET stable
(pas de descente rapide qui continue apres).

---

## Gates

| Gate | Verification | Seuil |
|------|-------------|-------|
| G0 | GPU T4, VRAM >= 14 GB | assert |
| G1 | TAPT checkpoint-88 charge, ~270M params | assert |
| G2 | Reading tasks >= 500 | assert |
| G3 | Loss finale < loss initiale | step 202 < step 1 |
| G4 | Token accuracy finale > 50% | trainer_state |
| G5 | Aucun NaN | logging_nan_inf_filter |

---

## Inputs Kaggle

| Dataset | Slug | Statut |
|---------|------|--------|
| TAPT checkpoint | pguillemin/gemma-270m-tapt-checkpoint | DONE (checkpoint-88) |
| Training data | pguillemin/pocket-arbiter-gen-data | OK |

---

## Sources

- Spec v2 : docs/superpowers/specs/2026-03-23-sft-v2-assembly-design.md
- Memory : feedback_loss_analysis.md, project_generation_state.md
- AdaptLLM (Cheng ICLR 2024), NEFTune (Jain ICLR 2024)
