# Training Parameters Correction — Design Spec

> **Date**: 2026-03-24
> **Statut**: En cours
> **ISO References**: ISO/IEC 42001:2023 (A.6.2.6), ISO/IEC 5338:2023 (6.3.2, 6.3.4)
> **Prerequis**: SFT v3 DONE (checkpoint-140), eval kernel v4 en cours
> **Scope**: TAPT + SFT complets — correction parametres + benchmark chaque etape

---

## Contexte

### Probleme identifie

Revue exhaustive de la litterature (2024-2026) et du guide officiel Google Gemma FFT.
3 parametres sous-optimaux affectent TAPT ET SFT :

| Parametre | Actuel | Google/Litterature | Impact |
|-----------|--------|-------------------|--------|
| attention_dropout | 0.1 (injecte) | **0.0** (Gemma default) | Reduit capacite, nuit en 1-5 epochs (arXiv:2505.24788) |
| lr_scheduler | cosine | **constant** (Google FFT guide) | Cosine decay premature sur ~100-200 steps (WSO arXiv:2603.16127) |
| Loss masking SFT | Full sequence | **Assistant-only** | Gradient gaspille sur tokens prompt (TRL docs) |
| **LR TAPT** | **5e-6** (script) | **5e-5** (Google FFT guide) | **10x trop bas !** model_card.json documentait 5e-5 par erreur |

### Eval v4 — Reference (2026-03-24)

Resultats avec prompt v2 + gen params state-of-the-art (TAPT v1 + SFT v3) :

| Modele | Citations | Empty | Median mots | Comportement |
|--------|-----------|-------|-------------|--------------|
| **Base** | **43.9%** | 0 | 36 | Lit contexte, cite, utilise "non trouvee" |
| TAPT v1 | 36.4% | 0 | 17 | Post-rationalise (pretend citer, fabule) |
| SFT v3 | 28.8% | 0 | 15 | Echo questions (full-seq loss) |

**Paradoxe confirme** : plus de FFT = moins de faithfulness.
**Objectif TAPT v2 + SFT v4** : depasser base (43.9%) avec les corrections.

### Pourquoi retrainer depuis TAPT

Un SFT optimal sur un TAPT sous-optimal herite des problemes :
- Le TAPT v1 a ete entraine avec dropout 0.1 et cosine → poids sous-optimaux
- Meme si SFT v4 corrige ses propres params, la base TAPT est corrompue
- **Rigueur ISO** : chaque etape doit etre optimale independamment

### Parametres secondaires (Tier 2)

| Parametre | Actuel | Propose | Impact |
|-----------|--------|---------|--------|
| warmup | 10% | **5%** | -10 steps de LR sous-optimal |
| optimizer | adamw_torch | **adamw_torch_fused** | ~5% speedup, zero impact qualite |

---

## Pipeline versionne avec benchmarks

```
TAPT v2 (params corriges)
    │
    ├─ Benchmark T1: perplexite TAPT v2 vs v1
    │
    ▼
SFT v4 sur TAPT v2 (params corriges + prompt v2 + assistant-only loss)
    │
    ├─ Benchmark T2: loss/accuracy SFT v4 vs v3
    │
    ▼
Eval v4 (prompt v2 + gen params state-of-the-art)
    │
    ├─ Benchmark T3: base vs TAPT v2 vs SFT v4 (298 questions)
    │   Comparaison avec eval v3 (TAPT v1 + SFT v3)
    │
    ▼
Decision: gain justifie-t-il la complexite ?
```

### Versions et tracabilite

| Version | Base | Params corriges | Benchmark |
|---------|------|----------------|-----------|
| TAPT v1 | Gemma 270M IT | dropout=0.1, cosine, 5 epochs LR 5e-5 | ppl 37.74→7.98 |
| **TAPT v2** | Gemma 270M IT | **dropout=0.0, constant, 5 epochs LR 5e-5** | ppl a mesurer |
| SFT v3 | TAPT v1 ep4 | dropout=0.1, cosine, 2ep LR 1e-5, full-seq loss | MA loss 1.716 |
| **SFT v4** | **TAPT v2 best** | **dropout=0.0, constant, 2ep LR 1e-5, assistant-only, prompt v2** | a mesurer |

---

## TAPT v2 — Changements

### Script: train_tapt_v2.py (fork de kernel TAPT existant)

| Parametre | v1 (reel) | v2 | Source |
|-----------|-----------|-----|--------|
| **lr** | **5e-6** (10x trop bas !) | **5e-5** | Google FFT guide. model_card disait 5e-5 par erreur |
| attention_dropout | 0.1 (injecte) | **0.0** (pas d'injection) | Google Gemma default, arXiv:2505.24788 |
| lr_scheduler_type | cosine | **constant_with_warmup** | Google FFT guide, WSO arXiv:2603.16127 |
| warmup_pct | 0.1 | **0.05** | Secret Recipe arXiv:2412.13337 |
| optim | adamw_torch | **adamw_torch_fused** | Google FFT guide |
| epochs | 5 | **5** | Inchange |
| batch_size | 1 | **1** | Inchange (script reel, pas 2 comme documente) |
| grad_accum | 16 | **16** | Inchange |
| fp32 + AMP | oui | **oui** | Inchange |

### Benchmark T1 : TAPT v2 vs v1

| Metrique | TAPT v1 | TAPT v2 | Gate |
|----------|---------|---------|------|
| Perplexite finale | 7.98 | a mesurer | < 7.98 (amelioration) |
| Perplexite epoch 4 | ~8.5 | a mesurer | Selectionner meilleur epoch |
| Repetitions (5-gram) | 77% | a mesurer | Informatif |

---

## SFT v4 — Changements

### Script: train_sft_v4.py (fork de train_sft_v3.py)

| Parametre | v3 | v4 | Source |
|-----------|----|----|--------|
| Base model | TAPT v1 ep4 | **TAPT v2 best** | Pipeline corrige |
| attention_dropout | 0.1 (injecte) | **0.0** (pas d'injection) | Google default |
| lr_scheduler_type | cosine | **constant_with_warmup** | Google FFT, WSO |
| warmup_pct | 0.1 | **0.05** | Secret Recipe |
| optim | adamw_torch | **adamw_torch_fused** | Google FFT |
| Loss | Full sequence | **Assistant-only** (prompt-completion format) | TRL docs |
| Prompt dans data | AdaptLLM generique | **Prompt RAG v2** | Alignment train/inference |
| epochs | 2 | **2** | Inchange |
| lr | 1e-5 | **1e-5** | Inchange |
| save_steps | 20 | **20** | Inchange |
| save_only_model | True | **True** | Inchange |

### Reformatage donnees SFT

Les 1802 tasks passent de format `messages` a format `prompt-completion` :

```python
# AVANT (messages, full-sequence loss)
{"messages": [
    {"role": "user", "content": "Resumez: [passage]"},
    {"role": "assistant", "content": "[resume]"}
]}

# APRES (prompt-completion, assistant-only loss)
{"prompt": "SYSTEM_PROMPT\n\nCONTEXTE:\n[passage]\n\nQUESTION: Resumez ce passage.",
 "completion": "[resume]"}
```

Le prompt RAG v2 est integre dans le champ `prompt` → le modele apprend a repondre
dans le format qu'il verra a l'inference.

### Benchmark T2 : SFT v4 vs v3

| Metrique | SFT v3 | SFT v4 | Gate |
|----------|--------|--------|------|
| MA(10) loss min | 1.716 | a mesurer | < 1.716 |
| MA(10) accuracy | 0.620 | a mesurer | > 0.620 |
| Overfit ratio | 1.08 | a mesurer | < 1.5 |
| Best checkpoint step | 140 | a mesurer | MA(10) analysis |

### Benchmark T3 : Eval comparative finale

3 modeles × 298 questions × prompt v2 + gen params state-of-the-art :

| Metrique | Base | TAPT v2 | SFT v4 | vs SFT v3 |
|----------|------|---------|--------|-----------|
| Empty responses | a mesurer | a mesurer | a mesurer | SFT v3: TBD |
| Auto-citations % | a mesurer | a mesurer | a mesurer | SFT v3: TBD |
| Longueur mediane | a mesurer | a mesurer | a mesurer | SFT v3: TBD |
| Repetitions 5-gram | a mesurer | a mesurer | a mesurer | SFT v3: TBD |
| Faithfulness | a mesurer | a mesurer | a mesurer | Nouveau critere |

---

## Ordre d'execution

1. **Attendre eval v4** (en cours, SFT v3 + prompt v2 + gen params) → reference
2. **TAPT v2** : creer kernel, push, download, benchmark T1
3. **Upload TAPT v2 best** checkpoint sur Kaggle
4. **Reformater donnees SFT** : messages → prompt-completion avec prompt RAG v2
5. **Upload dataset generation v2** sur Kaggle
6. **SFT v4** : creer kernel, push, download, benchmark T2, MA(10) analysis
7. **Upload SFT v4 best** checkpoint sur Kaggle
8. **Eval v5** : base vs TAPT v2 vs SFT v4, benchmark T3
9. **Comparer** eval v5 vs eval v4 → decision

---

## Gates ISO

### TAPT v2
| Gate | Verification | Seuil |
|------|-------------|-------|
| GT0 | GPU T4, VRAM >= 14 GB | assert |
| GT1 | Corpus paragraphs >= 5000 | assert |
| GT2 | Perplexite finale < baseline (37.74) | assert |
| GT3 | Aucun NaN | assert |

### SFT v4
| Gate | Verification | Seuil |
|------|-------------|-------|
| GS0 | TAPT v2 checkpoint charge, ~270M params | assert |
| GS1 | Reading tasks >= 500 | assert |
| GS2 | Loss finale < loss initiale | assert |
| GS3 | Prompt-completion format verifie (loss masking) | assert |
| GS4 | Overfit ratio < 1.5 | warning |

### Eval v5
| Gate | Verification | Seuil |
|------|-------------|-------|
| GE0 | 3 fichiers, 298 questions chacun | assert |
| GE1 | SFT v4 empty <= SFT v3 empty | warning |
| GE2 | SFT v4 citations >= SFT v3 citations | informatif |

---

## Ce qu'on ne fait PAS (Tier 3, futur)

- Pas de packing (v5, isoler variables)
- Pas de curriculum learning (v5)
- Pas de knowledge distillation (necessite API budget)
- Pas de DPO/ORPO (necessite preference pairs)
- Pas de label smoothing (inefficace avec vocab 262K, arXiv:2508.00264)
- Pas de changement beta2 (impact faible, isoler variables)

---

## Sources

- [Google Gemma FFT Guide](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune)
- [Google Gemma 270M Blog](https://developers.googleblog.com/en/own-your-ai-fine-tune-gemma-3-270m-for-on-device/)
- [Drop Dropout on Single-Epoch (arXiv:2505.24788)](https://arxiv.org/abs/2505.24788)
- [WSO: Pre-Training without LR Decay (arXiv:2603.16127)](https://arxiv.org/abs/2603.16127)
- [Secret Recipe SFT Small LLMs (arXiv:2412.13337)](https://arxiv.org/abs/2412.13337)
- [NEFTune (ICLR 2024, arXiv:2310.05914)](https://arxiv.org/abs/2310.05914)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- [Calibrated LMs Label Smoothing (arXiv:2508.00264)](https://arxiv.org/abs/2508.00264)
