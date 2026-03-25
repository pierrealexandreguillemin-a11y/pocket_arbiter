# Cleanup Reference — 2026-03-25

Fichiers supprimes localement pour liberer ~93 GB. Tous recuperables.

## Checkpoints Kaggle (kernel outputs)

| Repertoire supprime | Taille | Source | Commande recovery |
|---------------------|--------|--------|-------------------|
| models/kaggle-tapt-v2-output/ | 17 GB | kernel pguillemin/pocket-arbiter-tapt-v2 v1 | `kaggle kernels output pguillemin/pocket-arbiter-tapt-v2 -p models/kaggle-tapt-v2-output/` |
| models/kaggle-sft-v3-output/ | 13 GB | kernel pguillemin/pocket-arbiter-sft-generation v3 | `kaggle kernels output pguillemin/pocket-arbiter-sft-generation -p models/kaggle-sft-v3-output/` |
| models/kaggle-sft-output/ | 11 GB | kernel pguillemin/pocket-arbiter-cpt-sft-generation v? | `kaggle kernels output pguillemin/pocket-arbiter-cpt-sft-generation -p models/kaggle-sft-output/` |
| models/kaggle-sft-v2-output/ | 7.3 GB | kernel pguillemin/pocket-arbiter-sft-generation v2 | `kaggle kernels output pguillemin/pocket-arbiter-sft-generation --version 2 -p models/kaggle-sft-v2-output/` |
| models/kaggle-output/ | 7.1 GB | kernel pguillemin/pocket-arbiter-cpt-sft-generation v? | `kaggle kernels output pguillemin/pocket-arbiter-cpt-sft-generation -p models/kaggle-output/` |

## Modeles HuggingFace

| Repertoire supprime | Taille | Source | Commande recovery |
|---------------------|--------|--------|-------------------|
| models/embeddinggemma-chess-arbiter-fr/ | 12 GB | HF Pierrax/embeddinggemma-chess-arbiter-fr | `huggingface_hub.snapshot_download('Pierrax/embeddinggemma-chess-arbiter-fr', local_dir='models/embeddinggemma-chess-arbiter-fr')` |

## Artefacts reproductibles

| Repertoire supprime | Taille | Comment reproduire |
|---------------------|--------|--------------------|
| models/unsloth/embeddinggemma-300M-Q8_0.gguf | 314 MB | Convertir depuis google/embeddinggemma-300m avec llama.cpp |

## DVC cache

| Repertoire supprime | Taille | Recovery |
|---------------------|--------|----------|
| .dvc/cache | 25 GB | `dvc checkout` (re-telecharge depuis dvc_remote C:/Dev/dvc_remote) |
