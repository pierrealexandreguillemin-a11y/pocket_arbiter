# Pocket Arbiter

> Application Android 100% offline pour arbitres d'échecs - Q&A sur les règlements avec IA

[![ISO 25010](https://img.shields.io/badge/ISO-25010-blue)](docs/QUALITY_REQUIREMENTS.md)
[![ISO 42001](https://img.shields.io/badge/ISO-42001-green)](docs/AI_POLICY.md)
[![Android](https://img.shields.io/badge/Android-10%2B-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

---

## 🎯 Objectif

Permettre aux arbitres d'échecs de trouver rapidement les informations réglementaires en posant des questions en langage naturel. L'application fonctionne **100% hors ligne** et cite toujours ses sources.

### Fonctionnalités clés

- 📚 **2 corpus** : Règlements français (FFE) et internationaux (FIDE)
- 🔍 **Recherche sémantique** : Comprend le sens, pas juste les mots-clés
- 🤖 **Synthèse IA** : Explique et interprète les règles
- 📝 **Citations verbatim** : Texte exact + source + page
- ✈️ **100% offline** : Aucune connexion requise
- 🔒 **Vie privée** : Aucune donnée collectée

---

## 📋 Documentation projet

| Document | Description | Norme ISO |
|----------|-------------|-----------|
| [VISION.md](docs/VISION.md) | Vision et objectifs du projet | ISO 12207 |
| [AI_POLICY.md](docs/AI_POLICY.md) | Politique IA responsable | ISO 42001 |
| [QUALITY_REQUIREMENTS.md](docs/QUALITY_REQUIREMENTS.md) | Exigences qualité | ISO 25010 |
| [TEST_PLAN.md](docs/TEST_PLAN.md) | Plan de tests | ISO 29119 |
| [CLAUDE_CODE_INSTRUCTIONS.md](CLAUDE_CODE_INSTRUCTIONS.md) | Instructions pour Claude Code | - |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION ANDROID                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   UI    │───▶│  Embedder   │───▶│  Vector Search      │ │
│  │ (Query) │    │ (MediaPipe) │    │  (FAISS/sqlite-vec) │ │
│  └─────────┘    └─────────────┘    └──────────┬──────────┘ │
│       │                                       │            │
│       │         ┌─────────────┐               │            │
│       │         │   LLM       │◀──────────────┘            │
│       │         │ (Phi-3.5)   │                            │
│       │         └──────┬──────┘                            │
│       │                │                                   │
│       ▼                ▼                                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                    RÉPONSE                           │  │
│  │  • Synthèse interprétative                          │  │
│  │  • Citation verbatim                                │  │
│  │  • Source (règlement + page)                        │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Roadmap

| Phase | Description | Statut |
|-------|-------------|--------|
| 0 | Fondations et gouvernance | 🟢 En cours |
| 1 | Pipeline de données | ⚪ À faire |
| 2 | Prototype Android - Retrieval | ⚪ À faire |
| 3 | Synthèse LLM + Interprétation | ⚪ À faire |
| 4 | Qualité et optimisation | ⚪ À faire |
| 5 | Validation et beta | ⚪ À faire |
| 6 | Production | ⚪ À faire |

---

## 🛠️ Stack technique

### Application Android
- **Langage** : Kotlin
- **UI** : Jetpack Compose
- **Embeddings** : MediaPipe Text Embedder (EmbeddingGemma-300M)
- **LLM** : MediaPipe LLM Inference (Phi-3.5-mini / Gemma)
- **Vector Search** : FAISS ou sqlite-vec
- **Min SDK** : Android 10 (API 29)

### Pipeline de données
- **Langage** : Python 3.10+
- **Extraction PDF** : PyMuPDF (fitz)
- **Embeddings** : sentence-transformers
- **Index** : FAISS

---

## 📂 Structure du projet

```
pocket_arbiter/
├── android/          # Projet Android Studio
├── scripts/          # Scripts Python preprocessing
├── corpus/           # PDF sources (FR + INTL)
├── docs/             # Documentation projet (ISO)
├── prompts/          # Prompts LLM versionnés
├── tests/            # Données et rapports de test
└── README.md
```

---

## 🏁 Démarrage rapide

### Prérequis

- Android Studio Hedgehog+
- Python 3.10+
- Git

### Installation

```bash
# Cloner le repo
git clone https://github.com/[user]/pocket_arbiter.git
cd pocket_arbiter

# Setup Python (pour le pipeline)
cd scripts
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
pip install -r requirements.txt

# Ouvrir le projet Android
# → Ouvrir android/ dans Android Studio
```

### Ajouter les PDF sources

1. Copier les PDF FFE dans `corpus/fr/`
2. Copier les PDF FIDE dans `corpus/intl/`
3. Mettre à jour `corpus/INVENTORY.md`

---

## ⚠️ Avertissement IA

Cette application utilise l'intelligence artificielle pour aider à trouver des informations dans les règlements officiels.

- Les réponses sont des **interprétations indicatives**
- Référez-vous **toujours** au texte officiel cité
- L'arbitre reste **seul responsable** de ses décisions
- **Aucune donnée** n'est collectée ni transmise

---

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE)

---

## 🤝 Contribution

Ce projet est développé avec l'aide de Claude Code (Anthropic).

Pour contribuer :
1. Lire [CLAUDE_CODE_INSTRUCTIONS.md](CLAUDE_CODE_INSTRUCTIONS.md)
2. Respecter les normes ISO documentées
3. Suivre la Definition of Done

---

## 📞 Contact

[À compléter]

