# 🚀 Prompt Claude Code - Phase 1 Setup

> Copier ce prompt dans Claude Code (VS Code) pour initialiser le projet

---

## Instructions pour Claude Code

Tu travailles sur le projet **Pocket Arbiter** dans `C:\Dev\pocket_arbiter\`.

> 📱 **Pocket Arbiter** = Application mobile d'assistance à l'arbitrage d'échecs
> Le RAG est un détail d'implémentation interne, pas exposé à l'utilisateur.

### Contexte
- Application Android RAG 100% offline pour arbitres d'échecs
- Règlements FFE (~227 pages) et FIDE (~70 pages) à indexer
- Respect des normes ISO (25010, 42001, 12207, 29119)
- Fichier `CLAUDE_CODE_INSTRUCTIONS.md` contient les règles à respecter

### Ce qui vient d'être ajouté (à ne pas recréer)
```
.github/workflows/ci.yml    # Pipeline CI/CD GitHub Actions
.dvc/config                 # Configuration DVC
models/model_card.json      # Métadonnées modèle
docs/DVC_GUIDE.md          # Guide DVC
```

### Tâches Phase 1

#### 1. Initialiser Git et DVC
```bash
cd C:\Dev\pocket_arbiter
git init
git add .
git commit -m "Phase 0: Initial project structure"

dvc init
git add .dvc
git commit -m "Initialize DVC"
```

#### 2. Créer le repo GitHub
- Nom suggéré: `pocket_arbiter`
- Privé recommandé (contient potentiellement des PDFs sous copyright)
- Ajouter le remote et push

#### 3. Configurer DVC remote (Google Drive)
```bash
# Créer un dossier "pocket-arbiter-dvc" dans Google Drive
# Copier l'ID du dossier depuis l'URL
dvc remote add -d storage gdrive://FOLDER_ID
git add .dvc/config
git commit -m "Configure DVC remote"
```

#### 4. Vérifier que le CI fonctionne
- Push vers GitHub
- Vérifier l'onglet Actions
- Le workflow doit passer (avec warnings OK car projet vide)

### ⚠️ Rappels importants

1. **Lire `CLAUDE_CODE_INSTRUCTIONS.md`** avant toute action
2. **Ne pas créer de code Android** tant que Phase 2 (pipeline données) n'est pas terminée
3. **Documenter chaque décision** dans les fichiers appropriés
4. **Tester localement** avant de dire "c'est fait"

### Prochaine étape après setup

Phase 2 : Pipeline de données
- `scripts/extract_pdf.py` - Extraction texte des PDFs
- `scripts/chunk_text.py` - Découpage en chunks
- Les PDFs sont dans `/mnt/project/` (LA-octobre2025.pdf, FIDE_Arbiters_Manual_2025.pdf)

---

**Commence par l'étape 1 (git init) et confirme quand c'est fait.**
