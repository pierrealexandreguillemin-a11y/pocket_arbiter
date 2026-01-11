# Changelog - Prompts LLM

> Historique des versions des prompts utilisés par Pocket Arbiter

**Format** : [Semantic Versioning](https://semver.org/)

---

## [Unreleased]

### À venir
- `interpretation_v1.txt` - Prompt principal de synthèse/interprétation (Phase 3)

---

## Historique

| Version | Date | Fichier | Changements |
|---------|------|---------|-------------|
| - | - | - | Aucun prompt créé pour l'instant |

---

## Convention de nommage

```
{fonction}_{version}.txt
```

Exemples :
- `interpretation_v1.txt` - Synthèse et interprétation des règles
- `retrieval_v1.txt` - Instructions pour le retrieval (si nécessaire)
- `validation_v1.txt` - Vérification des réponses

---

## Processus de mise à jour

1. Créer nouvelle version : `{fonction}_v{N+1}.txt`
2. Documenter les changements ici
3. Tester sur le jeu de test (`tests/data/`)
4. Archiver l'ancienne version (ne pas supprimer)

---

## Notes

- Les prompts sont versionnés pour traçabilité ISO 42001
- Chaque modification majeure = nouvelle version
- Tests de régression obligatoires avant déploiement
