# Checklist Phase 1 - Pipeline de données

> ISO/IEC 12207 - Processus de développement

## Pré-requis (AVANT de coder)

- [ ] Specs extraction PDF documentées
- [ ] Specs chunking documentées
- [ ] Specs embeddings documentées
- [ ] Tests planifiés dans TEST_PLAN.md
- [ ] Corpus PDF présent et inventorié

## Definition of Done - Scripts

### extract_pdf.py
- [ ] Script exécutable sans erreur
- [ ] Arguments CLI documentés (--help)
- [ ] Gère tous les PDF du corpus
- [ ] Output : fichiers texte structurés
- [ ] Logs informatifs (progression)
- [ ] Test unitaire ou manuel documenté
- [ ] Gestion erreurs (PDF corrompu, etc.)

### chunk_text.py
- [ ] Script exécutable sans erreur
- [ ] Arguments CLI documentés
- [ ] Chunking par taille configurable
- [ ] Overlap configurable
- [ ] Métadonnées préservées (source, page)
- [ ] Output : JSON avec chunks
- [ ] Test unitaire

### generate_embeddings.py
- [ ] Script exécutable sans erreur
- [ ] Modèle embeddings choisi et documenté
- [ ] Output : fichier numpy ou similar
- [ ] Performance mesurée (temps/chunk)
- [ ] Test de qualité embeddings

### create_index.py
- [ ] Script exécutable sans erreur
- [ ] Index FAISS créé
- [ ] Test de retrieval basique
- [ ] Métriques de recall documentées

## Validation finale Phase 1

- [ ] Pipeline complet : PDF → Index
- [ ] Tous les scripts testés
- [ ] requirements.txt à jour
- [ ] Documentation dans docs/
- [ ] Recall ≥ 80% sur test set

## Approbation

| Rôle | Date | Signature |
|------|------|-----------|
| Dev | | |
| Review | | |
