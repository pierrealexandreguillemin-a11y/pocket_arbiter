# Checklist Feature IA : [NOM_FEATURE]

> Template ISO/IEC 42001 - AI Management System

## 1. CONCEPTION IA

### Évaluation du risque IA
- [ ] Type de feature IA identifié (retrieval/génération/classification)
- [ ] Risque d'hallucination évalué (Low/Medium/High)
- [ ] Impact d'une erreur évalué (Low/Medium/High)
- [ ] Mesures de mitigation définies

### Sources de données
- [ ] Sources documentées (corpus/INVENTORY.md)
- [ ] Qualité des sources vérifiée
- [ ] Droits d'utilisation confirmés
- [ ] Biais potentiels identifiés

### Design anti-hallucination
- [ ] Architecture RAG (Retrieval-Augmented Generation)
- [ ] Pas de génération sans context retrieval
- [ ] Format de citation défini
- [ ] Seuil de confiance défini (si applicable)

---

## 2. PLANIFICATION IA

### Prompt engineering
- [ ] Prompt initial rédigé
- [ ] Prompt versionné dans `prompts/`
- [ ] Instructions anti-hallucination incluses
- [ ] Format de sortie spécifié

### Tests planifiés
- [ ] Test set défini (questions + réponses attendues)
- [ ] Cas adversaires définis
- [ ] Métriques de succès définies
  - [ ] Recall retrieval ≥ 80%
  - [ ] Précision citations ≥ 95%
  - [ ] Taux hallucination = 0%

---

## 3. RÉALISATION IA

### Implémentation
- [ ] Retrieval implémenté
- [ ] LLM intégré avec grounding
- [ ] Citations automatiques
- [ ] Disclaimer IA affiché

### Tests exécutés
- [ ] Tests retrieval passent (recall ≥ 80%)
- [ ] Tests citation passent
- [ ] Tests adversaires passent
- [ ] Aucune hallucination détectée

### Documentation
- [ ] Prompt documenté
- [ ] Model card à jour
- [ ] Limitations documentées

---

## 4. RÉSULTAT IA (Definition of Done ISO 42001)

### Transparence
- [ ] L'utilisateur sait qu'il interagit avec une IA
- [ ] Le disclaimer est visible
- [ ] Les sources sont citées

### Fiabilité
- [ ] Toute réponse a une source vérifiable
- [ ] Pas de génération "créative" non sourcée
- [ ] Cas d'incertitude gérés ("Je ne sais pas")

### Traçabilité
- [ ] Prompt versionné
- [ ] Tests documentés
- [ ] Métriques enregistrées

### Responsabilité
- [ ] L'utilisateur reste décisionnaire
- [ ] L'IA ne prend pas de décision à sa place
- [ ] Recours humain possible

---

## Métriques finales

| Métrique | Cible | Mesuré | Status |
|----------|-------|--------|--------|
| Recall retrieval | ≥ 80% | | |
| Précision citations | ≥ 95% | | |
| Hallucination rate | 0% | | |
| Latence moyenne | < 5s | | |

---

## Approbation ISO 42001

| Vérification | Date | Validé par |
|--------------|------|------------|
| Design anti-hallucination | | |
| Tests adversaires | | |
| Disclaimer visible | | |
| Citations vérifiées | | |
