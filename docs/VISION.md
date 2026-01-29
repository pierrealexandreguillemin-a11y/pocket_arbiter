# Vision Projet - Arbitre Echecs RAG

> **Document ID**: SPEC-VIS-001
> **ISO Reference**: ISO/IEC 12207:2017 - Processus du cycle de vie logiciel
> **Version**: 2.1
> **Date**: 2026-01-29
> **Statut**: En cours
> **Classification**: Interne
> **Auteur**: Equipe projet
> **Mots-cles**: vision, objectifs, RAG, echecs, arbitre, mobile, offline

---

## 1. Résumé exécutif

Application mobile Android 100% offline permettant aux arbitres d'échecs d'interroger les règlements fédéraux via recherche sémantique et d'obtenir des réponses synthétisées avec citations verbatim des sources officielles.

### 1.1 Architecture Dual-RAG (v2.0)

**Evolution majeure**: Deux systemes RAG distincts et independants.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ARCHITECTURE DUAL-RAG v2.0                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐        │
│  │      RAG FRANCE (FR)        │    │   RAG INTERNATIONAL (INTL)  │        │
│  ├─────────────────────────────┤    ├─────────────────────────────┤        │
│  │ Corpus:                     │    │ Corpus:                     │        │
│  │ ├── Lois des Echecs FFE     │    │ ├── Laws of Chess FIDE      │        │
│  │ ├── Reglements championnats │    │ ├── FIDE Handbook           │        │
│  │ ├── Reglements jeunes       │    │ └── Regles arbitrage FIDE   │        │
│  │ ├── Annales DNA (477 Q)     │    │                             │        │
│  │ └── 29 documents FFE        │    │ A COMPLETER                 │        │
│  ├─────────────────────────────┤    ├─────────────────────────────┤        │
│  │ Focus: Specifications FR    │    │ Focus: Regles generalistes  │        │
│  │ - Championnats nationaux    │    │ - Tournois internationaux   │        │
│  │ - Homologation FFE          │    │ - Standards FIDE            │        │
│  │ - Cadences FR specifiques   │    │ - Regles universelles       │        │
│  ├─────────────────────────────┤    ├─────────────────────────────┤        │
│  │ Status: OPERATIONNEL        │    │ Status: A CONSTRUIRE        │        │
│  │ ├── GS FR v7.0 ✓            │    │ ├── GS INTL v2.1 obsolete   │        │
│  │ ├── Chunking LangChain ✓    │    │ ├── Corpus incomplet        │        │
│  │ ├── EmbeddingGemma 330M ✓   │    │ ├── Chunking a refaire      │        │
│  │ └── chunk_id mapping TODO   │    │ └── Embeddings a refaire    │        │
│  └─────────────────────────────┘    └─────────────────────────────┘        │
│                                                                              │
│  RELATION: Les reglements FR derivent des reglements FIDE.                  │
│            Coherents mais FR = specifications championnats francais.        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Justification Dual-RAG

**Cause principale**: Pollution mutuelle des corpus due a leur specificite metier et scopes differents.

| Probleme RAG Unique | Consequence |
|---------------------|-------------|
| Corpus FR tres specifique (championnats FFE) | Chunks FR "noient" les regles FIDE generales |
| Corpus INTL generaliste (Laws of Chess) | Chunks INTL interferent avec specs FR |
| Vocabulaire metier different (FR vs EN) | Embeddings compromis, recall degrade |
| Scopes non alignes | Retrieval ramene chunks hors contexte |

| Aspect | RAG Unique | Dual-RAG |
|--------|-----------|----------|
| Pertinence | Melange FR/INTL polluant | Resultats cibles par contexte |
| Embeddings | Compromis FR/EN degrade | Optimises par langue/corpus |
| Gold Standard | Mixte, evaluation biaisee | GS FR separe de GS INTL |
| Maintenance | Mise a jour complexe | Independance des corpus |
| Benchmarks | MTEB seul (compromis) | MTEB (FR) + MMTEB (INTL) |
| Recall | Dilue par chunks hors scope | Maximise par specialisation |

---

## 2. Problème à résoudre

### 2.1 Contexte
Les arbitres d'échecs doivent régulièrement consulter des règlements complexes pendant les compétitions :
- Règlements FFE (Fédération Française des Échecs) pour les tournois nationaux
- Règlements FIDE (Fédération Internationale des Échecs) pour les tournois internationaux

### 2.2 Problèmes actuels
| Problème | Impact |
|----------|--------|
| PDF volumineux (~300 pages total) | Recherche lente, stress en situation de jeu |
| Pas de recherche sémantique | Questions pratiques mal couvertes par Ctrl+F |
| Besoin de connexion internet | Salles de tournoi souvent mal connectées |
| Interprétation des règles | Arbitres juniors manquent d'expérience |

### 2.3 Conséquences
- Décisions arbitrales retardées
- Incohérences entre arbitres
- Frustration des joueurs et organisateurs

---

## 3. Solution proposée

### 3.1 Description
Application mobile RAG (Retrieval-Augmented Generation) fonctionnant 100% en local sur le téléphone de l'arbitre.

### 3.2 Fonctionnalités clés

| ID | Fonctionnalité | Priorité | Description |
|----|----------------|----------|-------------|
| F01 | Sélection corpus | Must | Choix entre règles FR (FFE) et règles INTL (FIDE) |
| F02 | Question en langage naturel | Must | L'arbitre pose sa question comme à un collègue |
| F03 | Retrieval sémantique | Must | Recherche des passages pertinents même si mots différents |
| F04 | Synthèse interprétative | Must | Réponse formulée avec explication pratique |
| F05 | Citation verbatim | Must | Texte exact du règlement + référence (doc, page) |
| F06 | Fonctionnement offline | Must | Aucune connexion requise après installation |
| F07 | Multilinguisme | Should | Questions FR, réponses adaptées même si règle EN |
| F08 | Historique questions | Could | Retrouver ses recherches précédentes |

### 3.3 Ce que l'app ne fait PAS
- ❌ Remplacer le jugement de l'arbitre
- ❌ Donner des décisions officielles
- ❌ Se connecter à internet pendant l'usage
- ❌ Collecter des données personnelles

---

## 4. Parties prenantes

| Rôle | Nom/Description | Intérêt | Influence |
|------|-----------------|---------|-----------|
| Utilisateur principal | Arbitres d'échecs (FR + INTL) | Élevé | Élevé |
| Sponsor | [À définir] | Moyen | Élevé |
| Développeur | Toi + Claude Code | Élevé | Élevé |
| Fédérations | FFE, FIDE | Faible | Moyen |
| Joueurs | Bénéficiaires indirects | Moyen | Faible |

---

## 5. Contraintes

### 5.1 Contraintes techniques
| Contrainte | Justification |
|------------|---------------|
| 100% offline | Salles de tournoi mal connectées |
| Android first | Majorité des arbitres sur Android |
| APK installable | Distribution hors Play Store possible |
| RAM < 500MB | Compatibilité téléphones mid-range |
| Latence < 5s | Décisions rapides en tournoi |

### 5.2 Contraintes légales/éthiques
| Contrainte | Justification |
|------------|---------------|
| Pas de données personnelles | RGPD, simplicité |
| Citation obligatoire | Traçabilité des décisions |
| Disclaimer IA | ISO 42001, responsabilité |
| Sources officielles uniquement | Fiabilité juridique |

### 5.3 Contraintes projet
| Contrainte | Valeur |
|------------|--------|
| Budget | 0€ (projet personnel) |
| Timeline cible | 3-4 mois |
| Équipe | 1 développeur + Claude Code |

---

## 6. Critères de succès

### 6.1 Objectifs SMART

| Objectif | Mesure | Cible | Échéance |
|----------|--------|-------|----------|
| Précision retrieval | Recall sur test set | > 80% | Phase 2 |
| Fidélité réponses | Score évaluation humaine | > 85% | Phase 3 |
| Performance | Temps réponse total | < 5s | Phase 3 |
| Satisfaction | NPS arbitres beta | > 7/10 | Phase 5 |
| Stabilité | Crash-free rate | > 99% | Phase 5 |
| Hallucination | Réponses sans source | 0% | Phase 3 |

### 6.2 Definition of Done - Projet global
Le projet est "Done" quand :
- [ ] App installable sur Android 10+
- [ ] 2 corpus indexés (FR + INTL)
- [ ] Retrieval + synthèse fonctionnels offline
- [ ] Citations verbatim avec sources
- [ ] Testé par 5 arbitres réels
- [ ] Documentation utilisateur complète
- [ ] Code source documenté

---

## 7. Risques identifiés

| ID | Risque | Probabilité | Impact | Mitigation |
|----|--------|-------------|--------|------------|
| R01 | Qualité extraction PDF insuffisante | Moyenne | Élevé | Tests manuels échantillon, nettoyage itératif |
| R02 | Hallucinations LLM | Moyenne | Critique | Grounding strict, disclaimer, tests adversaires |
| R03 | Performance insuffisante sur mobile | Moyenne | Élevé | Modèles quantifiés, benchmarks continus |
| R04 | Taille APK trop grande | Moyenne | Moyen | Assets téléchargeables séparément |
| R05 | Mises à jour règlements | Faible | Moyen | Process de mise à jour documenté |
| R06 | Complexité MediaPipe | Moyenne | Moyen | Prototypage précoce, fallback |

---

## 8. Périmètre exclus (hors scope v1.0)

- Support iOS (prévu v2.0)
- Mode connecté avec mises à jour auto
- Autres sports que les échecs
- Interface web
- Multi-utilisateurs / comptes

---

## 9. Dépendances externes

| Dépendance | Type | Criticité | Alternative |
|------------|------|-----------|-------------|
| MediaPipe | Librairie Google | Élevée | ONNX Runtime |
| EmbeddingGemma | Modèle embeddings | Élevée | all-MiniLM |
| Phi-3.5-mini / Gemma | Modèle LLM | Élevée | Autre mini LLM q4 |
| PDF règlements | Contenu | Critique | Aucune |

---

## 10. Historique du document

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-10 | Equipe Pocket Arbiter | Création initiale |
| 2.0 | 2026-01-24 | Claude Opus 4.5 | **EVOLUTION MAJEURE**: Architecture Dual-RAG (FR + INTL separes). Cause: pollution mutuelle des corpus due a specificite metier et scopes differents. Status: RAG FR operationnel (GS v6.7.0, chunking LangChain, EmbeddingGemma 330M), RAG INTL a construire (corpus incomplet, GS/chunking/embeddings obsoletes) |
| 2.1 | 2026-01-29 | Claude Opus 4.5 | GS FR v6.7.0→v7.0, reranker supprime (VRAM incompatible mobile) |

---

## 11. Approbations

| Rôle | Nom | Date | Signature |
|------|-----|------|-----------|
| Product Owner | | | |
| Tech Lead | | | |
