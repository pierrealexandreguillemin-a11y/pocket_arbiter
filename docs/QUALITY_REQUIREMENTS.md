# Exigences Qualite - Arbitre Echecs RAG

> **Document ID**: SPEC-REQ-001
> **ISO Reference**: ISO/IEC 25010:2023 - Modele de qualite produit
> **Version**: 1.1
> **Date**: 2026-01-18
> **Statut**: En cours
> **Classification**: Interne
> **Auteur**: Equipe projet
> **Mots-cles**: qualite, ISO 25010, exigences, performance, fiabilite, securite, maintenabilite

---

## 1. Introduction

### 1.1 Objet
Ce document définit les exigences qualité du projet selon les 9 caractéristiques du modèle ISO/IEC 25010:2023. Chaque caractéristique est déclinée en critères mesurables avec des cibles et méthodes de vérification.

### 1.2 Structure ISO 25010
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ISO/IEC 25010:2023 - QUALITÉ PRODUIT                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Functional    2. Performance   3. Compatibility  4. Interaction    │
│     Suitability      Efficiency                         Capability     │
│                                                                         │
│  5. Reliability   6. Security      7. Maintainability 8. Flexibility   │
│                                                                         │
│  9. Safety                                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Caractéristiques qualité

### 2.1 Adéquation fonctionnelle (Functional Suitability)

> Degré auquel le produit fournit des fonctions qui répondent aux besoins déclarés et implicites.

#### 2.1.1 Complétude fonctionnelle

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| FS-01 | Sélection corpus FR/INTL | 100% implémenté | Test fonctionnel |
| FS-02 | Saisie question langage naturel | 100% implémenté | Test fonctionnel |
| FS-03 | Retrieval sémantique | 100% implémenté | Test fonctionnel |
| FS-04 | Génération synthèse | 100% implémenté | Test fonctionnel |
| FS-05 | Affichage citation verbatim | 100% implémenté | Test fonctionnel |
| FS-06 | Affichage source (doc + page) | 100% implémenté | Test fonctionnel |

#### 2.1.2 Exactitude fonctionnelle

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| FA-01 | Recall retrieval sur test set | ≥ 80% (cible 90%) | Tests automatisés + benchmark |
| FA-02 | Precision retrieval sur test set | ≥ 70% | Tests automatisés |
| FA-03 | Fidélité réponses (éval humaine) | ≥ 85% | Évaluation manuelle 30 questions |
| FA-04 | Taux hallucination | 0% | Tests adversaires |
| FA-05 | Exactitude citations (source + page) | 100% | Tests automatisés |

#### 2.1.3 Pertinence fonctionnelle

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| FR-01 | Réponses utiles pour décisions arbitrales | NPS ≥ 7/10 | Enquête beta testeurs |
| FR-02 | Temps gagné vs recherche PDF manuelle | ≥ 50% | Benchmark comparatif |

---

### 2.2 Efficacité de performance (Performance Efficiency)

> Performance relative à la quantité de ressources utilisées.

#### 2.2.1 Comportement temporel

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| PT-01 | Temps embedding question | < 50ms | Benchmark device |
| PT-02 | Temps retrieval top-5 | < 200ms | Benchmark device |
| PT-03 | Temps génération LLM | < 4s | Benchmark device |
| PT-04 | Temps réponse total (E2E) | < 5s | Benchmark device |
| PT-05 | Temps démarrage app (cold start) | < 3s | Benchmark device |
| PT-06 | Temps chargement corpus | < 2s | Benchmark device |

**Devices de référence pour benchmark** :
- Minimum : Samsung Galaxy A33 (Exynos 1280, 6GB RAM)
- Cible : Google Pixel 6 (Tensor, 8GB RAM)

#### 2.2.2 Utilisation des ressources

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| PR-01 | RAM usage max | < 500MB | Profiler Android |
| PR-02 | CPU usage moyen (pendant query) | < 80% | Profiler Android |
| PR-03 | Battery drain par session (30 min) | < 5% | Test manuel |
| PR-04 | Stockage total app | < 1.5GB | Mesure APK + assets |
| PR-05 | Stockage APK seul | < 100MB | Mesure APK |

#### 2.2.3 Capacité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| PC-01 | Taille corpus supporté | ≥ 500 pages | Test de charge |
| PC-02 | Nombre de chunks indexables | ≥ 5000 | Test de charge |
| PC-03 | Longueur question max | ≥ 500 caractères | Test fonctionnel |

---

### 2.3 Compatibilité (Compatibility)

> Capacité à échanger des informations et coexister avec d'autres systèmes.

#### 2.3.1 Coexistence

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| CC-01 | Pas d'interférence avec autres apps | 100% | Test manuel |
| CC-02 | Fonctionne en arrière-plan | N/A (pas nécessaire) | - |

#### 2.3.2 Interopérabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| CI-01 | Format PDF sources standard | PDF/A-1, PDF 1.4+ | Test extraction |
| CI-02 | Export réponses (copier/coller) | Supporté | Test fonctionnel |

---

### 2.4 Capacité d'interaction (Interaction Capability)

> Capacité à être utilisé efficacement par des utilisateurs spécifiés.

#### 2.4.1 Reconnaissance de l'adéquation

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| IA-01 | Fonction principale identifiable < 5s | 100% utilisateurs | Test utilisabilité |
| IA-02 | Sélection corpus intuitive | 100% utilisateurs | Test utilisabilité |

#### 2.4.2 Apprenabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| IL-01 | Temps apprentissage fonctions de base | < 2 min | Test utilisabilité |
| IL-02 | Utilisation sans documentation | Possible | Test utilisabilité |
| IL-03 | Guide d'aide intégré | Présent | Test fonctionnel |

#### 2.4.3 Opérabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| IO-01 | Nombre de clics pour poser question | ≤ 3 | Test UX |
| IO-02 | Navigation clavier/swipe | Supportée | Test fonctionnel |
| IO-03 | Feedback visuel actions | Présent | Test UX |
| IO-04 | Loading indicator pendant génération | Présent | Test UX |

#### 2.4.4 Accessibilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| AC-01 | Compatibilité TalkBack | Basique | Test TalkBack |
| AC-02 | Contraste texte/fond | Ratio ≥ 4.5:1 | Outil contraste |
| AC-03 | Taille police ajustable | Supporté | Test système |
| AC-04 | Mode sombre | Supporté (Android system) | Test fonctionnel |

#### 2.4.5 Assistance utilisateur

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| IU-01 | Messages d'erreur explicites | 100% | Revue code |
| IU-02 | Suggestions si pas de résultat | Présentes | Test fonctionnel |
| IU-03 | Disclaimer IA visible | Permanent | Test UI |

---

### 2.5 Fiabilité (Reliability)

> Capacité à fonctionner correctement dans des conditions définies.

#### 2.5.1 Maturité (absence de défauts)

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| RM-01 | Crash-free rate | ≥ 99% | Firebase Crashlytics / logs |
| RM-02 | Bugs critiques en production | 0 | Bug tracking |
| RM-03 | Bugs majeurs en production | < 3 | Bug tracking |

#### 2.5.2 Disponibilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| RA-01 | App disponible 100% offline | Oui | Test avion mode |
| RA-02 | Temps de réponse stable | Variance < 20% | Benchmark répété |

#### 2.5.3 Tolérance aux pannes

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| RF-01 | Gestion corpus corrompu | Graceful error | Test injection faute |
| RF-02 | Gestion question vide | Pas de crash | Test edge case |
| RF-03 | Gestion mémoire insuffisante | Warning + dégradation | Test low memory |

#### 2.5.4 Récupérabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| RR-01 | Reprise après crash | État précédent restauré | Test kill app |
| RR-02 | Reprise après interruption (appel) | Transparent | Test interruption |

---

### 2.6 Sécurité (Security)

> Protection des informations et des données.

#### 2.6.1 Confidentialité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| SC-01 | Aucune donnée transmise réseau | 100% | Analyse trafic (Wireshark) |
| SC-02 | Pas de tracking/analytics | 100% | Revue code |
| SC-03 | Questions non stockées (par défaut) | 100% | Revue code |

#### 2.6.2 Intégrité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| SI-01 | Corpus non modifiable par utilisateur | 100% | Test tentative modif |
| SI-02 | Pas d'injection dans prompts | 100% | Tests sécurité |

#### 2.6.3 Authenticité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| SA-01 | APK signé | Oui | Vérification signature |
| SA-02 | Source des règlements vérifiable | Métadonnées présentes | Test fonctionnel |

---

### 2.7 Maintenabilité (Maintainability)

> Facilité à modifier, corriger, améliorer le produit.

#### 2.7.1 Modularité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| MM-01 | Séparation UI / Logic / Data | Architecture clean | Revue architecture |
| MM-02 | Module embeddings indépendant | Oui | Revue code |
| MM-03 | Module LLM interchangeable | Oui | Revue code |

#### 2.7.2 Réutilisabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| MR-01 | Pipeline preprocessing réutilisable | Oui | Revue scripts |
| MR-02 | Prompts externalisés (config) | Oui | Revue code |

#### 2.7.3 Analysabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| MA-01 | Logging des erreurs | Complet | Revue logs |
| MA-02 | Métriques performance accessibles | Oui | Revue code |
| MA-03 | Documentation code (KDoc) | Fonctions publiques | Revue code |

#### 2.7.4 Modifiabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| MO-01 | Temps ajout nouveau corpus | < 1 jour | Test procédure |
| MO-02 | Temps changement modèle LLM | < 1 jour | Test procédure |
| MO-03 | Couverture tests unitaires | ≥ 60% | Rapport coverage |

#### 2.7.5 Testabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| MT-01 | Tests unitaires automatisés | Présents | CI/CD |
| MT-02 | Tests UI automatisés | Basiques | CI/CD |
| MT-03 | Tests retrieval automatisés | Présents | CI/CD |
| MT-04 | Mocks pour LLM (tests rapides) | Disponibles | Revue code |

---

### 2.8 Flexibilité (Flexibility)

> Capacité à être adapté à différents contextes.

#### 2.8.1 Adaptabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| FA-01 | Support Android 10 à 14 | 100% | Tests multi-version |
| FA-02 | Support différentes tailles écran | Phone + tablet | Tests multi-device |
| FA-03 | Support FR + EN interface | FR v1, EN v2 | Roadmap |

#### 2.8.2 Installabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| FI-01 | Installation via APK direct | Supporté | Test install |
| FI-02 | Installation via Play Store | Supporté (v1.1) | Test publish |
| FI-03 | Mise à jour sans perte données | Supporté | Test upgrade |

#### 2.8.3 Remplaçabilité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| FP-01 | Export données utilisateur | N/A (pas de données) | - |
| FP-02 | Désinstallation propre | 100% données supprimées | Test uninstall |

---

### 2.9 Sûreté (Safety)

> Capacité à éviter les risques pour les personnes et l'environnement.

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| SF-01 | Pas de risque physique | N/A | - |
| SF-02 | Pas de contenu offensant généré | 100% | Tests filtrage |
| SF-03 | Disclaimer responsabilité décision | Présent | Revue UI |

---

## 3. Qualité en usage (Quality in Use)

> Évaluation depuis la perspective de l'utilisateur final.

### 3.1 Efficacité

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| QU-E01 | Taux de réponses correctes (user eval) | ≥ 85% | Enquête beta |
| QU-E02 | Tâche complétée sans aide externe | ≥ 90% | Test utilisabilité |

### 3.2 Efficience

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| QU-F01 | Temps moyen pour obtenir réponse | < 30s | Observation beta |
| QU-F02 | Nombre d'actions pour réponse | ≤ 4 | Analyse UX |

### 3.3 Satisfaction

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| QU-S01 | Score NPS | ≥ 7/10 | Enquête beta |
| QU-S02 | Intention de recommander | ≥ 70% | Enquête beta |
| QU-S03 | Intention d'utilisation régulière | ≥ 60% | Enquête beta |

### 3.4 Absence de risque

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| QU-R01 | Décisions erronées dues à l'app | 0 | Suivi incidents |
| QU-R02 | Compréhension du disclaimer IA | 100% | Test compréhension |

### 3.5 Couverture du contexte

| ID | Exigence | Cible | Méthode vérification |
|----|----------|-------|---------------------|
| QU-C01 | Utilisable en salle de tournoi | Oui | Test terrain |
| QU-C02 | Utilisable sous pression (temps limité) | Oui | Test stress |
| QU-C03 | Utilisable par arbitre débutant | Oui | Test persona |

---

## 4. Matrice de traçabilité

| Caractéristique ISO 25010 | Phase test | Priorité |
|---------------------------|------------|----------|
| Functional Suitability | Phase 2, 3 | Must |
| Performance Efficiency | Phase 2, 3, 4 | Must |
| Compatibility | Phase 2 | Should |
| Interaction Capability | Phase 2, 4, 5 | Must |
| Reliability | Phase 4, 5 | Must |
| Security | Phase 4 | Must |
| Maintainability | Continue | Should |
| Flexibility | Phase 4, 5 | Should |
| Safety | Phase 3 | Must |

---

## 5. Historique du document

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-10 | Equipe Pocket Arbiter | Creation initiale |
| 1.1 | 2026-01-18 | Claude Opus 4.5 | FA-01 cible recall 90%, chunking v3 strategy |
