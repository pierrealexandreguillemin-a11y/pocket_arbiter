# Instructions Claude Code - Pocket Arbiter

> 📱 **Pocket Arbiter** = Application mobile d'assistance à l'arbitrage d'échecs
> Ce fichier définit les règles et garde-fous pour Claude Code dans VS Code.
> Toute action de développement doit respecter ces instructions.

**Version** : 1.0
**Date** : 2026-01-10

---

## 🎯 Mission

Développer une application Android RAG pour arbitres d'échecs, en respectant :
- Les normes ISO (25010, 42001, 12207, 29119)
- Une qualité de code professionnelle
- Une Definition of Done honnête et vérifiable

---

## 🚫 Règles absolues (NE JAMAIS VIOLER)

### 1. Avant de coder une nouvelle feature

```
☐ Document de specs existe dans /docs/
☐ Critères DoD sont définis pour cette feature
☐ Tests sont planifiés (au moins listés)
☐ Feature est dans le scope de la phase actuelle
```

**Si une case n'est pas cochée → DEMANDER CLARIFICATION, ne pas coder**

### 2. Avant de marquer "Done"

```
☐ Code compile sans erreurs
☐ Tests unitaires passent (existants + nouveaux)
☐ Pas de TODO/FIXME critiques non résolus
☐ Code documenté (KDoc pour fonctions publiques)
☐ Build Gradle réussi
☐ Pas de warnings Lint critiques
```

**Si une case n'est pas cochée → NE PAS dire "c'est done"**

### 3. Pour tout code IA (LLM/embeddings)

```
☐ Toute réponse DOIT citer sa source (ISO 42001)
☐ Jamais de génération sans context retrieval (anti-hallucination)
☐ Disclaimer IA visible pour l'utilisateur
☐ Prompt versionné dans /prompts/
☐ Test d'hallucination ajouté si nouvelle feature IA
```

**Violation = risque critique pour le projet**

---

## 📁 Structure projet obligatoire

```
pocket_arbiter/
├── android/                    # Projet Android Studio
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── kotlin/        # Code Kotlin
│   │   │   ├── res/           # Resources Android
│   │   │   └── assets/        # Modèles, indexes (ou téléchargés)
│   │   └── src/test/          # Tests unitaires
│   └── build.gradle.kts
│
├── scripts/                    # Scripts Python preprocessing
│   ├── extract_pdf.py
│   ├── chunk_text.py
│   ├── generate_embeddings.py
│   ├── create_index.py
│   └── requirements.txt
│
├── corpus/                     # Données sources
│   ├── fr/                    # PDF règlements français
│   ├── intl/                  # PDF règlements internationaux
│   └── INVENTORY.md           # Inventaire des fichiers
│
├── docs/                       # Documentation projet (ISO)
│   ├── VISION.md
│   ├── AI_POLICY.md
│   ├── QUALITY_REQUIREMENTS.md
│   ├── TEST_PLAN.md
│   └── USER_GUIDE.md          # (Phase 5)
│
├── prompts/                    # Prompts LLM versionnés
│   ├── interpretation_v1.txt
│   └── CHANGELOG.md
│
├── tests/                      # Données de test
│   ├── data/
│   │   ├── questions_fr.json
│   │   ├── questions_intl.json
│   │   └── adversarial.json
│   └── reports/               # Rapports de test
│
├── CLAUDE_CODE_INSTRUCTIONS.md # Ce fichier
├── README.md
└── .gitignore
```

**Ne pas créer de fichiers hors de cette structure sans justification**

---

## ✅ Definition of Done - Par type de tâche

### Feature Android (UI)

```
☐ UI implémentée selon specs
☐ Navigation fonctionne
☐ États (loading, error, success) gérés
☐ Tests UI Espresso/Compose ajoutés
☐ Accessibilité basique (contentDescription)
☐ Testé sur émulateur
```

### Feature Android (Logic)

```
☐ Fonction implémentée selon specs
☐ Tests unitaires ajoutés (≥ 80% coverage de la fonction)
☐ Erreurs gérées (try/catch, Result)
☐ KDoc pour fonctions publiques
☐ Pas de memory leaks évidents
```

### Script Python (Pipeline)

```
☐ Script exécutable
☐ Arguments CLI documentés (--help)
☐ Logs informatifs
☐ Gestion erreurs (fichiers manquants, etc.)
☐ Test unitaire ou test manuel documenté
☐ requirements.txt à jour si nouvelle dépendance
```

### Feature IA (Retrieval/LLM)

```
☐ Fonctionnalité implémentée
☐ Grounding vérifié (réponse basée sur sources)
☐ Test retrieval (recall mesuré)
☐ Test hallucination (si applicable)
☐ Performance mesurée (latence)
☐ Prompt documenté dans /prompts/
```

---

## 🔄 Workflow de développement

### 1. Avant de commencer une tâche

```kotlin
// Claude Code doit vérifier :
fun checkBeforeStart(task: String): Boolean {
    return specsExist(task)
        && dodDefined(task)
        && testsPlanned(task)
        && inCurrentPhaseScope(task)
}
```

### 2. Pendant le développement

- Commiter régulièrement avec messages clairs
- Nommer les branches : `feature/xxx`, `fix/xxx`, `test/xxx`
- Tester localement avant de dire "terminé"

### 3. Après le développement

```kotlin
fun checkBeforeDone(task: String): Boolean {
    return codeCompiles()
        && testsPass()
        && noBlockingTodos()
        && documented()
        && lintClean()
}
```

---

## 📊 Métriques à surveiller

| Métrique | Cible | Action si écart |
|----------|-------|-----------------|
| Tests pass rate | 100% | Fix avant merge |
| Code coverage | ≥ 60% | Ajouter tests |
| Lint warnings | 0 critiques | Fix immédiat |
| Build time | < 2 min | Optimiser si dépasse |
| Recall retrieval | ≥ 80% | Améliorer embeddings/chunking |
| Hallucination rate | 0% | Fix prompt/grounding |

---

## 🚨 Alertes et escalades

### Si Claude Code ne sait pas

```
❓ Situation incertaine → DEMANDER à l'utilisateur
❓ Specs ambiguës → DEMANDER clarification
❓ Choix technique majeur → PROPOSER options, ne pas décider seul
```

### Si quelque chose semble mal

```
⚠️ Test qui échoue sans raison claire → SIGNALER
⚠️ Performance dégradée → MESURER et SIGNALER
⚠️ Code legacy problématique → PROPOSER refactoring
```

### Si violation ISO détectée

```
🔴 Hallucination dans réponse IA → BLOQUER le merge
🔴 Données personnelles collectées → BLOQUER et ALERTER
🔴 Citation manquante → BLOQUER et fixer
```

---

## 📝 Templates

### Commit message

```
[TYPE] Description courte

- Détail 1
- Détail 2

Refs: #issue (si applicable)
```

Types : `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`

### Documentation fonction (KDoc)

```kotlin
/**
 * Description courte de la fonction.
 *
 * Description détaillée si nécessaire.
 *
 * @param param1 Description du paramètre
 * @param param2 Description du paramètre
 * @return Description de la valeur retournée
 * @throws ExceptionType Si condition d'erreur
 *
 * @sample com.example.SampleClass.sampleUsage
 */
fun maFonction(param1: Type1, param2: Type2): ReturnType
```

### Test unitaire

```kotlin
@Test
fun `nomFonction devrait faireQuelqueChose quand condition`() {
    // Given
    val input = ...

    // When
    val result = functionUnderTest(input)

    // Then
    assertThat(result).isEqualTo(expected)
}
```

---

## 🔗 Références

- `/docs/VISION.md` - Vision projet
- `/docs/AI_POLICY.md` - Politique IA (ISO 42001)
- `/docs/QUALITY_REQUIREMENTS.md` - Exigences qualité (ISO 25010)
- `/docs/TEST_PLAN.md` - Plan de tests (ISO 29119)

---

## 📅 Changelog

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-10 | Création initiale |
