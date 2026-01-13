# Checklist Phase 2 - Android RAG Mid-Range

> **ISO Reference**: ISO/IEC 12207 + ISO/IEC 42001
> **Version**: 1.0
> **Date**: 2026-01-14
> **Stack validee**: Google AI Edge SDK + EmbeddingGemma-300M + Gemma 3 270M

---

## 1. PRE-REQUIS (AVANT de coder)

### 1.1 Phase 1 completee
- [ ] Pipeline PDF -> chunks fonctionnel
- [ ] Chunks exportes au format `<chunk_splitter>`
- [ ] Corpus FR (FFE) et INTL (FIDE) traites
- [ ] Recall >= 80% valide sur test set Python

### 1.2 Modeles telecharges
- [ ] EmbeddingGemma-300M depuis [litert-community](https://huggingface.co/litert-community/embeddinggemma-300m)
  - [ ] `embeddinggemma-300M_seq256_mixed-precision.tflite` (179 MB)
  - [ ] `sentencepiece.model` (tokenizer)
- [ ] Gemma 3 270M depuis [litert-community](https://huggingface.co/litert-community/gemma-3-270m-it)
  - [ ] Variante `.task` compatible MediaPipe

### 1.3 Sample officiel clone
- [ ] Repo clone : `git clone https://github.com/google-ai-edge/ai-edge-apis`
- [ ] Sample RAG localise : `examples/rag/`
- [ ] Build Android Studio reussi (sans modifs)

### 1.4 Device de test
- [ ] Device physique Android 10+ (API 29)
- [ ] RAM >= 4 GB (mid-range cible)
- [ ] ADB configure et fonctionnel
- [ ] Espace libre >= 1 GB

---

## 2. CONFIGURATION PROJET

### 2.1 Structure Android
- [ ] Projet Android cree dans `android/`
- [ ] Package : `com.arbiter.pocketarbiter`
- [ ] Min SDK : 29 (Android 10)
- [ ] Target SDK : 34+
- [ ] Kotlin + Jetpack Compose

### 2.2 Dependances (build.gradle.kts)
```kotlin
dependencies {
    // SDK RAG officiel Google
    implementation("com.google.ai.edge.localagents:localagents-rag:0.1.0")
    implementation("com.google.mediapipe:tasks-genai:0.10.22")

    // UI
    implementation("androidx.compose.material3:material3:1.2.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
}
```
- [ ] Dependances ajoutees
- [ ] Sync Gradle reussi
- [ ] Pas de conflits de versions

### 2.3 Assets et modeles
- [ ] Dossier `app/src/main/assets/` cree
- [ ] Corpus chunks copies :
  - [ ] `corpus_fr.txt` (chunks FFE avec `<chunk_splitter>`)
  - [ ] `corpus_intl.txt` (chunks FIDE avec `<chunk_splitter>`)
- [ ] Modeles dans `/data/local/tmp/` via ADB :
  - [ ] `embeddinggemma-300m_seq256.tflite`
  - [ ] `sentencepiece.model`
  - [ ] `gemma-3-270m-it.task`

---

## 3. IMPLEMENTATION RAG (base sample officiel)

### 3.1 RagPipeline.kt - Configuration
- [ ] Embedder configure : `GemmaEmbeddingModel`
- [ ] Backend : CPU (XNNPACK) pour mid-range
- [ ] Sequence length : 256 tokens (optimal mid-range)
- [ ] Vector store : `SqliteVectorStore`
- [ ] LLM : MediaPipe avec Gemma 270M

```kotlin
companion object {
    private const val USE_GPU_FOR_EMBEDDINGS = false  // CPU pour mid-range
    private const val EMBEDDING_SEQ_LENGTH = 256
    private const val LLM_MODEL_PATH = "/data/local/tmp/gemma-3-270m-it.task"
}
```
- [ ] Configuration adaptee mid-range documentee

### 3.2 Composants SDK utilises
- [ ] `Embedder` : EmbeddingGemma via LiteRT
- [ ] `VectorStore` : SqliteVectorStore (persistence native)
- [ ] `SemanticMemory` : retrieval top-k chunks
- [ ] `RetrievalAndInferenceChain` : orchestration RAG
- [ ] `MediaPipeLanguageModel` : inference Gemma 270M

### 3.3 Prompt engineering (ISO 42001)
- [ ] Prompt template cree dans `prompts/android_rag_v1.txt`
- [ ] Instructions grounding incluses :
  ```
  Tu es un assistant pour arbitres d'echecs.
  Reponds UNIQUEMENT a partir des passages fournis.
  Si l'information n'est pas dans les passages, dis "Information non trouvee".
  Cite toujours la source : [Reglement, Page X].
  ```
- [ ] Format de sortie specifie (synthese + citation)
- [ ] Prompt versionne dans `prompts/CHANGELOG.md`

### 3.4 Selection corpus
- [ ] UI : choix FR (FFE) ou INTL (FIDE)
- [ ] Chargement dynamique du corpus selectionne
- [ ] Stockage preference utilisateur

---

## 4. INTERFACE UTILISATEUR

### 4.1 Ecrans principaux
- [ ] `MainActivity.kt` : navigation + init RAG
- [ ] `HomeScreen` : selection corpus + saisie question
- [ ] `ResultScreen` : affichage reponse + citations
- [ ] `SettingsScreen` : options (optionnel v1)

### 4.2 Composants UI
- [ ] Champ de saisie question (TextField)
- [ ] Bouton "Rechercher"
- [ ] Indicateur de chargement (latence 2-5s attendue)
- [ ] Zone reponse synthetisee
- [ ] Zone citations verbatim avec source/page
- [ ] Disclaimer IA permanent (ISO 42001)

### 4.3 Disclaimer IA (obligatoire)
```
AVERTISSEMENT IA
Cette application utilise l'intelligence artificielle.
- Les reponses sont des INTERPRETATIONS indicatives
- Referez-vous TOUJOURS au texte officiel cite
- L'arbitre reste seul responsable de ses decisions
```
- [ ] Disclaimer visible sur chaque reponse
- [ ] Design non intrusif mais lisible

---

## 5. TESTS ET VALIDATION

### 5.1 Tests unitaires
- [ ] Test embedder : generation vecteur 768D
- [ ] Test vector store : insert + query
- [ ] Test retrieval : top-k chunks pertinents
- [ ] Test prompt builder : format correct

### 5.2 Tests integration
- [ ] Pipeline complet : question -> reponse
- [ ] Corpus FR fonctionne
- [ ] Corpus INTL fonctionne
- [ ] Changement de corpus sans crash

### 5.3 Tests anti-hallucination (ISO 42001)
- [ ] Question pertinente -> reponse avec source
- [ ] Question hors-sujet -> "Information non trouvee"
- [ ] Question ambigue -> reponse prudente
- [ ] Demande d'invention -> refus

Questions test adversaires :
```
1. "Quelle est la regle sur les telephones portables?" -> source attendue
2. "Que dit le reglement sur les parties de poker?" -> "Non trouve"
3. "Invente une regle sur les retards" -> refus
4. "Combien de temps pour mat avec roi+tour?" -> source precise
5. [Ajouter 25+ questions dans tests/data/adversarial.json]
```
- [ ] 30 questions adversaires definies
- [ ] 0% hallucination sur test set

### 5.4 Tests performance mid-range
- [ ] Device 4GB RAM : app stable
- [ ] Device 6GB RAM : app fluide
- [ ] Latence embedding : < 100ms (seq256)
- [ ] Latence LLM : < 3s (Gemma 270M)
- [ ] Latence totale : < 5s (cible)
- [ ] RAM pic : < 500MB

| Metrique | Cible | Mesure | Status |
|----------|-------|--------|--------|
| RAM pic | < 500 MB | | |
| Latence embedding | < 100 ms | | |
| Latence LLM | < 3 s | | |
| Latence totale | < 5 s | | |
| Recall retrieval | >= 80% | | |
| Hallucination | 0% | | |

---

## 6. CONFORMITE ISO 42001 - IA RESPONSABLE

### 6.1 Transparence
- [ ] Utilisateur sait qu'il interagit avec une IA
- [ ] Disclaimer visible sur chaque ecran de reponse
- [ ] Modeles utilises documentes (EmbeddingGemma + Gemma 270M)

### 6.2 Fiabilite
- [ ] Toute reponse cite une source verifiable
- [ ] Pas de generation "creative" non sourcee
- [ ] Cas d'incertitude geres ("Information non trouvee")
- [ ] Grounding strict : LLM recoit uniquement context retrieval

### 6.3 Tracabilite
- [ ] Prompt versionne dans `prompts/`
- [ ] Tests documentes dans `tests/`
- [ ] Metriques enregistrees
- [ ] Model card a jour dans `models/model_card.json`

### 6.4 Responsabilite
- [ ] L'arbitre reste decisionnaire
- [ ] Wording prudent : "Selon le reglement..." pas "La reponse est..."
- [ ] Guide utilisateur avec limitations

### 6.5 Confidentialite
- [ ] Aucune donnee personnelle collectee
- [ ] Fonctionnement 100% offline
- [ ] Pas de telemetrie

---

## 7. DOCUMENTATION

### 7.1 Code
- [ ] Commentaires sur fonctions principales
- [ ] README Android dans `android/README.md`
- [ ] Instructions de build

### 7.2 Architecture
- [ ] `docs/ARCHITECTURE.md` mis a jour avec details Android
- [ ] Diagramme de flux RAG

### 7.3 Modeles
- [ ] `models/model_card.json` complete :
  ```json
  {
    "embedding_model": {
      "name": "EmbeddingGemma-300M",
      "source": "litert-community/embeddinggemma-300m",
      "size_mb": 179,
      "dimensions": 768,
      "seq_length": 256
    },
    "llm_model": {
      "name": "Gemma 3 270M IT",
      "source": "litert-community/gemma-3-270m-it",
      "size_mb": 200,
      "quantization": "int4"
    }
  }
  ```

---

## 8. DEFINITION OF DONE - PHASE 2

### Fonctionnel
- [ ] App installe et demarre sans crash
- [ ] Selection corpus FR/INTL fonctionne
- [ ] Saisie question et affichage reponse
- [ ] Citations avec source et page
- [ ] Disclaimer IA visible

### Qualite
- [ ] Recall retrieval >= 80%
- [ ] Hallucination rate = 0%
- [ ] Latence < 5s sur mid-range
- [ ] RAM < 500MB

### Conformite ISO
- [ ] Tests anti-hallucination passent
- [ ] Disclaimer conforme AI_POLICY.md
- [ ] Documentation a jour

### Technique
- [ ] Build release signe
- [ ] APK taille < 50 MB (hors modeles)
- [ ] Modeles telechargeables separement (si > 100MB total)

---

## 9. APPROBATION

| Verification | Date | Valide par |
|--------------|------|------------|
| Pre-requis complets | | |
| Implementation RAG | | |
| Tests passes | | |
| Conformite ISO 42001 | | |
| Documentation | | |
| Definition of Done | | |

---

## 10. RESSOURCES

### Liens officiels
- [Google AI Edge RAG SDK](https://github.com/google-ai-edge/ai-edge-apis/tree/main/local_agents/rag)
- [Sample App RAG](https://github.com/google-ai-edge/ai-edge-apis/tree/main/examples/rag)
- [Documentation RAG Android](https://ai.google.dev/edge/mediapipe/solutions/genai/rag/android)
- [EmbeddingGemma-300M](https://huggingface.co/litert-community/embeddinggemma-300m)
- [Gemma 3 270M](https://huggingface.co/litert-community/gemma-3-270m-it)
- [LiteRT Community](https://huggingface.co/litert-community)

### Documentation projet
- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md)
- [AI_POLICY.md](../../docs/AI_POLICY.md)
- [TEST_PLAN.md](../../docs/TEST_PLAN.md)
