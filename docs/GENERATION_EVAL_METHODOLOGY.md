# Methodologie d'evaluation de la generation — Pocket Arbiter

> **Document ID**: DOC-EVAL-GEN-001
> **ISO Reference**: ISO/IEC 42001:2023 (A.7.2, A.7.3, A.9.2), ISO/IEC 25010:2023 (FA-03/04/05), ISO/IEC 29119:2021
> **Version**: 1.0
> **Date**: 2026-03-28
> **Statut**: Approuve
> **Classification**: Interne
> **Auteur**: Claude Opus 4.6
> **Scope**: Evaluation des modeles de generation (Gemma 3 270M IT et checkpoints fine-tunes)

---

## 1. Objet

Ce document definit la methodologie d'evaluation des resultats de generation du pipeline RAG
Pocket Arbiter. Il couvre les metriques utilisees, leurs limites, les standards de l'industrie
applicables, et les decisions pragmatiques prises sous contrainte (budget 0EUR, offline, T4 GPU).

Ce document est le **referentiel** pour interpreter les resultats des kernels eval v4, v5+.

---

## 2. Metriques implementees

### 2.1 Auto-citation (regex) — Metrique primaire

**Definition** : pourcentage des 264 questions annales ou la reponse contient une mention
d'un document source OU d'un numero de page attendu, detectee par regex.

```python
# Patterns utilises (eval_generation.py)
SOURCE_PATTERNS = {
    "LA": r"(?:livre.{0,10}arbitre|l\.?a\.?\b)",
    "R01": r"(?:r[eè]gles?\s+g[eé]n[eé]rales?|r\.?01)",
    # ... (5 patterns total)
}
# Page pattern: r"\bpage\s*{p}\b|\bp\.?\s*{p}\b"
```

| Propriete | Valeur |
|-----------|--------|
| Scope | 264 questions annales (pas les 34 humaines) |
| Type | Binaire (cite / ne cite pas) |
| Ce que ca mesure | Le modele MENTIONNE-t-il un document/page |
| Ce que ca NE mesure PAS | Si la citation est CORRECTE, si le contenu est FIDELE |

**Baselines historiques** :

| Modele | cited_pct | Source |
|--------|-----------|--------|
| Base (Gemma 3 270M IT) | 43.9% | Eval v4 (2026-03-24), TAPT v3 sweep (2026-03-25) |
| TAPT ep1 (1 epoch mild TAPT) | 46.2% | TAPT v3 sweep (2026-03-25) |
| TAPT ep5 | 40.2% | TAPT v3 sweep (2026-03-25) |
| SFT v3 (regex data) | 28.8% | Eval v4 (2026-03-24) |

### 2.2 Empty responses — Metrique secondaire

Nombre de reponses vides ou whitespace-only sur 298 questions. Gate K6 : warning si > 10%.

### 2.3 Response length — Metrique diagnostique

Median et mean du nombre de mots sur les 34 reponses humaines. Detecte le sous-apprentissage
(median < 10 mots = SFT v2 pattern) ou le sur-apprentissage (echo de la question).

### 2.4 Eval humaine — Metrique qualitative (post-kernel)

34 questions humaines avec 3 scores binaires a remplir manuellement :
- **useful** : la reponse aide l'arbitre
- **faithful** : la reponse est coherente avec le contexte fourni
- **cited** : la reponse cite une source verifiable

Gate G4a : qualite >= 70% sur les 34 questions.

### 2.5 Reproducibilite

Seed=42 (torch + CUDA) avant chaque modele. `cudnn.benchmark=False`, `cudnn.deterministic=True`.
Resultats reproductibles sur meme hardware/driver (PyTorch docs).

---

## 3. Standards ISO applicables

### 3.1 ISO/IEC 42001:2023 — Gouvernance IA

| Clause | Exigence | Notre implementation |
|--------|----------|---------------------|
| A.7.2 Verification IA | Tests unitaires + tests d'integration | 312 tests, 80% coverage |
| A.7.3 Validation IA | Tests utilisateurs, evaluation humaine | 34 Q humaines + eval manuelle |
| A.9.2 Surveillance performance | Metriques continues | cited_pct, empty_count, response length |
| A.6.2.2 Documentation modeles | Fiche technique chaque modele | models/model_card.json |
| A.6.2.3 Tracabilite donnees | Inventaire sources, versions | DVC tracking, dataset_composition.json |

**ISO/IEC 42005:2025** (nouveau, avril 2025) : guide pour les evaluations d'impact des systemes IA.
Non encore implemente — a evaluer pour v2.0.

Source : [ISO 42001](https://www.iso.org/standard/42001), [ISO 42005](https://www.aarc-360.com/understanding-iso-iec-42005-2025/)

### 3.2 ISO/IEC 25010:2023 — Qualite logicielle

| Exigence | Cible | Metrique kernel | Status |
|----------|-------|----------------|--------|
| FA-03 Fidelite reponses | >= 85% (eval humaine) | scores.faithful (manual) | PENDING |
| FA-04 Taux hallucination | 0% | Non mesure automatiquement | GAP |
| FA-05 Exactitude citations | 100% | cited_pct (regex proxy) | PROXY |
| TB-04 RAGAS Faithfulness | >= 0.85 | Non mesure | GAP — necessite LLM judge |
| TB-05 ARES Context Relevance | >= 0.80 | Non mesure | GAP — necessite LLM judge |

### 3.3 ISO/IEC 29119:2021 — Tests logiciels

| Exigence | Implementation |
|----------|----------------|
| Reproductibilite | Seed=42, cudnn deterministic |
| Donnees de test | GS v9 : 403 questions (298 testables) |
| Validation set | 100% GS, separation stricte |
| Couverture | 264 annales (auto) + 34 humaines (manual) |

---

## 4. Standards industrie RAG/IA

### 4.1 Frameworks d'evaluation

| Framework | Methode | LLM requis ? | Offline T4 ? | Ref |
|-----------|---------|-------------|-------------|-----|
| **RAGAS** | Decompose en claims, verifie entailment vs contexte | Oui (LLM judge) ou HHEM | Partiel (HHEM only) | arXiv:2309.15217 |
| **HHEM-2.1-Open** (Vectara) | Classificateur T5 hallucination, supporte FR/EN/DE | **Non** (T5 standalone) | **OUI** (~250 MB) | [HuggingFace](https://huggingface.co/vectara/hallucination_evaluation_model) |
| **FaithJudge** (Vectara, EMNLP 2025) | LLM-as-judge avec exemples humains annotes | Oui (o3-mini) | Non | arXiv:2505.04847 |
| **FACTS Grounding** (Google 2025) | 3 frontier LLM judges (Gemini, GPT-4o, Claude) | Oui (3 cloud LLMs) | Non | arXiv:2501.03200 |
| **BERTScore** (ICLR 2020) | Similarite semantique token-level via BERT | **Non** (BERT standalone) | **OUI** (~400 MB) | arXiv:1904.09675 |
| **ROUGE** | Overlap n-grams entre reponse et reference | **Non** | **OUI** (CPU) | Lin 2004 |
| **GaRAGe** (2025) | Benchmark avec annotations grounding passage-level | Non (dataset) | N/A | arXiv:2506.07671 |

### 4.2 HHEM-2.1-Open — Alternative offline recommandee

HHEM (Hughes Hallucination Evaluation Model) est un classificateur T5-base entraine pour
detecter les hallucinations dans les textes generes par LLM. Points cles :

- **T5-base** : ~250 MB, tourne sur CPU ou GPU sans probleme sur T4
- **Supporte le francais** (FR, EN, DE depuis HHEM-2.0, avril 2024)
- **Contexte illimite** (HHEM-2.1 vs 512 tokens pour HHEM-1.0)
- **Pas de LLM-as-judge** : classificateur pur, bien plus rapide et moins cher
- **Score 0-1** : probabilite que la reponse soit factuellement coherente avec le contexte
- **Utilise par RAGAS** comme alternative au LLM judge pour le calcul de faithfulness

**Decision** : non inclus dans eval v5 (risque de complexite sur premier run).
A evaluer pour eval v5.1 si les resultats regex ne sont pas conclusifs.

Source : [HHEM-2.1-Open](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model), [HHEM HuggingFace](https://huggingface.co/vectara/hallucination_evaluation_model)

### 4.3 Distinction citation correctness vs citation faithfulness

**Finding critique** : le paper "Correctness is not Faithfulness in RAG Attributions"
(Wallat et al., ICTIR 2025, **Best Paper Honorable Mention**) etablit que :

| Concept | Definition | Ce que notre kernel mesure |
|---------|------------|--------------------------|
| **Citation correctness** | Le document cite supporte-t-il l'affirmation ? | Partiellement (regex verifie la MENTION, pas le support) |
| **Citation faithfulness** | Le modele a-t-il REELLEMENT utilise le document cite pour generer sa reponse ? | **Non mesure** |

**Finding cle** : jusqu'a **57% des citations manquent de faithfulness** — le modele repond
de memoire puis post-rationalise en trouvant une citation superficielle a posteriori.

**Impact** : un modele avec 80% cited_pct pourrait avoir seulement ~35% de vraie faithfulness.
La comparaison relative entre modeles reste valide (meme biais pour tous), mais la valeur
absolue de cited_pct ne doit PAS etre interpretee comme un score de faithfulness.

Source : [Wallat et al. 2025](https://doi.org/10.1145/3731120.3744592)

### 4.4 FaithBench et le Hallucination Leaderboard

FaithBench (Vectara, EMNLP 2025) fournit des annotations d'hallucination au niveau span
pour 10 LLMs differents, avec 4 niveaux de severite :

| Severite | Definition |
|----------|------------|
| Consistent | Factuel et supporte par le contexte |
| Benign | Ajout mineur non problematique |
| Questionable | Affirmation non verifiable dans le contexte |
| Unwanted | Hallucination factuelle contredisant le contexte |

Notre eval binaire (cite/pas cite) ne distingue pas ces niveaux.
L'eval humaine post-kernel (useful/faithful/cited) s'en approche mais reste binaire.

Source : [FaithBench arXiv:2505.04847](https://arxiv.org/abs/2505.04847)

---

## 5. Analyse des gaps

### 5.1 Gaps documentes et justifies

| Gap | Standard | Notre choix | Justification |
|-----|----------|-------------|---------------|
| RAGAS faithfulness (TB-04 >= 0.85) | Score continu 0-1 via LLM judge | Non mesure | Budget 0EUR, offline. HHEM alternative pour v5.1 |
| ARES context relevance (TB-05 >= 0.80) | Score continu via LLM judge | Non mesure | Meme contrainte |
| LLM-as-judge | Recommande 2025-2026 (FACTS, FaithJudge) | Non utilise | T4 15 GB = pas assez pour judge + modele evalue |
| Citation faithfulness (ICTIR 2025) | Detection post-rationalisation | Non mesure | Necessite intervention causale, pas faisable sur 270M |
| Hallucination granulaire (FaithBench) | 4 niveaux severite | Binaire seulement | 270M trop variable pour scoring granulaire |
| Taille eval humaine | >= 50 golden questions (PremAI 2026) | 34 questions | GS contient 34 humaines non-QCM |
| Score continu | RAGAS 0-1 | Binaire 0/1 | Actionnable pour decision gate go/no-go |

### 5.2 Metriques faisables offline sur T4 — non implementees

| Metrique | Modele | Taille | Effort | Priorite |
|----------|--------|--------|--------|----------|
| **HHEM-2.1-Open faithfulness** | T5-base | ~250 MB | +50 lignes kernel | HAUTE |
| **BERTScore** (reponse vs contexte) | BERT | ~400 MB | +30 lignes | MOYENNE |
| **ROUGE-L** (reponse vs contexte) | CPU only | 0 MB | +20 lignes | BASSE |

**Recommandation** : HHEM-2.1-Open est le meilleur ratio effort/valeur pour un eval v5.1.

---

## 6. Interpretation des resultats

### 6.1 Ce que les resultats DISENT

- **Comparaison relative** : si SFT v5 ckpt120 > TAPT ep1 en cited_pct, c'est un signal
  que le SFT ameliore la propension du modele a mentionner ses sources
- **Empty responses** : un modele avec 0 empty vs 71 empty a clairement appris a repondre
- **Response length** : median < 10 mots = sous-apprentissage, > 100 mots = sur-generation

### 6.2 Ce que les resultats NE DISENT PAS

- **cited_pct = X% ne signifie PAS X% de faithfulness** (ICTIR 2025 : 57% post-rationalisees)
- **cited_pct ne verifie PAS** que la citation est CORRECTE (bon doc, bonne page)
- **cited_pct ne mesure PAS** si le CONTENU de la reponse est supporte par le contexte
- **L'absence de citation ne signifie PAS** une hallucination (le modele peut repondre correctement sans citer)

### 6.3 Decision gates

| Resultat | Decision |
|----------|----------|
| Un SFT v5 checkpoint > 46.2% cited_pct | SFT v5 ameliore la generation, utiliser ce checkpoint |
| Tous SFT v5 < 46.2% | SFT inutile, TAPT ep1 + prompting = modele final |
| Un SFT v5 > 46.2% MAIS empty > 10% | Investiguer — le modele cite mais ne repond pas toujours |
| Tous modeles < 43.9% (base) | Regression — rollback au base model |

---

## 7. Protocole eval humaine (post-kernel)

### 7.1 Scope

170 reponses a scorer : 5 modeles x 34 questions humaines.

### 7.2 Criteres (binaire Pass/Fail)

| Critere | Question a se poser | Pass si |
|---------|---------------------|---------|
| **useful** | La reponse aide-t-elle l'arbitre a prendre une decision ? | Oui, contient l'information pertinente |
| **faithful** | La reponse est-elle coherente avec le contexte fourni ? | Aucune affirmation contredite par le contexte |
| **cited** | La reponse cite-t-elle une source verifiable ? | Mention d'un document et/ou page |

### 7.3 Gate G4a

Qualite >= 70% = moyenne des 3 criteres sur 34 questions.
Si FAIL → rollback ADR-001 : Gemma 3 1B IT.

### 7.4 Conformite ISO 29119

- Evaluateur : Pierre (arbitre FFE, connaissance du corpus)
- Pas de double-aveugle (1 seul evaluateur) — documente comme limitation
- ISO 29119 P3-F01 recommande 2 arbitres independants — non faisable (equipe = 1 personne)

---

## 8. Ameliorations futures

| Priorite | Amelioration | Standard | Effort |
|----------|-------------|----------|--------|
| P0 (v5.1) | Ajouter HHEM-2.1-Open comme score faithfulness | Vectara 2024 | +50 lignes |
| P1 (v6) | BERTScore reponse vs contexte | ICLR 2020 | +30 lignes |
| P2 (Android) | LLM-as-judge via API cloud (si budget) | FACTS Grounding 2025 | Nouveau kernel |
| P3 (Android) | RAGAS complet (faithfulness + relevancy) | RAGAS 2023 | Integration framework |
| P4 (v2.0) | ISO 42005:2025 impact assessment | ISO 2025 | Nouveau document |

---

## 9. References

### 9.1 Standards ISO

| Standard | Titre | Application |
|----------|-------|-------------|
| ISO/IEC 42001:2023 | AI Management Systems | Gouvernance IA, verification, validation |
| ISO/IEC 42005:2025 | AI System Impact Assessment | Evaluation d'impact (nouveau, avril 2025) |
| ISO/IEC 25010:2023 | Software Quality Model | Fidelite, hallucination, citations |
| ISO/IEC 29119:2021 | Software Testing | Reproductibilite, donnees de test |
| ISO/IEC 12207:2017 | Software Lifecycle | Tracabilite, commits conventionnels |

### 9.2 Litterature scientifique

| Ref | Titre | Annee | Contribution |
|-----|-------|-------|-------------|
| [Wallat et al.](https://doi.org/10.1145/3731120.3744592) | Correctness is not Faithfulness in RAG Attributions | ICTIR 2025 | 57% citations post-rationalisees, Best Paper HM |
| [Es et al.](https://arxiv.org/abs/2309.15217) | RAGAS: Automated Evaluation of RAG | 2023 | Framework faithfulness/relevancy/precision |
| [Google DeepMind](https://arxiv.org/abs/2501.03200) | FACTS Grounding Leaderboard | 2025 | Benchmark grounded generation, 3 LLM judges |
| [Siriwardhana et al.](https://arxiv.org/abs/2505.04847) | FaithJudge / FaithBench | EMNLP 2025 | Hallucination annotations span-level, leaderboard RAG |
| [Vectara](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model) | HHEM-2.1-Open | 2024 | T5-base hallucination classifier, FR support |
| [Zhang et al.](https://arxiv.org/abs/1904.09675) | BERTScore | ICLR 2020 | Similarite semantique token-level |
| [Huang et al.](https://arxiv.org/abs/2504.14891) | RAG Evaluation Survey | 2025 | Survey complet metriques RAG 2024-2025 |
| [arXiv:2506.07671](https://arxiv.org/abs/2506.07671) | GaRAGe Benchmark | 2025 | Grounding annotations pour RAG evaluation |
| [Li et al.](https://arxiv.org/abs/2210.15097) | Repetition in Small LMs | 2022 | repetition_penalty critique pour < 1B |
| [Google](https://arxiv.org/abs/2501.03200) | FACTS: faithfulness DECLINES with more training | 2025 | Coherent avec notre finding TAPT |

### 9.3 Ressources techniques

| Ressource | URL | Usage |
|-----------|-----|-------|
| RAGAS docs | https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ | Metriques disponibles |
| HHEM-2.1-Open | https://huggingface.co/vectara/hallucination_evaluation_model | Modele T5 hallucination |
| PyTorch Reproducibility | https://docs.pytorch.org/docs/stable/notes/randomness.html | Seed, cudnn.deterministic |
| HF Transformers #3154 | https://github.com/huggingface/transformers/issues/3154 | Seed pour model.generate() |
| FACTS Leaderboard | https://www.kaggle.com/facts-leaderboard | Benchmark Google |
| FaithJudge GitHub | https://github.com/vectara/FaithJudge | Framework eval Vectara |

### 9.4 Documents projet

| Document | Chemin | Contenu lie |
|----------|--------|-------------|
| Eval kernel v5 | kaggle/kernel-eval-v5/eval_v5_generation_kaggle.py | Implementation metriques |
| Eval design spec | docs/superpowers/specs/2026-03-23-eval-generation-kaggle-design.md | Architecture kernel eval |
| Quality Requirements | docs/QUALITY_REQUIREMENTS.md | Cibles TB-04, TB-05, FA-03/04/05 |
| AI Policy | docs/AI_POLICY.md | Risques IA, hallucination gates |
| Test Plan | docs/TEST_PLAN.md | P3-F01 eval humaine, P3-H01 hallucination |
| Model Card | models/model_card.json | Resultats eval, baselines |
| Eval functions | kaggle/eval-data/eval_generation.py | check_citation(), load_*() |
| Prompt v2 | kaggle/eval-data/generation_prompt.py | 7 regles RAG, anti-hallucination |

---

## 10. Historique

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-03-28 | Claude Opus 4.6 | Creation. Web research: ICTIR 2025, FACTS Grounding, HHEM, FaithBench, RAGAS, ISO 42005. Gap analysis, decision gates, protocole eval humaine. |

---

*Document ISO 42001/25010/29119 — Pocket Arbiter Project*
*Referentiel pour l'interpretation des resultats d'evaluation de la generation.*
