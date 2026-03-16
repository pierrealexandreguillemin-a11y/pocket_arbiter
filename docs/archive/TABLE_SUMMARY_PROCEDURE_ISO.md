# Procedure ISO - Generation de Table Summaries

> **Document**: TABLE_SUMMARY_PROCEDURE_ISO.md
> **Version**: 1.0
> **Date**: 2026-01-19
> **Auteur**: Claude Code
> **Normes**: ISO/IEC 42001:2023, ISO/IEC 25010:2023

## 1. Objet et Portee

Ce document definit la procedure conforme ISO pour la generation de summaries
de tables destines au retrieval semantique RAG dans Pocket Arbiter.

### 1.1 Normes de Reference

| Norme | Section | Exigence |
|-------|---------|----------|
| ISO/IEC 42001 | A.7.4 | Qualite des donnees AI: accuracy, completeness, consistency |
| ISO/IEC 42001 | A.7.3 | Provenance et tracabilite des donnees |
| ISO/IEC 25010 | 4.2.1 | Functional Correctness: resultats precis |
| ISO/IEC 25010 | 4.2.2 | Functional Completeness: couverture exhaustive |

### 1.2 Definitions

- **Summary**: Description semantique d'une table pour embedding vectoriel
- **Table Parent**: Contenu brut de la table (markdown/texte)
- **Table Child**: Summary embeddable pour recherche semantique
- **Discriminant**: Capacite a distinguer une table d'une autre

## 2. Exigences de Qualite (ISO 42001 A.7.4)

### 2.1 Criteres de Qualite

Chaque summary DOIT respecter:

| Critere | Definition | Validation |
|---------|------------|------------|
| **Accuracy** | Description fidele au contenu reel | Verification croisee avec table source |
| **Completeness** | Informations essentielles presentes | Checklist elements obligatoires |
| **Consistency** | Terminologie uniforme | Glossaire de reference |
| **Discriminant** | Unique parmi toutes les tables | Test de non-duplication |
| **Non-hallucination** | 0% d'information inventee | Audit tracabilite |

### 2.2 Elements Obligatoires d'un Summary

Un summary DOIT contenir:

1. **Type de table** (classification correcte):
   - Appariements Berger (toutes rondes par equipes)
   - Appariements Scheveningen (schema specifique)
   - Appariements Suisse (systeme suisse)
   - Appariements toutes rondes (round-robin individuel)
   - Bareme/Grille (calcul Elo, frais, etc.)
   - Categories (age, titres, etc.)
   - Questionnaire (medical, formulaire)
   - Table des matieres
   - Autre (avec description)

2. **Parametres specifiques** (si applicable):
   - Nombre d'equipes
   - Nombre de joueurs par equipe
   - Nombre de rondes
   - Cadence/temps

3. **Source documentaire**:
   - Document FFE vs FIDE
   - Reference article/chapitre si connue

4. **Langue du summary**:
   - FR pour corpus FFE
   - EN pour corpus FIDE

### 2.3 Interdictions

Un summary NE DOIT PAS:

- Etre identique a un autre summary (test unicite)
- Contenir "..." ou troncatures
- Utiliser des termes generiques seuls ("Tableau technique")
- Inventer des informations non presentes dans la table
- Confondre les types d'appariements (Berger ≠ Scheveningen ≠ Suisse)

## 3. Procedure de Generation

### 3.1 Etape 1: Analyse de la Table Source

```
POUR chaque table:
  1. Lire headers complets
  2. Lire 5 premieres lignes de contenu
  3. Identifier le type de table (classification)
  4. Extraire parametres specifiques
  5. Verifier source documentaire
```

### 3.2 Etape 2: Classification

**Regles de classification des appariements echecs:**

| Pattern Headers | Classification |
|-----------------|----------------|
| "Tableau N pour N equipes a N joueurs" | Berger (toutes rondes equipes) |
| "Ronde 1, Ronde 2..." + "Echiquier" | Berger ou Scheveningen |
| "R1, 1-(N), 2-N..." | Suisse (grille appariements) |
| Joueurs individuels + rondes | Toutes rondes individuel |
| "A1-B1, B2-A2" pattern | Scheveningen (croise) |

### 3.3 Etape 3: Redaction du Summary

**Template FR:**
```
[Type] pour [parametres specifiques]: [description contenu] ([source]).
```

**Exemples conformes:**
```
Appariements Berger 3 equipes x 4 joueurs: grille toutes rondes interclubs (LA FFE).
Appariements Suisse 8 joueurs: table appariements rondes 1-4 systeme hollandais (LA FFE).
Categories age FFE U8-S65: codes, appellations officielles et tranches age (R01).
Bareme frais deplacement FFE: constantes et prix/km par tranche distance (Reglement Financier).
```

### 3.4 Etape 4: Validation

Checklist de validation (TOUS les points requis):

- [ ] Summary ≤ 150 caracteres
- [ ] Type de table correct (classification verifiee)
- [ ] Parametres specifiques presents
- [ ] Pas de troncature "..."
- [ ] Pas de duplication avec autre summary
- [ ] Terminologie conforme glossaire
- [ ] Verification croisee avec contenu table

## 4. Glossaire de Reference

### 4.1 Types d'Appariements Echecs

| Terme | Definition | Contexte |
|-------|------------|----------|
| **Berger** | Toutes rondes par equipes, chaque equipe rencontre toutes les autres | Interclubs, CdF |
| **Scheveningen** | Schema croise, equipe A vs equipe B sur plusieurs echiquiers | Matches internationaux |
| **Suisse** | Appariements dynamiques selon score, pas de toutes rondes | Opens, grands tournois |
| **Toutes rondes** | Round-robin individuel, chaque joueur rencontre tous les autres | Petits tournois |

### 4.2 Abreviations

| Abbrev | Signification |
|--------|---------------|
| LA | Livre de l'Arbitre FFE |
| R01 | Regles Generales FFE |
| CdF | Coupe de France |
| FFE | Federation Francaise des Echecs |
| FIDE | Federation Internationale des Echecs |

## 5. Tracabilite (ISO 42001 A.7.3)

### 5.1 Metadata Obligatoires

Chaque fichier de summaries DOIT inclure:

```json
{
  "summaries": {...},
  "metadata": {
    "source": "claude_code",
    "procedure": "TABLE_SUMMARY_PROCEDURE_ISO.md",
    "version": "1.0",
    "generated_at": "ISO8601 timestamp",
    "validated_by": "claude_code",
    "validation_checklist": {
      "accuracy_verified": true,
      "no_duplicates": true,
      "no_truncation": true,
      "classification_verified": true
    }
  }
}
```

### 5.2 Audit Trail

Chaque modification DOIT etre tracee:
- Commit Git avec message conventionnel
- Reference a cette procedure dans le commit

## 6. References

### Sources ISO
- [ISO/IEC 42001:2023 - AI Management Systems](https://www.iso.org/standard/42001)
- [ISO/IEC 42001 A.7.4 - Quality of Data for AI Systems](https://www.isms.online/iso-42001/annex-a-controls/a-7-data-for-ai-systems/a-7-4-quality-of-data-for-ai-systems/)
- [ISO/IEC 25010:2023 - Product Quality Model](https://www.iso.org/standard/78176.html)

### Sources RAG Best Practices
- [AWS RAG Writing Best Practices](https://docs.aws.amazon.com/pdfs/prescriptive-guidance/latest/writing-best-practices-rag/writing-best-practices-rag.pdf)
- [RAG with Summaries as Metadata](https://link.springer.com/chapter/10.1007/978-981-96-8343-7_8)

---

**Validation**: Ce document a ete cree conformement aux exigences ISO/IEC 42001 A.7.3 (provenance)
et A.7.4 (qualite des donnees).
