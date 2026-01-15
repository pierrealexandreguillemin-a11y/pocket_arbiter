"""
Create Gold Standard v4 - ISO 42001 Compliant (Non-Circular)

Méthodologie:
1. Utilise expected_pages du v1 (créées manuellement)
2. Valide par présence keywords dans corpus extrait
3. Indépendant du système de retrieval testé

ISO 42001 A.7.3: Validation IA indépendante du système testé.
"""

import json
from pathlib import Path

print("=== CRÉATION GOLD STANDARD V4 ISO 42001 ===")
print("Méthodologie: Validation indépendante (non-circulaire)")
print()

# Charger v1 backup (pages originales manuelles)
with open("tests/data/questions_fr_v1_backup.json", "r", encoding="utf-8") as f:
    v1_data = json.load(f)

# Charger extractions PDF
raw_dir = Path("corpus/processed/raw_fr")

# Index: source -> {page -> text}
corpus_index: dict[str, dict[int, str]] = {}
for json_file in raw_dir.glob("*.json"):
    if json_file.name == "extraction_report.json":
        continue
    with open(json_file, "r", encoding="utf-8") as f:
        doc = json.load(f)
    source = doc["filename"]
    corpus_index[source] = {}
    for page in doc["pages"]:
        corpus_index[source][page["page_num"]] = page["text"].lower()

print(f"Corpus indexé: {len(corpus_index)} documents")
print()

# Valider chaque question
gold_questions = []
validated_count = 0
issues = []

for q in v1_data["questions"]:
    qid = q["id"]
    question = q["question"]
    keywords = q["keywords"]
    expected_docs = q["expected_docs"]
    expected_pages_v1 = q["expected_pages"]

    # Valider les pages v1 contre le corpus
    validated_pages = []
    validation_details = []

    for doc in expected_docs:
        if doc not in corpus_index:
            issues.append(f"{qid}: Document {doc} non trouvé")
            continue

        for page_num in expected_pages_v1:
            if page_num not in corpus_index[doc]:
                # Page hors bornes, chercher dans tout le corpus
                continue

            page_text = corpus_index[doc][page_num]

            # Vérifier présence d'au moins un keyword
            found_keywords = [kw for kw in keywords if kw.lower() in page_text]

            if found_keywords:
                validated_pages.append(page_num)
                validation_details.append(
                    {"page": page_num, "source": doc, "keywords_found": found_keywords}
                )

    # Si pages v1 ne matchent pas, chercher alternatives dans le corpus
    if not validated_pages:
        # Recherche étendue dans tout le document
        for doc in expected_docs:
            if doc not in corpus_index:
                continue
            for page_num, page_text in corpus_index[doc].items():
                found_keywords = [kw for kw in keywords if kw.lower() in page_text]
                if len(found_keywords) >= 2:  # Au moins 2 keywords
                    validated_pages.append(page_num)
                    validation_details.append(
                        {
                            "page": page_num,
                            "source": doc,
                            "keywords_found": found_keywords,
                        }
                    )
                    if len(validated_pages) >= 3:
                        break

    status = "VALIDATED" if validated_pages else "UNVALIDATED"
    if status == "VALIDATED":
        validated_count += 1
    else:
        issues.append(f"{qid}: Aucune page validée pour '{question[:50]}...'")

    gold_questions.append(
        {
            "id": qid,
            "question": question,
            "category": q.get("category", "general"),
            "expected_pages": validated_pages[:3]
            if validated_pages
            else expected_pages_v1,
            "expected_docs": expected_docs,
            "keywords": keywords,
            "validation": {
                "status": status,
                "method": "keyword_in_corpus",
                "original_pages_v1": expected_pages_v1,
                "validated_pages": validated_pages[:3],
                "details": validation_details[:3],
            },
        }
    )

    status_icon = "✓" if status == "VALIDATED" else "✗"
    print(f"{qid}: {validated_pages[:3] if validated_pages else 'FAILED'} [{status}]")

# Sauvegarder gold standard v4
gold_standard = {
    "version": "4.0",
    "description": "Gold standard ISO 42001 - Validation indépendante (non-circulaire)",
    "methodology": {
        "base": "expected_pages v1 (manuelles)",
        "validation": "Keyword presence in extracted PDF corpus",
        "independence": "Indépendant du système de retrieval testé",
        "corpus": "FFE Règlements (28 PDFs extraits)",
    },
    "statistics": {
        "total_questions": len(gold_questions),
        "validated_questions": validated_count,
        "validation_rate": f"{validated_count/len(gold_questions):.1%}",
    },
    "issues": issues,
    "questions": gold_questions,
}

with open("tests/data/questions_fr.json", "w", encoding="utf-8") as f:
    json.dump(gold_standard, f, ensure_ascii=False, indent=2)

print()
print("=== RÉSULTAT ===")
print(
    f"Questions validées: {validated_count}/{len(gold_questions)} ({validated_count/len(gold_questions):.1%})"
)
if issues:
    print(f"Issues: {len(issues)}")
    for issue in issues[:5]:
        print(f"  - {issue}")
print("Sauvegardé: tests/data/questions_fr.json")
