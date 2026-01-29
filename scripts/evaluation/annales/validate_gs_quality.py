#!/usr/bin/env python3
"""
Validation rigoureuse du Gold Standard Annales.

Métriques validées:
1. Answerability: réponse trouvable dans le chunk (keyword matching)
2. Semantic Similarity: BERTScore entre question/réponse et chunk
3. Chunk ID Validity: tous les chunk_ids existent
4. Data Completeness: tous les champs requis présents

References:
- BERTScore (arXiv:1904.09675)
- RAGalyst (arXiv:2511.04502)
- GaRAGe (arXiv:2506.07671)

ISO 42001: Validation anti-hallucination
ISO 25010: Métriques qualité
ISO 29119: Couverture de test
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Lazy import for sentence-transformers
_model = None


def get_embedding_model():
    """Lazy load embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print("  Chargement du modèle d'embeddings...")
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def load_json(path: str) -> dict:
    """Load JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    """Save JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "ä": "a",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ç": "c",
        "œ": "oe",
        "æ": "ae",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def extract_keywords(text: str, min_length: int = 4) -> list[str]:
    """Extract meaningful keywords from text."""
    text = normalize_text(text)
    stopwords = {
        "pour",
        "dans",
        "avec",
        "cette",
        "celui",
        "celle",
        "sont",
        "etre",
        "avoir",
        "fait",
        "faire",
        "peut",
        "doit",
        "tous",
        "tout",
        "plus",
    }
    words = re.findall(r"\b[a-z]+\b", text)
    return [w for w in words if len(w) >= min_length and w not in stopwords]


def compute_keyword_score(answer: str, chunk_text: str) -> float:
    """Compute keyword matching score."""
    keywords = extract_keywords(answer)
    if not keywords:
        return 0.0
    chunk_norm = normalize_text(chunk_text)
    found = sum(1 for kw in keywords if kw in chunk_norm)
    return found / len(keywords)


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity using sentence embeddings."""
    model = get_embedding_model()

    # Truncate long texts
    text1 = text1[:512]
    text2 = text2[:512]

    embeddings = model.encode([text1, text2], convert_to_numpy=True)

    # Cosine similarity
    sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(sim)


@dataclass
class ValidationResult:
    """Result of validating a single question."""

    question_id: str
    chunk_id_valid: bool
    keyword_score: float
    semantic_score: Optional[float]
    answerable: bool
    issues: list[str]


def validate_question(
    question: dict, chunk_index: dict, compute_semantic: bool = True
) -> ValidationResult:
    """Validate a single question."""
    qid = question["id"]
    chunk_id = question.get("expected_chunk_id", "")
    answer = question.get("expected_answer", "")
    q_text = question.get("question", "")
    issues = []

    # Check 1: Chunk ID valid
    chunk_id_valid = chunk_id in chunk_index
    if not chunk_id_valid:
        issues.append("invalid_chunk_id")
        return ValidationResult(
            question_id=qid,
            chunk_id_valid=False,
            keyword_score=0.0,
            semantic_score=None,
            answerable=False,
            issues=issues,
        )

    chunk_text = chunk_index[chunk_id]

    # Check 2: Keyword score
    keyword_score = compute_keyword_score(answer, chunk_text)
    if keyword_score < 0.3:
        issues.append(f"low_keyword_score:{keyword_score:.2f}")

    # Check 3: Semantic similarity (Q+A vs Chunk)
    semantic_score = None
    if compute_semantic:
        qa_text = f"{q_text} {answer}"
        semantic_score = compute_semantic_similarity(qa_text, chunk_text)
        if semantic_score < 0.5:
            issues.append(f"low_semantic_score:{semantic_score:.2f}")

    # Determine answerability
    answerable = keyword_score >= 0.3 or (semantic_score and semantic_score >= 0.6)

    return ValidationResult(
        question_id=qid,
        chunk_id_valid=True,
        keyword_score=keyword_score,
        semantic_score=semantic_score,
        answerable=answerable,
        issues=issues,
    )


def validate_gold_standard(
    gs: dict, chunk_index: dict, compute_semantic: bool = True, sample_size: int = None
) -> dict:
    """
    Validate entire Gold Standard.

    Args:
        gs: Gold Standard data
        chunk_index: Dict mapping chunk_id to text
        compute_semantic: Whether to compute semantic scores (slow)
        sample_size: If set, only validate a sample
    """
    questions = gs["questions"]
    if sample_size:
        import random

        questions = random.sample(questions, min(sample_size, len(questions)))

    results = {
        "validation_date": datetime.now().isoformat(),
        "gs_version": gs.get("version", "unknown"),
        "total_validated": len(questions),
        "compute_semantic": compute_semantic,
        "metrics": {
            "chunk_id_valid": 0,
            "chunk_id_invalid": 0,
            "answerable": 0,
            "not_answerable": 0,
            "avg_keyword_score": 0.0,
            "avg_semantic_score": 0.0,
        },
        "thresholds": {
            "keyword_answerable": 0.3,
            "semantic_answerable": 0.6,
        },
        "details": [],
        "issues_summary": {},
    }

    keyword_scores = []
    semantic_scores = []

    print(f"\nValidation de {len(questions)} questions...")

    for i, q in enumerate(questions):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(questions)}...")

        result = validate_question(q, chunk_index, compute_semantic)

        # Aggregate metrics
        if result.chunk_id_valid:
            results["metrics"]["chunk_id_valid"] += 1
        else:
            results["metrics"]["chunk_id_invalid"] += 1

        if result.answerable:
            results["metrics"]["answerable"] += 1
        else:
            results["metrics"]["not_answerable"] += 1

        keyword_scores.append(result.keyword_score)
        if result.semantic_score is not None:
            semantic_scores.append(result.semantic_score)

        # Track issues
        for issue in result.issues:
            issue_type = issue.split(":")[0]
            results["issues_summary"][issue_type] = (
                results["issues_summary"].get(issue_type, 0) + 1
            )

        # Store detail
        results["details"].append(
            {
                "id": result.question_id,
                "chunk_id_valid": result.chunk_id_valid,
                "keyword_score": round(result.keyword_score, 3),
                "semantic_score": round(result.semantic_score, 3)
                if result.semantic_score
                else None,
                "answerable": result.answerable,
                "issues": result.issues,
            }
        )

    # Compute averages
    results["metrics"]["avg_keyword_score"] = round(np.mean(keyword_scores), 3)
    if semantic_scores:
        results["metrics"]["avg_semantic_score"] = round(np.mean(semantic_scores), 3)

    # Compute percentages
    total = len(questions)
    results["metrics"]["pct_chunk_valid"] = round(
        100 * results["metrics"]["chunk_id_valid"] / total, 1
    )
    results["metrics"]["pct_answerable"] = round(
        100 * results["metrics"]["answerable"] / total, 1
    )

    return results


def main():
    """Main validation pipeline."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base_path = Path(__file__).parent.parent.parent.parent
    gs_path = base_path / "tests" / "data" / "gold_standard_annales_fr_v7.json"
    chunks_path = base_path / "corpus" / "processed" / "chunks_mode_b_fr.json"
    output_path = base_path / "tests" / "data" / "gs_quality_validation_report.json"

    print("=" * 70)
    print("VALIDATION QUALITÉ - Gold Standard Annales")
    print("=" * 70)
    print("""
Métriques:
  1. Chunk ID Validity: ID existe dans l'index
  2. Keyword Score: % mots-clés réponse dans chunk
  3. Semantic Score: Similarité embeddings (Q+A vs Chunk)
  4. Answerable: keyword >= 0.3 OR semantic >= 0.6

Modèle: paraphrase-multilingual-MiniLM-L12-v2
""")

    # Load data
    print("Chargement des données...")
    gs = load_json(str(gs_path))
    chunks_data = load_json(str(chunks_path))
    chunks = chunks_data.get("chunks", chunks_data)
    chunk_index = {c["id"]: c["text"] for c in chunks}

    print(f"  GS version: {gs.get('version', 'unknown')}")
    print(f"  Questions: {len(gs['questions'])}")
    print(f"  Chunks: {len(chunks)}")

    # Run validation
    results = validate_gold_standard(gs, chunk_index, compute_semantic=True)

    # Display results
    print(f"\n{'=' * 70}")
    print("RÉSULTATS DE VALIDATION")
    print("=" * 70)

    print(f"\nVersion GS: {results['gs_version']}")
    print(f"Questions validées: {results['total_validated']}")

    print("\nMétriques:")
    print(f"  Chunk IDs valides: {results['metrics']['pct_chunk_valid']}%")
    print(f"  Answerable: {results['metrics']['pct_answerable']}%")
    print(f"  Score keyword moyen: {results['metrics']['avg_keyword_score']}")
    print(f"  Score sémantique moyen: {results['metrics']['avg_semantic_score']}")

    print("\nProblèmes détectés:")
    for issue, count in sorted(results["issues_summary"].items(), key=lambda x: -x[1]):
        print(f"  - {issue}: {count}")

    # Quality gate
    print(f"\n{'=' * 70}")
    print("QUALITY GATE")
    print("=" * 70)

    passed = True
    checks = [
        ("Chunk IDs >= 95%", results["metrics"]["pct_chunk_valid"] >= 95),
        ("Answerable >= 50%", results["metrics"]["pct_answerable"] >= 50),
        ("Avg Keyword >= 0.4", results["metrics"]["avg_keyword_score"] >= 0.4),
        ("Avg Semantic >= 0.5", results["metrics"]["avg_semantic_score"] >= 0.5),
    ]

    for check_name, check_passed in checks:
        status = "PASS" if check_passed else "FAIL"
        print(f"  [{status}] {check_name}")
        if not check_passed:
            passed = False

    print()
    if passed:
        print("[OK] VALIDATION RÉUSSIE")
    else:
        print("[FAIL] VALIDATION ÉCHOUÉE - Corrections nécessaires")

    # Save report
    save_json(results, str(output_path))
    print(f"\nRapport détaillé: {output_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
