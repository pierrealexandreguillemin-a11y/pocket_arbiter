#!/usr/bin/env python3
"""
Optimize Gold Standard Annales v7 for triplet generation.

Optimizations:
- P0.1: Add "?" to questions missing it
- P0.2: Complete question_type for None values
- P0.3: Add reasoning_class (Know Your RAG taxonomy)
- P0.4: Fix encoding issues in chunk_ids

Standards:
- Know Your RAG (arXiv:2411.19710) - COLING 2025
- NV-Embed-v2 (arXiv:2405.17428)
- Sentence Transformers v3

ISO Reference: ISO 42001, ISO 25010
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


# Mapping for inferring question_type from cognitive_level + answer_type
QUESTION_TYPE_INFERENCE = {
    ("Remember", "extractive"): "factual",
    ("Remember", "multiple_choice"): "factual",
    ("Remember", "abstractive"): "factual",
    ("Remember", "list"): "factual",
    ("Remember", "yes_no"): "factual",
    ("Apply", "multiple_choice"): "scenario",
    ("Apply", "extractive"): "scenario",
    ("Apply", "abstractive"): "scenario",
    ("Understand", "abstractive"): "procedural",
    ("Understand", "extractive"): "procedural",
    ("Understand", "multiple_choice"): "procedural",
    ("Analyze", "multiple_choice"): "comparative",
    ("Analyze", "extractive"): "comparative",
    ("Analyze", "abstractive"): "comparative",
}

# Encoding fixes for chunk_ids
ENCODING_FIXES = {
    "parit�": "parite",
    "Loubati�re": "Loubatiere",
    "r�glement": "reglement",
    "r�gionale": "regionale",
}

# Question starters that indicate it should end with ?
QUESTION_STARTERS_FR = [
    "que ",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "quand",
    "comment",
    "pourquoi",
    "où",
    "qui",
    "est-ce",
    "peut-on",
    "doit-on",
    "faut-il",
    "combien",
    "lequel",
    "laquelle",
    "lesquels",
    "à qui",
    "à quoi",
    "de quoi",
    "avec quoi",
    "un joueur",
    "une équipe",
    "un arbitre",
    "un club",
    "vous êtes",
    "lors de",
    "en cas de",
    "si un",
    "pour un",
    "en nationale",
    "en coupe",
    "au cours",
    "l'arbitre",
    "le joueur",
    "la partie",
    "une partie",
    "dans le cas",
    "dans un",
    "suite à",
    "après",
    "avant",
    "pendant",
    "lorsque",
    "lorsqu",
    "il est",
    "elle est",
    "c'est",
    "ce sont",
]

# Patterns that indicate it's a scenario/question even without question words
SCENARIO_PATTERNS = [
    r"que (?:faites|fait|doit|devez|pouvez)-",
    r"quelle est (?:la|votre)",
    r"quel est (?:le|votre)",
    r"\.\s*(?:que|quel|quelle|comment|pourquoi)",
    r"\.{3}\s*$",  # Ends with ...
]


def normalize_question(
    question: str, has_mcq_choices: bool = False
) -> tuple[str, bool]:
    """
    Normalize question format.
    Returns (normalized_question, was_modified).

    For MCQ scenarios with choices, the question format is acceptable
    even without "?" because the question is implied in the choices.
    """
    q = question.strip()

    if q.endswith("?"):
        return q, False

    # MCQ scenarios are acceptable without "?" (question implied in choices)
    if has_mcq_choices:
        return q, False

    # Check if it looks like a question
    q_lower = q.lower()
    is_question = any(q_lower.startswith(starter) for starter in QUESTION_STARTERS_FR)

    # Also check for scenario patterns
    if not is_question:
        for pattern in SCENARIO_PATTERNS:
            if re.search(pattern, q_lower):
                is_question = True
                break

    if is_question:
        # Add space before ? if needed
        if q.endswith((".", ",", "...")):
            q = q.rstrip(".,")
        q = q + " ?"
        return q, True

    return q, False


def infer_question_type(
    cognitive_level: str | None, answer_type: str | None, reasoning_type: str | None
) -> str:
    """Infer question_type from other metadata fields."""

    # Try cognitive_level + answer_type mapping
    if cognitive_level and answer_type:
        key = (cognitive_level, answer_type)
        if key in QUESTION_TYPE_INFERENCE:
            return QUESTION_TYPE_INFERENCE[key]

    # Fallback based on cognitive_level alone
    if cognitive_level == "Remember":
        return "factual"
    elif cognitive_level == "Apply":
        return "scenario"
    elif cognitive_level == "Understand":
        return "procedural"
    elif cognitive_level == "Analyze":
        return "comparative"

    # Fallback based on reasoning_type
    if reasoning_type == "multi-hop":
        return "scenario"
    elif reasoning_type == "temporal":
        return "procedural"

    return "factual"  # Default


def infer_reasoning_class(reasoning_type: str | None, question_type: str | None) -> str:
    """
    Infer reasoning_class from reasoning_type and question_type.

    Know Your RAG taxonomy (arXiv:2411.19710):
    - fact_single: Answer is 1 unit of info in context
    - summary: Answer has multiple units of info
    - reasoning: Answer inferable but not explicit
    """
    rt = reasoning_type or "single-hop"
    qt = question_type or "factual"

    if rt == "single-hop" and qt in ["factual", "procedural"]:
        return "fact_single"
    elif rt == "multi-hop" and qt in ["scenario", "factual"]:
        return "summary"
    elif rt in ["multi-hop", "temporal"] and qt == "comparative":
        return "reasoning"
    elif qt == "scenario":
        return "summary"
    elif qt == "comparative":
        return "reasoning"

    return "fact_single"


def fix_encoding(text: str) -> tuple[str, bool]:
    """Fix encoding issues in text."""
    original = text
    for bad, good in ENCODING_FIXES.items():
        text = text.replace(bad, good)
    return text, text != original


def optimize_question(question: dict) -> dict:
    """Apply all optimizations to a single question."""
    q = question.copy()
    metadata = q.get("metadata", {}).copy()
    modifications = []

    # Check if this is an MCQ with choices
    has_mcq_choices = bool(metadata.get("choices"))

    # P0.1: Normalize question format (add ?)
    q_text, modified = normalize_question(q.get("question", ""), has_mcq_choices)
    if modified:
        q["question"] = q_text
        modifications.append("added_question_mark")

    # P0.2: Complete question_type if None
    if metadata.get("question_type") is None:
        inferred_type = infer_question_type(
            metadata.get("cognitive_level"),
            metadata.get("answer_type"),
            metadata.get("reasoning_type"),
        )
        metadata["question_type"] = inferred_type
        metadata["question_type_method"] = "inferred"
        modifications.append(f"inferred_question_type:{inferred_type}")

    # P0.3: Add reasoning_class (Know Your RAG)
    if "reasoning_class" not in metadata:
        reasoning_class = infer_reasoning_class(
            metadata.get("reasoning_type"), metadata.get("question_type")
        )
        metadata["reasoning_class"] = reasoning_class
        metadata["reasoning_class_method"] = "inferred"
        modifications.append(f"added_reasoning_class:{reasoning_class}")

    # P0.4: Fix encoding in chunk_id
    chunk_id = q.get("expected_chunk_id", "")
    fixed_chunk_id, encoding_fixed = fix_encoding(chunk_id)
    if encoding_fixed:
        q["expected_chunk_id"] = fixed_chunk_id
        modifications.append("fixed_chunk_id_encoding")

    # Fix encoding in expected_docs
    expected_docs = q.get("expected_docs", [])
    fixed_docs = []
    for doc in expected_docs:
        fixed_doc, _ = fix_encoding(doc)
        fixed_docs.append(fixed_doc)
    if fixed_docs != expected_docs:
        q["expected_docs"] = fixed_docs
        modifications.append("fixed_docs_encoding")

    # Set triplet_ready flag
    metadata["triplet_ready"] = len(modifications) == 0 or all(
        "inferred" in m or "added" in m for m in modifications
    )

    q["metadata"] = metadata

    return q, modifications


def compute_stats(questions: list[dict]) -> dict:
    """Compute statistics for the optimized dataset."""
    stats = {
        "total": len(questions),
        "with_question_mark": sum(
            1 for q in questions if q["question"].strip().endswith("?")
        ),
        "question_type_distribution": dict(
            Counter(q.get("metadata", {}).get("question_type") for q in questions)
        ),
        "reasoning_class_distribution": dict(
            Counter(q.get("metadata", {}).get("reasoning_class") for q in questions)
        ),
        "reasoning_type_distribution": dict(
            Counter(q.get("metadata", {}).get("reasoning_type") for q in questions)
        ),
        "cognitive_level_distribution": dict(
            Counter(q.get("metadata", {}).get("cognitive_level") for q in questions)
        ),
        "triplet_ready": sum(
            1 for q in questions if q.get("metadata", {}).get("triplet_ready")
        ),
    }
    return stats


def main():
    """Main optimization pipeline."""
    input_path = Path("tests/data/gold_standard_annales_fr_v7.json")
    output_path = Path("tests/data/gold_standard_annales_fr_v7.json")  # Overwrite
    report_path = Path(
        "tests/data/gold_standard_annales_fr_v7_optimization_report.json"
    )

    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_version = data.get("version", "7.3.0")
    questions = data.get("questions", [])
    print(f"Loaded {len(questions)} questions (v{original_version})")

    # Pre-optimization stats
    pre_stats = {
        "questions_without_mark": sum(
            1 for q in questions if not q["question"].strip().endswith("?")
        ),
        "question_type_none": sum(
            1 for q in questions if q.get("metadata", {}).get("question_type") is None
        ),
        "missing_reasoning_class": sum(
            1 for q in questions if "reasoning_class" not in q.get("metadata", {})
        ),
    }
    print("\nPre-optimization:")
    print(f"  - Questions without '?': {pre_stats['questions_without_mark']}")
    print(f"  - question_type=None: {pre_stats['question_type_none']}")
    print(f"  - Missing reasoning_class: {pre_stats['missing_reasoning_class']}")

    # Apply optimizations
    print("\nApplying optimizations...")
    optimized_questions = []
    all_modifications = []

    for q in questions:
        optimized_q, mods = optimize_question(q)
        optimized_questions.append(optimized_q)
        if mods:
            all_modifications.append({"id": q.get("id"), "modifications": mods})

    # Post-optimization stats
    post_stats = compute_stats(optimized_questions)

    print("\nPost-optimization:")
    print(
        f"  - Questions with '?': {post_stats['with_question_mark']}/{post_stats['total']}"
    )
    print(f"  - question_type distribution: {post_stats['question_type_distribution']}")
    print(
        f"  - reasoning_class distribution: {post_stats['reasoning_class_distribution']}"
    )
    print(f"  - triplet_ready: {post_stats['triplet_ready']}/{post_stats['total']}")

    # Update version and metadata
    new_version = "7.4.0"
    data["version"] = new_version
    data["questions"] = optimized_questions
    data["optimization"] = {
        "date": datetime.now().isoformat(),
        "from_version": original_version,
        "to_version": new_version,
        "changes": [
            "P0.1: Added '?' to questions missing it",
            "P0.2: Inferred question_type for None values",
            "P0.3: Added reasoning_class (Know Your RAG taxonomy)",
            "P0.4: Fixed encoding issues in chunk_ids",
        ],
        "standards": [
            "Know Your RAG (arXiv:2411.19710) - COLING 2025",
            "NV-Embed-v2 (arXiv:2405.17428)",
            "Sentence Transformers v3",
        ],
        "stats": post_stats,
    }

    # Update taxonomy_standards if present
    if "taxonomy_standards" not in data:
        data["taxonomy_standards"] = {}
    data["taxonomy_standards"]["reasoning_class"] = [
        "fact_single",
        "summary",
        "reasoning",
    ]

    # Save optimized data
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_version": original_version,
        "output_version": new_version,
        "pre_optimization": pre_stats,
        "post_optimization": post_stats,
        "modifications_count": len(all_modifications),
        "modifications_sample": all_modifications[:10],
        "modification_types": dict(
            Counter(mod for item in all_modifications for mod in item["modifications"])
        ),
    }

    print(f"Saving report to {report_path}...")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Optimization complete: v{original_version} -> v{new_version}")
    print(f"   Modified {len(all_modifications)} questions")

    return 0


if __name__ == "__main__":
    exit(main())
