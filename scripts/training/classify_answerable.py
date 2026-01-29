#!/usr/bin/env python3
"""Classify answerable questions according to academic taxonomies.

References:
- arXiv:2107.12708 "QA Dataset Explosion: A Taxonomy of NLP Resources"
- arXiv:1606.05250 "SQuAD: 100,000+ Questions for Machine Comprehension"
- Bloom's Taxonomy (Revised, Anderson & Krathwohl 2001)

Taxonomies:
1. ANSWER_TYPE (what kind of answer is expected)
   - FACTUAL: Direct fact lookup (what/who/when/where)
   - PROCEDURAL: How-to process (how)
   - CAUSAL: Explanation/reasoning (why)
   - DEFINITIONAL: Term definition (what is)
   - COMPARATIVE: Comparison between entities
   - CONDITIONAL: Context-dependent rules (if/when)
   - LIST: Multiple items expected

2. REASONING_TYPE (cognitive process required)
   - LEXICAL_MATCH: Direct word overlap with source
   - PARAPHRASE: Semantic matching
   - SINGLE_SENTENCE: Answer in one sentence
   - MULTI_SENTENCE: Synthesis from multiple sentences
   - DOMAIN_KNOWLEDGE: Requires chess knowledge

3. COGNITIVE_LEVEL (Bloom's Taxonomy)
   - REMEMBER: Recall facts
   - UNDERSTAND: Explain/interpret
   - APPLY: Use in new situation
   - ANALYZE: Break down/compare
"""

import json
import re
from collections import defaultdict
from pathlib import Path


def classify_answer_type(question: str) -> str:
    """Classify question by expected answer type."""
    q = question.lower()

    # Procedural: How to do something
    if re.search(r"^comment\s|comment\s.*\?$|^how\s", q):
        return "PROCEDURAL"

    # Causal: Why something happens
    if re.search(r"^pourquoi\s|^why\s", q):
        return "CAUSAL"

    # Definitional: What is X
    if re.search(r"^qu'est-ce qu|^what is\s|^define\s|definition", q):
        return "DEFINITIONAL"

    # Comparative: Difference between X and Y
    if re.search(r"difference|comparer|versus|vs\.|distinction|^which\s.*better", q):
        return "COMPARATIVE"

    # Conditional: If/when conditions
    if re.search(r"^si\s|^quand\s|^dans quel cas|^when\s|^if\s|peut-on", q):
        return "CONDITIONAL"

    # List: Multiple items expected
    if re.search(r"quels sont|quelles sont|list|enumerate|^what are\s", q):
        return "LIST"

    # Default: Factual (what/who/when/where)
    return "FACTUAL"


def classify_reasoning_type(question: str, has_multi_page: bool) -> str:
    """Classify question by reasoning required."""
    q = question.lower()

    # Multi-sentence if references multiple concepts or has complex conditions
    complex_patterns = [
        r"et\s.*et\s",  # Multiple "and"
        r"ou\s.*ou\s",  # Multiple "or"
        r"difference.*entre",  # Comparison
        r"toutes les",  # All of
        r"plusieurs",  # Multiple
    ]
    for pattern in complex_patterns:
        if re.search(pattern, q):
            return "MULTI_SENTENCE"

    # Domain knowledge for chess-specific terms
    chess_terms = [
        "zeitnot",
        "roque",
        "echec",
        "mat",
        "pat",
        "promotion",
        "prise en passant",
        "cadence",
        "elo",
        "fide",
        "ffe",
        "castling",
        "checkmate",
        "stalemate",
        "en passant",
    ]
    for term in chess_terms:
        if term in q:
            return "DOMAIN_KNOWLEDGE"

    # If multi-page reference, likely multi-sentence
    if has_multi_page:
        return "MULTI_SENTENCE"

    # Simple questions likely single sentence
    if len(question) < 50:
        return "LEXICAL_MATCH"

    return "SINGLE_SENTENCE"


def classify_cognitive_level(answer_type: str, reasoning_type: str) -> str:
    """Classify question by Bloom's cognitive level."""
    # Remember: Direct fact recall
    if answer_type == "FACTUAL" and reasoning_type in [
        "LEXICAL_MATCH",
        "SINGLE_SENTENCE",
    ]:
        return "REMEMBER"

    # Understand: Explain/interpret
    if answer_type in ["DEFINITIONAL", "CAUSAL"]:
        return "UNDERSTAND"

    # Apply: Use knowledge in context
    if answer_type in ["PROCEDURAL", "CONDITIONAL"]:
        return "APPLY"

    # Analyze: Compare/break down
    if answer_type in ["COMPARATIVE", "LIST"] or reasoning_type == "MULTI_SENTENCE":
        return "ANALYZE"

    return "REMEMBER"


def load_json(path: Path) -> dict:
    """Load JSON file with UTF-8 encoding."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """Save JSON file with UTF-8 encoding and formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    """Classify all answerable questions."""
    gs_fr_path = Path("tests/data/gold_standard_fr.json")
    gs_intl_path = Path("tests/data/gold_standard_intl.json")

    gs_fr = load_json(gs_fr_path)
    gs_intl = load_json(gs_intl_path)

    stats = {
        "answer_type": defaultdict(int),
        "reasoning_type": defaultdict(int),
        "cognitive_level": defaultdict(int),
        "classified": 0,
        "skipped_unanswerable": 0,
    }

    for gs, name in [(gs_fr, "FR"), (gs_intl, "INTL")]:
        for q in gs["questions"]:
            # Skip unanswerable (no expected_pages and has hard_type)
            hard_type = q.get("metadata", {}).get("hard_type")
            if not q.get("expected_pages") and hard_type and hard_type != "ANSWERABLE":
                stats["skipped_unanswerable"] += 1
                continue

            # Initialize metadata if needed
            if "metadata" not in q:
                q["metadata"] = {}

            # Classify
            has_multi_page = len(q.get("expected_pages", [])) > 1
            answer_type = classify_answer_type(q["question"])
            reasoning_type = classify_reasoning_type(q["question"], has_multi_page)
            cognitive_level = classify_cognitive_level(answer_type, reasoning_type)

            # Store classification
            q["metadata"]["answer_type"] = answer_type
            q["metadata"]["reasoning_type"] = reasoning_type
            q["metadata"]["cognitive_level"] = cognitive_level

            # Update stats
            stats["answer_type"][answer_type] += 1
            stats["reasoning_type"][reasoning_type] += 1
            stats["cognitive_level"][cognitive_level] += 1
            stats["classified"] += 1

    # Save updated files
    save_json(gs_fr_path, gs_fr)
    save_json(gs_intl_path, gs_intl)

    # Print report
    print("=" * 70)
    print("CLASSIFICATION DES QUESTIONS ANSWERABLE")
    print("Reference: arXiv:2107.12708 'QA Dataset Explosion'")
    print("=" * 70)
    print()
    print(f"Questions classifiees: {stats['classified']}")
    print(f"Questions unanswerable (skipped): {stats['skipped_unanswerable']}")
    print()

    print("1. ANSWER_TYPE (type de reponse attendue)")
    print("-" * 40)
    for t, count in sorted(stats["answer_type"].items(), key=lambda x: -x[1]):
        pct = count / stats["classified"] * 100
        print(f"   {t}: {count} ({pct:.1f}%)")
    print()

    print("2. REASONING_TYPE (raisonnement requis)")
    print("-" * 40)
    for t, count in sorted(stats["reasoning_type"].items(), key=lambda x: -x[1]):
        pct = count / stats["classified"] * 100
        print(f"   {t}: {count} ({pct:.1f}%)")
    print()

    print("3. COGNITIVE_LEVEL (niveau Bloom)")
    print("-" * 40)
    for t, count in sorted(stats["cognitive_level"].items(), key=lambda x: -x[1]):
        pct = count / stats["classified"] * 100
        print(f"   {t}: {count} ({pct:.1f}%)")
    print()

    print("References academiques:")
    print("  - arXiv:2107.12708 'QA Dataset Explosion: A Taxonomy'")
    print("  - arXiv:1606.05250 'SQuAD: 100,000+ Questions'")
    print("  - Anderson & Krathwohl (2001) Bloom's Revised Taxonomy")


if __name__ == "__main__":
    main()
