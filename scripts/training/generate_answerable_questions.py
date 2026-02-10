"""
Génération de questions answerable pour Gold Standard - Pocket Arbiter

Analyse les chunks non couverts et génère des questions answerable conformes aux
standards académiques (SQuAD 2.0, QA Taxonomy, Bloom's Taxonomy).

ISO Reference: ISO/IEC 42001 A.6.2.2, ISO/IEC 25010 S4.2

Usage:
    python -m scripts.training.generate_answerable_questions \
        --gs tests/data/gold_standard_fr.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output data/training/new_questions_fr.json \
        --analyze
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Distribution cibles selon standards académiques
ANSWER_TYPE_DISTRIBUTION = {
    "FACTUAL": 0.36,  # What/Who/When facts
    "PROCEDURAL": 0.23,  # How to do something
    "LIST": 0.24,  # Multiple items
    "CONDITIONAL": 0.09,  # If/When conditions
    "DEFINITIONAL": 0.07,  # What is / definition
}

COGNITIVE_LEVEL_DISTRIBUTION = {
    "REMEMBER": 0.25,  # Recall facts
    "UNDERSTAND": 0.25,  # Explain concepts
    "APPLY": 0.25,  # Use in new situations
    "ANALYZE": 0.25,  # Break down, compare
}

REASONING_TYPE_DISTRIBUTION = {
    "LEXICAL_MATCH": 0.10,  # Direct word match
    "SINGLE_SENTENCE": 0.30,  # Answer in one sentence
    "MULTI_SENTENCE": 0.30,  # Answer spans multiple sentences
    "DOMAIN_KNOWLEDGE": 0.30,  # Requires chess expertise
}


def load_gold_standard(gs_path: Path) -> dict:
    """Charge le Gold Standard complet."""
    with open(gs_path, encoding="utf-8") as f:
        return json.load(f)


def load_chunks(chunks_path: Path) -> list[dict]:
    """Charge les chunks Mode B."""
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def get_covered_pages(questions: list[dict]) -> set[int]:
    """Retourne les pages couvertes par les questions answerable."""
    pages = set()
    for q in questions:
        hard_type = q.get("metadata", {}).get("hard_type", "ANSWERABLE")
        if hard_type == "ANSWERABLE":
            for p in q.get("expected_pages", []):
                pages.add(p)
    return pages


def analyze_coverage(
    questions: list[dict],
    chunks: list[dict],
    target_unanswerable_ratio: float = 0.33,
) -> dict:
    """Analyse la couverture et calcule les besoins.

    Args:
        questions: Liste des questions GS
        chunks: Liste des chunks
        target_unanswerable_ratio: Ratio cible unanswerable (default 33%)

    Returns:
        Dictionnaire avec analyse de couverture
    """
    # Count answerable vs unanswerable
    answerable = 0
    unanswerable = 0
    for q in questions:
        hard_type = q.get("metadata", {}).get("hard_type", "ANSWERABLE")
        if hard_type == "ANSWERABLE":
            answerable += 1
        else:
            unanswerable += 1

    total = len(questions)
    current_ratio = unanswerable / total if total > 0 else 0

    # Calculate need for target ratio
    # target_ratio = unanswerable / (answerable + need + unanswerable)
    # Solving: need = unanswerable / target_ratio - total
    target_total = (
        int(unanswerable / target_unanswerable_ratio) if unanswerable > 0 else total
    )
    need_answerable = max(0, target_total - total)

    # Analyze chunk coverage
    covered_pages = get_covered_pages(questions)
    all_pages = {c["page"] for c in chunks}
    uncovered_pages = all_pages - covered_pages

    # Group uncovered chunks by source
    source_chunks = defaultdict(list)
    for c in chunks:
        if c["page"] not in covered_pages:
            source_chunks[c["source"]].append(c)

    return {
        "total_questions": total,
        "answerable": answerable,
        "unanswerable": unanswerable,
        "current_ratio": round(current_ratio, 4),
        "target_ratio": target_unanswerable_ratio,
        "need_answerable": need_answerable,
        "target_total": target_total,
        "covered_pages": len(covered_pages),
        "total_pages": len(all_pages),
        "uncovered_pages": len(uncovered_pages),
        "uncovered_by_source": {
            src: len(chunks)
            for src, chunks in sorted(source_chunks.items(), key=lambda x: -len(x[1]))[
                :10
            ]
        },
    }


def sample_distribution(distribution: dict[str, float], count: int) -> list[str]:
    """Échantillonne selon une distribution cible.

    Args:
        distribution: Dict {category: probability}
        count: Nombre d'éléments à générer

    Returns:
        Liste de catégories
    """
    categories = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(categories, weights=weights, k=count)


def generate_question_templates(
    need: int,
    answer_types: list[str] | None = None,
    cognitive_levels: list[str] | None = None,
    reasoning_types: list[str] | None = None,
) -> list[dict]:
    """Génère des templates de questions avec classifications.

    Args:
        need: Nombre de templates à générer
        answer_types: Liste pré-générée ou None pour générer
        cognitive_levels: Liste pré-générée ou None pour générer
        reasoning_types: Liste pré-générée ou None pour générer

    Returns:
        Liste de templates avec metadata
    """
    if answer_types is None:
        answer_types = sample_distribution(ANSWER_TYPE_DISTRIBUTION, need)
    if cognitive_levels is None:
        cognitive_levels = sample_distribution(COGNITIVE_LEVEL_DISTRIBUTION, need)
    if reasoning_types is None:
        reasoning_types = sample_distribution(REASONING_TYPE_DISTRIBUTION, need)

    templates = []
    for i in range(need):
        templates.append(
            {
                "id": f"NEW-Q{i + 1:03d}",
                "question": "",
                "category": "",
                "expected_docs": [],
                "expected_pages": [],
                "keywords": [],
                "metadata": {
                    "hard_type": "ANSWERABLE",
                    "hard_reason": "Standard question - to be filled",
                    "answer_type": answer_types[i % len(answer_types)],
                    "cognitive_level": cognitive_levels[i % len(cognitive_levels)],
                    "reasoning_type": reasoning_types[i % len(reasoning_types)],
                },
                "difficulty": "medium",
            }
        )

    return templates


def suggest_themes_from_chunks(
    uncovered_chunks: list[dict],
    limit: int = 20,
) -> list[dict]:
    """Suggère des thèmes à partir des chunks non couverts.

    Args:
        uncovered_chunks: Liste de chunks non couverts
        limit: Nombre max de suggestions

    Returns:
        Liste de suggestions {source, page, section, text_preview}
    """
    suggestions = []
    seen_sections = set()

    for chunk in uncovered_chunks:
        section = chunk.get("section", "")
        if section and section not in seen_sections:
            seen_sections.add(section)
            text_preview = (
                chunk["text"][:200] + "..."
                if len(chunk["text"]) > 200
                else chunk["text"]
            )
            suggestions.append(
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "section": section,
                    "text_preview": text_preview,
                }
            )

        if len(suggestions) >= limit:
            break

    return suggestions


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Generate answerable questions for Gold Standard"
    )
    parser.add_argument("--gs", type=Path, required=True, help="Gold Standard JSON")
    parser.add_argument("--chunks", type=Path, required=True, help="Chunks Mode B JSON")
    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON for new questions"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze only, don't generate"
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.33,
        help="Target unanswerable ratio (default: 0.33)",
    )
    parser.add_argument(
        "--suggest", type=int, default=20, help="Number of theme suggestions"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    random.seed(args.seed)

    # Load data
    gs = load_gold_standard(args.gs)
    chunks = load_chunks(args.chunks)

    # Analyze coverage
    analysis = analyze_coverage(gs["questions"], chunks, args.target_ratio)

    print("\n" + "=" * 60)
    print("GOLD STANDARD ANALYSIS")
    print("=" * 60)
    print(f"Total questions: {analysis['total_questions']}")
    print(f"  Answerable: {analysis['answerable']}")
    print(f"  Unanswerable: {analysis['unanswerable']}")
    print(f"  Current ratio: {analysis['current_ratio']:.1%}")
    print(f"  Target ratio: {analysis['target_ratio']:.1%}")
    print()
    print(f"Need to add: {analysis['need_answerable']} answerable questions")
    print(f"Target total: {analysis['target_total']} questions")
    print()
    print("Chunk coverage:")
    print(f"  Covered pages: {analysis['covered_pages']}")
    print(f"  Total pages: {analysis['total_pages']}")
    print(f"  Uncovered pages: {analysis['uncovered_pages']}")
    print()
    print("Uncovered chunks by source:")
    for src, count in list(analysis["uncovered_by_source"].items())[:5]:
        print(f"  {src}: {count} chunks")

    if args.analyze:
        return

    # Get uncovered chunks for suggestions
    covered_pages = get_covered_pages(gs["questions"])
    uncovered_chunks = [c for c in chunks if c["page"] not in covered_pages]

    # Suggest themes
    suggestions = suggest_themes_from_chunks(uncovered_chunks, args.suggest)
    print()
    print("=" * 60)
    print("SUGGESTED THEMES (from uncovered chunks)")
    print("=" * 60)
    for i, s in enumerate(suggestions, 1):
        print(f"\n{i}. {s['section']}")
        print(f"   Source: {s['source']} (p{s['page']})")
        print(f"   Preview: {s['text_preview'][:100]}...")

    # Generate templates if output specified
    if args.output and analysis["need_answerable"] > 0:
        templates = generate_question_templates(analysis["need_answerable"])

        output_data = {
            "description": f"New answerable questions to add ({len(templates)} questions)",
            "analysis": analysis,
            "target_distributions": {
                "answer_type": ANSWER_TYPE_DISTRIBUTION,
                "cognitive_level": COGNITIVE_LEVEL_DISTRIBUTION,
                "reasoning_type": REASONING_TYPE_DISTRIBUTION,
            },
            "suggested_themes": suggestions,
            "question_templates": templates,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print()
        print(f"Generated {len(templates)} question templates to {args.output}")
        print("Fill in the 'question' and 'expected_pages' fields manually.")


if __name__ == "__main__":
    main()
