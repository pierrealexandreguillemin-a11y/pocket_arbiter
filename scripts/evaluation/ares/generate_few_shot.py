"""Generate few-shot examples for ARES LLM judge.

ISO Reference: ISO 42001 A.7.3 - Explainable AI
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


# Path constants
BASE_DIR = Path(__file__).parent.parent.parent.parent
TESTS_DATA_DIR = BASE_DIR / "tests" / "data"
CORPUS_DIR = BASE_DIR / "corpus" / "processed"
DATA_TRAINING_DIR = BASE_DIR / "data" / "training"
OUTPUT_DIR = BASE_DIR / "data" / "evaluation" / "ares"

# Curated few-shot examples with reasoning (French)
FEW_SHOT_POSITIVE_TEMPLATES_FR = [
    {
        "reasoning": "Le document explique directement la règle demandée dans la question. "
        "Les mots-clés de la question apparaissent dans le contexte et "
        "la réponse peut être extraite sans ambiguïté.",
        "pattern": "direct_match",
    },
    {
        "reasoning": "Le contexte contient l'article de règlement pertinent qui répond "
        "à la question posée. L'information est factuelle et vérifiable.",
        "pattern": "article_reference",
    },
    {
        "reasoning": "La question porte sur une procédure spécifique et le document "
        "décrit exactement cette procédure avec les étapes à suivre.",
        "pattern": "procedure",
    },
    {
        "reasoning": "Le document traite du même sujet que la question et fournit "
        "les informations nécessaires pour y répondre complètement.",
        "pattern": "topic_match",
    },
    {
        "reasoning": "Le contexte contient une définition ou explication qui répond "
        "directement à la question de compréhension posée.",
        "pattern": "definition",
    },
]

FEW_SHOT_NEGATIVE_TEMPLATES_FR = [
    {
        "reasoning": "Le document traite d'un sujet complètement différent de la question. "
        "Aucun des mots-clés pertinents n'apparaît dans le contexte.",
        "pattern": "topic_mismatch",
    },
    {
        "reasoning": "Bien que le document soit du même domaine (échecs), il concerne "
        "un aspect réglementaire différent qui ne répond pas à la question.",
        "pattern": "domain_adjacent",
    },
    {
        "reasoning": "Le contexte mentionne des termes similaires mais dans un contexte "
        "différent qui ne permet pas de répondre à la question posée.",
        "pattern": "false_positive",
    },
    {
        "reasoning": "Le document est une table de données ou annexe technique "
        "qui ne contient pas d'explication pour la question.",
        "pattern": "data_table",
    },
    {
        "reasoning": "Le contexte traite d'une catégorie de joueurs ou compétition "
        "différente de celle mentionnée dans la question.",
        "pattern": "category_mismatch",
    },
]

# Curated few-shot examples with reasoning (English for INTL)
FEW_SHOT_POSITIVE_TEMPLATES_EN = [
    {
        "reasoning": "The document directly explains the rule asked in the question. "
        "Keywords from the question appear in the context and "
        "the answer can be extracted unambiguously.",
        "pattern": "direct_match",
    },
    {
        "reasoning": "The context contains the relevant regulation article that answers "
        "the question. The information is factual and verifiable.",
        "pattern": "article_reference",
    },
    {
        "reasoning": "The question is about a specific procedure and the document "
        "describes exactly that procedure with the steps to follow.",
        "pattern": "procedure",
    },
    {
        "reasoning": "The document covers the same topic as the question and provides "
        "the necessary information to answer it completely.",
        "pattern": "topic_match",
    },
    {
        "reasoning": "The context contains a definition or explanation that directly "
        "answers the comprehension question asked.",
        "pattern": "definition",
    },
]

FEW_SHOT_NEGATIVE_TEMPLATES_EN = [
    {
        "reasoning": "The document covers a completely different topic from the question. "
        "None of the relevant keywords appear in the context.",
        "pattern": "topic_mismatch",
    },
    {
        "reasoning": "Although the document is from the same domain (chess), it concerns "
        "a different regulatory aspect that does not answer the question.",
        "pattern": "domain_adjacent",
    },
    {
        "reasoning": "The context mentions similar terms but in a different context "
        "that does not allow answering the question asked.",
        "pattern": "false_positive",
    },
    {
        "reasoning": "The document is a data table or technical appendix "
        "that does not contain an explanation for the question.",
        "pattern": "data_table",
    },
    {
        "reasoning": "The context deals with a different category of players or competition "
        "than the one mentioned in the question.",
        "pattern": "category_mismatch",
    },
]


def _get_templates(corpus: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Get language-appropriate templates for corpus.

    Args:
        corpus: Either 'fr' or 'intl'

    Returns:
        Tuple of (positive_templates, negative_templates)
    """
    if corpus == "intl":
        return FEW_SHOT_POSITIVE_TEMPLATES_EN, FEW_SHOT_NEGATIVE_TEMPLATES_EN
    return FEW_SHOT_POSITIVE_TEMPLATES_FR, FEW_SHOT_NEGATIVE_TEMPLATES_FR


def load_gold_standard(corpus: str) -> dict[str, Any]:
    """Load gold standard questions for a corpus."""
    path = TESTS_DATA_DIR / f"gold_standard_{corpus}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_chunks(corpus: str) -> dict[str, dict[str, Any]]:
    """Load chunks indexed by ID."""
    path = CORPUS_DIR / f"chunks_mode_b_{corpus}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {chunk["id"]: chunk for chunk in data.get("chunks", [])}


def load_triplets() -> list[dict[str, str]]:
    """Load gold triplets."""
    path = DATA_TRAINING_DIR / "gold_triplets_mode_b.jsonl"
    triplets = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    triplets.append(json.loads(line))
    return triplets


def _get_negative_chunk(
    chunks: dict[str, dict[str, Any]], exclude_id: str
) -> dict[str, Any] | None:
    """Get a random chunk different from the excluded one."""
    positive_chunk = chunks.get(exclude_id)
    if not positive_chunk:
        return None

    positive_source = positive_chunk.get("source", "")
    positive_pages = set(positive_chunk.get("pages", []))

    candidates = [
        c
        for cid, c in chunks.items()
        if cid != exclude_id
        and (
            c.get("source") != positive_source
            or not set(c.get("pages", [])).intersection(positive_pages)
        )
    ]

    return random.choice(candidates) if candidates else None


def generate_few_shot_examples(
    corpus: str = "fr",
    n_positive: int = 5,
    n_negative: int = 5,
    output_dir: Path | None = None,
    seed: int = 42,
) -> Path:
    """Generate few-shot examples with reasoning for LLM judge.

    Creates a TSV file with columns:
    - Query: The question
    - Document: The context document
    - Answer: The expected answer (proxy from triplet)
    - Context_Relevance_Label: 1 for positive, 0 for negative
    - Reasoning: Explanation for the label (for judge prompting)

    Args:
        corpus: Either 'fr' or 'intl'
        n_positive: Number of positive examples
        n_negative: Number of negative examples
        output_dir: Output directory
        seed: Random seed

    Returns:
        Path to the created TSV file
    """
    random.seed(seed)

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_standard = load_gold_standard(corpus)
    chunks = load_chunks(corpus)
    triplets = load_triplets() if corpus == "fr" else []

    questions = gold_standard.get("questions", [])

    # Get answerable questions with chunk_id
    answerable = [
        q
        for q in questions
        if q.get("expected_chunk_id")
        and q.get("validation", {}).get("status") == "VALIDATED"
        and q.get("metadata", {}).get("hard_type", "ANSWERABLE") == "ANSWERABLE"
    ]

    # Select diverse positive examples
    selected_positive = _select_diverse_examples(answerable, n_positive)

    # Get language-appropriate templates
    positive_templates, negative_templates = _get_templates(corpus)

    few_shot_samples = []

    # Create positive examples
    for i, q in enumerate(selected_positive):
        chunk_id = q["expected_chunk_id"]
        chunk = chunks.get(chunk_id)
        if not chunk:
            continue

        # Find matching triplet for answer (FR only, INTL uses chunk text)
        triplet = next((t for t in triplets if t.get("anchor") == q["question"]), None)
        answer = triplet["positive"] if triplet else chunk["text"][:500]

        reasoning_template = positive_templates[i % len(positive_templates)]

        few_shot_samples.append(
            {
                "Query": q["question"],
                "Document": chunk["text"],
                "Answer": answer,
                "Context_Relevance_Label": 1,
                "Reasoning": reasoning_template["reasoning"],
            }
        )

    # Create negative examples
    for i in range(n_negative):
        if not answerable:
            break

        q = random.choice(answerable)
        chunk_id = q["expected_chunk_id"]
        neg_chunk = _get_negative_chunk(chunks, chunk_id)

        if not neg_chunk:
            continue

        reasoning_template = negative_templates[i % len(negative_templates)]

        few_shot_samples.append(
            {
                "Query": q["question"],
                "Document": neg_chunk["text"],
                "Answer": "",
                "Context_Relevance_Label": 0,
                "Reasoning": reasoning_template["reasoning"],
            }
        )

    # Shuffle to mix positives and negatives
    random.shuffle(few_shot_samples)

    # Write TSV
    output_path = output_dir / f"few_shot_{corpus}.tsv"

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "Query",
            "Document",
            "Answer",
            "Context_Relevance_Label",
            "Reasoning",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(few_shot_samples)

    print(f"Created {len(few_shot_samples)} few-shot examples at {output_path}")
    print(
        f"  - Positive: {sum(1 for s in few_shot_samples if s['Context_Relevance_Label'] == 1)}"
    )
    print(
        f"  - Negative: {sum(1 for s in few_shot_samples if s['Context_Relevance_Label'] == 0)}"
    )

    return output_path


def _select_diverse_examples(
    questions: list[dict[str, Any]], n: int
) -> list[dict[str, Any]]:
    """Select diverse examples covering different categories.

    Args:
        questions: List of question dicts
        n: Number to select

    Returns:
        Selected questions
    """
    # Group by category
    by_category: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        cat = q.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(q)

    # Select one from each category until we have n
    selected: list[dict[str, Any]] = []
    categories = list(by_category.keys())
    random.shuffle(categories)

    idx = 0
    while len(selected) < n and any(by_category.values()):
        cat = categories[idx % len(categories)]
        if by_category[cat]:
            selected.append(by_category[cat].pop())
        idx += 1
        # Remove empty categories
        categories = [c for c in categories if by_category[c]]

    return selected


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate few-shot examples for ARES LLM judge"
    )
    parser.add_argument(
        "--corpus",
        choices=["fr", "intl"],
        default="fr",
        help="Corpus to use (default: fr)",
    )
    parser.add_argument(
        "--n-positive",
        type=int,
        default=5,
        help="Number of positive examples (default: 5)",
    )
    parser.add_argument(
        "--n-negative",
        type=int,
        default=5,
        help="Number of negative examples (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    generate_few_shot_examples(
        corpus=args.corpus,
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
