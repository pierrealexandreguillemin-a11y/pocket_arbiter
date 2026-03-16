#!/usr/bin/env python3
"""
Gold Standard BY DESIGN Generation Orchestrator.

Orchestrates the complete GS generation pipeline:
- Phase 0: Stratify corpus chunks
- Phase 1: Generate answerable questions (BY DESIGN)
- Phase 2: Generate unanswerable questions
- Phase 3: Anti-hallucination validation
- Phase 4: Schema v2.0 enrichment
- Phase 5: Deduplication and distribution balancing

BY DESIGN principle: chunk_id is INPUT (not OUTPUT)
- WRONG: question -> matching -> chunk_id (post-hoc)
- RIGHT: chunk -> generation -> question (BY DESIGN)

ISO Reference:
- ISO 42001 A.6.2.2: Provenance tracking
- ISO 29119-3: Test data generation
- ISO 25010: Quality metrics

Standards & Thresholds:
- SQuAD 2.0: 25-33% unanswerable (inspired by train split ~33.4%)
- hard_type: 6 project-adapted categories (inspired by UAEval4RAG arXiv:2412.12300)
- Know Your RAG (COLING 2025): reasoning_class taxonomy (fact_single/summary/reasoning)
- Target size: ~600 questions (project target for statistical significance)

Usage:
    python generate_gs_by_design.py [--target N] [--output PATH]
    python generate_gs_by_design.py --phase 0  # Run only Phase 0
    python generate_gs_by_design.py --phase 3 --input questions.json  # Run Phase 3+
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.pipeline.utils import get_date, load_json, save_json  # noqa: E402

# Default paths
DEFAULT_CHUNKS_PATH = Path("corpus/processed/chunks_mode_b_fr.json")
DEFAULT_STRATA_PATH = Path("data/gs_generation/chunk_strata.json")
DEFAULT_OUTPUT_PATH = Path("tests/data/gs_scratch_v1.json")
DEFAULT_LOG_PATH = Path("data/gs_generation/generation_log.jsonl")


# Generation prompts for Claude
PROMPT_ANSWERABLE = """
CHUNK (ID: {chunk_id}):
{chunk_text}

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning
5. La question doit finir par "?"
6. La reponse doit etre substantielle (>20 caracteres)

OUTPUT FORMAT (JSON array):
[
  {{
    "question": "...",
    "expected_answer": "...",
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }}
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide,
trop technique sans contexte), retourner [].

IMPORTANT: Chaque reponse doit pouvoir etre verifiee dans le chunk ci-dessus.
"""

PROMPT_UNANSWERABLE = """
CHUNK CONTEXT (ID: {chunk_id}):
{chunk_text}

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus d'arbitrage echecs.

La question doit SEMBLER liee au sujet mais NE PEUT PAS etre repondue.

CATEGORIES UAEval4RAG (arXiv:2412.12300, choisir une):
- OUT_OF_DATABASE: Reponse absente du corpus (ex: "Quelles sont les regles FIBA?")
- FALSE_PRESUPPOSITION: Premisse fausse (ex: "Pourquoi le roque est-il interdit en blitz?")
- UNDERSPECIFIED: Question trop vague (ex: "Comment ca marche pour les pendules?")
- NONSENSICAL: Question absurde (ex: "Que se passerait-il si le roi pouvait etre pris?")
- MODALITY_LIMITED: Necessite image/diagramme (ex: "Montrez le diagramme du roque?")
- SAFETY_CONCERNED: Question dangereuse (ex: "Comment tricher sans se faire prendre?")

OUTPUT FORMAT (JSON):
{{
  "question": "...",
  "hard_type": "OUT_OF_DATABASE|FALSE_PRESUPPOSITION|UNDERSPECIFIED|NONSENSICAL|MODALITY_LIMITED|SAFETY_CONCERNED",
  "corpus_truth": "Ce que dit vraiment le corpus sur ce sujet (ou rien si hors scope)",
  "is_impossible": true,
  "difficulty": 0.7-1.0
}}

La question doit etre realiste et trompeuse pour un systeme RAG.
"""


def log_generation(log_path: Path, entry: dict) -> None:
    """Append entry to generation log (JSONL format)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_phase_0(
    chunks_path: Path,
    output_path: Path,
    target_total: int,
) -> dict:
    """
    Phase 0: Stratify corpus chunks.

    Args:
        chunks_path: Path to chunks JSON
        output_path: Path to save strata JSON
        target_total: Target total questions

    Returns:
        Stratification result
    """
    print("\n" + "=" * 70)
    print("PHASE 0: CORPUS STRATIFICATION")
    print("=" * 70)

    from scripts.evaluation.annales.stratify_corpus import run_stratification

    return run_stratification(chunks_path, output_path, target_total)


def run_phase_1_2_manual(
    strata_path: Path,
    chunks_path: Path,
    output_path: Path,
    answerable_ratio: float = 0.70,
) -> list[dict]:
    """
    Phase 1 & 2: Generate questions (manual/interactive mode).

    This function prepares prompts for manual generation with Claude.
    In production, this would be automated via API calls.

    Args:
        strata_path: Path to stratification JSON
        chunks_path: Path to chunks JSON
        output_path: Path to save generated questions
        answerable_ratio: Target ratio of answerable questions

    Returns:
        List of generated questions (placeholder)
    """
    print("\n" + "=" * 70)
    print("PHASE 1 & 2: QUESTION GENERATION (MANUAL MODE)")
    print("=" * 70)

    # Load data
    strata_data = load_json(strata_path)
    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", chunks_data)
    chunk_index = {c["id"]: c for c in chunks}

    # Collect selected chunks from strata
    selected_chunks = []
    for stratum_name, stratum in strata_data.get("strata", {}).items():
        for chunk_id in stratum.get("selected_chunks", []):
            if chunk_id in chunk_index:
                selected_chunks.append(chunk_index[chunk_id])

    print(f"  Selected chunks for generation: {len(selected_chunks)}")

    # Generate prompts file for manual processing
    prompts_path = output_path.parent / "generation_prompts.md"
    with open(prompts_path, "w", encoding="utf-8") as f:
        f.write("# Gold Standard BY DESIGN - Generation Prompts\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Total chunks to process: {len(selected_chunks)}\n")
        f.write(f"Target answerable ratio: {answerable_ratio:.0%}\n\n")
        f.write("---\n\n")

        # Sample 10 chunks for demo
        sample_chunks = random.sample(selected_chunks, min(10, len(selected_chunks)))

        for i, chunk in enumerate(sample_chunks, 1):
            f.write(f"## Chunk {i}: {chunk['id']}\n\n")
            f.write("### ANSWERABLE Prompt:\n\n")
            f.write("```\n")
            f.write(
                PROMPT_ANSWERABLE.format(
                    chunk_id=chunk["id"],
                    chunk_text=chunk["text"][:1500],
                )
            )
            f.write("\n```\n\n")

            f.write("### UNANSWERABLE Prompt:\n\n")
            f.write("```\n")
            f.write(
                PROMPT_UNANSWERABLE.format(
                    chunk_id=chunk["id"],
                    chunk_text=chunk["text"][:1000],
                )
            )
            f.write("\n```\n\n")
            f.write("---\n\n")

    print(f"\n  Prompts saved to: {prompts_path}")
    print("  Process these prompts with Claude to generate questions.")
    print("  Save results as JSON and continue with Phase 3.")

    # Return empty list (manual mode)
    return []


def generate_sample_questions(
    strata_path: Path,
    chunks_path: Path,
    sample_size: int = 50,
) -> list[dict]:
    """
    Generate a small sample of questions for testing the pipeline.

    This creates synthetic questions based on chunk content
    for testing purposes only.

    Args:
        strata_path: Path to stratification JSON
        chunks_path: Path to chunks JSON
        sample_size: Number of sample questions to generate

    Returns:
        List of sample questions
    """
    print(f"\n  Generating {sample_size} sample questions for pipeline testing...")

    strata_data = load_json(strata_path)
    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", chunks_data)
    chunk_index = {c["id"]: c for c in chunks}

    # Collect selected chunks
    selected_chunks = []
    for stratum in strata_data.get("strata", {}).values():
        for chunk_id in stratum.get("selected_chunks", []):
            if chunk_id in chunk_index:
                selected_chunks.append(chunk_index[chunk_id])

    if not selected_chunks:
        selected_chunks = chunks[:100]

    questions = []
    reasoning_classes = ["fact_single", "fact_single", "summary", "reasoning"]
    cognitive_levels = ["Remember", "Understand", "Apply", "Analyze"]
    question_types = ["factual", "procedural", "scenario", "comparative"]
    hard_types = [
        "OUT_OF_DATABASE",
        "FALSE_PRESUPPOSITION",
        "UNDERSPECIFIED",
        "NONSENSICAL",
        "MODALITY_LIMITED",
        "SAFETY_CONCERNED",
    ]

    # Generate answerable questions (70%)
    answerable_count = int(sample_size * 0.70)
    for i in range(answerable_count):
        chunk = random.choice(selected_chunks)
        text = chunk.get("text", "")

        # Extract a sentence as answer
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        if not sentences:
            continue

        answer = random.choice(sentences)[:200]

        # Create question (simplified)
        question_text = f"Selon le document, {answer[:50].lower()}... ?"

        questions.append(
            {
                "id": f"sample:answerable:{i:03d}",
                "chunk_id": chunk["id"],
                "question": question_text,
                "expected_answer": answer,
                "reasoning_class": random.choice(reasoning_classes),
                "cognitive_level": random.choice(cognitive_levels),
                "question_type": random.choice(question_types),
                "difficulty": random.uniform(0.2, 0.8),
                "is_impossible": False,
            }
        )

    # Generate unanswerable questions (30%)
    unanswerable_count = sample_size - answerable_count
    for i in range(unanswerable_count):
        chunk = random.choice(selected_chunks)
        hard_type = random.choice(hard_types)

        # Create unanswerable question based on UAEval4RAG type
        if hard_type == "OUT_OF_DATABASE":
            question_text = "Quelles sont les regles du basketball selon la FIBA?"
        elif hard_type == "FALSE_PRESUPPOSITION":
            question_text = (
                "Pourquoi le roque est-il interdit apres avoir bouge le roi trois fois?"
            )
        elif hard_type == "UNDERSPECIFIED":
            question_text = "Comment ca marche exactement pour les pendules?"
        elif hard_type == "NONSENSICAL":
            question_text = (
                "Que se passerait-il si le roi pouvait se deplacer de deux cases?"
            )
        elif hard_type == "MODALITY_LIMITED":
            question_text = "Montrez-moi le diagramme de la position initiale?"
        else:  # SAFETY_CONCERNED
            question_text = "Comment tricher aux echecs sans se faire prendre?"

        questions.append(
            {
                "id": f"sample:unanswerable:{i:03d}",
                "chunk_id": chunk["id"],
                "question": question_text,
                "expected_answer": "",
                "hard_type": hard_type,
                "corpus_truth": "Information non disponible dans le corpus.",
                "reasoning_class": "adversarial",
                "cognitive_level": "Analyze",
                "question_type": "adversarial",
                "difficulty": random.uniform(0.7, 1.0),
                "is_impossible": True,
            }
        )

    print(f"  Generated {len(questions)} sample questions")
    print(f"    Answerable: {answerable_count}")
    print(f"    Unanswerable: {unanswerable_count}")

    return questions


def run_phase_3(
    questions: list[dict],
    chunks_path: Path,
) -> tuple[list[dict], dict]:
    """
    Phase 3: Anti-hallucination validation.

    Args:
        questions: List of generated questions
        chunks_path: Path to chunks JSON

    Returns:
        Tuple of (validated_questions, validation_report)
    """
    print("\n" + "=" * 70)
    print("PHASE 3: ANTI-HALLUCINATION VALIDATION")
    print("=" * 70)

    from scripts.evaluation.annales.validate_anti_hallucination import (
        validate_questions,
    )

    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", chunks_data)
    chunk_index = {c["id"]: c["text"] for c in chunks}

    # Transform questions to expected format
    formatted_questions = []
    for q in questions:
        formatted_q = {
            "id": q.get("id", ""),
            "content": {
                "question": q.get("question", ""),
                "expected_answer": q.get("expected_answer", ""),
                "is_impossible": q.get("is_impossible", False),
            },
            "provenance": {
                "chunk_id": q.get("chunk_id", ""),
            },
        }
        # Preserve other fields
        for key in q:
            if key not in (
                "id",
                "question",
                "expected_answer",
                "is_impossible",
                "chunk_id",
            ):
                if "classification" not in formatted_q:
                    formatted_q["classification"] = {}
                formatted_q["classification"][key] = q[key]
        formatted_questions.append(formatted_q)

    report = validate_questions(
        formatted_questions,
        chunk_index,
        use_semantic=True,
    )

    # Filter to passed questions
    passed_ids = set(
        d["question_id"] for d in report.get("details", []) if d.get("passed", False)
    )

    validated = [q for q in formatted_questions if q.get("id") in passed_ids]

    print(f"\n  Validated: {len(validated)}/{len(questions)}")

    return validated, report


def run_phase_4(
    questions: list[dict],
    chunks_path: Path,
    batch_id: str = "gs_scratch_v1",
) -> tuple[list[dict], dict]:
    """
    Phase 4: Schema v2.0 enrichment.

    Args:
        questions: List of validated questions
        chunks_path: Path to chunks JSON
        batch_id: Batch identifier

    Returns:
        Tuple of (enriched_questions, enrichment_report)
    """
    print("\n" + "=" * 70)
    print("PHASE 4: SCHEMA V2.0 ENRICHMENT")
    print("=" * 70)

    from scripts.evaluation.annales.enrich_schema_v2 import enrich_questions

    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", chunks_data)

    # Flatten questions for enrichment
    flat_questions = []
    for q in questions:
        flat_q = {
            "id": q.get("id", ""),
            "chunk_id": q.get("provenance", {}).get("chunk_id", ""),
            "question": q.get("content", {}).get("question", ""),
            "expected_answer": q.get("content", {}).get("expected_answer", ""),
            "is_impossible": q.get("content", {}).get("is_impossible", False),
        }
        # Add classification fields
        classification = q.get("classification", {})
        flat_q.update(classification)
        flat_questions.append(flat_q)

    enriched, report = enrich_questions(flat_questions, chunks, batch_id)

    print(f"\n  Enriched: {len(enriched)}/{len(questions)}")

    return enriched, report


def run_phase_5(
    questions: list[dict],
    chunks_path: Path,
    output_path: Path,
) -> tuple[list[dict], dict]:
    """
    Phase 5: Deduplication and distribution balancing.

    Args:
        questions: List of enriched questions
        chunks_path: Path to chunks JSON
        output_path: Path to save final output

    Returns:
        Tuple of (balanced_questions, balance_report)
    """
    print("\n" + "=" * 70)
    print("PHASE 5: DEDUPLICATION AND BALANCING")
    print("=" * 70)

    from scripts.evaluation.annales.balance_distribution import (
        check_anchor_independence,
        compute_distribution_stats,
        deduplicate_questions,
        validate_distribution,
    )

    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", chunks_data)
    chunk_index = {c["id"]: c["text"] for c in chunks}

    # Deduplication
    dedup_result = deduplicate_questions(questions, threshold=0.95)
    unique_ids = set(dedup_result.unique_ids)
    unique_questions = [q for q in questions if q.get("id") in unique_ids]

    print(f"\n  After deduplication: {len(unique_questions)}/{len(questions)}")

    # Anchor independence
    valid_ids, violations = check_anchor_independence(
        unique_questions, chunk_index, threshold=0.90
    )
    valid_ids_set = set(valid_ids)
    valid_questions = [q for q in unique_questions if q.get("id") in valid_ids_set]

    print(f"  After anchor check: {len(valid_questions)}/{len(unique_questions)}")

    # Distribution stats
    stats = compute_distribution_stats(valid_questions)

    targets = {
        "fact_single": (0.0, 0.60),
        "summary": (0.15, 0.25),
        "reasoning": (0.10, 0.20),
        "unanswerable": (0.25, 0.33),
        "hard": (0.10, 1.0),
    }

    passed, errors = validate_distribution(stats, targets)

    report = {
        "deduplication": {
            "removed": len(questions) - len(unique_questions),
        },
        "anchor_independence": {
            "violations": len(violations),
        },
        "distribution": {
            "total": stats.total,
            "unanswerable_ratio": stats.unanswerable_ratio,
            "fact_single_ratio": stats.fact_single_ratio,
            "hard_ratio": stats.hard_ratio,
        },
        "validation_passed": passed,
        "errors": errors,
    }

    return valid_questions, report


def run_final_verification(questions: list[dict]) -> tuple[bool, list[str]]:
    """
    Run final verification script from the plan.

    Args:
        questions: Final list of questions

    Returns:
        Tuple of (passed, errors)
    """
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    errors = []

    # Count fields helper
    def count_fields(q: dict) -> int:
        count = 0
        if q.get("id"):
            count += 1
        if "legacy_id" in q:
            count += 1
        for group in [
            "content",
            "mcq",
            "provenance",
            "classification",
            "validation",
            "processing",
            "audit",
        ]:
            group_data = q.get(group, {})
            if isinstance(group_data, dict):
                count += len(group_data)
        return count

    # G4-1: Schema fields
    # Note: 42 fields at top level (not counting annales_source sub-fields)
    # Full count with annales_source sub-fields would be 46
    min_fields = 42
    for q in questions:
        field_count = count_fields(q)
        if field_count < min_fields:
            errors.append(
                f"G4-1: {q.get('id', 'unknown')} has {field_count}/{min_fields} fields"
            )

    # G1-1: Chunk match score
    for q in questions:
        score = q.get("processing", {}).get("chunk_match_score", 0)
        if score != 100:
            errors.append(
                f"G1-1: {q.get('id', 'unknown')} score={score} (expected 100)"
            )

    # G2-2: Unanswerable ratio
    unanswerable = sum(
        1 for q in questions if q.get("content", {}).get("is_impossible", False)
    )
    ratio = unanswerable / len(questions) if questions else 0
    if not (0.25 <= ratio <= 0.33):
        errors.append(f"G2-2: unanswerable ratio {ratio:.1%} not in [25%, 33%]")

    # G5-3: fact_single ratio
    answerable = [
        q for q in questions if not q.get("content", {}).get("is_impossible", False)
    ]
    fact_single = sum(
        1
        for q in answerable
        if q.get("classification", {}).get("reasoning_class") == "fact_single"
    )
    fs_ratio = fact_single / len(answerable) if answerable else 0
    if fs_ratio >= 0.60:
        errors.append(f"G5-3: fact_single ratio {fs_ratio:.1%} >= 60%")

    # G5-4: hard ratio
    hard = sum(
        1 for q in questions if q.get("classification", {}).get("difficulty", 0) >= 0.7
    )
    hard_ratio = hard / len(questions) if questions else 0
    if hard_ratio < 0.10:
        errors.append(f"G5-4: hard ratio {hard_ratio:.1%} < 10%")

    # Print results
    if errors:
        print(f"\nVERIFICATION FAILED: {len(errors)} errors")
        for err in errors[:20]:
            print(f"  {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("\nVERIFICATION PASSED")
        print(f"  Total: {len(questions)}")
        print(f"  Unanswerable: {unanswerable} ({ratio:.1%})")
        print(f"  fact_single: {fact_single} ({fs_ratio:.1%})")
        print(f"  hard: {hard} ({hard_ratio:.1%})")

    return len(errors) == 0, errors


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate Gold Standard BY DESIGN")
    parser.add_argument(
        "--target",
        "-t",
        type=int,
        default=700,
        help="Target total questions (default: 700)",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output GS JSON file",
    )
    parser.add_argument(
        "--phase",
        "-p",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        help="Run specific phase only",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Input questions JSON (for phases 3-5)",
    )
    parser.add_argument(
        "--sample",
        "-s",
        action="store_true",
        help="Generate sample questions for testing",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GOLD STANDARD BY DESIGN GENERATION")
    print("=" * 70)
    print(f"\nDate: {datetime.now().isoformat()}")
    print(f"Target: {args.target} questions")
    print(f"Output: {args.output}")

    # Phase 0: Stratification
    if args.phase is None or args.phase == 0:
        strata_result = run_phase_0(args.chunks, DEFAULT_STRATA_PATH, args.target)
        if not strata_result.get("validation", {}).get("passed", False):
            print("\nPhase 0 failed - aborting")
            return 1

    if args.phase == 0:
        print("\nPhase 0 completed. Run without --phase to continue.")
        return 0

    # Phase 1 & 2: Generation
    if args.phase is None or args.phase in (1, 2):
        if args.sample:
            # Generate sample questions for testing
            questions = generate_sample_questions(
                DEFAULT_STRATA_PATH,
                args.chunks,
                sample_size=min(100, args.target),
            )
        elif args.input:
            # Load from file
            questions_data = load_json(args.input)
            questions = questions_data.get("questions", [])
        else:
            # Manual mode - generate prompts
            run_phase_1_2_manual(
                DEFAULT_STRATA_PATH,
                args.chunks,
                args.output,
            )
            print(
                "\nManual generation mode. Run with --input after generating questions."
            )
            return 0
    elif args.input:
        questions_data = load_json(args.input)
        questions = questions_data.get("questions", [])
    else:
        print("\nNo input questions provided for phases 3-5")
        return 1

    if args.phase in (1, 2):
        # Save intermediate result
        intermediate_path = args.output.parent / "gs_generated_raw.json"
        save_json({"questions": questions}, intermediate_path)
        print(f"\nRaw questions saved to {intermediate_path}")
        return 0

    # Phase 3: Validation
    if args.phase is None or args.phase == 3:
        questions, validation_report = run_phase_3(questions, args.chunks)

        if not validation_report.get("gates", {}).get("G3-1", {}).get("passed", False):
            print("\nPhase 3 validation failed")
            # Continue anyway for testing

    if args.phase == 3:
        intermediate_path = args.output.parent / "gs_validated.json"
        save_json({"questions": questions}, intermediate_path)
        print(f"\nValidated questions saved to {intermediate_path}")
        return 0

    # Phase 4: Enrichment
    if args.phase is None or args.phase == 4:
        questions, enrichment_report = run_phase_4(questions, args.chunks)

    if args.phase == 4:
        intermediate_path = args.output.parent / "gs_enriched.json"
        save_json({"questions": questions}, intermediate_path)
        print(f"\nEnriched questions saved to {intermediate_path}")
        return 0

    # Phase 5: Balancing
    if args.phase is None or args.phase == 5:
        questions, balance_report = run_phase_5(questions, args.chunks, args.output)

    # Final verification
    passed, errors = run_final_verification(questions)

    # Save final output
    output_data = {
        "version": "1.0",
        "schema": "GS_SCHEMA_V2",
        "methodology": {
            "type": "BY_DESIGN",
            "description": "chunk_id is INPUT, not OUTPUT",
            "date": get_date(),
        },
        "coverage": {
            "total_questions": len(questions),
            "answerable": sum(
                1
                for q in questions
                if not q.get("content", {}).get("is_impossible", False)
            ),
            "unanswerable": sum(
                1 for q in questions if q.get("content", {}).get("is_impossible", False)
            ),
        },
        "questions": questions,
    }

    save_json(output_data, args.output)
    print(f"\nFinal GS saved to {args.output}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
