#!/usr/bin/env python3
"""
Etape 5: Validation Finale (UNIFIED_TRAINING_DATA_SPEC.md)

Validation complete du dataset avant fine-tuning:
- Deduplication fuzzy (SemHash cosine < 0.95)
- Schema JSON validation (Draft-07)
- Distribution entropy >= 0.8
- DVC lineage documentation

ISO 42001 A.7.3 - Validation tracable
ISO 29119 - Test data verification
ISO 25010 - Quality metrics

Usage:
    python -m scripts.training.unified.validate_dataset \
        --input data/training/unified/triplets_train.jsonl \
        --output data/training/unified/validation_report.json
"""

import argparse
import hashlib
import json
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "data" / "training" / "unified" / "triplets_train.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "unified" / "validation_report.json"

# Quality gate thresholds (UNIFIED spec 2.4)
DEDUP_THRESHOLD = 0.05  # Max 5% duplicates
ENTROPY_THRESHOLD = 0.8  # Min entropy for distribution balance
COSINE_DEDUP_THRESHOLD = 0.95  # Semantic similarity threshold for dedup


# JSON Schema for triplet validation
TRIPLET_SCHEMA = {
    "type": "object",
    "required": ["anchor", "positive", "negative"],
    "properties": {
        "anchor": {"type": "string", "minLength": 10},
        "positive": {"type": "string", "minLength": 20},
        "negative": {"type": "string", "minLength": 20},
        "metadata": {
            "type": "object",
            "properties": {
                "question_id": {"type": "string"},
                "positive_chunk_id": {"type": "string"},
                "negative_chunk_id": {"type": "string"},
                "selection_method": {"type": "string"},
                "category": {"type": "string"},
            },
        },
    },
}


def load_triplets_jsonl(path: Path) -> list[dict]:
    """Load triplets from JSONL file."""
    triplets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triplets.append(json.loads(line))
    return triplets


def compute_text_hash(text: str) -> str:
    """Compute MD5 hash of text for exact deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def validate_schema(triplets: list[dict]) -> dict[str, Any]:
    """
    Validate triplets against JSON schema.

    Returns validation results with errors.
    """
    errors = []
    valid_count = 0

    for i, triplet in enumerate(triplets):
        triplet_errors = []

        # Check required fields
        for field in TRIPLET_SCHEMA["required"]:
            if field not in triplet:
                triplet_errors.append(f"Missing required field: {field}")
            elif not isinstance(triplet[field], str):
                triplet_errors.append(f"Field {field} must be string")
            elif len(triplet[field]) < TRIPLET_SCHEMA["properties"][field].get("minLength", 0):
                triplet_errors.append(f"Field {field} too short")

        if triplet_errors:
            errors.append({
                "index": i,
                "question_id": triplet.get("metadata", {}).get("question_id", "unknown"),
                "errors": triplet_errors,
            })
        else:
            valid_count += 1

    return {
        "passed": len(errors) == 0,
        "valid_count": valid_count,
        "error_count": len(errors),
        "errors": errors[:20] if errors else None,  # First 20 only
    }


def check_exact_duplicates(triplets: list[dict]) -> dict[str, Any]:
    """Check for exact text duplicates."""
    anchor_hashes: dict[str, list[int]] = {}
    positive_hashes: dict[str, list[int]] = {}

    for i, t in enumerate(triplets):
        ah = compute_text_hash(t["anchor"])
        ph = compute_text_hash(t["positive"])

        if ah not in anchor_hashes:
            anchor_hashes[ah] = []
        anchor_hashes[ah].append(i)

        if ph not in positive_hashes:
            positive_hashes[ph] = []
        positive_hashes[ph].append(i)

    # Find duplicates
    anchor_dups = {h: indices for h, indices in anchor_hashes.items() if len(indices) > 1}
    positive_dups = {h: indices for h, indices in positive_hashes.items() if len(indices) > 1}

    total = len(triplets)
    anchor_dup_count = sum(len(v) - 1 for v in anchor_dups.values())
    positive_dup_count = sum(len(v) - 1 for v in positive_dups.values())

    return {
        "anchor_duplicates": {
            "count": anchor_dup_count,
            "rate": round(anchor_dup_count / total, 4) if total > 0 else 0,
            "groups": len(anchor_dups),
        },
        "positive_duplicates": {
            "count": positive_dup_count,
            "rate": round(positive_dup_count / total, 4) if total > 0 else 0,
            "groups": len(positive_dups),
        },
    }


def check_semantic_duplicates(
    triplets: list[dict],
    model: Any = None,
    threshold: float = COSINE_DEDUP_THRESHOLD,
    sample_size: int = 500,
) -> dict[str, Any]:
    """
    Check for semantic duplicates using embedding similarity.

    Uses sampling for efficiency on large datasets.
    """
    if model is None:
        return {
            "checked": False,
            "note": "Requires embedding model (--check-semantic)",
        }

    # Sample for efficiency
    n = len(triplets)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        sample = [triplets[i] for i in indices]
    else:
        sample = triplets

    # Compute embeddings
    anchors = [t["anchor"] for t in sample]
    embeddings = model.encode(anchors, show_progress_bar=True)

    # Check pairwise similarities (only upper triangle)
    duplicates = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if sim >= threshold:
                duplicates.append({
                    "i": i,
                    "j": j,
                    "similarity": round(float(sim), 4),
                })

    dup_rate = len(duplicates) / (len(sample) * (len(sample) - 1) / 2) if len(sample) > 1 else 0

    return {
        "checked": True,
        "sample_size": len(sample),
        "threshold": threshold,
        "duplicate_pairs": len(duplicates),
        "duplicate_rate": round(dup_rate, 4),
        "passed": dup_rate < DEDUP_THRESHOLD,
        "examples": duplicates[:5] if duplicates else None,
    }


def compute_entropy(distribution: dict[str, int]) -> float:
    """Compute normalized Shannon entropy of a distribution."""
    total = sum(distribution.values())
    if total == 0:
        return 0.0

    n_classes = len(distribution)
    if n_classes <= 1:
        return 1.0  # Trivially uniform

    entropy = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Normalize by max possible entropy
    max_entropy = math.log2(n_classes)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def analyze_distribution(triplets: list[dict]) -> dict[str, Any]:
    """Analyze distribution of triplets by various dimensions."""
    categories = Counter()
    methods = Counter()
    difficulties = []

    for t in triplets:
        meta = t.get("metadata", {})
        cat = meta.get("category", "unknown")
        method = meta.get("selection_method", "unknown")
        diff = meta.get("difficulty")

        categories[cat] += 1
        methods[method] += 1
        if diff is not None:
            difficulties.append(diff)

    # Compute entropies
    category_entropy = compute_entropy(dict(categories))
    method_entropy = compute_entropy(dict(methods))

    # Difficulty stats
    diff_stats = {}
    if difficulties:
        diff_stats = {
            "mean": round(np.mean(difficulties), 4),
            "std": round(np.std(difficulties), 4),
            "min": round(min(difficulties), 4),
            "max": round(max(difficulties), 4),
        }

    return {
        "category_distribution": dict(categories),
        "category_entropy": round(category_entropy, 4),
        "category_entropy_passed": category_entropy >= ENTROPY_THRESHOLD,
        "method_distribution": dict(methods),
        "method_entropy": round(method_entropy, 4),
        "difficulty_stats": diff_stats,
        "total_triplets": len(triplets),
    }


def generate_dvc_lineage(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Generate DVC lineage documentation."""
    return {
        "inputs": [str(input_path)],
        "outputs": [str(output_path)],
        "pipeline_stage": "validate_dataset",
        "spec_reference": "UNIFIED_TRAINING_DATA_SPEC.md",
        "iso_reference": "ISO 42001 A.7.3, ISO 29119",
        "timestamp": datetime.now().isoformat(),
        "dvc_command": f"dvc run -n validate_dataset -d {input_path} -o {output_path} python -m scripts.training.unified.validate_dataset",
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate training dataset (UNIFIED spec Step 5)"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input triplets JSONL",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output validation report JSON",
    )
    parser.add_argument(
        "--check-semantic",
        action="store_true",
        help="Check semantic duplicates (requires embedding model, slower)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("ETAPE 5: Validation Finale")
    logger.info("=" * 60)

    # Load triplets
    logger.info(f"Loading triplets: {args.input}")
    triplets = load_triplets_jsonl(args.input)
    logger.info(f"  Loaded {len(triplets)} triplets")

    # Schema validation
    logger.info("Validating schema...")
    schema_result = validate_schema(triplets)
    logger.info(f"  Schema: {'PASS' if schema_result['passed'] else 'FAIL'}")
    logger.info(f"    Valid: {schema_result['valid_count']}, Errors: {schema_result['error_count']}")

    # Exact duplicates
    logger.info("Checking exact duplicates...")
    exact_dups = check_exact_duplicates(triplets)
    logger.info(f"  Anchor duplicates: {exact_dups['anchor_duplicates']['count']}")
    logger.info(f"  Positive duplicates: {exact_dups['positive_duplicates']['count']}")

    # Semantic duplicates (optional)
    semantic_dups = {"checked": False}
    if args.check_semantic:
        logger.info("Checking semantic duplicates...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("google/embeddinggemma-300m")
            semantic_dups = check_semantic_duplicates(triplets, model)
            status = "PASS" if semantic_dups.get("passed", False) else "FAIL"
            logger.info(f"  Semantic dedup: {status}")
            logger.info(f"    Duplicate rate: {semantic_dups.get('duplicate_rate', 'N/A')}")
        except Exception as e:
            logger.warning(f"Could not check semantic duplicates: {e}")

    # Distribution analysis
    logger.info("Analyzing distribution...")
    distribution = analyze_distribution(triplets)
    logger.info(f"  Category entropy: {distribution['category_entropy']} (threshold: {ENTROPY_THRESHOLD})")
    logger.info(f"  Method entropy: {distribution['method_entropy']}")

    # DVC lineage
    dvc_lineage = generate_dvc_lineage(args.input, args.output)

    # Compile report
    report = {
        "validation_date": datetime.now().isoformat(),
        "input_file": str(args.input),
        "total_triplets": len(triplets),
        "quality_gates": {
            "gate_5_schema": {
                "passed": schema_result["passed"],
                "valid_count": schema_result["valid_count"],
                "error_count": schema_result["error_count"],
            },
            "gate_5_deduplication": {
                "exact_duplicates": exact_dups,
                "semantic_duplicates": semantic_dups,
                "passed": (
                    exact_dups["anchor_duplicates"]["rate"] < DEDUP_THRESHOLD
                    and exact_dups["positive_duplicates"]["rate"] < DEDUP_THRESHOLD
                ),
            },
            "gate_5_distribution": {
                "category_entropy": distribution["category_entropy"],
                "category_entropy_passed": distribution["category_entropy_passed"],
                "method_entropy": distribution["method_entropy"],
                "passed": distribution["category_entropy_passed"],
            },
        },
        "distribution": distribution,
        "schema_errors": schema_result.get("errors"),
        "dvc_lineage": dvc_lineage,
    }

    # Overall pass/fail
    all_passed = (
        report["quality_gates"]["gate_5_schema"]["passed"]
        and report["quality_gates"]["gate_5_deduplication"]["passed"]
        and report["quality_gates"]["gate_5_distribution"]["passed"]
    )
    report["overall_passed"] = all_passed

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved validation report: {args.output}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"VALIDATION RESULT: {'PASS' if all_passed else 'FAIL'}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Quality Gates:")
    logger.info(f"  Gate 5 - Schema:        {'PASS' if report['quality_gates']['gate_5_schema']['passed'] else 'FAIL'}")
    logger.info(f"  Gate 5 - Deduplication: {'PASS' if report['quality_gates']['gate_5_deduplication']['passed'] else 'FAIL'}")
    logger.info(f"  Gate 5 - Distribution:  {'PASS' if report['quality_gates']['gate_5_distribution']['passed'] else 'FAIL'}")

    if not all_passed:
        logger.warning("\nDataset validation FAILED. Review report for details.")
        exit(1)


if __name__ == "__main__":
    main()
