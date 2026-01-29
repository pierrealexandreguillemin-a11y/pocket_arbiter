#!/usr/bin/env python3
"""
Etape 4: Export Multi-Format (UNIFIED_TRAINING_DATA_SPEC.md)

Exporte les triplets vers plusieurs formats standards:
- TRIPLETS (SentenceTransformers): triplets_train.jsonl, triplets_val.jsonl
- ARES: ares_gold_label.tsv (context relevance)
- BEIR: queries.jsonl, corpus.jsonl, qrels.tsv
- RAGAS: ragas_evaluation.jsonl

ISO 42001 A.6.2.2 - Provenance tracable
ISO 29119 - Test data documentation

Usage:
    python -m scripts.training.unified.export_formats \
        --input data/training/unified/triplets_raw.jsonl \
        --output-dir data/training/unified/ \
        --formats triplets,ares,beir,ragas \
        --train-ratio 0.8 \
        --seed 42
"""

import argparse
import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "data" / "training" / "unified" / "triplets_raw.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "training" / "unified"

ALL_FORMATS = ["triplets", "ares", "beir", "ragas"]


def load_triplets_jsonl(path: Path) -> list[dict]:
    """Load triplets from JSONL file."""
    triplets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triplets.append(json.loads(line))
    return triplets


def split_train_val(
    triplets: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split triplets into train and validation sets."""
    random.seed(seed)
    shuffled = triplets.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def export_triplets_format(
    train: list[dict],
    val: list[dict],
    output_dir: Path,
) -> dict:
    """Export to SentenceTransformers triplet format (JSONL)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export train
    train_path = output_dir / "triplets_train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for t in train:
            # SentenceTransformers format: anchor, positive, negative
            out = {
                "anchor": t["anchor"],
                "positive": t["positive"],
                "negative": t["negative"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Export val
    val_path = output_dir / "triplets_val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for t in val:
            out = {
                "anchor": t["anchor"],
                "positive": t["positive"],
                "negative": t["negative"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_count": len(train),
        "val_count": len(val),
    }


def export_ares_format(
    triplets: list[dict],
    output_dir: Path,
) -> dict:
    """Export to ARES format (TSV with Context_Relevance_Label)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gold labeled (positives = 1, negatives = 0)
    gold_path = output_dir / "ares_gold_label.tsv"
    with open(gold_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Query", "Document", "Answer", "Context_Relevance_Label"])

        for t in triplets:
            metadata = t.get("metadata", {})
            answer = metadata.get("original_question", "")

            # Positive example
            writer.writerow([t["anchor"], t["positive"], answer, 1])
            # Negative example
            writer.writerow([t["anchor"], t["negative"], "", 0])

    # Unlabeled version (for ARES PPI)
    unlabeled_path = output_dir / "ares_unlabeled.tsv"
    with open(unlabeled_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Query", "Document", "Answer"])

        for t in triplets:
            metadata = t.get("metadata", {})
            answer = metadata.get("original_question", "")
            writer.writerow([t["anchor"], t["positive"], answer])

    return {
        "gold_path": str(gold_path),
        "unlabeled_path": str(unlabeled_path),
        "positive_count": len(triplets),
        "negative_count": len(triplets),
        "total_rows": len(triplets) * 2,
    }


def export_beir_format(
    triplets: list[dict],
    output_dir: Path,
) -> dict:
    """Export to BEIR format (queries.jsonl, corpus.jsonl, qrels.tsv)."""
    beir_dir = output_dir / "beir"
    beir_dir.mkdir(parents=True, exist_ok=True)

    # Build unique queries and corpus
    queries: dict[str, str] = {}  # query_id -> query_text
    corpus: dict[str, dict] = {}  # doc_id -> {title, text}
    qrels: list[tuple] = []  # (query_id, doc_id, score)

    for i, t in enumerate(triplets):
        metadata = t.get("metadata", {})

        # Query
        query_id = metadata.get("question_id", f"q{i}")
        queries[query_id] = t["anchor"]

        # Positive document
        pos_doc_id = metadata.get("positive_chunk_id", f"pos_{i}")
        if pos_doc_id not in corpus:
            corpus[pos_doc_id] = {
                "title": "",
                "text": t["positive"],
            }
        qrels.append((query_id, pos_doc_id, 1))

        # Negative document (score 0)
        neg_doc_id = metadata.get("negative_chunk_id", f"neg_{i}")
        if neg_doc_id not in corpus:
            corpus[neg_doc_id] = {
                "title": "",
                "text": t["negative"],
            }
        # Note: BEIR qrels typically only include positive relevance
        # but we include negatives with score 0 for completeness

    # Export queries.jsonl
    queries_path = beir_dir / "queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")

    # Export corpus.jsonl
    corpus_path = beir_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc_id, doc in corpus.items():
            f.write(
                json.dumps(
                    {
                        "_id": doc_id,
                        "title": doc["title"],
                        "text": doc["text"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Export qrels.tsv (BEIR format: query-id, corpus-id, score)
    qrels_path = beir_dir / "qrels.tsv"
    with open(qrels_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for query_id, doc_id, score in qrels:
            writer.writerow([query_id, doc_id, score])

    return {
        "queries_path": str(queries_path),
        "corpus_path": str(corpus_path),
        "qrels_path": str(qrels_path),
        "queries_count": len(queries),
        "corpus_count": len(corpus),
        "qrels_count": len(qrels),
    }


def export_ragas_format(
    triplets: list[dict],
    output_dir: Path,
) -> dict:
    """Export to RAGAS format (jsonl for evaluation)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ragas_path = output_dir / "ragas_evaluation.jsonl"
    with open(ragas_path, "w", encoding="utf-8") as f:
        for t in triplets:
            metadata = t.get("metadata", {})

            # RAGAS format: question, answer (to generate), contexts, ground_truth
            out = {
                "question": t["anchor"],
                "answer": "",  # To be generated by RAG system
                "contexts": [t["positive"]],  # Ground truth context
                "ground_truth": metadata.get("original_question", ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return {
        "ragas_path": str(ragas_path),
        "count": len(triplets),
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export triplets to multiple evaluation formats"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input triplets JSONL",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--formats",
        "-f",
        type=str,
        default="triplets,ares,beir,ragas",
        help=f"Comma-separated formats to export: {','.join(ALL_FORMATS)}",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    formats = [f.strip().lower() for f in args.formats.split(",")]
    for fmt in formats:
        if fmt not in ALL_FORMATS:
            raise ValueError(f"Unknown format: {fmt}. Available: {ALL_FORMATS}")

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("ETAPE 4: Export Multi-Format")
    logger.info("=" * 60)

    # Load triplets
    logger.info(f"Loading triplets: {args.input}")
    triplets = load_triplets_jsonl(args.input)
    logger.info(f"  Loaded {len(triplets)} triplets")

    # Split for training
    train, val = split_train_val(triplets, args.train_ratio, args.seed)
    logger.info(f"  Split: {len(train)} train, {len(val)} val")

    # Export each format
    report: dict[str, Any] = {
        "input": str(args.input),
        "total_triplets": len(triplets),
        "train_count": len(train),
        "val_count": len(val),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "formats": {},
    }

    for fmt in formats:
        logger.info(f"\nExporting {fmt.upper()}...")

        if fmt == "triplets":
            result = export_triplets_format(train, val, args.output_dir)
        elif fmt == "ares":
            result = export_ares_format(triplets, args.output_dir)
        elif fmt == "beir":
            result = export_beir_format(triplets, args.output_dir)
        elif fmt == "ragas":
            result = export_ragas_format(triplets, args.output_dir)
        else:
            continue

        report["formats"][fmt] = result
        logger.info(f"  Done: {result}")

    # Save composition report
    report["timestamp"] = datetime.now().isoformat()
    composition_path = args.output_dir / "dataset_composition.json"
    with open(composition_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved composition report: {composition_path}")

    # Print summary
    logger.info("")
    logger.info("SUMMARY:")
    logger.info(f"  Total triplets: {len(triplets)}")
    logger.info(f"  Train: {len(train)} ({args.train_ratio * 100:.0f}%)")
    logger.info(f"  Val: {len(val)} ({(1 - args.train_ratio) * 100:.0f}%)")
    logger.info(f"  Formats exported: {', '.join(formats)}")
    logger.info(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
