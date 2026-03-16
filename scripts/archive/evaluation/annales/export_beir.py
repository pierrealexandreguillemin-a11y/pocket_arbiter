"""
Export BEIR Format - Pocket Arbiter

Exporte le Gold Standard et corpus au format BEIR pour benchmarking standard.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Interoperabilite donnees
    - ISO/IEC 25010 CM-01 - Compatibilite

Format BEIR:
    beir_export/
        corpus.jsonl      # {"_id": str, "title": str, "text": str}
        queries.jsonl     # {"_id": str, "text": str}
        qrels/
            test.tsv      # query_id \\t corpus_id \\t score

Usage:
    python -m scripts.evaluation.annales.export_beir \
        --gs tests/data/gold_standard_annales_fr_v7.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output tests/data/beir_annales_fr/
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from scripts.pipeline.utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def export_corpus_jsonl(chunks: list[dict], output_path: Path) -> int:
    """
    Export chunks to BEIR corpus.jsonl format.

    Format: {"_id": str, "title": str, "text": str}
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            doc = {
                "_id": chunk["id"],
                "title": chunk.get("section", "") or chunk.get("source", ""),
                "text": chunk["text"],
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return len(chunks)


def export_queries_jsonl(questions: list[dict], output_path: Path) -> int:
    """
    Export questions to BEIR queries.jsonl format.

    Format: {"_id": str, "text": str}
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            # Skip requires_context questions
            if q.get("metadata", {}).get("requires_context", False):
                continue
            # Skip questions without chunk_id
            if not q.get("expected_chunk_id"):
                continue

            query = {
                "_id": q["id"],
                "text": q["question"],
            }
            f.write(json.dumps(query, ensure_ascii=False) + "\n")
            count += 1

    return count


def export_qrels_tsv(
    questions: list[dict],
    output_path: Path,
    relevance_score: int = 1,
) -> int:
    """
    Export relevance judgments to BEIR qrels/test.tsv format.

    Format: query_id \\t corpus_id \\t score
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        # Header (optional in BEIR but good practice)
        f.write("query-id\tcorpus-id\tscore\n")

        for q in questions:
            # Skip requires_context questions
            if q.get("metadata", {}).get("requires_context", False):
                continue
            # Skip questions without chunk_id
            if not q.get("expected_chunk_id"):
                continue

            q_id = q["id"]
            chunk_id = q["expected_chunk_id"]
            f.write(f"{q_id}\t{chunk_id}\t{relevance_score}\n")
            count += 1

    return count


def export_beir(
    gs_path: Path,
    chunks_path: Path,
    output_dir: Path,
) -> dict:
    """
    Export gold standard and corpus to BEIR format.

    Args:
        gs_path: Path to gold standard JSON.
        chunks_path: Path to chunks JSON.
        output_dir: Output directory for BEIR files.

    Returns:
        Export report with statistics.
    """
    logger.info(f"Loading gold standard: {gs_path}")
    gs_data = load_json(gs_path)
    questions = gs_data.get("questions", [])

    logger.info(f"Loading chunks: {chunks_path}")
    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", [])

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir = output_dir / "qrels"
    qrels_dir.mkdir(exist_ok=True)

    # Export corpus
    corpus_path = output_dir / "corpus.jsonl"
    corpus_count = export_corpus_jsonl(chunks, corpus_path)
    logger.info(f"Exported corpus: {corpus_count} documents")

    # Export queries
    queries_path = output_dir / "queries.jsonl"
    queries_count = export_queries_jsonl(questions, queries_path)
    logger.info(f"Exported queries: {queries_count} queries")

    # Export qrels
    qrels_path = qrels_dir / "test.tsv"
    qrels_count = export_qrels_tsv(questions, qrels_path)
    logger.info(f"Exported qrels: {qrels_count} relevance judgments")

    # Create dataset info
    dataset_info = {
        "name": "chess-arbiters-fr",
        "description": "French chess arbitration rules Q&A dataset from FFE annales",
        "source": "Pocket Arbiter Gold Standard",
        "version": gs_data.get("version", "unknown"),
        "language": "fr",
        "split": "test",
        "statistics": {
            "corpus_size": corpus_count,
            "queries": queries_count,
            "qrels": qrels_count,
        },
        "created": datetime.now().isoformat(),
    }
    save_json(dataset_info, output_dir / "dataset_info.json")

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "gold_standard_file": str(gs_path),
        "gold_standard_version": gs_data.get("version", "unknown"),
        "chunks_file": str(chunks_path),
        "output_directory": str(output_dir),
        "files_created": {
            "corpus.jsonl": str(corpus_path),
            "queries.jsonl": str(queries_path),
            "qrels/test.tsv": str(qrels_path),
            "dataset_info.json": str(output_dir / "dataset_info.json"),
        },
        "statistics": {
            "corpus_documents": corpus_count,
            "queries": queries_count,
            "relevance_judgments": qrels_count,
            "skipped_requires_context": sum(
                1
                for q in questions
                if q.get("metadata", {}).get("requires_context", False)
            ),
        },
        "beir_compatible": True,
    }

    return report


def validate_beir_export(output_dir: Path) -> dict:
    """
    Validate BEIR export structure and consistency.
    """
    issues = []

    # Check required files
    required_files = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]
    for f in required_files:
        if not (output_dir / f).exists():
            issues.append(f"Missing required file: {f}")

    # Load and validate
    corpus_ids = set()
    query_ids = set()

    # Check corpus
    corpus_path = output_dir / "corpus.jsonl"
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as corpus_file:
            for line in corpus_file:
                doc = json.loads(line)
                if "_id" not in doc or "text" not in doc:
                    issues.append(
                        f"Corpus document missing required fields: {doc.get('_id', 'unknown')}"
                    )
                corpus_ids.add(doc["_id"])

    # Check queries
    queries_path = output_dir / "queries.jsonl"
    if queries_path.exists():
        with open(queries_path, encoding="utf-8") as queries_file:
            for line in queries_file:
                query = json.loads(line)
                if "_id" not in query or "text" not in query:
                    issues.append(
                        f"Query missing required fields: {query.get('_id', 'unknown')}"
                    )
                query_ids.add(query["_id"])

    # Check qrels
    qrels_path = output_dir / "qrels" / "test.tsv"
    if qrels_path.exists():
        with open(qrels_path, encoding="utf-8") as qrels_file:
            for i, line in enumerate(qrels_file):
                if i == 0 and "query-id" in line:
                    continue  # Skip header
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    issues.append(f"Invalid qrels line {i}: {line[:50]}")
                    continue
                q_id, c_id, score = parts
                if q_id not in query_ids:
                    issues.append(f"Qrels references unknown query: {q_id}")
                if c_id not in corpus_ids:
                    issues.append(f"Qrels references unknown corpus doc: {c_id}")

    return {
        "valid": len(issues) == 0,
        "corpus_size": len(corpus_ids),
        "queries_size": len(query_ids),
        "issues": issues[:20],  # Limit reported issues
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export gold standard to BEIR benchmark format"
    )

    parser.add_argument(
        "--gs",
        "-g",
        type=Path,
        required=True,
        help="Gold standard JSON file",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        required=True,
        help="Chunks JSON file from corpus",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for BEIR files",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate export after creation",
    )

    args = parser.parse_args()

    report = export_beir(args.gs, args.chunks, args.output)

    # Print summary
    stats = report["statistics"]
    logger.info("=" * 60)
    logger.info("BEIR EXPORT REPORT")
    logger.info("=" * 60)
    logger.info(f"Output directory: {report['output_directory']}")
    logger.info(f"Corpus documents: {stats['corpus_documents']}")
    logger.info(f"Queries: {stats['queries']}")
    logger.info(f"Relevance judgments: {stats['relevance_judgments']}")
    logger.info(f"Skipped (requires_context): {stats['skipped_requires_context']}")

    # Validate if requested
    if args.validate:
        logger.info("-" * 60)
        logger.info("Validating export...")
        validation = validate_beir_export(args.output)
        if validation["valid"]:
            logger.info("BEIR export VALID")
        else:
            logger.warning(f"BEIR export has {len(validation['issues'])} issues:")
            for issue in validation["issues"][:10]:
                logger.warning(f"  - {issue}")

    # Save report
    report_path = args.output / "export_report.json"
    save_json(report, report_path)
    logger.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
