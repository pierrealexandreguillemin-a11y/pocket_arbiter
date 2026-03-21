"""Prepare Kaggle dataset for generation training.

Reads docling JSONs, splits into paragraphs for TAPT corpus.
Copies reading_tasks.jsonl for SFT. Writes to kaggle/dataset-generation/.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_paragraphs(input_dir: Path) -> list[dict]:
    """Extract paragraphs from docling JSONs with source metadata.

    Args:
        input_dir: Directory containing docling JSON files.

    Returns:
        List of dicts with ``text`` and ``source`` keys.
        Paragraphs shorter than 20 characters are skipped.
    """
    paragraphs: list[dict] = []
    json_files = sorted(input_dir.glob("*.json"))
    for f in json_files:
        data: dict = json.loads(f.read_text(encoding="utf-8"))
        md = data.get("markdown", "")
        source = data.get("source", f.stem)
        for para in md.split("\n\n"):
            text = para.strip()
            if len(text) > 20:  # skip tiny fragments
                paragraphs.append({"text": text, "source": source})
    logger.info(
        "Extracted %d paragraphs from %d documents",
        len(paragraphs),
        len(json_files),
    )
    return paragraphs


def main() -> None:
    """Entry point for dataset preparation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare Kaggle dataset for generation training"
    )
    parser.add_argument("--corpus", required=True, help="Path to docling_v2_fr/")
    parser.add_argument("--tasks", required=True, help="Path to reading_tasks.jsonl")
    parser.add_argument("--output", required=True, help="Output dir for Kaggle dataset")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Corpus paragraphs for TAPT
    paragraphs = extract_paragraphs(Path(args.corpus))
    para_path = output / "corpus_paragraphs.jsonl"
    with open(para_path, "w", encoding="utf-8") as fh:
        for p in paragraphs:
            json.dump(p, fh, ensure_ascii=False)
            fh.write("\n")
    logger.info("Wrote %d paragraphs to %s", len(paragraphs), para_path)

    # Copy reading tasks for SFT
    tasks_src = Path(args.tasks)
    if not tasks_src.exists():
        logger.error("Missing tasks file: %s", tasks_src)
        raise FileNotFoundError(tasks_src)
    tasks_dst = output / "reading_tasks.jsonl"
    shutil.copy2(tasks_src, tasks_dst)
    task_count = sum(1 for _ in open(tasks_dst, encoding="utf-8"))
    logger.info("Copied %d reading tasks to %s", task_count, tasks_dst)


if __name__ == "__main__":
    main()
