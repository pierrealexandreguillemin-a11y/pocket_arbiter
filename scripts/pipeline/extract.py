"""PDF extraction with hierarchical heading levels.

Uses docling + docling-hierarchical-pdf to produce markdown
with real heading levels (#, ##, ###) from PDF structure.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# Try hierarchical post-processor, fallback gracefully
try:
    from hierarchical.postprocessor import ResultPostprocessor
    HAS_HIERARCHICAL = True
except ImportError:
    HAS_HIERARCHICAL = False
    logger.warning("docling-hierarchical-pdf not installed, headings will be flat")

# Directories to exclude from corpus extraction (not RAG content)
EXCLUDE_DIRS = {"Annales", "annales"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _strip_page_headers(markdown: str) -> str:
    """Remove repeated page headers/footers that docling extracts as headings.

    Detects headings whose exact text appears 3+ times (page headers
    repeated on every page) and removes all occurrences except the first.
    """
    # Count heading text occurrences
    counts: dict[str, int] = {}
    for match in _HEADING_RE.finditer(markdown):
        text = match.group(2).strip()
        counts[text] = counts.get(text, 0) + 1

    # Identify repeated headings (3+ = likely page header)
    repeated = {text for text, count in counts.items() if count >= 3}
    if not repeated:
        return markdown

    seen: set[str] = set()
    lines = markdown.split("\n")
    result = []
    for line in lines:
        m = _HEADING_RE.match(line)
        if m and m.group(2).strip() in repeated:
            text = m.group(2).strip()
            if text in seen:
                continue  # skip duplicate
            seen.add(text)
        result.append(line)

    return "\n".join(result)


def _extract_heading_pages(doc: object) -> dict[str, int]:
    """Extract heading text → page number mapping from docling provenance.

    Args:
        doc: DoclingDocument with texts having prov[].page_no.

    Returns:
        Dict mapping heading text (stripped) to first page number.
    """
    heading_pages: dict[str, int] = {}
    texts = doc.texts if hasattr(doc, "texts") else []
    for item in texts:
        if (
            hasattr(item, "label")
            and item.label == "section_header"
            and hasattr(item, "prov")
            and item.prov
        ):
            text = ""
            if hasattr(item, "text") and item.text:
                text = item.text.strip()
            elif hasattr(item, "orig") and item.orig:
                text = str(item.orig).strip()
            if text:
                heading_pages[text] = item.prov[0].page_no
    return heading_pages


def extract_pdf(pdf_path: Path) -> dict:
    """Extract a single PDF with hierarchical headings.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with keys: markdown, tables, source.
    """
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    if HAS_HIERARCHICAL:
        ResultPostprocessor(result).process()

    doc = result.document
    markdown = doc.export_to_markdown()

    # Remove repeated page headers that docling picks up as headings
    markdown = _strip_page_headers(markdown)

    # Build heading → page mapping from provenance data
    heading_pages = _extract_heading_pages(doc)

    # Extract tables (pass doc to avoid deprecation warning)
    tables = []
    for table_ix, table in enumerate(doc.tables):
        try:
            table_md = table.export_to_markdown(doc=doc)
        except TypeError:
            table_md = table.export_to_markdown()
        # Get table page from provenance
        table_page = None
        if hasattr(table, "prov") and table.prov:
            table_page = table.prov[0].page_no
        tables.append({
            "id": f"{pdf_path.stem}-table{table_ix}",
            "source": pdf_path.name,
            "text": table_md,
            "page": table_page,
        })

    return {
        "markdown": markdown,
        "tables": tables,
        "source": pdf_path.name,
        "heading_pages": heading_pages,
    }


def extract_corpus(
    corpus_dir: Path,
    output_dir: Path,
    exclude_dirs: set[str] | None = None,
) -> list[dict]:
    """Extract all PDFs in corpus_dir, excluding non-RAG directories.

    Args:
        corpus_dir: Root directory containing PDFs.
        output_dir: Where to save JSON extractions.
        exclude_dirs: Directory names to skip (default: Annales).

    Returns:
        List of extraction results.
    """
    if exclude_dirs is None:
        exclude_dirs = EXCLUDE_DIRS

    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(corpus_dir.rglob("*.pdf"))

    # Filter out excluded directories
    pdfs = [
        p for p in pdfs
        if not any(excl in p.parts for excl in exclude_dirs)
    ]

    results = []
    for pdf_path in pdfs:
        logger.info("Extracting %s", pdf_path.name)
        try:
            result = extract_pdf(pdf_path)
            out_path = output_dir / f"{pdf_path.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            results.append(result)
        except Exception:
            logger.exception("Failed to extract %s", pdf_path.name)

    logger.info("Extracted %d/%d PDFs", len(results), len(pdfs))
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    corpus_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus/fr")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corpus/processed/docling_v2_fr")
    extract_corpus(corpus_dir, output_dir)
