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

_HEADING_RE = re.compile(r"^(#{1,})\s+(.+)$", re.MULTILINE)


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
    for item in getattr(doc, "texts", []):
        if getattr(item, "label", None) != "section_header":
            continue
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            text = str(getattr(item, "orig", "") or "").strip()
        if text:
            heading_pages[text] = prov[0].page_no
    return heading_pages


def _extract_text_pages(doc: object) -> list[tuple[str, int]]:
    """Extract ordered (text_snippet, page_no) for ALL text items.

    Unlike _extract_heading_pages which only returns section_headers
    (sparse → off-by-1 page gaps), this returns ALL text items in
    document order, enabling dense page tracking in the chunker.

    Args:
        doc: DoclingDocument with texts having prov[].page_no.

    Returns:
        Ordered list of (text[:80], page_no) for every text item.
    """
    items: list[tuple[str, int]] = []
    for item in getattr(doc, "texts", []):
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        text = (getattr(item, "text", "") or "").strip()
        if text:
            items.append((text[:80], prov[0].page_no))
    return items


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
        tables.append(
            {
                "id": f"{pdf_path.stem}-table{table_ix}",
                "source": pdf_path.name,
                "text": table_md,
                "page": table_page,
            }
        )

    # Dense page tracking: ALL text items with page numbers (not just headings)
    text_pages = _extract_text_pages(doc)

    result_dict = {
        "markdown": markdown,
        "tables": tables,
        "source": pdf_path.name,
        "heading_pages": heading_pages,
        "text_pages": text_pages,
    }

    # Apply manual fixes for PDFs where docling produces flat/missing headings
    _apply_edge_case_fixes(result_dict)

    return result_dict


# Edge case fixes for PDFs where docling produces flat or missing headings.
# These are 1-page documents with no real heading hierarchy in the PDF.
# Verified manually against source PDFs (mars 2026).
_EDGE_CASE_FIXES: dict[str, list[tuple[str, str]]] = {
    "H01_2025_26_Conduite_pour_joueur_handicapes.pdf": [
        # No headings detected — add document title
        ("<!-- image -->\n\n", ""),  # remove leading image
        ("__PREPEND__", "## Conduite pour joueurs handicapes\n\n"),  # prepend title
    ],
    "H02_2025_26_Joueurs_a_mobilite_reduite.pdf": [
        ("<!-- image -->\n\n", ""),
        ("## Le cas g", "Le cas g"),  # intro text, not a heading
        ("## Phase I", "### Phase I"),
        ("## Phase II", "### Phase II"),
        ("__PREPEND__", "## Joueurs a mobilite reduite\n\n"),
    ],
    "R02_2025_26_Regles_generales_Annexes.pdf": [
        # All ## flat — promote title to #
        ("## ANNEXES AUX REGLES GENERALES", "# ANNEXES AUX REGLES GENERALES"),
    ],
}

# Pages for manually added headings (not in docling provenance)
_EDGE_CASE_PAGES: dict[str, dict[str, int]] = {
    "H01_2025_26_Conduite_pour_joueur_handicapes.pdf": {
        "Conduite pour joueurs handicapes": 1,
    },
    "H02_2025_26_Joueurs_a_mobilite_reduite.pdf": {
        "Joueurs a mobilite reduite": 1,
    },
}


def _apply_edge_case_fixes(result: dict) -> None:
    """Apply manual heading fixes for known problematic PDFs.

    Modifies result dict in place.
    """
    source = result["source"]
    fixes = _EDGE_CASE_FIXES.get(source)
    if not fixes:
        return

    md = result["markdown"]
    for old, new in fixes:
        md = new + md if old == "__PREPEND__" else md.replace(old, new, 1)
    result["markdown"] = md

    # Fix Phase II double-demotion (### → #### if already ###)
    if "#### Phase II" in result["markdown"]:
        result["markdown"] = result["markdown"].replace("#### Phase II", "### Phase II")

    # Add manual heading pages
    extra_pages = _EDGE_CASE_PAGES.get(source, {})
    result["heading_pages"].update(extra_pages)


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
    pdfs = [p for p in pdfs if not any(excl in p.parts for excl in exclude_dirs)]

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
    output_dir = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("corpus/processed/docling_v2_fr")
    )
    extract_corpus(corpus_dir, output_dir)
