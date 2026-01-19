"""
Multi-Vector Table Retriever - Pocket Arbiter

Implements multi-vector retrieval for tables:
- Embed table summaries (child) for semantic search
- Store raw tables (parent) in docstore for LLM synthesis
- Link via doc_id for parent-child retrieval

Leverages existing code:
- table_extractor.py: table_to_text()
- parent_child_chunker.py: architecture pattern

ISO Reference:
    - ISO/IEC 42001 - AI traceability (LLM summaries)
    - ISO/IEC 25010 S4.2 - Performance efficiency
    - ISO/IEC 12207 S7.3.3 - Code reuse (leverage existing modules)

Sources:
    - LangChain MultiVectorRetriever pattern
    - Google AI Edge SDK SqliteVectorStore compatibility

Changelog:
    - 2026-01-19: Initial implementation

Usage:
    python -m scripts.pipeline.table_multivector \
        --input corpus/processed/tables_fr.json \
        --output corpus/processed/tables_multivector_fr.json \
        --use-llm  # Optional: use LLM for summaries
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# Leverage existing modules
from scripts.pipeline.table_extractor import table_to_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --- Pydantic Models (ISO 25010 data quality) ---


class TableInput(BaseModel):
    """Input table schema from table_extractor."""

    id: str = Field(..., min_length=1)
    table_type: str = Field(default="other")
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    source: str = Field(default="unknown")
    page: int = Field(default=0, ge=0)
    text: str = Field(default="")
    accuracy: Optional[float] = Field(default=None, ge=0, le=100)

    @field_validator("headers", mode="before")
    @classmethod
    def coerce_headers(cls, v: Any) -> list[str]:
        """Coerce headers to list of strings."""
        if not v:
            return []
        return [str(h) if h else "" for h in v]


class ChildDocument(BaseModel):
    """Child document for vectorstore (summary)."""

    id: str
    doc_id: str  # Link to parent
    type: str = "table_summary"
    text: str = Field(..., min_length=1)
    source: Optional[str] = None
    page: Optional[int] = None
    table_type: str = "other"


class ParentDocument(BaseModel):
    """Parent document for docstore (raw table)."""

    id: str
    type: str = "table"
    table_type: str = "other"
    source: Optional[str] = None
    page: Optional[int] = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    markdown: str = ""
    text: str = ""
    accuracy: Optional[float] = None


class MultiVectorOutput(BaseModel):
    """Output schema for multi-vector table data."""

    corpus: str
    strategy: str = "multi_vector"
    use_llm: bool = False
    children: list[ChildDocument]
    parents: list[ParentDocument]
    parent_lookup: dict[str, int]
    config: dict[str, str] = Field(default_factory=lambda: {
        "child_type": "table_summary",
        "parent_type": "table_raw",
        "link_key": "doc_id",
    })


# --- Constants ---

TABLE_TYPE_DESCRIPTIONS = {
    "cadence": "Table de cadences de jeu (temps de reflexion, increment)",
    "penalty": "Grille de penalites et sanctions disciplinaires",
    "elo": "Table de classement Elo et coefficients",
    "tiebreak": "Criteres de departage (Buchholz, Sonneborn-Berger)",
    "other": "Tableau reglementaire",
}

LLM_SUMMARY_PROMPT = """Tu es un expert en reglements d'echecs.
Resume cette table en 1-2 phrases pour faciliter la recherche semantique.
Mentionne: le type de table, les colonnes principales, et le contexte d'usage.

Table:
{table_text}

Resume (1-2 phrases, en francais):"""


# --- Summary Generation ---


def generate_rule_based_summary(table: dict[str, Any]) -> str:
    """
    Generate table summary using rule-based approach.

    Uses table_type from input (already detected by table_extractor).

    Args:
        table: Table dict with headers, rows, table_type.

    Returns:
        Summary string optimized for semantic search.
    """
    table_type = table.get("table_type", "other")
    headers = table.get("headers", [])
    source = table.get("source", "document")
    page = table.get("page", "?")

    # Get description from type
    description = TABLE_TYPE_DESCRIPTIONS.get(table_type, TABLE_TYPE_DESCRIPTIONS["other"])

    # Format headers
    headers_str = ", ".join(str(h) for h in headers[:5])  # Max 5 headers
    if len(headers) > 5:
        headers_str += f" (+{len(headers) - 5} colonnes)"

    # Build summary
    summary = f"{description}. Colonnes: {headers_str}. Source: {source}, page {page}."

    return summary


def generate_llm_summary(
    table: dict[str, Any],
    api_key: Optional[str] = None,
    model: str = "gemini-1.5-flash",
) -> str:
    """
    Generate table summary using LLM (Gemini API).

    ISO 42001 compliant: traceable, deterministic prompt.

    Args:
        table: Table dict with text representation.
        api_key: Gemini API key (or from env GOOGLE_API_KEY).
        model: Model name (default: gemini-1.5-flash for cost).

    Returns:
        LLM-generated summary string.

    Raises:
        ImportError: If google-generativeai not installed.
        ValueError: If API key not provided.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "google-generativeai required for LLM summaries. "
            "Install with: pip install google-generativeai"
        ) from e

    # Get API key
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Gemini API key required. Set GOOGLE_API_KEY env or pass api_key."
        )

    genai.configure(api_key=key)

    # Get table text (leverage existing function)
    table_text = table.get("text", "")
    if not table_text:
        table_text = table_to_text(
            table.get("headers", []),
            table.get("rows", []),
            table.get("table_type", "other"),
        )

    # Build prompt
    prompt = LLM_SUMMARY_PROMPT.format(table_text=table_text[:2000])  # Truncate

    # Call API
    try:
        llm = genai.GenerativeModel(model)
        response = llm.generate_content(prompt)
        summary = response.text.strip()

        # Add source info (not in LLM output)
        source = table.get("source", "document")
        page = table.get("page", "?")
        summary = f"{summary} [Source: {source}, page {page}]"

        return summary

    except Exception as e:
        logger.warning(f"LLM summary failed: {e}. Falling back to rule-based.")
        return generate_rule_based_summary(table)


def generate_table_summary(
    table: dict[str, Any],
    use_llm: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate table summary for embedding.

    Args:
        table: Table dict from table_extractor.
        use_llm: If True, use LLM. Otherwise rule-based.
        api_key: Optional Gemini API key.

    Returns:
        Summary string optimized for semantic search.
    """
    if use_llm:
        return generate_llm_summary(table, api_key=api_key)
    return generate_rule_based_summary(table)


# --- Multi-Vector Structure ---


def table_to_markdown(table: dict[str, Any]) -> str:
    """
    Convert table to Markdown format for LLM synthesis.

    Args:
        table: Table dict with headers and rows.

    Returns:
        Markdown table string.
    """
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    if not headers:
        return ""

    lines = []

    # Header row
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")

    # Separator
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    # Data rows
    for row in rows:
        # Pad row if needed
        padded = list(row) + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(str(cell) for cell in padded[:len(headers)]) + " |")

    return "\n".join(lines)


def create_multivector_entry(
    table: dict[str, Any],
    use_llm: bool = False,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create multi-vector entry for a table.

    Structure compatible with:
    - LangChain MultiVectorRetriever
    - Google AI Edge SqliteVectorStore (via doc_id metadata)

    Args:
        table: Table dict from table_extractor.
        use_llm: Use LLM for summary generation.
        api_key: Optional Gemini API key.

    Returns:
        Dict with:
        - child: Summary for embedding (vectorstore)
        - parent: Raw table for synthesis (docstore)
        - doc_id: Link between child and parent
    """
    doc_id = table.get("id", f"table-{hash(str(table))}")

    # Generate summary (child for embedding)
    summary = generate_table_summary(table, use_llm=use_llm, api_key=api_key)

    # Create and validate parent (raw for LLM synthesis) with pydantic
    parent = ParentDocument(
        id=doc_id,
        type="table",
        table_type=table.get("table_type", "other"),
        source=table.get("source"),
        page=table.get("page"),
        headers=table.get("headers", []),
        rows=table.get("rows", []),
        markdown=table_to_markdown(table),
        text=table.get("text", ""),
        accuracy=table.get("accuracy"),
    )

    # Create and validate child (summary for vectorstore) with pydantic
    child = ChildDocument(
        id=f"{doc_id}-summary",
        doc_id=doc_id,  # Link to parent
        type="table_summary",
        text=summary,  # This gets embedded
        source=table.get("source"),
        page=table.get("page"),
        table_type=table.get("table_type", "other"),
    )

    return {
        "doc_id": doc_id,
        "child": child.model_dump(),
        "parent": parent.model_dump(),
    }


# --- Processing ---


def process_tables_multivector(
    input_file: Path,
    output_file: Path,
    use_llm: bool = False,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Process tables into multi-vector format.

    Args:
        input_file: Path to tables JSON (from table_extractor).
        output_file: Output path for multi-vector JSON.
        use_llm: Use LLM for summaries.
        api_key: Optional Gemini API key.

    Returns:
        Processing report.
    """
    logger.info(f"Loading tables from {input_file}")

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    tables = data.get("tables", [])
    corpus = data.get("corpus", "unknown")

    logger.info(f"Processing {len(tables)} tables (use_llm={use_llm})")

    children = []  # For vectorstore (summaries)
    parents = []  # For docstore (raw tables)
    parent_lookup = {}  # doc_id -> parent index
    skipped_empty = 0  # Count of tables skipped due to empty headers

    for i, table in enumerate(tables):
        try:
            # Filter tables with empty or invalid headers (quality gate)
            headers = table.get("headers", [])
            valid_headers = [h for h in headers if h and str(h).strip()]
            if len(valid_headers) < 2:
                skipped_empty += 1
                logger.debug(f"  Skipping table {table.get('id')}: insufficient headers")
                continue

            entry = create_multivector_entry(table, use_llm=use_llm, api_key=api_key)

            children.append(entry["child"])
            parents.append(entry["parent"])
            parent_lookup[entry["doc_id"]] = len(parents) - 1

            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{len(tables)} tables")

        except Exception as e:
            logger.warning(f"  Failed to process table {table.get('id')}: {e}")

    if skipped_empty > 0:
        logger.info(f"  Skipped {skipped_empty} tables with empty/invalid headers")

    # Build output
    output = {
        "corpus": corpus,
        "strategy": "multi_vector",
        "use_llm": use_llm,
        "children": children,  # Embed these
        "parents": parents,  # Store in docstore
        "parent_lookup": parent_lookup,  # doc_id -> index
        "config": {
            "child_type": "table_summary",
            "parent_type": "table_raw",
            "link_key": "doc_id",
        },
    }

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(children)} children + {len(parents)} parents to {output_file}")

    # Report
    by_type: dict[str, int] = {}
    for p in parents:
        t = p.get("table_type", "other")
        by_type[t] = by_type.get(t, 0) + 1

    report = {
        "corpus": corpus,
        "total_tables": len(tables),
        "tables_skipped": skipped_empty,
        "children_created": len(children),
        "parents_created": len(parents),
        "use_llm": use_llm,
        "by_type": by_type,
    }

    return report


# --- CLI ---


def main() -> None:
    """CLI for multi-vector table processing."""
    parser = argparse.ArgumentParser(
        description="Multi-Vector Table Processor - Pocket Arbiter",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input tables JSON (from table_extractor)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output multi-vector JSON",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM (Gemini) for summaries (requires GOOGLE_API_KEY)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GOOGLE_API_KEY env)",
    )

    args = parser.parse_args()

    report = process_tables_multivector(
        input_file=args.input,
        output_file=args.output,
        use_llm=args.use_llm,
        api_key=args.api_key,
    )

    logger.info(f"Report: {json.dumps(report, indent=2)}")


if __name__ == "__main__":
    main()
