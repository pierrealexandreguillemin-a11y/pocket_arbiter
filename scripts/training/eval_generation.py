"""Generation model evaluation on GS questions.

Runs CPU inference (slow, intentional — no GPU required).
Produces one output JSON per model: human scores + auto citation.

Usage:
    python -m scripts.training.eval_generation \\
        --model <path_or_hf_id> \\
        --gs tests/data/gold_standard_annales_fr_v8_adversarial.json \\
        --db corpus/processed/corpus_v2_fr.db \\
        --output-dir eval_results/ \\
        --output-prefix generation_eval_base
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

SOURCE_PATTERNS: dict[str, str] = {
    "LA": r"(?:livre.{0,10}arbitre|l\.?a\.?\b)",
    "R01": r"(?:r[eè]gles?\s+g[eé]n[eé]rales?|r\.?01)",
    "R02": r"(?:annexes?\s+aux\s+r[eè]gles|r\.?02)",
    "A01": r"(?:championnat\s+de\s+france\b|a\.?01)",
    "A02": r"(?:championnat\s+.{0,10}clubs?|a\.?02)",
}


def load_human_questions(gs_path: str) -> list[dict]:
    """Load 34 answerable human-authored GS questions.

    Args:
        gs_path: Path to gold standard JSON file.

    Returns:
        List of question dicts with id, content, provenance.
    """
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)
    return [
        q
        for q in gs["questions"]
        if q["id"].startswith("ffe:human:")
        and not q.get("content", {}).get("is_impossible", False)
    ]


def load_annales_questions(gs_path: str) -> list[dict]:
    """Load 264 answerable annales GS questions.

    Args:
        gs_path: Path to gold standard JSON file.

    Returns:
        List of question dicts with annales_source set and not impossible.
    """
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)
    return [
        q
        for q in gs["questions"]
        if q.get("provenance", {}).get("annales_source") is not None
        and not q.get("content", {}).get("is_impossible", False)
    ]


def load_chunk_context(db_path: str, source: str, page: int) -> str:
    """Retrieve chunk text for a given source document and page.

    Concatenates all matching children (v2 DB uses source+page,
    not v1 chunk_id format).

    Args:
        db_path: Path to corpus_v2_fr.db.
        source: Document filename (e.g. 'LA-octobre2025.pdf').
        page: Page number.

    Returns:
        Concatenated text of all children on that page, or empty string.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT text FROM children WHERE source = ? AND page = ?",
            (source, page),
        ).fetchall()
        return "\n\n".join(r[0] for r in rows)
    finally:
        conn.close()


def check_citation(
    response: str,
    expected_docs: list[str],
    expected_pages: list[int],
) -> bool:
    """Check whether a response cites an expected document or page.

    Args:
        response: Model response text.
        expected_docs: List of expected source filenames.
        expected_pages: List of expected page numbers.

    Returns:
        True if at least one doc pattern or page number is cited.
    """
    response_lower = response.lower()
    doc_cited = any(
        re.search(pat, response_lower)
        for doc in expected_docs
        for key, pat in SOURCE_PATTERNS.items()
        if key in doc.upper()
    )
    page_cited = any(
        re.search(rf"\bpage\s*{p}\b|\bp\.?\s*{p}\b", response_lower)
        for p in expected_pages
    )
    return doc_cited or page_cited


def generate_response(model: object, tokenizer: object, messages: list[dict]) -> str:
    """Run CPU inference and return decoded response text.

    Args:
        model: HuggingFace CausalLM model (CPU, float32).
        tokenizer: Corresponding tokenizer.
        messages: Chat messages list from build_rag_prompt().

    Returns:
        Generated response string (assistant turn only).
    """
    import torch  # local import — not required for tests

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def main() -> None:
    """CLI entry point for generation evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate generation model on GS.")
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument(
        "--gs",
        default="tests/data/gold_standard_annales_fr_v8_adversarial.json",
        help="GS JSON path",
    )
    parser.add_argument(
        "--db",
        default="corpus/processed/corpus_v2_fr.db",
        help="DB path",
    )
    parser.add_argument("--output-dir", default="eval_results", help="Output dir")
    parser.add_argument(
        "--output-prefix",
        default="generation_eval",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None, help="Limit questions (testing)"
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.training.generation_prompt import build_rag_prompt

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    human_qs = load_human_questions(args.gs)
    annales_qs = load_annales_questions(args.gs)

    if args.max_questions:
        human_qs = human_qs[: args.max_questions]
        annales_qs = annales_qs[: args.max_questions]

    # --- Human questions: full response + manual scores placeholder ---
    human_results = []
    for i, q in enumerate(human_qs):
        prov = q.get("provenance", {})
        source = (prov.get("docs") or [""])[0]
        page = (prov.get("pages") or [0])[0]
        context = load_chunk_context(args.db, source, page)
        messages = build_rag_prompt(q["content"]["question"], context)
        print(f"[{i+1}/{len(human_qs)}] {q['id']}")
        response = generate_response(model, tokenizer, messages)
        human_results.append(
            {
                "id": q["id"],
                "question": q["content"]["question"],
                "context": context,
                "response": response,
                "scores": {"useful": None, "faithful": None, "cited": None},
            }
        )

    # --- Annales: auto citation check only ---
    cited_count = 0
    for i, q in enumerate(annales_qs):
        prov = q.get("provenance", {})
        source = (prov.get("docs") or [""])[0]
        page = (prov.get("pages") or [0])[0]
        context = load_chunk_context(args.db, source, page)
        messages = build_rag_prompt(q["content"]["question"], context)
        print(f"[annales {i+1}/{len(annales_qs)}] {q['id']}")
        response = generate_response(model, tokenizer, messages)
        if check_citation(response, prov.get("docs", []), prov.get("pages", [])):
            cited_count += 1

    total_annales = len(annales_qs)
    cited_pct = round(100 * cited_count / total_annales, 1) if total_annales else 0.0

    output = {
        "model": args.model,
        "questions": human_results,
        "auto_citation": {
            "total": total_annales,
            "cited_count": cited_count,
            "cited_pct": cited_pct,
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_prefix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Auto-citation: {cited_count}/{total_annales} = {cited_pct}%")


if __name__ == "__main__":
    main()
