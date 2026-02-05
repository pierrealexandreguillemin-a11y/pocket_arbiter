"""
Parse full Docling document to extract questions with explanations.

This script uses the `docling_document.texts` structure to extract:
- Questions with their choices
- Article references
- Detailed explanations from corrigé

ISO Reference:
    - ISO 42001 A.6.2.2 - Provenance tracking
    - ISO 29119-3 - Test data documentation
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# UV corrigé page ranges for jun2025
UV_CORRIGE_PAGES = {
    "UVR": range(12, 20),  # pages 12-19
    "UVC": range(27, 35),  # pages 27-34
    "UVO": range(41, 47),  # pages 41-46
    "UVT": range(58, 73),  # pages 58-72
}

# UV grille pages for jun2025
UV_GRILLE_PAGES = {
    "UVR": 11,
    "UVC": 26,
    "UVO": 40,
    "UVT": 57,
}


def get_page(text_item: dict) -> int:
    """Extract page number from text item provenance."""
    prov = text_item.get("prov", [{}])
    return prov[0].get("page_no", -1) if prov else -1


def is_question_start(text: str) -> tuple[bool, int | None]:
    """Check if text is a question start and return question number."""
    match = re.match(r"Question\s+(\d+)\s*:", text)
    if match:
        return True, int(match.group(1))
    return False, None


def is_article_reference(text: str) -> bool:
    """Check if text is an article reference."""
    patterns = [
        r"Règles du jeu",
        r"LA\s*[-–]\s*Chapitre",
        r"LA\s+RIDNA",
        r"RIDNA",
        r"R0\d+\s+",
        r"Guide de l'Arbitrage",
        r"L\.A\.",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_explanation(text: str) -> bool:
    """Check if text is an explanation (not just article ref)."""
    if len(text) < 30:
        return False
    patterns = [
        r"ne peut pas s'appliquer",
        r"car\s+\w",
        r"donc\s+\w",
        r"permet\s+",
        r"stipule",
        r"[aà] partir du moment",
        r"on consid[eè]re",
        r"il faut",
        r"la bonne r[eé]ponse",
        r"est correct",
        r"n'est pas correct",
        r"on ne revient pas",
        r"est autoris[eé]",
        r"n'est pas autoris[eé]",
        r"doit\s+",
        r"peut\s+",
        r"signifie que",
        r"cela veut dire",
        r"en effet",
        r"par cons[eé]quent",
        r"seul[e]?\s+",
        r"uniquement",
        r"toujours",
        r"jamais",
        r"obligatoire",
        r"interdit",
        r"valable",
        r"valide",
        r"incorrect",
        # New patterns for more explanations
        r"est assimil[eé]",
        r"sa responsabilit[eé]",
        r"il [eé]tait de",
        r"n'ont pas besoin",
        r"sera dans l'obligation",
        r"est r[eé]activ[eé]",
        r"beaucoup de candidats",
        r"cette question",
        r"le taux de r[eé]ussite",
        r"attention",
        r"majoration",
        r"ne s'applique pas",
        r"l'arbitre (doit|peut|n'a)",
        r"le joueur (doit|peut|n'a)",
        r"les joueurs (doivent|peuvent)",
        r"sont interdits",
        r"est interdit",
        r"il est (donc|ainsi)",
        r"position (morte|ill[eé]gale)",
        r"coup (ill[eé]gal|l[eé]gal)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def parse_corrige_texts(
    texts: list[dict],
    uv: str,
    page_range: range,
) -> list[dict[str, Any]]:
    """Parse corrigé texts for a UV and extract questions with explanations."""
    # Group texts by page
    page_texts: dict[int, list[dict]] = defaultdict(list)
    for t in texts:
        page = get_page(t)
        if page in page_range:
            page_texts[page].append(t)

    questions = []
    current_q: dict[str, Any] | None = None

    for page in sorted(page_range):
        for t in page_texts[page]:
            text = t.get("text", "").strip()
            label = t.get("label", "")

            is_q, q_num = is_question_start(text)
            if is_q:
                if current_q:
                    questions.append(current_q)
                current_q = {
                    "uv": uv,
                    "num": q_num,
                    "page": page,
                    "question_text": "",  # Will be filled by next text block
                    "choices": [],
                    "article_ref": None,
                    "explanations": [],
                }
            elif current_q:
                # Capture question text (first text block after "Question N :")
                if not current_q["question_text"] and label == "text":
                    current_q["question_text"] = text
                    continue
                if label == "list_item":
                    # Check if it's an article ref embedded in list_item
                    if is_article_reference(text) and not current_q["article_ref"]:
                        current_q["article_ref"] = text
                    else:
                        current_q["choices"].append(text)
                elif is_article_reference(text):
                    # First article ref is the main reference
                    if not current_q["article_ref"]:
                        current_q["article_ref"] = text
                    # But it might also contain explanation
                    if is_explanation(text):
                        current_q["explanations"].append(text)
                elif is_explanation(text):
                    current_q["explanations"].append(text)
                elif label == "text" and len(text) > 50:
                    # Long text blocks after question might be explanations
                    # Check if it's not just the question continuation
                    if not text.startswith(
                        ("Vous ", "Un ", "Une ", "Le ", "La ", "Les ")
                    ):
                        current_q["explanations"].append(text)

    if current_q:
        questions.append(current_q)

    return questions


def parse_grille_table(
    tables: list[dict],
    uv: str,
    grille_page: int,
) -> dict[int, dict[str, Any]]:
    """Parse grille table to get answers and success rates."""
    grille_data = {}

    for table in tables:
        # Find table on grille page
        prov = table.get("prov", [{}])
        page = prov[0].get("page_no", -1) if prov else -1

        if page == grille_page:
            data = table.get("data", {})
            grid = data.get("grid", [])

            # Skip header row
            for row in grid[1:] if len(grid) > 1 else []:
                if len(row) >= 4:
                    try:
                        q_num = int(row[0].get("text", "").strip())
                        answer = row[1].get("text", "").strip()
                        article = row[2].get("text", "").strip()
                        taux = row[3].get("text", "").strip().rstrip("%")

                        grille_data[q_num] = {
                            "answer": answer,
                            "article_grille": article,
                            "success_rate": float(taux) / 100 if taux else None,
                        }
                    except (ValueError, IndexError):
                        continue

    return grille_data


def extract_full_questions(docling_path: Path) -> list[dict[str, Any]]:
    """Extract all questions from full Docling document."""
    with open(docling_path, encoding="utf-8") as f:
        data = json.load(f)

    doc = data.get("docling_document", {})
    texts = doc.get("texts", [])
    tables = doc.get("tables", [])

    all_questions = []

    for uv, page_range in UV_CORRIGE_PAGES.items():
        # Parse corrigé for questions and explanations
        questions = parse_corrige_texts(texts, uv, page_range)

        # Parse grille for answers and success rates
        grille_page = UV_GRILLE_PAGES.get(uv)
        grille_data = {}
        if grille_page:
            grille_data = parse_grille_table(tables, uv, grille_page)

        # Merge data
        for q in questions:
            q_num = q["num"]
            if q_num in grille_data:
                q["mcq_answer"] = grille_data[q_num].get("answer")
                q["success_rate"] = grille_data[q_num].get("success_rate")
                # Use grille article if corrigé doesn't have one
                if not q["article_ref"]:
                    q["article_ref"] = grille_data[q_num].get("article_grille")

            all_questions.append(q)

    return all_questions


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse full Docling document for questions with explanations"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(
            "corpus/processed/annales_juin_2025/Annales-Juin-2025-VF2_full.json"
        ),
        help="Input Docling JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tests/data/jun2025_docling_extracted.json"),
        help="Output JSON file",
    )

    args = parser.parse_args()

    print(f"Parsing: {args.input}")
    questions = extract_full_questions(args.input)

    # Summary
    print("\n=== EXTRACTION SUMMARY ===")
    from collections import Counter

    uv_counts = Counter(q["uv"] for q in questions)
    for uv, count in sorted(uv_counts.items()):
        with_exp = sum(1 for q in questions if q["uv"] == uv and q["explanations"])
        print(f"{uv}: {count} questions ({with_exp} with explanations)")

    total_exp = sum(1 for q in questions if q["explanations"])
    print(f"\nTotal: {len(questions)} questions ({total_exp} with explanations)")

    # Save
    output_data = {
        "session": "jun2025",
        "extraction_method": "docling_full_document",
        "total_questions": len(questions),
        "questions_with_explanations": total_exp,
        "questions": questions,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
