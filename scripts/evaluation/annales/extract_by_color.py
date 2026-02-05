"""
Extract questions from PDF corrigé using text color.

Color coding in jun2025 corrigé:
- #00b050 (green): Correct answer
- #7030a0 (purple): Article reference
- #0070c0 (blue): Explanation
- #000000 (black): Question text, notes

ISO Reference: ISO 42001 A.6.2.2 - Provenance tracking
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

# Color codes (as integers) - multiple variants per category
COLORS_GREEN = [
    45136,  # #00b050 - UVR/UVC
    52224,  # #00cc00 - UVT
]
COLORS_PURPLE = [
    7352480,  # #7030a0 - UVR/UVC
    6684927,  # #6600ff - UVT/UVO
]
COLORS_BLUE = [
    28864,  # #0070c0 - UVR/UVC
    255,  # #0000ff - UVT
    4485828,  # #4472c4 - UVO
]
COLOR_BLACK = 0  # #000000 - Normal text


def color_to_hex(color: int) -> str:
    """Convert integer color to hex string."""
    r = (color >> 16) & 0xFF
    g = (color >> 8) & 0xFF
    b = color & 0xFF
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_colored_text(page) -> dict:
    """Extract text grouped by color from a page."""
    blocks = page.get_text("dict")["blocks"]

    colored_texts = {
        "green": [],  # Correct answers
        "purple": [],  # Article references
        "blue": [],  # Explanations
        "black": [],  # Normal text
    }

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span.get("text", "").strip()
                color = span.get("color", 0)

                if not text:
                    continue

                if color in COLORS_GREEN:
                    colored_texts["green"].append(text)
                elif color in COLORS_PURPLE:
                    colored_texts["purple"].append(text)
                elif color in COLORS_BLUE:
                    colored_texts["blue"].append(text)
                elif color == COLOR_BLACK:
                    colored_texts["black"].append(text)

    return colored_texts


def parse_corrige_page(page, page_num: int) -> list[dict]:
    """Parse a corrigé page and extract questions with colored elements."""
    colored = extract_colored_text(page)

    questions = []

    # Combine all text to find question boundaries
    full_text = page.get_text()

    # Find question numbers
    q_matches = list(re.finditer(r"Question\s+(\d+)\s*:", full_text))

    for i, match in enumerate(q_matches):
        q_num = int(match.group(1))

        # Get text range for this question
        start = match.end()
        end = q_matches[i + 1].start() if i + 1 < len(q_matches) else len(full_text)
        q_text = full_text[start:end]

        question = {
            "num": q_num,
            "page": page_num,
            "correct_answers": [],
            "article_refs": [],
            "explanations": [],
        }

        # Match colored text to this question's range
        for green_text in colored["green"]:
            if green_text in q_text:
                question["correct_answers"].append(green_text)

        for purple_text in colored["purple"]:
            if purple_text in q_text:
                question["article_refs"].append(purple_text)

        for blue_text in colored["blue"]:
            if blue_text in q_text:
                question["explanations"].append(blue_text)

        questions.append(question)

    return questions


def extract_uv_corrige(pdf_path: Path, pages: range, uv_name: str) -> list[dict]:
    """Extract all questions from a UV's corrigé pages."""
    doc = fitz.open(pdf_path)
    all_questions = []

    for page_num in pages:
        page = doc[page_num - 1]  # 0-indexed
        questions = parse_corrige_page(page, page_num)
        for q in questions:
            q["uv"] = uv_name
        all_questions.extend(questions)

    doc.close()
    return all_questions


def main():
    pdf_path = Path("corpus/fr/Annales/Annales-Juin-2025-VF2.pdf")

    # UV corrigé page ranges
    uv_pages = {
        "UVR": range(12, 20),
        "UVC": range(27, 35),
        "UVO": range(41, 47),
        "UVT": range(58, 73),
    }

    all_questions = []

    print("Extracting by color...")
    for uv, pages in uv_pages.items():
        questions = extract_uv_corrige(pdf_path, pages, uv)
        all_questions.extend(questions)

        with_exp = sum(1 for q in questions if q["explanations"])
        print(f"  {uv}: {len(questions)} questions, {with_exp} with explanations")

    # Summary
    total = len(all_questions)
    with_exp = sum(1 for q in all_questions if q["explanations"])
    with_article = sum(1 for q in all_questions if q["article_refs"])
    with_answer = sum(1 for q in all_questions if q["correct_answers"])

    print("\n=== EXTRACTION BY COLOR ===")
    print(f"Total: {total} questions")
    print(f"With correct answer (green): {with_answer}")
    print(f"With article ref (purple): {with_article}")
    print(f"With explanation (blue): {with_exp}")

    # Save
    output_path = Path("tests/data/jun2025_color_extracted.json")
    output = {
        "session": "jun2025",
        "extraction_method": "pymupdf_color",
        "color_mapping": {
            "green (#00b050)": "correct_answer",
            "purple (#7030a0)": "article_reference",
            "blue (#0070c0)": "explanation",
        },
        "questions": all_questions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
