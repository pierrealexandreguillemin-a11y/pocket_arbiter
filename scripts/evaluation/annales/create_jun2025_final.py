"""
Create final jun2025 Gold Standard by merging:
- Grilles (verified answers, articles, success rates)
- Docling extraction (question text, choices)
- Color extraction (correct answers, explanations)

ISO Reference: ISO 42001 A.6.2.2 - Provenance tracking
"""

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_all_sources():
    """Merge grilles, docling, and color extractions."""
    grilles = load_json(Path("tests/data/jun2025_grilles.json"))
    docling = load_json(Path("tests/data/jun2025_docling_extracted.json"))
    colors = load_json(Path("tests/data/jun2025_color_extracted.json"))

    # Index docling by (uv, num)
    docling_idx = {(q["uv"], q["num"]): q for q in docling["questions"]}

    # Index colors by (uv, num)
    colors_idx = {(q["uv"], q["num"]): q for q in colors["questions"]}

    questions = []

    for uv_name, uv_data in grilles["grilles"].items():
        for grille_q in uv_data["questions"]:
            q_num = grille_q["num"]
            key = (uv_name, q_num)

            # Start with grille data (authoritative)
            question = {
                "uv": uv_name,
                "question_num": q_num,
                "mcq_answer": grille_q["answer"],
                "article_reference": grille_q["article"],
                "success_rate": grille_q["rate"],
                "grille_page": uv_data["page"],
            }

            # Add docling data
            if key in docling_idx:
                doc_q = docling_idx[key]
                question["corrige_page"] = doc_q.get("page")
                question["question_text"] = doc_q.get("question_text", "")
                question["choices"] = doc_q.get("choices", [])

            # Add color-extracted data
            if key in colors_idx:
                col_q = colors_idx[key]
                question["correct_answer_text"] = col_q.get("correct_answers", [])
                question["article_refs_color"] = col_q.get("article_refs", [])
                question["explanations_color"] = col_q.get("explanations", [])

            # Merge explanations from both sources
            docling_exp = docling_idx.get(key, {}).get("explanations", [])
            color_exp = colors_idx.get(key, {}).get("explanations", [])

            # Combine unique explanations
            all_exp = []
            seen = set()
            for exp in docling_exp + color_exp:
                exp_clean = exp.strip()
                if exp_clean and exp_clean not in seen:
                    all_exp.append(exp_clean)
                    seen.add(exp_clean)

            question["explanations"] = all_exp
            question["has_explanation"] = len(all_exp) > 0

            questions.append(question)

    return questions


def main():
    print("Creating final jun2025 Gold Standard...")
    questions = merge_all_sources()

    # Summary
    total = len(questions)
    with_exp = sum(1 for q in questions if q["has_explanation"])
    with_answer_text = sum(1 for q in questions if q.get("correct_answer_text"))

    print("\n=== FINAL MERGE SUMMARY ===")
    print(f"Total questions: {total}")
    print(f"With explanations: {with_exp} ({100*with_exp/total:.1f}%)")
    print(f"With answer text (color): {with_answer_text}")

    # Per UV
    from collections import Counter

    uv_counts = Counter(q["uv"] for q in questions)
    uv_exp = Counter(q["uv"] for q in questions if q["has_explanation"])
    print("\nPer UV:")
    for uv in sorted(uv_counts.keys()):
        print(f"  {uv}: {uv_counts[uv]} questions, {uv_exp[uv]} with explanations")

    # Save
    output = {
        "session": "jun2025",
        "source": "Annales-Juin-2025-VF2.pdf",
        "extraction_date": "2026-02-04",
        "extraction_methods": ["grilles_verified", "docling_full", "pymupdf_color"],
        "total_questions": total,
        "questions_with_explanations": with_exp,
        "questions": questions,
    }

    output_path = Path("tests/data/jun2025_final.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
