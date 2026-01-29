#!/usr/bin/env python3
"""Fix answer_type to match taxonomy standard.

Taxonomy (GOLD_STANDARD_V6_ANNALES.md):
- extractive: answer directly in chunk
- abstractive: requires synthesis
- yes_no: yes/no questions
- list: enumeration answers
- multiple_choice: MCQ (existing annales)
"""

import json
from pathlib import Path


def main() -> None:
    gs_path = Path("tests/data/gold_standard_annales_fr_v7.json")

    with open(gs_path, "r", encoding="utf-8") as f:
        gs = json.load(f)

    # Questions with list-type answers (enumerations)
    list_answer_ids = {
        "FR-ELO-003",  # 5 parties listées
        "FR-ELO-007",  # catégories listées
        "FR-ELO-008",  # comparaison catégories
    }

    fixed_count = {"NUMERICAL": 0, "PROCEDURAL": 0}

    for q in gs["questions"]:
        answer_type = q.get("metadata", {}).get("answer_type", "")

        if answer_type == "NUMERICAL":
            if q["id"] in list_answer_ids:
                q["metadata"]["answer_type"] = "list"
            else:
                q["metadata"]["answer_type"] = "extractive"
            fixed_count["NUMERICAL"] += 1

        elif answer_type == "PROCEDURAL":
            q["metadata"]["answer_type"] = "extractive"
            fixed_count["PROCEDURAL"] += 1

    # Update version
    gs["version"] = "7.1.0"

    with open(gs_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2, ensure_ascii=False)

    print(f"Fixed NUMERICAL: {fixed_count['NUMERICAL']} -> extractive/list")
    print(f"Fixed PROCEDURAL: {fixed_count['PROCEDURAL']} -> extractive")
    print("Version: 7.1.0")


if __name__ == "__main__":
    main()
