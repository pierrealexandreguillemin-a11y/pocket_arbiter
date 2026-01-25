#!/usr/bin/env python3
"""Fix cognitive_level to match Bloom's Taxonomy original.

Bloom's Taxonomy (1956, revised 2001):
- Remember: Recall facts and basic concepts
- Understand: Explain ideas or concepts
- Apply: Use information in new situations
- Analyze: Draw connections among ideas
- Evaluate: Justify a decision (not used in current GS)
- Create: Produce new or original work (not used in current GS)

Migration:
- RECALL -> Remember
- REMEMBER -> Remember
- UNDERSTAND -> Understand
- APPLY -> Apply
- ANALYZE -> Analyze
"""

import json
from pathlib import Path


def main() -> None:
    gs_path = Path("tests/data/gold_standard_annales_fr_v7.json")

    with open(gs_path, "r", encoding="utf-8") as f:
        gs = json.load(f)

    # Mapping from current values to Bloom's Taxonomy
    mapping = {
        "RECALL": "Remember",
        "REMEMBER": "Remember",
        "UNDERSTAND": "Understand",
        "APPLY": "Apply",
        "ANALYZE": "Analyze",
    }

    fixed_count = {k: 0 for k in mapping}

    for q in gs["questions"]:
        if "metadata" in q and "cognitive_level" in q["metadata"]:
            old_value = q["metadata"]["cognitive_level"]
            if old_value in mapping:
                q["metadata"]["cognitive_level"] = mapping[old_value]
                fixed_count[old_value] += 1

    # Update version
    gs["version"] = "7.2.1"

    with open(gs_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2, ensure_ascii=False)

    print("Bloom's Taxonomy migration:")
    for old, count in fixed_count.items():
        if count > 0:
            print(f"  {old} -> {mapping[old]}: {count}")
    print(f"Total fixed: {sum(fixed_count.values())}")
    print(f"Version: 7.2.1")


if __name__ == "__main__":
    main()
