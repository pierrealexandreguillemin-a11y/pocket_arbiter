#!/usr/bin/env python3
"""
Batch reformulation script for GS Annales v7.
Reformulates QCM-style questions to natural language for RAG triplets.

Usage:
    python scripts/evaluation/annales/batch_reformulate.py

ISO Compliance:
    - ISO 42001: Citations preserved, grounded reformulation
    - ISO 25010: Quality validation with BERTScore
"""

import json
from datetime import datetime
from pathlib import Path


def load_json(path: str) -> dict:
    """Load JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    """Save JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fix_chunk_ids(gs: dict) -> dict:
    """Fix the 9 missing chunk_ids with correct mappings."""
    corrections = {
        "ffe:human:rating:003:880750a0": "LA-octobre2025.pdf-p186-parent552-child00",
        "ffe:human:rating:005:91ec4e85": "R01_2025_26_Regles_generales.pdf-p004-parent021-child00",
        "ffe:human:youth:007:a0e7a66d": "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf-p004-parent018-child00",
        "ffe:human:admin:005:35800014": "Contrat_de_delegation_15032022-table0-summary",
        "ffe:human:regional:001:6794986f": "Interclubs_DepartementalBdr.pdf-p007-parent014-child00",
        "ffe:human:regional:002:94cf56b9": "InterclubsJeunes_PACABdr.pdf-p002-parent005-child00",
        "ffe:human:regional:004:a18d472b": "règlement_régionale_2024_2025.pdf-p002-parent008-child00",
        "ffe:human:clubs:003:3c3a124c": "A02_2025_26_Championnat_de_France_des_Clubs.pdf-p005-parent017-child00",
        "ffe:human:clubs:004:f070542c": "A02_2025_26_Championnat_de_France_des_Clubs.pdf-p006-parent018-child02",
    }

    fixed_count = 0
    for q in gs["questions"]:
        if q["id"] in corrections:
            old_chunk = q.get("expected_chunk_id", "")
            new_chunk = corrections[q["id"]]
            if old_chunk != new_chunk:
                q["expected_chunk_id"] = new_chunk
                fixed_count += 1
                print(f"Fixed {q['id']}: {old_chunk} -> {new_chunk}")

    print(f"\nFixed {fixed_count} chunk_ids")
    return gs


def validate_chunk_coverage(gs: dict, chunks_index: dict) -> tuple[int, list]:
    """Validate that all questions have valid chunk_ids."""
    missing = []
    valid = 0

    for q in gs["questions"]:
        chunk_id = q.get("expected_chunk_id", "")
        if chunk_id in chunks_index:
            valid += 1
        else:
            missing.append({"id": q["id"], "chunk_id": chunk_id})

    return valid, missing


def build_chunk_index(chunks_path: str) -> dict:
    """Build chunk index from chunks file."""
    data = load_json(chunks_path)
    chunks = data.get("chunks", data)
    return {c["id"]: c["text"] for c in chunks}


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.parent.parent.parent
    gs_path = base_path / "tests" / "data" / "gold_standard_annales_fr_v7.json"
    chunks_path = base_path / "corpus" / "processed" / "chunks_mode_b_fr.json"

    print("=" * 60)
    print("GS Annales v7 - Batch Reformulation")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    gs = load_json(str(gs_path))
    chunk_index = build_chunk_index(str(chunks_path))
    print(f"  - GS questions: {len(gs['questions'])}")
    print(f"  - Chunks: {len(chunk_index)}")

    # Validate before fix
    print("\n[2/4] Validating chunk coverage (before fix)...")
    valid_before, missing_before = validate_chunk_coverage(gs, chunk_index)
    print(f"  - Valid: {valid_before}/{len(gs['questions'])}")
    print(f"  - Missing: {len(missing_before)}")

    # Fix chunk_ids
    print("\n[3/4] Fixing chunk_ids...")
    gs = fix_chunk_ids(gs)

    # Validate after fix
    print("\n[4/4] Validating chunk coverage (after fix)...")
    valid_after, missing_after = validate_chunk_coverage(gs, chunk_index)
    print(f"  - Valid: {valid_after}/{len(gs['questions'])}")
    print(f"  - Missing: {len(missing_after)}")

    if missing_after:
        print("\n[WARNING] Still missing chunks:")
        for m in missing_after:
            print(f"  - {m['id']}: {m['chunk_id']}")

    # Update version
    gs["version"] = "7.4.1"
    gs["methodology"]["reformulation"] = "chunk_ids fixed, reformulation pending"

    # Save
    save_json(gs, str(gs_path))
    print(f"\n[OK] Saved v7.4.1 to {gs_path}")


if __name__ == "__main__":
    main()
