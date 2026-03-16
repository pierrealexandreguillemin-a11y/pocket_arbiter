#!/usr/bin/env python3
"""Validate that all proposed chunk_ids exist in the corpus (anti-hallucination check)."""

import json


def main():
    print("=== VALIDATION ANTI-HALLUCINATION ===\n")

    # Load valid chunk_ids
    with open("corpus/processed/chunks_mode_b_fr.json", encoding="utf-8") as f:
        valid_fr = {c["id"] for c in json.load(f)["chunks"]}
    with open("corpus/processed/chunks_mode_b_intl.json", encoding="utf-8") as f:
        valid_intl = {c["id"] for c in json.load(f)["chunks"]}

    valid_ids = valid_fr | valid_intl
    print(
        f"Total chunks valides: {len(valid_ids)} ({len(valid_fr)} FR + {len(valid_intl)} INTL)"
    )

    # Check each new chunk_id
    hallucinations = []
    valid_corrections = []

    for i in range(1, 7):
        filepath = f"data/semantic_validation/agent_{i}_results.json"
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        for r in data["results"]:
            new_id = r.get("new_chunk_id")
            if new_id and new_id != "null" and new_id is not None:
                if new_id in valid_ids:
                    valid_corrections.append(
                        {
                            "id": r["id"],
                            "old": r["current_chunk_id"],
                            "new": new_id,
                            "verdict": r.get("verdict", ""),
                        }
                    )
                else:
                    hallucinations.append(
                        {"id": r["id"], "fake_chunk_id": new_id, "agent": i}
                    )

    print(f"\nCorrections valides: {len(valid_corrections)}")
    print(f"HALLUCINATIONS DETECTEES: {len(hallucinations)}")

    if hallucinations:
        print("\n!!! ALERTE: CHUNK_IDs INVENTES !!!")
        for h in hallucinations[:30]:
            print(f"  - {h['id']}: {h['fake_chunk_id']} (Agent {h['agent']})")
        if len(hallucinations) > 30:
            print(f"  ... et {len(hallucinations) - 30} autres")
    else:
        print("\n✓ Aucune hallucination - tous les chunk_ids existent!")

    return hallucinations, valid_corrections


if __name__ == "__main__":
    hallucinations, corrections = main()
