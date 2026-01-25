#!/usr/bin/env python3
"""Migrate question IDs to URN-like scheme for multi-corpus support.

New ID format: {CORPUS}:{SOURCE}:{CAT}:{SEQ}:{HASH}

CORPUS: ffe, fide, adversarial, synthetic
SOURCE: annales, human, squad-adv, llm-gen
CAT: rules, rating, clubs, youth, women, access, admin, regional, open, tournament
SEQ: 001-999
HASH: 8-char SHA256 hash for global uniqueness
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path


def generate_hash(content: str) -> str:
    """Generate 8-char hash for uniqueness."""
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def generate_new_id(corpus: str, source: str, category: str, seq: int, question: str) -> str:
    """Generate new URN-like question ID."""
    base = f"{corpus}:{source}:{category}:{seq:03d}"
    hash_input = f"{base}:{question}"
    hash_suffix = generate_hash(hash_input)
    return f"{base}:{hash_suffix}"


def map_uv_to_category(uv_code: str) -> str:
    """Map UV code from annales to category."""
    mapping = {
        "UVR": "rules",      # Règles du jeu
        "UVC": "clubs",      # Compétitions/Clubs
        "UVO": "open",       # Open
        "UVT": "tournament", # Tournoi
    }
    return mapping.get(uv_code, "general")


def map_old_id_to_new(old_id: str, question: str, category: str) -> tuple[str, str, str]:
    """Map old ID to new (corpus, source, category)."""

    # Annales questions
    if old_id.startswith("FR-ANN-"):
        parts = old_id.split("-")
        uv_code = parts[2] if len(parts) >= 3 else "UVC"
        return "ffe", "annales", map_uv_to_category(uv_code)

    # Human-created questions - map by prefix or category
    prefix_mapping = {
        "FR-ELO": ("ffe", "human", "rating"),
        "FR-E02": ("ffe", "human", "rating"),
        "FR-J01": ("ffe", "human", "youth"),
        "FR-J02": ("ffe", "human", "youth"),
        "FR-J03": ("ffe", "human", "youth"),
        "FR-F01": ("ffe", "human", "women"),
        "FR-F02": ("ffe", "human", "women"),
        "FR-H01": ("ffe", "human", "access"),
        "FR-H02": ("ffe", "human", "access"),
        "FR-A01": ("ffe", "human", "clubs"),
        "FR-A03": ("ffe", "human", "clubs"),
        "FR-MED": ("ffe", "human", "admin"),
        "FR-FIN": ("ffe", "human", "admin"),
        "FR-STAT": ("ffe", "human", "admin"),
        "FR-RI": ("ffe", "human", "admin"),
        "FR-DEL": ("ffe", "human", "admin"),
        "FR-N6BDR": ("ffe", "human", "regional"),
        "FR-IJBDR": ("ffe", "human", "regional"),
        "FR-N4PACA": ("ffe", "human", "regional"),
        "FR-REGPACA": ("ffe", "human", "regional"),
        "FR-COMP": ("ffe", "human", "clubs"),
        "FR-NOYAU": ("ffe", "human", "clubs"),
    }

    for prefix, mapping in prefix_mapping.items():
        if old_id.startswith(prefix):
            return mapping

    # Fallback based on category field
    category_mapping = {
        "classement": "rating",
        "jeunes": "youth",
        "feminin": "women",
        "handicap": "access",
        "medical": "admin",
        "administratif": "admin",
        "competitions": "clubs",
        "interclubs": "clubs",
        "regional": "regional",
        "regles_jeu": "rules",
    }
    cat = category_mapping.get(category, "general")
    return "ffe", "human", cat


def main() -> None:
    gs_path = Path("tests/data/gold_standard_annales_fr_v7.json")

    with open(gs_path, "r", encoding="utf-8") as f:
        gs = json.load(f)

    # Track sequences per (corpus, source, category)
    sequences: dict[tuple[str, str, str], int] = defaultdict(int)

    # Migration mapping for traceability
    migration_map: list[dict] = []

    # Stats
    stats = defaultdict(int)

    for q in gs["questions"]:
        old_id = q["id"]
        question_text = q["question"]
        category = q.get("category", "")

        # Determine new ID components
        corpus, source, cat = map_old_id_to_new(old_id, question_text, category)

        # Increment sequence
        key = (corpus, source, cat)
        sequences[key] += 1
        seq = sequences[key]

        # Generate new ID
        new_id = generate_new_id(corpus, source, cat, seq, question_text)

        # Track migration
        migration_map.append({
            "old_id": old_id,
            "new_id": new_id,
            "corpus": corpus,
            "source": source,
            "category": cat,
        })

        # Update question
        q["id"] = new_id
        q["legacy_id"] = old_id  # Keep for traceability

        # Stats
        stats[f"{corpus}:{source}:{cat}"] += 1

    # Update version
    gs["version"] = "7.3.0"

    # Add schema info
    gs["id_schema"] = {
        "format": "{corpus}:{source}:{category}:{sequence}:{hash}",
        "corpus_values": ["ffe", "fide", "adversarial", "synthetic"],
        "source_values": ["annales", "human", "squad-adv", "llm-gen"],
        "category_values": ["rules", "rating", "clubs", "youth", "women", "access", "admin", "regional", "open", "tournament", "general"],
        "hash_algorithm": "sha256[:8]",
        "version": "1.0.0"
    }

    # Save updated gold standard
    with open(gs_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2, ensure_ascii=False)

    # Save migration map
    migration_path = Path("tests/data/id_migration_map.json")
    with open(migration_path, "w", encoding="utf-8") as f:
        json.dump({
            "migration_date": "2025-01-25",
            "old_version": "7.2.1",
            "new_version": "7.3.0",
            "total_migrated": len(migration_map),
            "mappings": migration_map
        }, f, indent=2, ensure_ascii=False)

    # Print summary
    print("=== Migration Complete ===")
    print(f"Total questions migrated: {len(migration_map)}")
    print(f"Version: 7.2.1 -> 7.3.0")
    print()
    print("Distribution by namespace:")
    for key in sorted(stats.keys()):
        print(f"  {key}: {stats[key]}")
    print()
    print(f"Migration map saved: {migration_path}")

    # Show examples
    print()
    print("Examples:")
    for m in migration_map[:3]:
        print(f"  {m['old_id']} -> {m['new_id']}")
    print("  ...")
    for m in migration_map[-3:]:
        print(f"  {m['old_id']} -> {m['new_id']}")


if __name__ == "__main__":
    main()
