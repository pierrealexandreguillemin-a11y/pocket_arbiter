#!/usr/bin/env python3
"""
Phase B Prep: Identify and stratify uncovered chunks for GS generation.

Compares chunks_mode_b_fr.json against gs_scratch_v1_step1.json to find
chunks with no question (answerable or unanswerable), then stratifies
them by priority tier per GS_CORRECTION_PLAN_V2.md Section 5.2.

Coverage definition (aligned with plan): a chunk is "covered" if ANY
question in the GS references it (answerable or unanswerable).
Phase B targets the 1387 uncovered chunks to reach >=80% coverage.

ISO Reference:
- ISO 29119-3: Test data coverage analysis
- ISO 42001 A.6.2.2: Provenance tracking

Usage:
    python analyze_uncovered_chunks.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_json(path: Path) -> dict | list:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_doc_prefix(chunk_id: str) -> str:
    """
    Extract the document prefix from a chunk_id.

    Examples:
        LA-octobre2025.pdf-p009-parent014-child00 -> LA-octobre2025
        2018_Reglement_Disciplinaire20180422.pdf-p001-parent000-child00
            -> 2018_Reglement_Disciplinaire20180422
        C01_CoupeDeFrance2025.pdf-p003-parent005-child01
            -> C01_CoupeDeFrance2025
    """
    # Remove .pdf and everything after it that starts with -p
    match = re.match(r"^(.+?)\.pdf", chunk_id)
    if match:
        return match.group(1)
    # Fallback: split on first -p followed by digits (page marker)
    match2 = re.match(r"^(.+?)-p\d", chunk_id)
    if match2:
        return match2.group(1)
    return chunk_id


def classify_priority(doc_prefix: str) -> tuple[int, str]:
    """
    Classify a document prefix into a priority tier.

    Tiers aligned with GS_CORRECTION_PLAN_V2.md Section 5.2.

    Returns:
        (priority_number, tier_description)
    """
    # Priority 1: LA-octobre2025 (core arbitrage rules)
    if doc_prefix.startswith("LA-octobre2025") or doc_prefix.startswith(
        "LA_octobre2025"
    ):
        return 1, "LA-octobre2025 (Lois arbitrage)"

    # Priority 2: C01-C04 (Coupes)
    if re.match(r"^C0[1-4]", doc_prefix):
        return 2, "C01-C04 (Coupes)"

    # Priority 3: A02, F01, J03 (Championnats)
    if re.match(r"^(A02|F01|J03)", doc_prefix):
        return 3, "A02/F01/J03 (Championnats)"

    # Priority 4: 2025_RI, Contrat, Statuts (Admin)
    if re.match(
        r"^(2025_R[eè]glement_Int|2024_Statuts|Contrat)",
        doc_prefix,
        re.IGNORECASE,
    ):
        return 4, "RI/Contrat/Statuts (Admin)"

    # Priority 5: Everything else
    return 5, "Other"


def main() -> None:
    """Run the uncovered chunk analysis."""
    # ── Load chunks ──
    chunks_path = PROJECT_ROOT / "corpus" / "processed" / "chunks_mode_b_fr.json"
    print(f"Loading chunks from {chunks_path.relative_to(PROJECT_ROOT)}...")
    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", chunks_data)
    if isinstance(chunks, dict):
        chunks = list(chunks.values())
    print(f"  Total chunks: {len(chunks)}")

    all_chunk_ids = {c["id"] for c in chunks}
    chunk_index = {c["id"]: c for c in chunks}

    # ── Load GS ──
    gs_path = PROJECT_ROOT / "tests" / "data" / "gs_scratch_v1_step1.json"
    print(f"\nLoading GS from {gs_path.relative_to(PROJECT_ROOT)}...")
    gs_data = load_json(gs_path)
    questions = gs_data["questions"]
    print(f"  Total questions: {len(questions)}")

    # Extract chunk_ids from ALL questions (answerable + unanswerable)
    # This aligns with plan definition: "covered = has at least one question"
    all_gs_chunk_ids: set[str] = set()
    answerable_chunk_ids: set[str] = set()
    answerable_count = 0
    unanswerable_count = 0
    for q in questions:
        chunk_id = q.get("provenance", {}).get("chunk_id", "")
        is_impossible = q.get("content", {}).get("is_impossible", True)
        if chunk_id:
            all_gs_chunk_ids.add(chunk_id)
        if not is_impossible:
            answerable_count += 1
            if chunk_id:
                answerable_chunk_ids.add(chunk_id)
        else:
            unanswerable_count += 1

    covered_chunk_ids = all_gs_chunk_ids & all_chunk_ids
    answerable_covered = answerable_chunk_ids & all_chunk_ids

    print(f"  Answerable questions: {answerable_count}")
    print(f"  Unanswerable questions: {unanswerable_count}")
    print(f"  Unique chunk_ids (answerable only): {len(answerable_covered)}")
    print(f"  Unique chunk_ids (all GS):          {len(covered_chunk_ids)}")

    # Check for chunk_ids in GS that don't exist in chunks file
    orphan_ids = all_gs_chunk_ids - all_chunk_ids
    if orphan_ids:
        print(
            f"\n  WARNING: {len(orphan_ids)} chunk_ids in GS not found in chunks file:"
        )
        for oid in sorted(orphan_ids)[:5]:
            print(f"    - {oid}")
        if len(orphan_ids) > 5:
            print(f"    ... and {len(orphan_ids) - 5} more")

    # ── Compute uncovered ──
    uncovered_ids = all_chunk_ids - covered_chunk_ids
    coverage_pct = len(covered_chunk_ids) / len(all_chunk_ids) * 100

    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY")
    print("=" * 70)
    print(f"  Total chunks in corpus:     {len(all_chunk_ids):>6}")
    print(f"  Covered chunks (with Q):    {len(covered_chunk_ids & all_chunk_ids):>6}")
    print(f"  Uncovered chunks (no Q):    {len(uncovered_ids):>6}")
    print(f"  Coverage:                   {coverage_pct:>5.1f}%")
    print(
        f"  Target (>=80%):             {int(len(all_chunk_ids) * 0.80):>6} chunks needed"
    )
    print(
        f"  Gap to 80%:                 {max(0, int(len(all_chunk_ids) * 0.80) - len(covered_chunk_ids & all_chunk_ids)):>6} more chunks"
    )

    # ── Stratify by document prefix ──
    doc_uncovered: Counter[str] = Counter()
    doc_total: Counter[str] = Counter()
    doc_covered: Counter[str] = Counter()

    for c in chunks:
        prefix = extract_doc_prefix(c["id"])
        doc_total[prefix] += 1
        if c["id"] in uncovered_ids:
            doc_uncovered[prefix] += 1
        if c["id"] in covered_chunk_ids:
            doc_covered[prefix] += 1

    # ── Stratify by priority tier ──
    tier_chunks: defaultdict[int, list[str]] = defaultdict(list)
    tier_labels: dict[int, str] = {}

    for cid in uncovered_ids:
        prefix = extract_doc_prefix(cid)
        prio, label = classify_priority(prefix)
        tier_chunks[prio].append(cid)
        tier_labels[prio] = label

    # Also compute token stats per tier
    tier_tokens: defaultdict[int, list[int]] = defaultdict(list)
    tier_small_chunks: defaultdict[int, int] = defaultdict(int)  # <50 tokens
    for prio, cids in tier_chunks.items():
        for cid in cids:
            chunk = chunk_index.get(cid)
            if chunk:
                tokens = chunk.get("tokens", 0)
                tier_tokens[prio].append(tokens)
                if tokens < 50:
                    tier_small_chunks[prio] += 1

    print("\n" + "=" * 70)
    print("UNCOVERED CHUNKS BY PRIORITY TIER")
    print("=" * 70)
    print(f"{'Tier':>6} {'Label':<35} {'Count':>7} {'<50tok':>7} {'Med.tok':>8}")
    print("-" * 70)

    total_uncov = 0
    total_small = 0
    for prio in sorted(tier_chunks.keys()):
        count = len(tier_chunks[prio])
        small = tier_small_chunks[prio]
        toks = tier_tokens[prio]
        median_tok = sorted(toks)[len(toks) // 2] if toks else 0
        label = tier_labels.get(prio, "Unknown")
        print(f"  P{prio:<4} {label:<35} {count:>7} {small:>7} {median_tok:>8}")
        total_uncov += count
        total_small += small
    print("-" * 70)
    print(f"  {'TOTAL':<40} {total_uncov:>7} {total_small:>7}")
    print(
        f"\n  Chunks <50 tokens (likely headers/TOC): {total_small} "
        f"({total_small/total_uncov*100:.1f}% of uncovered)"
    )
    print(f"  Substantive uncovered (>=50 tokens):     {total_uncov - total_small}")

    # ── Top 10 documents by uncovered chunk count ──
    print("\n" + "=" * 70)
    print("TOP 15 DOCUMENTS BY UNCOVERED CHUNK COUNT")
    print("=" * 70)
    print(f"{'Doc prefix':<50} {'Uncov':>6} {'Total':>6} {'Cov%':>6} {'Tier':>5}")
    print("-" * 75)

    for doc, uncov_count in doc_uncovered.most_common(15):
        total = doc_total[doc]
        covered = doc_covered[doc]
        cov_pct = covered / total * 100 if total else 0
        prio, _ = classify_priority(doc)
        print(f"  {doc:<48} {uncov_count:>6} {total:>6} {cov_pct:>5.1f}% P{prio}")

    # ── Documents with 0% coverage (fully uncovered) ──
    zero_coverage_docs = [
        (doc, doc_total[doc]) for doc in doc_total if doc_covered[doc] == 0
    ]
    zero_coverage_docs.sort(key=lambda x: -x[1])

    print("\n" + "=" * 70)
    print(f"DOCUMENTS WITH 0% COVERAGE ({len(zero_coverage_docs)} documents)")
    print("=" * 70)
    print(f"{'Doc prefix':<50} {'Chunks':>7} {'Tier':>5}")
    print("-" * 65)
    for doc, count in zero_coverage_docs:
        prio, _ = classify_priority(doc)
        print(f"  {doc:<48} {count:>7} P{prio}")

    # ── Detailed tier breakdown by document ──
    print("\n" + "=" * 70)
    print("DETAILED TIER BREAKDOWN (uncovered chunks per doc per tier)")
    print("=" * 70)

    for prio in sorted(tier_chunks.keys()):
        label = tier_labels.get(prio, "Unknown")
        cids = tier_chunks[prio]

        # Group by document
        doc_counts: Counter[str] = Counter()
        for cid in cids:
            doc_counts[extract_doc_prefix(cid)] += 1

        print(f"\n  --- P{prio}: {label} ({len(cids)} chunks) ---")
        for doc, cnt in doc_counts.most_common():
            total = doc_total[doc]
            covered = doc_covered[doc]
            print(
                f"    {doc:<46} uncov={cnt:>4}  total={total:>4}  "
                f"covered={covered:>4} ({covered/total*100:.0f}%)"
            )

    # ── Estimated generation output ──
    substantive = total_uncov - total_small
    print("\n" + "=" * 70)
    print("GENERATION ESTIMATES (Phase B)")
    print("=" * 70)
    print(f"  Uncovered chunks:           {total_uncov}")
    print(f"  Substantive (>=50 tokens):  {substantive}")
    print(f"  Estimated Q at 0.6/chunk:   {int(substantive * 0.6)}")
    print(f"  Estimated Q at 0.8/chunk:   {int(substantive * 0.8)}")
    print(f"  Current answerable:         {answerable_count}")
    print(f"  Projected total (0.6):      {answerable_count + int(substantive * 0.6)}")
    print(f"  Projected total (0.8):      {answerable_count + int(substantive * 0.8)}")
    target_80_chunks = int(len(all_chunk_ids) * 0.80)
    print(
        f"  Chunks needed for 80%:      {target_80_chunks - len(covered_chunk_ids & all_chunk_ids)} more"
    )


if __name__ == "__main__":
    main()
