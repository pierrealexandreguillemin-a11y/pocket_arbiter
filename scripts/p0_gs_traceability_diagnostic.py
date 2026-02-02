"""
Phase 0 - GS v8.0 Traceability Chain Diagnostic
READ-ONLY analysis: checks M2, M2b, M3, M4, M5, CB-01, CB-02, F-01, F-04, M-01/M-02, CB-09
"""

from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────
GS_PATH = Path(r"C:\Dev\pocket_arbiter\tests\data\gold_standard_annales_fr_v7.json")
REF_PATH = Path(
    r"C:\Dev\pocket_arbiter\data\evaluation\annales\docling_mcq_reference.json"
)
CHUNKS_PATH = Path(r"C:\Dev\pocket_arbiter\corpus\processed\chunks_mode_b_fr.json")


# ── Load data ──────────────────────────────────────────────────────────
def load_json(p: Path) -> Any:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


print("=" * 90)
print("  PHASE 0 — GS v8.0 TRACEABILITY CHAIN DIAGNOSTIC")
print("=" * 90)

gs_data = load_json(GS_PATH)
ref_data = load_json(REF_PATH)
chunks_data = load_json(CHUNKS_PATH)

questions: list[dict] = gs_data["questions"]
print(
    f"\nLoaded GS: {len(questions)} questions  (version {gs_data.get('version', '?')})"
)
print(f"Loaded MCQ reference: {len(ref_data)} entries")

# Build chunk lookup
chunk_map: dict[str, dict] = {}
for c in chunks_data.get("chunks", []):
    chunk_map[c["id"]] = c
print(f"Loaded chunks corpus: {len(chunk_map)} chunks")

# Build MCQ ref lookup: (session, uv, question_num) -> ref entry
ref_map: dict[tuple[str, str, int], dict] = {}
for entry in ref_data:
    key = (entry["session"], entry["uv"], entry["question_num"])
    ref_map[key] = entry

print(f"MCQ ref index: {len(ref_map)} unique (session, uv, qnum) keys")

# ── Check definitions ──────────────────────────────────────────────────
# Each check returns (pass: bool, detail: str)


def check_m2(q: dict) -> tuple[bool, str]:
    """M2: mcq_answer matches docling_mcq_reference correct_letter."""
    meta = q.get("metadata", {})
    src = meta.get("annales_source")
    if not src:
        return True, "SKIP (no annales_source)"
    key = (src["session"], src["uv"], src["question_num"])
    ref = ref_map.get(key)
    if ref is None:
        return False, f"NO REF for {key}"
    gs_answer = meta.get("mcq_answer", "")
    ref_answer = ref.get("correct_letter", "")
    if gs_answer == ref_answer:
        return True, f"OK ({gs_answer})"
    return False, f"GS={gs_answer} vs REF={ref_answer}"


def check_m2b(q: dict) -> tuple[bool, str]:
    """M2b: mcq_answer letter(s) present in choices keys."""
    meta = q.get("metadata", {})
    mcq = meta.get("mcq_answer")
    choices = meta.get("choices", {})
    if not mcq:
        return True, "SKIP (no mcq_answer)"
    if not choices:
        return True, "SKIP (no choices dict)"
    missing = []
    for letter in mcq:
        if letter not in choices:
            missing.append(letter)
    if missing:
        return False, f"letter(s) {missing} not in choices keys {list(choices.keys())}"
    return True, "OK"


def check_m3(q: dict) -> tuple[bool, str]:
    """M3: expected_answer == choices[mcq_answer] for single-letter; note multi."""
    meta = q.get("metadata", {})
    mcq = meta.get("mcq_answer", "")
    choices = meta.get("choices", {})
    expected = q.get("expected_answer", "")
    if not mcq:
        return True, "SKIP (no mcq_answer)"
    if len(mcq) > 1:
        return True, f"MULTI-LETTER ({mcq}) — skipped equality check"
    if not choices:
        return True, "SKIP (no choices)"
    choice_text = choices.get(mcq, "")
    if not choice_text:
        return False, f"choices[{mcq}] is empty/missing"
    if expected.strip() == choice_text.strip():
        return True, "OK (exact match)"
    # Check if expected_answer looks like an article reference
    if ("Art" in expected or "Article" in expected) and len(expected) < 80:
        return (
            False,
            f"ARTICLE-REF in expected_answer: '{expected}' vs choice '{choice_text[:60]}...'",
        )
    return (
        False,
        f"MISMATCH: expected='{expected[:60]}...' vs choice='{choice_text[:60]}...'",
    )


def check_m4(q: dict) -> tuple[bool, str]:
    """M4: answer_explanation present and not null."""
    meta = q.get("metadata", {})
    expl = meta.get("answer_explanation")
    if expl is None:
        return False, "answer_explanation is null"
    if not str(expl).strip():
        return False, "answer_explanation is empty string"
    return True, f"OK ({len(str(expl))} chars)"


def check_m5(q: dict) -> tuple[bool, str]:
    """M5: article_reference present; compare with ref for annales questions."""
    meta = q.get("metadata", {})
    gs_ref = meta.get("article_reference", "")
    src = meta.get("annales_source")
    if not gs_ref:
        return False, "article_reference missing/empty"
    if not src:
        return True, f"OK (non-annales): {gs_ref[:40]}"
    key = (src["session"], src["uv"], src["question_num"])
    ref_entry = ref_map.get(key)
    if ref_entry is None:
        return True, f"OK (no ref entry): {gs_ref[:40]}"
    ref_art = ref_entry.get("article_reference", "")
    if not ref_art:
        return True, f"OK (ref has no article_reference): {gs_ref[:40]}"
    if gs_ref.strip() == ref_art.strip():
        return True, f"OK (exact match): {gs_ref[:40]}"
    return False, f"DIFF: GS='{gs_ref[:50]}' vs REF='{ref_art[:50]}'"


def check_cb01(q: dict) -> tuple[bool, str]:
    """CB-01: expected_answer text appears in the chunk text."""
    chunk_id = q.get("expected_chunk_id", "")
    expected = q.get("expected_answer", "")
    if not chunk_id or not expected:
        return True, "SKIP (no chunk_id or answer)"
    chunk = chunk_map.get(chunk_id)
    if chunk is None:
        return False, f"CHUNK NOT FOUND: {chunk_id}"
    chunk_text = chunk.get("text", "")
    if expected.lower() in chunk_text.lower():
        return True, "OK (substring match)"
    # Try first 40 chars as partial match
    if len(expected) > 10 and expected[:40].lower() in chunk_text.lower():
        return (
            False,
            f"PARTIAL (first 40 chars found, full not found): answer='{expected[:50]}...'",
        )
    return False, f"NOT FOUND in chunk: answer='{expected[:60]}...' chunk={chunk_id}"


def check_cb02(q: dict) -> tuple[bool, str]:
    """CB-02: expected_chunk_id exists in corpus."""
    chunk_id = q.get("expected_chunk_id", "")
    if not chunk_id:
        return False, "expected_chunk_id missing"
    if chunk_id in chunk_map:
        return True, "OK"
    return False, f"MISSING chunk: {chunk_id}"


def check_f01(q: dict) -> tuple[bool, str]:
    """F-01: question ends with '?'."""
    question = q.get("question", "").strip()
    if question.endswith("?"):
        return True, "OK"
    return False, f"Does not end with '?': ...'{question[-30:]}'"


def check_f04(q: dict) -> tuple[bool, str]:
    """F-04: len(expected_answer) > 5."""
    expected = q.get("expected_answer", "")
    length = len(expected)
    if length > 5:
        return True, f"OK ({length} chars)"
    return False, f"TOO SHORT ({length} chars): '{expected}'"


def check_m01m02(q: dict) -> tuple[bool, str]:
    """M-01/M-02: difficulty present and in [0, 1]."""
    meta = q.get("metadata", {})
    diff = meta.get("difficulty")
    if diff is None:
        return False, "difficulty missing"
    try:
        val = float(diff)
    except (ValueError, TypeError):
        return False, f"difficulty not numeric: {diff}"
    if 0.0 <= val <= 1.0:
        return True, f"OK ({val})"
    return False, f"OUT OF RANGE: {val}"


def check_cb09(q: dict) -> tuple[bool, str]:
    """CB-09: requires_context==true => requires_context_reason present."""
    meta = q.get("metadata", {})
    rc = meta.get("requires_context", False)
    if not rc:
        return True, "SKIP (requires_context is false/absent)"
    reason = meta.get("requires_context_reason")
    if reason and str(reason).strip():
        return True, f"OK: {str(reason)[:40]}"
    return False, "requires_context=true but NO reason"


# ── All checks ─────────────────────────────────────────────────────────
ALL_CHECKS = {
    "M2:mcq_answer_correct": check_m2,
    "M2b:mcq_in_choices": check_m2b,
    "M3:expected_eq_choice": check_m3,
    "M4:answer_explanation": check_m4,
    "M5:article_reference": check_m5,
    "CB-01:answer_in_chunk": check_cb01,
    "CB-02:chunk_exists": check_cb02,
    "F-01:ends_with_?": check_f01,
    "F-04:answer_gt5": check_f04,
    "M-01/02:difficulty": check_m01m02,
    "CB-09:context_reason": check_cb09,
}

# ── Run all checks ─────────────────────────────────────────────────────
# Results: per-question {check_name: (pass, detail)}
results: list[dict[str, tuple[bool, str]]] = []
# Per-batch tracking
batch_results: dict[str, dict[str, dict[str, int]]] = defaultdict(
    lambda: defaultdict(lambda: {"pass": 0, "fail": 0, "skip": 0})
)
# Global counts
global_counts: dict[str, dict[str, int]] = defaultdict(
    lambda: {"pass": 0, "fail": 0, "skip": 0}
)

for q in questions:
    q_results: dict[str, tuple[bool, str]] = {}
    # Determine batch (session) from annales_source or id
    meta = q.get("metadata", {})
    src = meta.get("annales_source")
    if src:
        batch = f"{src['session']}:{src['uv']}"
    else:
        # Derive from id pattern ffe:annales:xxx:nnn:hash
        parts = q.get("id", "").split(":")
        batch = f"other:{parts[2] if len(parts) > 2 else 'unknown'}"

    for check_name, check_fn in ALL_CHECKS.items():
        passed, detail = check_fn(q)
        q_results[check_name] = (passed, detail)
        if "SKIP" in detail:
            cat = "skip"
        elif passed:
            cat = "pass"
        else:
            cat = "fail"
        batch_results[batch][check_name][cat] += 1
        global_counts[check_name][cat] += 1

    results.append(q_results)

# ── OUTPUT ─────────────────────────────────────────────────────────────

# 1. Per-batch summary
print("\n" + "=" * 90)
print("  SECTION 1: PER-BATCH (SESSION:UV) SUMMARY")
print("=" * 90)

for batch in sorted(batch_results.keys()):
    checks = batch_results[batch]
    total_qs = sum(next(iter(checks.values())).values()) if checks else 0
    print(f"\n{'─' * 70}")
    print(f"  Batch: {batch}  ({total_qs} questions)")
    print(f"{'─' * 70}")
    print(f"  {'Check':<28s} {'PASS':>6s} {'FAIL':>6s} {'SKIP':>6s}  {'Rate':>7s}")
    print(f"  {'─'*28} {'─'*6} {'─'*6} {'─'*6}  {'─'*7}")
    for check_name in ALL_CHECKS:
        c = checks[check_name]
        p, f_, s = c["pass"], c["fail"], c["skip"]
        total_applicable = p + f_
        rate = f"{100*p/total_applicable:.1f}%" if total_applicable > 0 else "N/A"
        flag = " <<" if f_ > 0 else ""
        print(f"  {check_name:<28s} {p:>6d} {f_:>6d} {s:>6d}  {rate:>7s}{flag}")

# 2. Per-question FAIL details
print("\n" + "=" * 90)
print("  SECTION 2: PER-QUESTION FAILURE DETAILS")
print("=" * 90)

fail_count = 0
for i, q in enumerate(questions):
    q_results = results[i]
    failures = {k: v for k, v in q_results.items() if not v[0] and "SKIP" not in v[1]}
    if not failures:
        continue
    fail_count += 1
    qid = q["id"]
    short_q = q["question"][:70].replace("\n", " ")
    print(f"\n  [{i+1:03d}] {qid}")
    print(f"        Q: {short_q}...")
    for check_name, (_, detail) in failures.items():
        print(f"        FAIL {check_name}: {detail}")

print(f"\n  Total questions with at least 1 failure: {fail_count} / {len(questions)}")

# 3. Global summary
print("\n" + "=" * 90)
print("  SECTION 3: GLOBAL SUMMARY")
print("=" * 90)

print(
    f"\n  {'Check':<28s} {'PASS':>6s} {'FAIL':>6s} {'SKIP':>6s}  {'Rate':>7s}  {'Status':>8s}"
)
print(f"  {'═'*28} {'═'*6} {'═'*6} {'═'*6}  {'═'*7}  {'═'*8}")

total_checks = 0
total_pass = 0
total_fail = 0
for check_name in ALL_CHECKS:
    c = global_counts[check_name]
    p, f_, s = c["pass"], c["fail"], c["skip"]
    total_applicable = p + f_
    rate = f"{100*p/total_applicable:.1f}%" if total_applicable > 0 else "N/A"
    status = "PASS" if f_ == 0 else "FAIL"
    print(f"  {check_name:<28s} {p:>6d} {f_:>6d} {s:>6d}  {rate:>7s}  {status:>8s}")
    total_checks += total_applicable
    total_pass += p
    total_fail += f_

print(f"\n  {'─' * 70}")
overall_rate = 100 * total_pass / total_checks if total_checks > 0 else 0
print(
    f"  OVERALL: {total_pass} pass / {total_fail} fail / {total_checks} applicable checks"
)
print(f"  OVERALL RATE: {overall_rate:.2f}%")

# 4. Quick cross-check counters
print("\n" + "=" * 90)
print("  SECTION 4: ADDITIONAL STATISTICS")
print("=" * 90)

annales_count = sum(1 for q in questions if q.get("metadata", {}).get("annales_source"))
multi_letter_count = sum(
    1 for q in questions if len(q.get("metadata", {}).get("mcq_answer", "")) > 1
)
no_choices_count = sum(1 for q in questions if not q.get("metadata", {}).get("choices"))
requires_ctx_count = sum(
    1 for q in questions if q.get("metadata", {}).get("requires_context", False)
)
null_explanation_count = sum(
    1 for q in questions if q.get("metadata", {}).get("answer_explanation") is None
)

print(f"\n  Questions with annales_source:      {annales_count}")
print(f"  Multi-letter mcq_answer:            {multi_letter_count}")
print(f"  Questions with empty choices dict:   {no_choices_count}")
print(f"  Questions with requires_context:     {requires_ctx_count}")
print(f"  Questions with null explanation:      {null_explanation_count}")
print(f"  Total GS questions:                  {len(questions)}")

# 5. M2 mismatches detail table
print("\n" + "=" * 90)
print("  SECTION 5: M2 MISMATCH DETAIL (GS mcq_answer != REF correct_letter)")
print("=" * 90)

m2_mismatches = []
for i, q in enumerate(questions):
    passed, detail = results[i]["M2:mcq_answer_correct"]
    if not passed and "SKIP" not in detail:
        src = q["metadata"].get("annales_source", {})
        m2_mismatches.append((i, q["id"], src, detail))

if m2_mismatches:
    print(f"\n  Found {len(m2_mismatches)} M2 mismatches:\n")
    for idx, qid, src, detail in m2_mismatches:
        session = src.get("session", "?")
        uv = src.get("uv", "?")
        qnum = src.get("question_num", "?")
        print(f"  [{idx+1:03d}] {qid}  ({session}/{uv}/Q{qnum})  => {detail}")
else:
    print("\n  No M2 mismatches found. All mcq_answers match reference.")

# 6. CB-01 failures detail
print("\n" + "=" * 90)
print("  SECTION 6: CB-01 FAILURES (answer NOT in chunk)")
print("=" * 90)

cb01_fails = []
for i, q in enumerate(questions):
    passed, detail = results[i]["CB-01:answer_in_chunk"]
    if not passed and "SKIP" not in detail:
        cb01_fails.append((i, q["id"], q.get("expected_chunk_id", ""), detail))

print(f"\n  Found {len(cb01_fails)} CB-01 failures:\n")
for idx, qid, cid, detail in cb01_fails[:50]:  # Show first 50
    print(f"  [{idx+1:03d}] {qid}")
    print(f"        chunk: {cid}")
    print(f"        {detail}")
if len(cb01_fails) > 50:
    print(f"\n  ... and {len(cb01_fails) - 50} more")

print("\n" + "=" * 90)
print("  END OF DIAGNOSTIC")
print("=" * 90)
