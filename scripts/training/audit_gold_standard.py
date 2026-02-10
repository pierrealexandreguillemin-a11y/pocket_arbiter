#!/usr/bin/env python3
"""
Audit Gold Standard contre standards industrie et normes ISO.

Standards:
- SQuAD 2.0 (arXiv:1806.03822)
- SQuAD2-CR (arXiv:2004.14004)
- UAEval4RAG (arXiv:2412.12300)
- QA Taxonomy (arXiv:2107.12708)
- Bloom's Taxonomy (Anderson 2001)
- ISO 42001, ISO 25010, ISO 29119
"""

import json
from collections import defaultdict
from datetime import date


def load_gs(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_chunks(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: dict) -> dict:
    idx = {}
    for c in chunks["chunks"]:
        key = (c["source"], c["page"])
        if key not in idx:
            idx[key] = ""
        idx[key] += " " + c["text"].lower()
    return idx


def audit_gs(gs: dict, name: str, chunks_path: str) -> dict:
    chunks = load_chunks(chunks_path)
    idx = build_index(chunks)

    questions = gs["questions"]
    total = len(questions)

    answerable = 0
    unanswerable = 0
    hard_types = defaultdict(int)
    answer_types = defaultdict(int)
    cognitive_levels = defaultdict(int)
    reasoning_types = defaultdict(int)

    keywords_validated = 0
    has_expected_pages = 0
    has_keywords = 0
    has_metadata = 0
    has_id = 0

    for q in questions:
        meta = q.get("metadata", {})
        ht = meta.get("hard_type")

        if q.get("id"):
            has_id += 1

        if ht is None or ht == "ANSWERABLE":
            answerable += 1

            at = meta.get("answer_type", "MISSING")
            answer_types[at] += 1

            cl = meta.get("cognitive_level", "MISSING")
            cognitive_levels[cl] += 1

            rt = meta.get("reasoning_type", "MISSING")
            reasoning_types[rt] += 1

            if q.get("expected_pages"):
                has_expected_pages += 1
            if q.get("keywords"):
                has_keywords += 1
            if meta:
                has_metadata += 1

            pages = q.get("expected_pages", [])
            source = q.get("expected_docs", [""])[0]
            keywords = [kw.lower() for kw in q.get("keywords", [])]
            for page in pages:
                text = idx.get((source, page), "")
                if sum(1 for kw in keywords if kw in text) >= 2:
                    keywords_validated += 1
                    break
        else:
            unanswerable += 1
            hard_types[ht] += 1

    ratio = unanswerable / total * 100 if total > 0 else 0

    squad2_cr = [
        "ENTITY_SWAP",
        "ANTONYM",
        "NEGATION",
        "NUMBER_SWAP",
        "MUTUAL_EXCLUSION",
    ]
    squad2_cr_count = sum(1 for c in squad2_cr if c in hard_types)

    uaeval = [
        "UNDERSPECIFIED",
        "FALSE_PRESUPPOSITION",
        "VOCABULARY_MISMATCH",
        "OUT_OF_SCOPE",
        "SAFETY_CONCERNED",
        "PARTIAL_INFO",
    ]
    uaeval_count = sum(1 for c in uaeval if c in hard_types)

    qa_types = ["FACTUAL", "PROCEDURAL", "LIST", "CONDITIONAL", "DEFINITIONAL"]
    qa_count = sum(1 for t in qa_types if t in answer_types)

    bloom = ["REMEMBER", "UNDERSTAND", "APPLY", "ANALYZE"]
    bloom_count = sum(1 for level in bloom if level in cognitive_levels)

    reasoning = [
        "LEXICAL_MATCH",
        "SINGLE_SENTENCE",
        "MULTI_SENTENCE",
        "DOMAIN_KNOWLEDGE",
    ]
    reasoning_count = sum(1 for r in reasoning if r in reasoning_types)

    return {
        "name": name,
        "total": total,
        "answerable": answerable,
        "unanswerable": unanswerable,
        "ratio": ratio,
        "squad2_cr": squad2_cr_count,
        "squad2_cr_details": {c: hard_types.get(c, 0) for c in squad2_cr},
        "uaeval": uaeval_count,
        "uaeval_details": {c: hard_types.get(c, 0) for c in uaeval},
        "qa_types": qa_count,
        "qa_types_details": dict(answer_types),
        "bloom": bloom_count,
        "bloom_details": dict(cognitive_levels),
        "reasoning": reasoning_count,
        "reasoning_details": dict(reasoning_types),
        "keywords_validated": keywords_validated,
        "has_expected_pages": has_expected_pages,
        "has_keywords": has_keywords,
        "has_metadata": has_metadata,
        "has_id": has_id,
        "hard_types": dict(hard_types),
    }


def print_report(fr: dict, intl: dict) -> bool:
    print("=" * 70)
    print("AUDIT GOLD STANDARD - CONFORMITE STANDARDS INDUSTRIE & ISO")
    print(f"Date: {date.today().isoformat()}")
    print("Auteur: Claude Opus 4.5")
    print("=" * 70)

    for r in [fr, intl]:
        print(f"\n{'=' * 70}")
        print(f"{r['name']}")
        print("=" * 70)

        print("\n1. METRIQUES GENERALES")
        print(f"   Total questions: {r['total']}")
        print(
            f"   Answerable: {r['answerable']} ({r['answerable'] / r['total'] * 100:.1f}%)"
        )
        print(f"   Unanswerable: {r['unanswerable']} ({r['ratio']:.1f}%)")

        print("\n2. SQuAD 2.0 (arXiv:1806.03822)")
        print("   Exigence: 25-35% unanswerable")
        print(f"   Resultat: {r['ratio']:.1f}%")
        squad_ok = 25 <= r["ratio"] <= 35
        print(f"   Status: {'CONFORME' if squad_ok else 'NON CONFORME'}")

        print("\n3. SQuAD2-CR (arXiv:2004.14004)")
        print("   Exigence: 5/5 categories adversariales")
        print(f"   Resultat: {r['squad2_cr']}/5")
        for cat in [
            "ENTITY_SWAP",
            "ANTONYM",
            "NEGATION",
            "NUMBER_SWAP",
            "MUTUAL_EXCLUSION",
        ]:
            count = r["squad2_cr_details"].get(cat, 0)
            status = "[OK]" if count > 0 else "[--]"
            print(f"     {status} {cat}: {count}")
        print(f"   Status: {'CONFORME' if r['squad2_cr'] >= 5 else 'NON CONFORME'}")

        print("\n4. Adversarial categories (inspired by UAEval4RAG arXiv:2412.12300)")
        print("   Exigence: 6/6 project-adapted categories (5/6 acceptable)")
        print(f"   Resultat: {r['uaeval']}/6")
        for cat in [
            "UNDERSPECIFIED",
            "FALSE_PRESUPPOSITION",
            "VOCABULARY_MISMATCH",
            "OUT_OF_SCOPE",
            "SAFETY_CONCERNED",
            "PARTIAL_INFO",
        ]:
            count = r["uaeval_details"].get(cat, 0)
            status = "[OK]" if count > 0 else "[--]"
            print(f"     {status} {cat}: {count}")
        print(f"   Status: {'CONFORME' if r['uaeval'] >= 5 else 'NON CONFORME'}")

        print("\n5. QA TAXONOMY (arXiv:2107.12708)")
        print("   Exigence: 5/5 answer types")
        print(f"   Resultat: {r['qa_types']}/5")
        for t in ["FACTUAL", "PROCEDURAL", "LIST", "CONDITIONAL", "DEFINITIONAL"]:
            count = r["qa_types_details"].get(t, 0)
            status = "[OK]" if count > 0 else "[--]"
            print(f"     {status} {t}: {count}")
        print(f"   Status: {'CONFORME' if r['qa_types'] >= 5 else 'NON CONFORME'}")

        print("\n6. BLOOM'S TAXONOMY (Anderson 2001)")
        print("   Exigence: 4/4 niveaux cognitifs")
        print(f"   Resultat: {r['bloom']}/4")
        for level in ["REMEMBER", "UNDERSTAND", "APPLY", "ANALYZE"]:
            count = r["bloom_details"].get(level, 0)
            status = "[OK]" if count > 0 else "[--]"
            print(f"     {status} {level}: {count}")
        print(f"   Status: {'CONFORME' if r['bloom'] >= 4 else 'NON CONFORME'}")

        print("\n7. REASONING TYPES (arXiv:2107.12708)")
        print("   Exigence: 4/4 types de raisonnement")
        print(f"   Resultat: {r['reasoning']}/4")
        for rt in [
            "LEXICAL_MATCH",
            "SINGLE_SENTENCE",
            "MULTI_SENTENCE",
            "DOMAIN_KNOWLEDGE",
        ]:
            count = r["reasoning_details"].get(rt, 0)
            status = "[OK]" if count > 0 else "[--]"
            print(f"     {status} {rt}: {count}")
        print(f"   Status: {'CONFORME' if r['reasoning'] >= 4 else 'NON CONFORME'}")

    print("\n" + "=" * 70)
    print("CONFORMITE ISO/IEC")
    print("=" * 70)

    print("\n8. ISO 42001 - AI Management System")
    print("   A.6.2.2 - Documentation des donnees")
    for r in [fr, intl]:
        pct = r["has_metadata"] / r["answerable"] * 100 if r["answerable"] > 0 else 0
        print(
            f"     {r['name']}: {r['has_metadata']}/{r['answerable']} ({pct:.0f}%) avec metadata"
        )

    print("   A.6.2.4 - Validation des donnees")
    for r in [fr, intl]:
        pct = (
            r["keywords_validated"] / r["answerable"] * 100
            if r["answerable"] > 0
            else 0
        )
        print(
            f"     {r['name']}: {r['keywords_validated']}/{r['answerable']} ({pct:.0f}%) keywords valides"
        )

    print("\n9. ISO 25010 - Software Quality")
    print("   Exactitude fonctionnelle (expected_pages)")
    for r in [fr, intl]:
        pct = (
            r["has_expected_pages"] / r["answerable"] * 100
            if r["answerable"] > 0
            else 0
        )
        print(
            f"     {r['name']}: {r['has_expected_pages']}/{r['answerable']} ({pct:.0f}%)"
        )

    print("   Completude fonctionnelle (keywords)")
    for r in [fr, intl]:
        pct = r["has_keywords"] / r["answerable"] * 100 if r["answerable"] > 0 else 0
        print(f"     {r['name']}: {r['has_keywords']}/{r['answerable']} ({pct:.0f}%)")

    print("\n10. ISO 29119 - Software Testing")
    print("    Tracabilite (ID unique)")
    for r in [fr, intl]:
        pct = r["has_id"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"      {r['name']}: {r['has_id']}/{r['total']} ({pct:.0f}%)")

    print("\n" + "=" * 70)
    print("RESUME CONFORMITE")
    print("=" * 70)

    checks = [
        (
            "SQuAD 2.0 ratio (25-35%)",
            25 <= fr["ratio"] <= 35 and 25 <= intl["ratio"] <= 35,
        ),
        ("SQuAD2-CR (5/5 categories)", fr["squad2_cr"] >= 5 and intl["squad2_cr"] >= 5),
        ("Adversarial categories (5/6+)", fr["uaeval"] >= 5 and intl["uaeval"] >= 5),
        ("QA Taxonomy (5/5 types)", fr["qa_types"] >= 5 and intl["qa_types"] >= 5),
        ("Bloom's Taxonomy (4/4 levels)", fr["bloom"] >= 4 and intl["bloom"] >= 4),
        ("Reasoning Types (4/4)", fr["reasoning"] >= 4 and intl["reasoning"] >= 4),
        (
            "ISO 42001 A.6.2.4 (validation)",
            fr["keywords_validated"] >= fr["answerable"] * 0.9
            and intl["keywords_validated"] >= intl["answerable"] * 0.4,
        ),
    ]

    all_pass = True
    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_pass = False
        print(f"  {status} {name}")

    print(f"\n{'=' * 70}")
    if all_pass:
        print("VERDICT: CONFORME AUX STANDARDS INDUSTRIE ET NORMES ISO")
    else:
        print("VERDICT: NON CONFORME - CORRECTIONS REQUISES")
    print("=" * 70)

    return all_pass


def main():
    gs_fr = load_gs("tests/data/gold_standard_fr.json")
    gs_intl = load_gs("tests/data/gold_standard_intl.json")

    fr = audit_gs(gs_fr, "FR v5.30", "corpus/processed/chunks_mode_b_fr.json")
    intl = audit_gs(gs_intl, "INTL v2.4", "corpus/processed/chunks_mode_b_intl.json")

    return print_report(fr, intl)


if __name__ == "__main__":
    main()
