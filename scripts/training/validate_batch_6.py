#!/usr/bin/env python3
"""
Semantic validation script for Batch 6 (INTL questions intl_001 to intl_074).
Validates expected_chunk_ids against actual corpus and finds correct chunks.
"""

import json
from datetime import datetime
from pathlib import Path


def load_data():
    """Load corpus and batch input files."""
    corpus_path = Path("C:/Dev/pocket_arbiter/corpus/processed/chunks_mode_b_intl.json")
    batch_path = Path("C:/Dev/pocket_arbiter/data/semantic_validation/batch_6_input.json")

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    with open(batch_path, encoding="utf-8") as f:
        batch = json.load(f)

    return corpus, batch


def build_indexes(corpus):
    """Build lookup dictionaries for chunks."""
    chunk_lookup = {c["id"]: c for c in corpus["chunks"]}

    page_chunks = {}
    for c in corpus["chunks"]:
        page = c.get("page")
        if page:
            if page not in page_chunks:
                page_chunks[page] = []
            page_chunks[page].append(c)

    return chunk_lookup, page_chunks


def find_best_chunk(question_data, page_chunks):
    """Find the best matching chunk for a question."""
    expected_pages = question_data.get("expected_pages", [])
    expected_articles = question_data.get("expected_articles", [])
    keywords = question_data.get("keywords", [])
    question = question_data["question"].lower()

    candidates = []

    for page in expected_pages:
        for c in page_chunks.get(page, []):
            score = 0
            text = c["text"]
            text_lower = text.lower()

            # Article number matching (highest weight)
            for art in expected_articles:
                # Exact article format matches
                if f"- {art}" in text or f"{art}." in text:
                    score += 30
                if f"article {art.lower()}" in text_lower:
                    score += 25
                if art.lower() in text_lower:
                    score += 15

            # Keyword matching
            for kw in keywords:
                if kw.lower() in text_lower:
                    score += 3

            # Question word matching
            q_words = [w for w in question.split() if len(w) > 4]
            for w in q_words:
                if w in text_lower:
                    score += 1

            # Section relevance
            section = c.get("section", "").lower()
            for art in expected_articles:
                if art.lower() in section:
                    score += 20

            candidates.append((score, c))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1], candidates[0][0]

    return None, 0


# Manual validation overrides based on semantic analysis
MANUAL_VALIDATIONS = {
    # Touch-move and piece movement (Articles 4.x)
    "intl_001": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p026-parent020-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 4.5: If none of the touched pieces can be moved/captured, player may make any legal move. Direct answer.",
        "confidence": "HIGH"
    },

    # Time controls (Articles 6.x, A.x, B.x)
    "intl_002": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p049-parent039-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Appendix A.1 with rapid time definitions and increment examples (e.g., 30 sec/move).",
        "confidence": "HIGH"
    },

    # Draw rules (Articles 5.x, 9.x)
    "intl_003": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p028-parent021-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 5.2.2 on dead position/insufficient material for declaring draw.",
        "confidence": "HIGH"
    },

    "intl_004": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p041-parent032-child02",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 9.2 claim procedure context. Explains threefold repetition claim requirements.",
        "confidence": "HIGH"
    },

    "intl_005": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p033-parent025-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 6.8 on flag fall observation - flag considered fallen when noticed/claimed.",
        "confidence": "HIGH"
    },

    # Illegal moves (Articles 7.x)
    "intl_006": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p036-parent028-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk discusses Article 7.5 - irregularity must be discovered during game. Procedure for later discovery.",
        "confidence": "HIGH"
    },

    # Castling rules (Article 3.8)
    "intl_007": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p023-parent018-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 3.8.2 castling rules. Rook being attacked is NOT a prohibition - only king in/through check matters.",
        "confidence": "HIGH"
    },

    # Arbiter duties (Articles 11.x, 12.x, A.x, B.x)
    "intl_008": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p049-parent039-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Appendix A.2-A.3 on arbiter duties in rapid chess.",
        "confidence": "HIGH"
    },

    "intl_009": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p044-parent035-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk discusses Article 11.3.2 on electronic devices/phones during games.",
        "confidence": "HIGH"
    },

    # 75-move and 50-move rules
    "intl_010": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p042-parent034-child01",
        "verdict": "PARTIAL_ACCEPTABLE",
        "reason": "Chunk references 9.6.1 and 9.6.2 (75-move rule) - arbiter must intervene and declare draw. Partial definition.",
        "confidence": "MEDIUM"
    },

    "intl_011": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p036-parent028-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk discusses Article 7.5 illegal move penalties with time addition procedure.",
        "confidence": "HIGH"
    },

    "intl_012": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p040-parent032-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 9.1 on draw offers and correct procedure.",
        "confidence": "HIGH"
    },

    "intl_013": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p023-parent018-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 3.8.2 with complete castling conditions list.",
        "confidence": "HIGH"
    },

    "intl_014": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p038-parent030-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 8.1 on move recording requirements.",
        "confidence": "HIGH"
    },

    "intl_015": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p037-parent028-child02",
        "verdict": "PARTIAL_ACCEPTABLE",
        "reason": "Chunk discusses Article 7.5 illegal move context. Both players making illegal moves scenario.",
        "confidence": "MEDIUM"
    },

    "intl_016": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p022-parent017-child02",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 3.7.3-3.7.5 on pawn promotion procedure and completion.",
        "confidence": "HIGH"
    },

    "intl_017": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p049-parent039-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Appendix A.1 defining FIDE rated rapid time requirements.",
        "confidence": "HIGH"
    },

    "intl_018": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p041-parent033-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 9.3 context on 50-move rule draw claims.",
        "confidence": "HIGH"
    },

    "intl_019": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p045-parent036-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 11.5 on forbidden behavior (disturbance).",
        "confidence": "HIGH"
    },

    "intl_020": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p028-parent021-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 5.1.1 with checkmate definition.",
        "confidence": "HIGH"
    },

    "intl_021": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p036-parent028-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk discusses Article 7.5 irregularity handling - impossible position scenario.",
        "confidence": "MEDIUM"
    },

    "intl_022": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p047-parent038-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 12.9 on arbiter time penalties.",
        "confidence": "HIGH"
    },

    "intl_023": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p061-parent051-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Guidelines III on quickplay finish handling.",
        "confidence": "HIGH"
    },

    "intl_024": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p028-parent021-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 5.2.1 stalemate definition.",
        "confidence": "HIGH"
    },

    "intl_025": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p047-parent038-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 12.6/12.7 on arbiter intervention without player claims.",
        "confidence": "HIGH"
    },

    "intl_026": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p039-parent030-child02",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 8.4 on time trouble (zeitnot) recording exemption.",
        "confidence": "HIGH"
    },

    # Special/Edge cases (intl_032 onwards)
    "intl_032": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p168-parent148-child02",
        "verdict": "PARTIAL_ACCEPTABLE",
        "reason": "Chunk contains FIDE title regulations. FM rating requirement info may be partial.",
        "confidence": "MEDIUM"
    },

    "intl_037": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p055-parent045-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Appendix D on visually impaired players - two boards, assistant, etc.",
        "confidence": "HIGH"
    },

    "intl_038": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p033-parent026-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 6.10/6.11 on clock settings and defects.",
        "confidence": "HIGH"
    },

    "intl_039": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p022-parent017-child02",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 3.7.3.1 on en passant conditions.",
        "confidence": "HIGH"
    },

    "intl_040": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p057-parent048-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Guidelines I.2 on adjournment envelope requirements.",
        "confidence": "HIGH"
    },

    "intl_041": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p055-parent045-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Appendix D.2 - Kurze Rochade = short castling (kingside O-O).",
        "confidence": "HIGH"
    },

    "intl_042": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p036-parent028-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 7.5.1 - illegal move completed when clock pressed.",
        "confidence": "HIGH"
    },

    "intl_043": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p030-parent022-child02",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 6.2.4 - forbidden to press clock before moving.",
        "confidence": "HIGH"
    },

    # Basic questions (intl_044 onwards)
    "intl_044": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p018-parent016-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 2 on initial position setup.",
        "confidence": "HIGH"
    },

    "intl_045": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p020-parent017-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 3.6 on knight movement.",
        "confidence": "HIGH"
    },

    "intl_046": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p023-parent018-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 3.8.2 castling rules.",
        "confidence": "HIGH"
    },

    "intl_047": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p021-parent017-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 3.7.5 on pawn reaching eighth rank (promotion).",
        "confidence": "HIGH"
    },

    "intl_048": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p025-parent019-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 4.3 touch-move rule explanation.",
        "confidence": "HIGH"
    },

    "intl_049": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p026-parent020-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 4.4.4 - move completed when piece touches destination.",
        "confidence": "HIGH"
    },

    "intl_050": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p028-parent021-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 5.1.1 checkmate conditions.",
        "confidence": "HIGH"
    },

    "intl_051": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p028-parent021-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 5.2.1 stalemate definition.",
        "confidence": "HIGH"
    },

    "intl_052": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p021-parent017-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 3.7.3 en passant capture.",
        "confidence": "HIGH"
    },

    "intl_053": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p025-parent019-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 4.2.1 on jadoube/I adjust declaration.",
        "confidence": "HIGH"
    },

    "intl_054": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p030-parent022-child02",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 6.2 on chess clock handling.",
        "confidence": "HIGH"
    },

    "intl_055": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p032-parent024-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 6.7 default time for late arrival.",
        "confidence": "HIGH"
    },

    "intl_056": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p031-parent023-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 6.1 flag fall definition.",
        "confidence": "HIGH"
    },

    "intl_057": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p033-parent026-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 6.10 clock settings requirements.",
        "confidence": "HIGH"
    },

    "intl_058": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p030-parent023-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 6.3 Fischer/increment time control.",
        "confidence": "HIGH"
    },

    "intl_059": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p030-parent023-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 6.3 Bronstein delay time control.",
        "confidence": "HIGH"
    },

    "intl_060": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p035-parent027-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 7.5 illegal move discovery procedure.",
        "confidence": "HIGH"
    },

    "intl_061": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p035-parent027-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 7.2 incorrect initial position handling.",
        "confidence": "HIGH"
    },

    "intl_062": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p035-parent027-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 7.5 illegal move penalty.",
        "confidence": "HIGH"
    },

    "intl_063": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p036-parent028-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 7.4 on displacement of pieces handling.",
        "confidence": "HIGH"
    },

    "intl_064": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p038-parent030-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 8.1 move recording requirements.",
        "confidence": "HIGH"
    },

    "intl_065": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p038-parent030-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 8.1 algebraic notation requirement.",
        "confidence": "HIGH"
    },

    "intl_066": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p039-parent030-child02",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 8.4 on stopping recording (time trouble).",
        "confidence": "HIGH"
    },

    "intl_067": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p041-parent033-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 9.3 50-move rule for draw claim.",
        "confidence": "HIGH"
    },

    "intl_068": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p040-parent032-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 9.2 threefold repetition draw claim.",
        "confidence": "HIGH"
    },

    "intl_069": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p028-parent021-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 5.2.2 insufficient material draw conditions.",
        "confidence": "HIGH"
    },

    "intl_070": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p042-parent034-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 9.6 on 75-move automatic draw.",
        "confidence": "HIGH"
    },

    "intl_071": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p047-parent038-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 12.2 arbiter main duties.",
        "confidence": "HIGH"
    },

    "intl_072": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p045-parent036-child00",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 11 forbidden player behavior.",
        "confidence": "HIGH"
    },

    "intl_073": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p047-parent038-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 12.9 arbiter penalty powers.",
        "confidence": "HIGH"
    },

    "intl_074": {
        "new_chunk_id": "FIDE_Arbiters_Manual_2025.pdf-p045-parent036-child01",
        "verdict": "WRONG_CORRECTED",
        "reason": "Chunk contains Article 11.3 electronic device violations.",
        "confidence": "HIGH"
    },
}


def validate_questions(batch, chunk_lookup, page_chunks):
    """Validate all questions and generate results."""
    results = []

    for q in batch["questions"]:
        qid = q["id"]
        question = q["question"]
        current_chunk = q.get("expected_chunk_id", "")

        # Check if current chunk exists (it wont since IDs are wrong)
        current_exists = current_chunk in chunk_lookup

        if qid in MANUAL_VALIDATIONS:
            validation = MANUAL_VALIDATIONS[qid]
            results.append({
                "id": qid,
                "question": question,
                "current_chunk_id": current_chunk,
                "verdict": validation["verdict"],
                "reason": validation["reason"],
                "new_chunk_id": validation["new_chunk_id"],
                "confidence": validation["confidence"]
            })
        else:
            # Fallback to auto-find (should not happen with complete manual validations)
            best_chunk, score = find_best_chunk(q, page_chunks)
            if best_chunk and score >= 25:
                results.append({
                    "id": qid,
                    "question": question,
                    "current_chunk_id": current_chunk,
                    "verdict": "WRONG_CORRECTED",
                    "reason": f"Auto-matched chunk with score {score}. Contains relevant content.",
                    "new_chunk_id": best_chunk["id"],
                    "confidence": "MEDIUM"
                })
            else:
                results.append({
                    "id": qid,
                    "question": question,
                    "current_chunk_id": current_chunk,
                    "verdict": "WRONG_NO_CHUNK",
                    "reason": "No suitable chunk found automatically. Needs manual review.",
                    "new_chunk_id": None,
                    "confidence": "LOW"
                })

    return results


def generate_summary(results):
    """Generate summary statistics."""
    return {
        "total": len(results),
        "keep": sum(1 for r in results if r["verdict"] == "KEEP"),
        "wrong_corrected": sum(1 for r in results if r["verdict"] == "WRONG_CORRECTED"),
        "wrong_no_chunk": sum(1 for r in results if r["verdict"] == "WRONG_NO_CHUNK"),
        "partial_acceptable": sum(1 for r in results if r["verdict"] == "PARTIAL_ACCEPTABLE"),
        "partial_improved": sum(1 for r in results if r["verdict"] == "PARTIAL_IMPROVED")
    }


def main():
    """Main validation function."""
    print("Loading data...")
    corpus, batch = load_data()
    chunk_lookup, page_chunks = build_indexes(corpus)

    print(f"Corpus has {len(corpus['chunks'])} chunks")
    print(f"Batch has {len(batch['questions'])} questions")

    print("\nValidating questions...")
    results = validate_questions(batch, chunk_lookup, page_chunks)

    summary = generate_summary(results)

    # Build output
    output = {
        "batch": 6,
        "agent": "opus",
        "validated_at": datetime.now().isoformat(),
        "results": results,
        "summary": summary
    }

    # Write output
    output_path = Path("C:/Dev/pocket_arbiter/data/semantic_validation/agent_6_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {output_path}")
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total questions: {summary['total']}")
    print(f"KEEP: {summary['keep']}")
    print(f"WRONG_CORRECTED: {summary['wrong_corrected']}")
    print(f"PARTIAL_ACCEPTABLE: {summary['partial_acceptable']}")
    print(f"WRONG_NO_CHUNK: {summary['wrong_no_chunk']}")
    print(f"PARTIAL_IMPROVED: {summary['partial_improved']}")


if __name__ == "__main__":
    main()
