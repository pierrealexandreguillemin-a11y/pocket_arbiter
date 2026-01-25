#!/usr/bin/env python3
"""
Semantic Page Finder - 3-Layer Analysis.

Trouve LA page unique pour chaque question via analyse sémantique:
- Couche 1: Matching exact article_reference → page_no docling
- Couche 2: Overlap textuel answer_text → texte docling par page
- Couche 3: Validation chunk DB contient le contenu

ISO 42001 A.6.2.2: Provenance tracable
ISO 29119: Test data completeness
"""

import json
import re
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# === CONSTANTS ===

DIFFICULTY_SINGLE_HOP = 0.3
DIFFICULTY_MULTI_HOP = 0.6
DIFFICULTY_REASONING = 0.9

GS_PATH = Path("tests/data/gold_standard_annales_fr.json")
DB_PATH = Path("corpus/processed/corpus_mode_b_fr.db")
DOCLING_DIR = Path("corpus/processed/docling_fr")

# Document name mappings (expected_docs → docling filename)
DOC_TO_DOCLING = {
    "LA-octobre2025.pdf": "LA-octobre2025",
    "A02_2025_26_Championnat_de_France_des_Clubs.pdf": "A02_2025_26_Championnat_de_France_des_Clubs",
    "R01_2025_26_Regles_generales.pdf": "R01_2025_26_Regles_generales",
    "R02_2025_26_Regles_generales_Annexes.pdf": "R02_2025_26_Regles_generales_Annexes",
    "R03_2025_26_Competitions_homologuees.pdf": "R03_2025_26_Competitions_homologuees",
    "C01_2025_26_Coupe_de_France.pdf": "C01_2025_26_Coupe_de_France",
    "C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf": "C03_2025_26_Coupe_Jean_Claude_Loubatiere",
    "C04_2025_26_Coupe_de_la_parité.pdf": "C04_2025_26_Coupe_de_la_parité",  # avec accent
    "J01_2025_26_Championnat_de_France_Jeunes.pdf": "J01_2025_26_Championnat_de_France_Jeunes",
    "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf": "J02_2025_26_Championnat_de_France_Interclubs_Jeunes",
    "2018_Reglement_Disciplinaire20180422.pdf": "2018_Reglement_Disciplinaire20180422",
}


@dataclass
class PageScore:
    """Score for a candidate page."""
    page_no: int
    layer1_score: float = 0.0  # Article reference match
    layer2_score: float = 0.0  # Answer text overlap
    layer3_valid: bool = False  # Chunk exists in DB
    chunk_id: Optional[str] = None
    debug_info: dict = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        """Combined score with layer weights."""
        return (self.layer1_score * 10) + (self.layer2_score * 5) + (3 if self.layer3_valid else 0)


@dataclass
class QuestionResult:
    """Result for one question."""
    id: str
    best_page: Optional[int] = None
    chunk_id: Optional[str] = None
    difficulty_human: float = 0.5
    difficulty_retrieval: float = DIFFICULTY_SINGLE_HOP
    confidence: str = "low"
    method: str = "none"
    error: Optional[str] = None


def load_docling(doc_name: str) -> Optional[dict]:
    """Load docling JSON, build page→texts index."""
    docling_name = DOC_TO_DOCLING.get(doc_name)
    if not docling_name:
        # Try to find by prefix
        prefix = doc_name.replace(".pdf", "").split("_")[0]
        for k, v in DOC_TO_DOCLING.items():
            if k.startswith(prefix):
                docling_name = v
                break

    if not docling_name:
        return None

    docling_path = DOCLING_DIR / f"{docling_name}.json"
    if not docling_path.exists():
        return None

    with open(docling_path, encoding="utf-8") as f:
        data = json.load(f)

    # Build page → texts index
    page_texts: dict[int, list[str]] = defaultdict(list)
    doc = data.get("docling_document", data)
    texts = doc.get("texts", [])

    for text_obj in texts:
        prov = text_obj.get("prov", [])
        if not prov:
            continue
        page_no = prov[0].get("page_no", 0)
        text = text_obj.get("text", "")
        if page_no > 0 and text:
            page_texts[page_no].append(text)

    return dict(page_texts)


def extract_article_numbers(article_ref: str) -> list[str]:
    """
    Extract article numbers from reference.

    Examples:
    - "Article 3.2 des règles" → ["3.2"]
    - "R01 - 3.2. Forfait" → ["3.2"]
    - "Article 5.1.2 et 7.5.1" → ["5.1.2", "7.5.1"]
    - "Annexe A.5.1" → ["A.5.1"]
    """
    patterns = [
        r"Article\s*([A-D]?\d+(?:\.\d+)*)",  # Article 3.2, Article A.5
        r"Annexe\s*([A-D])",  # Annexe A
        r"(\d+\.\d+(?:\.\d+)*)",  # 3.2, 3.2.1
        r"-\s*(\d+(?:\.\d+)*)",  # R01 - 3.2
    ]

    found = []
    for p in patterns:
        matches = re.findall(p, article_ref, re.IGNORECASE)
        found.extend(matches)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for num in found:
        if num not in seen:
            seen.add(num)
            result.append(num)

    return result


def layer1_article_match(
    page_texts: dict[int, list[str]],
    article_ref: str
) -> dict[int, float]:
    """
    Couche 1: Match article numbers in docling text.

    Returns {page_no: score} where score = number of article matches.
    """
    article_nums = extract_article_numbers(article_ref)
    if not article_nums:
        return {}

    scores: dict[int, float] = {}

    for page_no, texts in page_texts.items():
        page_text = " ".join(texts).lower()
        score = 0.0

        for art_num in article_nums:
            # Exact match patterns
            patterns = [
                rf"article\s*{re.escape(art_num)}",
                rf"\b{re.escape(art_num)}[\.\s\)]",
                rf"art\.\s*{re.escape(art_num)}",
            ]
            for p in patterns:
                if re.search(p, page_text, re.IGNORECASE):
                    score += 1.0
                    break

        if score > 0:
            scores[page_no] = score

    return scores


def layer2_text_overlap(
    page_texts: dict[int, list[str]],
    answer_text: str,
    question_text: str
) -> dict[int, float]:
    """
    Couche 2: Keyword overlap between answer/question and page text.

    Returns {page_no: score} where score = overlap ratio.
    """
    if not answer_text:
        return {}

    # Extract keywords (words > 3 chars, not stopwords)
    stopwords = {
        "dans", "pour", "avec", "sans", "cette", "cette", "être", "avoir",
        "fait", "faire", "plus", "moins", "tout", "tous", "toute", "toutes",
        "peut", "doit", "sont", "été", "qui", "que", "quoi", "dont", "elle",
        "elles", "nous", "vous", "leur", "leurs", "même", "autres", "autre",
        "entre", "sous", "aussi", "ainsi", "donc", "alors", "comme", "mais",
        "avant", "après", "depuis", "pendant", "selon", "contre", "vers",
    }

    def extract_keywords(text: str) -> set[str]:
        words = re.findall(r"\b[a-zàâäéèêëïîôùûüç]{4,}\b", text.lower())
        return {w for w in words if w not in stopwords}

    answer_keywords = extract_keywords(answer_text)
    question_keywords = extract_keywords(question_text)
    all_keywords = answer_keywords | question_keywords

    if not all_keywords:
        return {}

    scores: dict[int, float] = {}

    for page_no, texts in page_texts.items():
        page_text = " ".join(texts).lower()
        page_words = set(re.findall(r"\b[a-zàâäéèêëïîôùûüç]{4,}\b", page_text))

        overlap = len(all_keywords & page_words)
        if overlap > 0:
            # Normalize by keyword count
            scores[page_no] = overlap / len(all_keywords)

    return scores


def layer3_chunk_validation(
    conn: sqlite3.Connection,
    source: str,
    page_no: int
) -> tuple[bool, Optional[str]]:
    """
    Couche 3: Verify chunk exists in DB for source+page.

    Returns (exists, chunk_id).
    """
    # Normalize source for matching
    source_pattern = f"%{source.replace('.pdf', '')}%"

    cur = conn.execute(
        """SELECT id FROM chunks
           WHERE source LIKE ? AND page = ?
           ORDER BY id LIMIT 1""",
        (source_pattern, page_no)
    )
    row = cur.fetchone()
    if row:
        return True, row[0]
    return False, None


def find_best_page(
    question: dict,
    page_texts: dict[int, list[str]],
    conn: sqlite3.Connection
) -> QuestionResult:
    """
    Find the best single page for a question using 3-layer analysis.
    """
    result = QuestionResult(id=question["id"])

    source = question.get("expected_docs", [""])[0]
    article_ref = question.get("article_reference", "")
    answer_text = question.get("answer_text", "")
    question_text = question.get("question", question.get("question_text", ""))

    if not page_texts:
        result.error = "No docling data"
        return result

    # Score all pages
    page_scores: dict[int, PageScore] = {}

    # Layer 1: Article reference matching
    layer1 = layer1_article_match(page_texts, article_ref)
    for page_no, score in layer1.items():
        if page_no not in page_scores:
            page_scores[page_no] = PageScore(page_no=page_no)
        page_scores[page_no].layer1_score = score

    # Layer 2: Text overlap
    layer2 = layer2_text_overlap(page_texts, answer_text, question_text)
    for page_no, score in layer2.items():
        if page_no not in page_scores:
            page_scores[page_no] = PageScore(page_no=page_no)
        page_scores[page_no].layer2_score = score

    # Layer 3: Chunk validation for top candidates
    for page_no in page_scores:
        valid, chunk_id = layer3_chunk_validation(conn, source, page_no)
        page_scores[page_no].layer3_valid = valid
        page_scores[page_no].chunk_id = chunk_id

    if not page_scores:
        # Fallback: use expected_pages if available
        expected = question.get("expected_pages", [])
        if expected:
            page_no = expected[0]
            valid, chunk_id = layer3_chunk_validation(conn, source, page_no)
            result.best_page = page_no
            result.chunk_id = chunk_id
            result.method = "fallback_expected"
            result.confidence = "low"
        else:
            result.error = "No page candidates found"
        return result

    # Find best page by total score
    best = max(page_scores.values(), key=lambda p: p.total_score)

    result.best_page = best.page_no
    result.chunk_id = best.chunk_id

    # Determine confidence
    if best.layer1_score > 0 and best.layer2_score > 0.3 and best.layer3_valid:
        result.confidence = "high"
        result.method = "semantic_3layer"
    elif best.layer1_score > 0 and best.layer3_valid:
        result.confidence = "medium"
        result.method = "article_match"
    elif best.layer2_score > 0.5 and best.layer3_valid:
        result.confidence = "medium"
        result.method = "text_overlap"
    else:
        result.confidence = "low"
        result.method = "best_guess"

    return result


def classify_retrieval_difficulty(question: dict) -> float:
    """Classify retrieval difficulty based on question complexity."""
    q_text = question.get("question", question.get("question_text", "")).lower()
    reasoning_type = question.get("reasoning_type", "single-hop")

    if reasoning_type == "multi-hop":
        return DIFFICULTY_MULTI_HOP

    # Reasoning indicators
    reasoning_patterns = [
        r"combien", r"calculer", r"quel.*score", r"quelle.*sanction",
        r"que.*faire", r"comment.*réagir", r"si.*alors",
        r"exception", r"priorité", r"différence"
    ]

    for p in reasoning_patterns:
        if re.search(p, q_text):
            return DIFFICULTY_REASONING

    return DIFFICULTY_SINGLE_HOP


def compute_difficulty_human(question: dict) -> float:
    """Compute human difficulty from annales success rate."""
    annales = question.get("annales_source", {})
    if annales and annales.get("success_rate") is not None:
        return round(1 - annales["success_rate"], 2)

    # Fallback to existing difficulty or default
    existing = question.get("difficulty")
    if existing is not None:
        return existing

    return 0.5  # Default


def process_all_questions() -> dict:
    """Process all 477 questions with semantic analysis."""
    # Load gold standard
    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)

    questions = gs["questions"]
    print(f"Processing {len(questions)} questions...")

    # Connect to DB
    conn = sqlite3.connect(str(DB_PATH))

    # Cache docling data by document
    docling_cache: dict[str, dict] = {}

    stats = {
        "total": len(questions),
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
        "errors": 0,
        "chunk_found": 0,
        "by_method": defaultdict(int),
    }

    for i, q in enumerate(questions):
        source = q.get("expected_docs", [""])[0]

        # Load docling (cached)
        if source not in docling_cache:
            docling_cache[source] = load_docling(source) or {}

        page_texts = docling_cache[source]

        # Find best page
        result = find_best_page(q, page_texts, conn)

        # Update question
        if result.best_page:
            q["expected_pages"] = [result.best_page]

        if result.chunk_id:
            q["expected_chunk_id"] = result.chunk_id
            stats["chunk_found"] += 1

        # Add difficulties
        q["difficulty_human"] = compute_difficulty_human(q)
        q["difficulty_retrieval"] = classify_retrieval_difficulty(q)

        # Keep legacy difficulty
        if not q.get("difficulty"):
            q["difficulty"] = q["difficulty_human"]

        # Track stats
        if result.error:
            stats["errors"] += 1
        elif result.confidence == "high":
            stats["high_confidence"] += 1
        elif result.confidence == "medium":
            stats["medium_confidence"] += 1
        else:
            stats["low_confidence"] += 1

        stats["by_method"][result.method] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(questions)}...")

    conn.close()

    # Save
    with open(GS_PATH, "w", encoding="utf-8") as f:
        json.dump(gs, f, ensure_ascii=False, indent=2)

    return stats


def validate_results() -> dict:
    """Validate the processed gold standard."""
    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))

    validation = {
        "total": len(gs["questions"]),
        "with_chunk_id": 0,
        "chunk_exists": 0,
        "single_page": 0,
        "page_3_count": 0,
        "with_difficulty_human": 0,
        "with_difficulty_retrieval": 0,
    }

    for q in gs["questions"]:
        if q.get("expected_chunk_id"):
            validation["with_chunk_id"] += 1
            cur = conn.execute(
                "SELECT 1 FROM chunks WHERE id = ?",
                (q["expected_chunk_id"],)
            )
            if cur.fetchone():
                validation["chunk_exists"] += 1

        pages = q.get("expected_pages", [])
        if len(pages) == 1:
            validation["single_page"] += 1
        if pages == [3]:
            validation["page_3_count"] += 1

        if q.get("difficulty_human") is not None:
            validation["with_difficulty_human"] += 1
        if q.get("difficulty_retrieval") is not None:
            validation["with_difficulty_retrieval"] += 1

    conn.close()
    return validation


if __name__ == "__main__":
    print("=" * 60)
    print("  SEMANTIC PAGE FINDER")
    print("  3-Layer Analysis: Article Match + Text Overlap + DB Validation")
    print("=" * 60)
    print()

    stats = process_all_questions()

    print()
    print("Processing complete:")
    print(f"  High confidence:   {stats['high_confidence']}")
    print(f"  Medium confidence: {stats['medium_confidence']}")
    print(f"  Low confidence:    {stats['low_confidence']}")
    print(f"  Errors:            {stats['errors']}")
    print(f"  Chunks found:      {stats['chunk_found']}/{stats['total']}")
    print()
    print("By method:")
    for method, count in sorted(stats["by_method"].items()):
        print(f"  {method}: {count}")

    print()
    print("Validating...")
    validation = validate_results()

    print()
    print("=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)
    total = validation["total"]
    print(f"  expected_chunk_id:      {validation['with_chunk_id']}/{total}")
    print(f"  chunks exist in DB:     {validation['chunk_exists']}/{total}")
    print(f"  single page:            {validation['single_page']}/{total}")
    print(f"  difficulty_human:       {validation['with_difficulty_human']}/{total}")
    print(f"  difficulty_retrieval:   {validation['with_difficulty_retrieval']}/{total}")
    print(f"  page [3] remaining:     {validation['page_3_count']}")
    print("=" * 60)

    # Check quality gates
    all_pass = (
        validation["with_chunk_id"] == total and
        validation["chunk_exists"] == total and
        validation["single_page"] == total and
        validation["with_difficulty_human"] == total and
        validation["with_difficulty_retrieval"] == total
    )

    if all_pass:
        print("  STATUS: ALL CRITERIA MET")
    else:
        print("  STATUS: CRITERIA NOT MET - Manual review needed")
    print("=" * 60)
