"""
Generate Synthetic Training Data - Claude Direct

Génère des paires (question, chunk) pour fine-tuning EmbeddingGemma.
Utilise l'analyse directe des chunks sans API externe.

ISO Reference:
    - ISO/IEC 25010 - Functional suitability (Recall >= 80%)
    - ISO/IEC 42001 - AI traceability

Document ID: SCRIPT-SYN-002
Version: 1.0
Date: 2026-01-16
Author: Claude Opus 4.5
"""

import json
import re
import sqlite3
from pathlib import Path

# Question templates by content type
QUESTION_TEMPLATES = {
    "article": [
        "Que dit l'article {ref} ?",
        "Quel est le contenu de l'article {ref} ?",
        "Comment s'applique l'article {ref} ?",
    ],
    "regle": [
        "Quelle est la règle concernant {topic} ?",
        "Comment fonctionne la règle de {topic} ?",
        "Que prévoit le règlement pour {topic} ?",
    ],
    "procedure": [
        "Quelle est la procédure pour {topic} ?",
        "Comment procéder en cas de {topic} ?",
        "Quelles sont les étapes pour {topic} ?",
    ],
    "sanction": [
        "Quelle sanction pour {topic} ?",
        "Quelles sont les sanctions prévues pour {topic} ?",
        "Comment est sanctionné {topic} ?",
    ],
    "definition": [
        "Qu'est-ce que {topic} ?",
        "Comment est défini {topic} ?",
        "Quelle est la définition de {topic} ?",
    ],
}

# Keywords to detect content type
CONTENT_PATTERNS = {
    "article": r"article\s+(\d+[\.\d]*)",
    "sanction": r"sanction|pénalité|avertissement|exclusion|amende",
    "procedure": r"procédure|démarche|étape|comment|doit",
    "temps": r"temps|pendule|cadence|minute|seconde|délai",
    "roque": r"roque|roi|tour|O-O",
    "nulle": r"nulle|pat|répétition|50 coups",
    "promotion": r"promotion|pion|dame|transformer",
    "notation": r"notation|feuille|algébrique|noter",
    "arbitre": r"arbitre|directeur|officiel|décision",
    "forfait": r"forfait|absent|retard|défaut",
}


def extract_topic(text: str) -> str:
    """Extract main topic from chunk text."""
    text_lower = text.lower()

    # Check for article reference
    match = re.search(CONTENT_PATTERNS["article"], text_lower)
    if match:
        return f"article {match.group(1)}"

    # Check for specific topics
    for topic, pattern in CONTENT_PATTERNS.items():
        if topic != "article" and re.search(pattern, text_lower):
            return topic

    # Default: extract first meaningful phrase
    sentences = text.split(".")
    if sentences:
        first = sentences[0].strip()[:50]
        return first if first else "ce sujet"

    return "ce sujet"


def detect_content_type(text: str) -> str:
    """Detect content type from chunk text."""
    text_lower = text.lower()

    if re.search(CONTENT_PATTERNS["article"], text_lower):
        return "article"
    if re.search(CONTENT_PATTERNS["sanction"], text_lower):
        return "sanction"
    if re.search(CONTENT_PATTERNS["procedure"], text_lower):
        return "procedure"

    return "regle"


def generate_questions_for_chunk(chunk_text: str, chunk_id: str) -> list[str]:
    """Generate 2 questions for a chunk based on its content."""
    content_type = detect_content_type(chunk_text)
    topic = extract_topic(chunk_text)

    templates = QUESTION_TEMPLATES.get(content_type, QUESTION_TEMPLATES["regle"])

    questions = []
    for i, template in enumerate(templates[:2]):
        try:
            if "{ref}" in template:
                # Extract article reference
                match = re.search(CONTENT_PATTERNS["article"], chunk_text.lower())
                if match:
                    q = template.format(ref=match.group(1))
                else:
                    continue
            else:
                q = template.format(topic=topic)
            questions.append(q)
        except (KeyError, IndexError):
            continue

    # Fallback: generic question
    if not questions:
        questions = [f"Que contient ce passage sur {topic} ?"]

    return questions


def generate_synthetic_dataset(
    db_path: Path,
    output_path: Path,
    max_chunks: int | None = None,
) -> dict:
    """Generate synthetic training pairs from corpus chunks."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = "SELECT id, text, source, page FROM chunks"
    if max_chunks:
        query += f" LIMIT {max_chunks}"

    cur.execute(query)
    chunks = cur.fetchall()
    conn.close()

    pairs = []
    skipped = 0

    for chunk_id, text, source, page in chunks:
        if not text or len(text) < 50:
            skipped += 1
            continue

        questions = generate_questions_for_chunk(text, chunk_id)

        for q in questions:
            pairs.append(
                {
                    "query": q,
                    "positive": text,
                    "chunk_id": chunk_id,
                    "source": source,
                    "page": page,
                }
            )

    # Save to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return {
        "total_chunks": len(chunks),
        "skipped_chunks": skipped,
        "total_pairs": len(pairs),
        "output_file": str(output_path),
    }


def main() -> None:
    """Generate synthetic training data."""
    print("=== GÉNÉRATION DONNÉES SYNTHÉTIQUES (CLAUDE DIRECT) ===")
    print()

    db_path = Path("corpus/processed/corpus_sentence_fr_qat.db")
    output_path = Path("data/training/synthetic_pairs.jsonl")

    if not db_path.exists():
        print(f"ERREUR: Base de données non trouvée: {db_path}")
        return

    result = generate_synthetic_dataset(db_path, output_path)

    print(f"Chunks traités: {result['total_chunks']}")
    print(f"Chunks ignorés: {result['skipped_chunks']}")
    print(f"Paires générées: {result['total_pairs']}")
    print(f"Fichier: {result['output_file']}")


if __name__ == "__main__":
    main()
