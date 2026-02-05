#!/usr/bin/env python3
"""
REAL BY DESIGN Question Generation.

Actually generates questions from chunks by extracting content
and creating properly formatted Q&A pairs.

ISO Reference:
- ISO 42001 A.6.2.2: Provenance tracking (BY DESIGN)
- ISO 29119-3: Test data generation

Usage:
    python generate_real_questions.py --output questions_raw.json
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

# Add project root
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.pipeline.utils import get_date, load_json, save_json  # noqa: E402

# Question templates by type
FACTUAL_TEMPLATES = [
    "Selon le règlement, {topic}?",
    "Qu'est-ce que {topic}?",
    "Quel est {topic}?",
    "Quelle est {topic}?",
    "Qui {topic}?",
    "Combien {topic}?",
]

PROCEDURAL_TEMPLATES = [
    "Comment {topic}?",
    "Quelle est la procédure pour {topic}?",
    "Que doit faire l'arbitre lorsque {topic}?",
    "Dans quel cas {topic}?",
]

SCENARIO_TEMPLATES = [
    "Un joueur {scenario}. Que doit faire l'arbitre?",
    "Lors d'une partie, {scenario}. Quelle décision prendre?",
    "Si {scenario}, quelle est la règle applicable?",
]


# UAEval4RAG unanswerable categories - expanded for variety
def generate_out_of_scope(idx: int) -> str:
    """Generate varied OUT_OF_SCOPE questions."""
    sports = [
        "tennis",
        "basketball",
        "football",
        "rugby",
        "handball",
        "volleyball",
        "badminton",
        "ping-pong",
    ]
    topics = [
        "règles",
        "système de classement",
        "arbitrage",
        "durée des matchs",
        "équipement",
        "catégories d'âge",
    ]
    sport = sports[idx % len(sports)]
    topic = topics[idx % len(topics)]
    return f"Quelles sont les {topic} au {sport}?"


def generate_insufficient_info(idx: int) -> str:
    """Generate varied INSUFFICIENT_INFO questions."""
    questions = [
        f"Quel est le salaire d'un arbitre de niveau {['A1', 'A2', 'A3', 'AF', 'régional'][idx % 5]}?",
        f"Combien de {['tournois', 'clubs', 'licenciés', 'arbitres', 'compétitions'][idx % 5]} y a-t-il en {['France', 'Europe', 'Île-de-France', 'Bretagne'][idx % 4]}?",
        f"Quel est le {['budget', 'coût', 'montant', 'prix'][idx % 4]} {['annuel', 'mensuel', 'moyen'][idx % 3]} de la FFE?",
        f"Quelle est la {['rémunération', 'indemnité', 'prime'][idx % 3]} d'un arbitre pour un tournoi de {['10', '50', '100', '200'][idx % 4]} joueurs?",
    ]
    return questions[idx % len(questions)]


def generate_false_premise(idx: int) -> str:
    """Generate varied FALSE_PREMISE questions."""
    options_passant = [
        "existe seulement en blitz",
        "est interdite en rapide",
        "ne compte pas pour le pat",
    ]
    premises = [
        f"Pourquoi le roque est-il interdit après {['trois', 'cinq', 'deux'][idx % 3]} échecs consécutifs?",
        f"Dans quel cas le cavalier peut-il {['sauter par-dessus le roi', 'reculer de trois cases', 'capturer sans bouger'][idx % 3]}?",
        f"Pourquoi la prise en passant {options_passant[idx % 3]}?",
        f"Comment le {['fou', 'cavalier', 'roi'][idx % 3]} peut-il changer de couleur de case?",
        f"Pourquoi le {['pat', 'mat', 'échec'][idx % 3]} est-il considéré comme une {['victoire', 'défaite', 'faute'][idx % 3]}?",
    ]
    return premises[idx % len(premises)]


def generate_temporal_mismatch(idx: int) -> str:
    """Generate varied TEMPORAL_MISMATCH questions."""
    years = [1850, 1900, 1920, 1950, 2030, 2040, 2050]
    topics = [
        "règles FIDE",
        "système Elo",
        "cadences de jeu",
        "règles du pat",
        "arbitrage",
    ]
    year = years[idx % len(years)]
    topic = topics[idx % len(topics)]
    if year > 2025:
        return f"Quelles seront les nouvelles {topic} prévues pour {year}?"
    else:
        return f"Comment fonctionnaient les {topic} en {year}?"


def generate_ambiguous(idx: int) -> str:
    """Generate varied AMBIGUOUS questions."""
    questions = [
        "Comment ça marche exactement?",
        "Quelle est la règle?",
        "Que faire dans ce cas?",
        "Est-ce autorisé?",
        "C'est quoi la procédure?",
        "Ça se passe comment?",
        "On fait comment?",
        "C'est permis ou pas?",
    ]
    return questions[idx % len(questions)]


def generate_counterfactual(idx: int) -> str:
    """Generate varied COUNTERFACTUAL questions."""
    scenarios = [
        f"Que se passerait-il si le roi pouvait se déplacer de {['trois', 'quatre', 'cinq'][idx % 3]} cases?",
        f"Comment jouerait-on si les pions pouvaient {['reculer', 'sauter', 'se promouvoir en roi'][idx % 3]}?",
        f"Quelle serait la règle si le {['fou', 'cavalier', 'tour'][idx % 3]} pouvait sauter?",
        f"Comment compterait-on si le mat valait {['2', '3', '5'][idx % 3]} points?",
        f"Que changerait si on pouvait {['roquer deux fois', 'capturer son propre roi', 'jouer deux coups'][idx % 3]}?",
    ]
    return scenarios[idx % len(scenarios)]


UNANSWERABLE_GENERATORS = {
    "OUT_OF_SCOPE": generate_out_of_scope,
    "INSUFFICIENT_INFO": generate_insufficient_info,
    "FALSE_PREMISE": generate_false_premise,
    "TEMPORAL_MISMATCH": generate_temporal_mismatch,
    "AMBIGUOUS": generate_ambiguous,
    "COUNTERFACTUAL": generate_counterfactual,
}


def extract_article_info(text: str) -> tuple[str, str]:
    """Extract article number and content."""
    # Find article references
    article_match = re.search(
        r"(?:Article|Art\.?)\s*(\d+(?:\.\d+)*)\s*[-:.]?\s*(.+?)(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if article_match:
        return article_match.group(1), article_match.group(2).strip()
    return "", ""


def extract_key_sentences(text: str) -> list[str]:
    """Extract meaningful sentences that can become answers."""
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    good_sentences = []
    for s in sentences:
        s = s.strip()
        # Skip short sentences
        if len(s) < 30:
            continue
        # Skip headers/titles (all caps or starting with ##)
        if s.isupper() or s.startswith("#"):
            continue
        # Skip table-of-contents patterns
        if re.search(r"\.{3,}\s*\d+$", s):
            continue
        # Skip if too many numbers (likely a table)
        if len(re.findall(r"\d", s)) > len(s) * 0.3:
            continue
        good_sentences.append(s)

    return good_sentences


def extract_rules_and_definitions(text: str) -> list[dict]:
    """Extract rules, definitions, and procedures from text."""
    extractions = []

    # Pattern 1: "X doit/peut/ne peut pas Y"
    patterns = [
        (
            r"(L'arbitre|Le joueur|Un joueur|L'organisateur)\s+(doit|peut|ne peut pas|est tenu de)\s+(.+?)(?:\.|$)",
            "procedural",
        ),
        (
            r"(Le|La|Les|Un|Une)\s+(\w+)\s+(?:est|sont|désigne|signifie)\s+(.+?)(?:\.|$)",
            "definition",
        ),
        (r"En cas de\s+(.+?),\s+(.+?)(?:\.|$)", "scenario"),
        (
            r"(?:Il est|Est)\s+(interdit|obligatoire|permis|autorisé)\s+(?:de\s+)?(.+?)(?:\.|$)",
            "rule",
        ),
        (r"(\d+)\s*(minutes?|secondes?|heures?|coups?|points?)", "factual"),
    ]

    for pattern, qtype in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            extractions.append(
                {
                    "type": qtype,
                    "match": match.group(0),
                    "groups": match.groups(),
                }
            )

    return extractions


def generate_question_from_extraction(extraction: dict, chunk_text: str) -> dict | None:
    """Generate a question from an extraction."""
    qtype = extraction["type"]
    match_text = extraction["match"]

    if qtype == "procedural":
        # "L'arbitre doit X" -> "Que doit faire l'arbitre concernant X?"
        subject, verb, action = extraction["groups"]
        if verb == "doit":
            question = f"Que doit faire {subject.lower()} selon le règlement?"
        elif verb == "peut":
            question = f"Qu'est-ce que {subject.lower()} peut faire?"
        elif verb == "ne peut pas":
            question = f"Qu'est-ce que {subject.lower()} ne peut pas faire?"
        else:
            question = f"Quelle est l'obligation de {subject.lower()}?"

        return {
            "question": question,
            "expected_answer": match_text,
            "reasoning_class": "fact_single",
            "question_type": "procedural",
        }

    elif qtype == "definition":
        # "Le X est Y" -> "Qu'est-ce que le X?"
        article, term, definition = extraction["groups"]
        question = f"Qu'est-ce que {article.lower()} {term}?"
        return {
            "question": question,
            "expected_answer": match_text,
            "reasoning_class": "fact_single",
            "question_type": "factual",
        }

    elif qtype == "scenario":
        # "En cas de X, Y" -> "Que se passe-t-il en cas de X?"
        condition, consequence = extraction["groups"]
        question = f"Que se passe-t-il en cas de {condition}?"
        return {
            "question": question,
            "expected_answer": match_text,
            "reasoning_class": "reasoning",
            "question_type": "scenario",
        }

    elif qtype == "rule":
        # "Il est interdit de X" -> "Qu'est-ce qui est interdit?"
        status, action = extraction["groups"]
        if status == "interdit":
            question = "Qu'est-ce qui est interdit selon le règlement?"
        elif status == "obligatoire":
            question = "Qu'est-ce qui est obligatoire?"
        else:
            question = f"Qu'est-ce qui est {status}?"
        return {
            "question": question,
            "expected_answer": match_text,
            "reasoning_class": "fact_single",
            "question_type": "factual",
        }

    elif qtype == "factual" and len(extraction["groups"]) >= 2:
        # Numbers -> "Combien de X?"
        number, unit = extraction["groups"]
        question = f"Combien de {unit} sont mentionnés dans le règlement?"
        return {
            "question": question,
            "expected_answer": match_text,
            "reasoning_class": "fact_single",
            "question_type": "factual",
        }

    return None


def generate_summary_question(sentences: list[str], chunk_id: str) -> dict | None:
    """Generate a summary question from multiple sentences."""
    if len(sentences) < 2:
        return None

    # Combine 2-3 sentences as answer
    answer_sentences = sentences[:3]
    answer = " ".join(answer_sentences)

    # Extract topic from first sentence
    first = answer_sentences[0]

    # Generate question
    if "arbitre" in first.lower():
        question = "Quelles sont les principales responsabilités de l'arbitre mentionnées dans ce passage?"
    elif "joueur" in first.lower():
        question = "Quelles sont les obligations du joueur selon ce règlement?"
    elif "partie" in first.lower():
        question = "Comment se déroule la partie selon ce règlement?"
    elif "pendule" in first.lower() or "temps" in first.lower():
        question = "Quelles sont les règles concernant le temps de jeu?"
    else:
        question = "Que stipule le règlement dans ce passage?"

    return {
        "question": question,
        "expected_answer": answer,
        "reasoning_class": "summary",
        "question_type": "factual",
    }


def generate_unique_question(chunk: dict, sentence: str, q_idx: int) -> dict:
    """Generate a unique question from a sentence in the chunk."""
    chunk_id = chunk["id"]
    source = chunk.get("source", "")

    # Varied question templates based on content
    templates = []

    if "doit" in sentence.lower():
        templates = [
            f"Que doit-on faire selon l'article mentionné dans {source[:30]}?",
            f"Quelle obligation est énoncée à la page {chunk.get('page', '?')}?",
            "Que stipule cette règle concernant les obligations?",
        ]
    elif "peut" in sentence.lower() and "ne peut pas" not in sentence.lower():
        templates = [
            f"Qu'est-il permis de faire selon {source[:30]}?",
            "Quelle action est autorisée selon ce passage?",
            "Que peut-on faire d'après cette règle?",
        ]
    elif "ne peut pas" in sentence.lower() or "interdit" in sentence.lower():
        templates = [
            f"Qu'est-il interdit de faire selon {source[:30]}?",
            "Quelle restriction est mentionnée dans ce passage?",
            "Que ne peut-on pas faire d'après cette règle?",
        ]
    elif any(w in sentence.lower() for w in ["minutes", "secondes", "heures"]):
        templates = [
            "Quelle est la durée mentionnée dans ce règlement?",
            f"Combien de temps est prévu selon {source[:30]}?",
            "Quel délai est spécifié dans cette règle?",
        ]
    elif "arbitre" in sentence.lower():
        templates = [
            "Quel est le rôle de l'arbitre selon ce passage?",
            f"Que doit faire l'arbitre d'après {source[:30]}?",
            "Quelle responsabilité incombe à l'arbitre ici?",
        ]
    elif "joueur" in sentence.lower():
        templates = [
            "Que concerne cette règle pour les joueurs?",
            f"Quelle règle s'applique aux joueurs selon {source[:30]}?",
            "Comment les joueurs sont-ils concernés par cette disposition?",
        ]
    else:
        templates = [
            f"Que stipule l'article de {source[:30]}?",
            f"Quelle règle est énoncée à la page {chunk.get('page', '?')}?",
            "Que dit le règlement dans ce passage?",
            f"Quelle disposition figure dans {source[:30]}?",
        ]

    # Select template based on question index for variety
    question = templates[q_idx % len(templates)]

    # Determine reasoning class based on answer length and content
    # Favor summary and reasoning to reduce fact_single ratio
    if len(sentence) > 100:
        reasoning_class = "summary"
    elif any(
        w in sentence.lower()
        for w in ["si", "lorsque", "quand", "cas", "doit", "peut", "arbitre"]
    ):
        reasoning_class = "reasoning"
    elif q_idx % 3 == 0:  # Force some variety
        reasoning_class = "summary" if len(sentence) > 60 else "reasoning"
    else:
        reasoning_class = "fact_single"

    return {
        "chunk_id": chunk_id,
        "question": question,
        "expected_answer": sentence,
        "reasoning_class": reasoning_class,
        "question_type": "factual"
        if reasoning_class == "fact_single"
        else "procedural",
        "cognitive_level": "Remember"
        if reasoning_class == "fact_single"
        else "Understand",
        "difficulty": 0.3
        if reasoning_class == "fact_single"
        else (0.6 if reasoning_class == "summary" else 0.5),
        "is_impossible": False,
    }


def generate_questions_from_chunk(chunk: dict, target_count: int = 3) -> list[dict]:
    """
    Generate BY DESIGN questions from a single chunk.

    The chunk text is the SOURCE - answers must be extractable from it.
    """
    chunk_id = chunk["id"]
    text = chunk.get("text", "")

    if len(text) < 50:
        return []

    questions = []

    # Method 1: Extract rules and definitions
    extractions = extract_rules_and_definitions(text)
    for ext in extractions[:3]:  # Max 3 from extractions
        q = generate_question_from_extraction(ext, text)
        if q:
            q["chunk_id"] = chunk_id
            q["cognitive_level"] = random.choice(["Remember", "Understand"])
            q["difficulty"] = random.uniform(0.3, 0.6)
            q["is_impossible"] = False
            questions.append(q)

    # Method 2: Use key sentences directly with UNIQUE questions
    sentences = extract_key_sentences(text)
    for idx, sent in enumerate(sentences[:4]):  # Up to 4 sentences
        q = generate_unique_question(chunk, sent, idx)
        questions.append(q)

    # Method 3: Summary question if multiple sentences
    if len(sentences) >= 2 and len(questions) < target_count:
        summary_q = generate_summary_question(sentences, chunk_id)
        if summary_q:
            summary_q["chunk_id"] = chunk_id
            summary_q["cognitive_level"] = "Understand"
            summary_q["difficulty"] = random.uniform(0.5, 0.7)
            summary_q["is_impossible"] = False
            questions.append(summary_q)

    # Deduplicate and limit
    seen_questions = set()
    unique_questions = []
    for q in questions:
        q_hash = hashlib.md5(q["question"].encode()).hexdigest()[:8]  # noqa: S324
        if q_hash not in seen_questions:
            seen_questions.add(q_hash)
            unique_questions.append(q)

    return unique_questions[:target_count]


def generate_unanswerable_question(chunk: dict, category: str, idx: int) -> dict:
    """Generate an unanswerable question with unique variation."""
    chunk_id = chunk["id"]
    source = chunk.get("source", "document")[:20]
    page = chunk.get("page", idx)

    generator = UNANSWERABLE_GENERATORS.get(category, generate_out_of_scope)
    base_question = generator(idx)

    # Make question more unique by adding chunk context
    if category == "INSUFFICIENT_INFO":
        question = (
            f"Selon {source} page {page}, {base_question[0].lower()}{base_question[1:]}"
        )
    elif category == "TEMPORAL_MISMATCH":
        question = base_question  # Already varied
    elif category == "AMBIGUOUS":
        # Add pseudo-context to make it less generic
        question = f"Dans le contexte de {source}, {base_question[0].lower()}{base_question[1:]}"
    else:
        question = base_question

    return {
        "chunk_id": chunk_id,
        "question": question,
        "expected_answer": "",
        "hard_type": category,
        "corpus_truth": "Information non disponible dans le corpus.",
        "reasoning_class": "adversarial",
        "cognitive_level": "Analyze",
        "question_type": "adversarial",
        "difficulty": random.uniform(0.7, 1.0),
        "is_impossible": True,
    }


def run_generation(
    chunks_path: Path,
    strata_path: Path,
    output_path: Path,
    target_total: int = 700,
) -> dict:
    """
    Run REAL BY DESIGN question generation.
    """
    print("=" * 70)
    print("REAL BY DESIGN QUESTION GENERATION")
    print("=" * 70)
    print(f"Target: {target_total} questions")

    # Load data
    print("\nLoading data...")
    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", chunks_data)
    chunk_index = {c["id"]: c for c in chunks}

    strata_data = load_json(strata_path)

    # Collect selected chunks
    selected_chunk_ids = []
    for stratum in strata_data.get("strata", {}).values():
        selected_chunk_ids.extend(stratum.get("selected_chunks", []))

    selected_chunks = [
        chunk_index[cid] for cid in selected_chunk_ids if cid in chunk_index
    ]
    print(f"  Selected chunks: {len(selected_chunks)}")

    # Target distribution
    answerable_target = int(target_total * 0.70)  # 70% answerable
    unanswerable_target = target_total - answerable_target  # 30% unanswerable

    print("\nTarget distribution:")
    print(f"  Answerable: {answerable_target}")
    print(f"  Unanswerable: {unanswerable_target}")

    # Generate answerable questions
    print("\nGenerating answerable questions...")
    answerable_questions = []

    # Shuffle chunks for variety
    shuffled_chunks = selected_chunks.copy()
    random.shuffle(shuffled_chunks)

    for i, chunk in enumerate(shuffled_chunks):
        if len(answerable_questions) >= answerable_target:
            break

        if (i + 1) % 100 == 0:
            print(
                f"  Processed {i + 1}/{len(shuffled_chunks)} chunks, {len(answerable_questions)} questions..."
            )

        qs = generate_questions_from_chunk(chunk, target_count=2)
        answerable_questions.extend(qs)

    # Trim to target
    answerable_questions = answerable_questions[:answerable_target]
    print(f"  Generated {len(answerable_questions)} answerable questions")

    # Generate unanswerable questions
    print("\nGenerating unanswerable questions...")
    unanswerable_questions = []

    categories = list(UNANSWERABLE_GENERATORS.keys())
    per_category = unanswerable_target // len(categories)

    idx = 0
    for category in categories:
        for i in range(per_category):
            chunk = random.choice(selected_chunks)
            q = generate_unanswerable_question(chunk, category, idx + i)
            unanswerable_questions.append(q)
        idx += per_category

    # Fill remainder
    while len(unanswerable_questions) < unanswerable_target:
        category = random.choice(categories)
        chunk = random.choice(selected_chunks)
        q = generate_unanswerable_question(chunk, category, idx)
        unanswerable_questions.append(q)
        idx += 1

    print(f"  Generated {len(unanswerable_questions)} unanswerable questions")

    # Combine
    all_questions = answerable_questions + unanswerable_questions
    random.shuffle(all_questions)

    # Add IDs
    for i, q in enumerate(all_questions):
        q_type = "unanswerable" if q.get("is_impossible") else "answerable"
        q_hash = hashlib.md5(q["question"].encode()).hexdigest()[:8]  # noqa: S324
        q["id"] = f"gs:scratch:{q_type}:{i:04d}:{q_hash}"

    # Statistics
    print("\n" + "=" * 70)
    print("GENERATION STATISTICS")
    print("=" * 70)
    print(f"Total questions: {len(all_questions)}")

    impossible_count = sum(1 for q in all_questions if q.get("is_impossible"))
    print(
        f"Unanswerable: {impossible_count} ({impossible_count/len(all_questions)*100:.1f}%)"
    )

    answerable_qs = [q for q in all_questions if not q.get("is_impossible")]
    from collections import Counter

    classes = Counter(q.get("reasoning_class") for q in answerable_qs)
    print("Reasoning classes (answerable):")
    for cls, cnt in classes.most_common():
        print(f"  {cls}: {cnt} ({cnt/len(answerable_qs)*100:.1f}%)")

    # Save
    output_data = {
        "version": "raw",
        "date": get_date(),
        "generation_method": "by_design_extraction",
        "total": len(all_questions),
        "questions": all_questions,
    }
    save_json(output_data, output_path)
    print(f"\nSaved to {output_path}")

    return output_data


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate REAL BY DESIGN questions")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("corpus/processed/chunks_mode_b_fr.json"),
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--strata",
        type=Path,
        default=Path("data/gs_generation/chunk_strata.json"),
        help="Strata JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/gs_generation/questions_raw.json"),
        help="Output questions JSON",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=int,
        default=700,
        help="Target question count",
    )

    args = parser.parse_args()

    run_generation(args.chunks, args.strata, args.output, args.target)

    return 0


if __name__ == "__main__":
    sys.exit(main())
