"""
Generate synthetic questions using Cerebras API (free tier: 24M tokens/day).

Usage:
    # Test on 5 chunks
    python scripts/pipeline/run_cerebras_generation.py --test

    # Full generation FFE
    python scripts/pipeline/run_cerebras_generation.py --corpus ffe

    # Full generation FIDE
    python scripts/pipeline/run_cerebras_generation.py --corpus fide
"""

import argparse
import json
import os
import time
from pathlib import Path

from cerebras.cloud.sdk import Cerebras

# Configuration
MODEL = "llama-3.3-70b"
QUESTIONS_PER_CHUNK = 3
MAX_RETRIES = 3
OUTPUT_DIR = Path("data/synthetic_triplets")

# Categories
QUESTION_CATEGORIES = {
    "ffe": {
        "arbitre_terrain": {
            "examples": [
                "Quel sera l'Elo d'un joueur apres ce tournoi avec ces resultats?",
                "Quels sont les affichages obligatoires en salle de jeu?",
                "Un portable sonne pendant la partie, quelle sanction?",
            ],
        },
        "arbitre_organisateur": {
            "examples": [
                "Quelles sont les conditions pour devenir arbitre AF2?",
                "Quel est le delai d'homologation d'un tournoi FFE?",
                "Quel materiel est obligatoire pour un tournoi homologue?",
            ],
        },
        "question_joueur": {
            "examples": [
                "J'ai le droit de proposer nulle quand exactement?",
                "Il a touche une piece, il doit la jouer non?",
                "C'est quoi le departage au juste?",
            ],
        },
    },
    "fide": {
        "arbiter_field": {
            "examples": [
                "What is the penalty for a mobile phone ringing?",
                "How to handle a claim of threefold repetition?",
                "What happens if a player makes an illegal move?",
            ],
        },
        "arbiter_organizer": {
            "examples": [
                "What are the requirements to become a FIDE Arbiter?",
                "What equipment is mandatory for a FIDE-rated tournament?",
            ],
        },
        "player_question": {
            "examples": [
                "Can I offer a draw before making my move?",
                "My opponent touched a piece, does he have to move it?",
            ],
        },
    },
}


def build_prompt(corpus: str, num_questions: int) -> str:
    """Build system prompt based on corpus."""
    cats = QUESTION_CATEGORIES[corpus]

    if corpus == "ffe":
        examples = "\n".join(
            f"- {ex}" for cat in cats.values() for ex in cat["examples"][:2]
        )
        return f"""Tu es un arbitre d'echecs FFE experimente.
Genere {num_questions} questions REALISTES basees sur le texte fourni.

CATEGORIES (varier):
1. ARBITRE TERRAIN - cas particuliers en competition
2. ARBITRE ORGANISATEUR - organisation tournoi, formation
3. JOUEUR - questions orales (langage familier)

EXEMPLES:
{examples}

REGLES:
- FRANCAIS uniquement
- La reponse DOIT etre dans le texte
- Style naturel, pas academique

JSON UNIQUEMENT:
{{"questions": [{{"question": "...", "category": "arbitre_terrain|arbitre_organisateur|question_joueur", "difficulty": "easy|medium|hard"}}]}}"""
    else:
        examples = "\n".join(
            f"- {ex}" for cat in cats.values() for ex in cat["examples"][:2]
        )
        return f"""You are an experienced FIDE arbiter.
Generate {num_questions} REALISTIC questions based on the provided text.

CATEGORIES (must vary):
1. ARBITER FIELD - specific cases during tournament
2. ARBITER ORGANIZER - organization, certification
3. PLAYER - oral questions (casual language)

EXAMPLES:
{examples}

RULES:
- ENGLISH only
- Answer MUST be in the text
- Natural style, not academic

JSON ONLY:
{{"questions": [{{"question": "...", "category": "arbiter_field|arbiter_organizer|player_question", "difficulty": "easy|medium|hard"}}]}}"""


def generate_questions(
    client: Cerebras,
    chunk_text: str,
    chunk_id: str,
    corpus: str = "ffe",
    num_questions: int = 3,
) -> list[dict]:
    """Generate questions for a chunk."""
    system_prompt = build_prompt(corpus, num_questions)

    if corpus == "fide":
        user_prompt = f'Text:\n"""\n{chunk_text[:2500]}\n"""\n\nGenerate {num_questions} questions.'
    else:
        user_prompt = f'Texte:\n"""\n{chunk_text[:2500]}\n"""\n\nGenere {num_questions} questions.'

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=800,
                temperature=0.7,
            )

            content = response.choices[0].message.content
            if not content:
                continue

            # Extract JSON
            content = content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]

            data = json.loads(content)
            questions = data.get("questions", [])

            for q in questions:
                q["chunk_id"] = chunk_id
                q["corpus"] = corpus

            return questions

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  Error {chunk_id}: {e}")
                return []
            time.sleep(1)

    return []


def run_generation(
    api_key: str,
    chunks_path: str,
    corpus: str = "ffe",
    max_chunks: int | None = None,
    questions_per_chunk: int = 3,
) -> list[dict]:
    """Run the generation pipeline."""
    client = Cerebras(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load chunks
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks", data)

    if max_chunks:
        chunks = chunks[:max_chunks]

    lang = "FR" if corpus == "ffe" else "EN"
    print(f"\n{'='*60}")
    print(f"CEREBRAS GENERATION - {corpus.upper()} ({lang})")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Chunks: {len(chunks)}")
    print(f"Questions/chunk: {questions_per_chunk}")
    print(f"{'='*60}\n")

    all_questions = []
    errors = 0
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("id", f"chunk-{i}")
        chunk_text = chunk.get("text", "")

        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = len(all_questions) / elapsed * 60 if elapsed > 0 else 0
            print(f"[{i:4d}/{len(chunks)}] {len(all_questions):4d} Q ({rate:.0f}/min)")

        questions = generate_questions(
            client=client,
            chunk_text=chunk_text,
            chunk_id=chunk_id,
            corpus=corpus,
            num_questions=questions_per_chunk,
        )

        if questions:
            all_questions.extend(questions)
        else:
            errors += 1

        # Checkpoint every 100
        if (i + 1) % 100 == 0:
            checkpoint = OUTPUT_DIR / f"checkpoint_{corpus}_{i+1}.json"
            with open(checkpoint, "w", encoding="utf-8") as f:
                json.dump(all_questions, f, ensure_ascii=False, indent=2)
            print(f"  >> Checkpoint: {checkpoint}")

    # Final save
    elapsed = time.time() - start_time
    output_path = OUTPUT_DIR / f"synthetic_questions_{corpus}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE - {corpus.upper()}")
    print(f"{'='*60}")
    print(f"Questions: {len(all_questions)}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Rate: {len(all_questions)/elapsed*60:.0f} Q/min")
    print(f"Output: {output_path}")

    return all_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["ffe", "fide"], default="ffe")
    parser.add_argument("--test", action="store_true", help="Test on 5 chunks")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("Error: CEREBRAS_API_KEY required")
        return

    chunks_path = (
        "corpus/processed/chunks_for_embedding_fr.json"
        if args.corpus == "ffe"
        else "corpus/processed/chunks_for_embedding_intl.json"
    )

    max_chunks = 5 if args.test else args.max_chunks

    run_generation(
        api_key=api_key,
        chunks_path=chunks_path,
        corpus=args.corpus,
        max_chunks=max_chunks,
    )


if __name__ == "__main__":
    main()
