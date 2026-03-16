"""Generate questions for remaining chunks only."""

import json
import os
import time
from pathlib import Path

from cerebras.cloud.sdk import Cerebras

MODEL = "llama-3.3-70b"
OUTPUT_DIR = Path("data/synthetic_triplets")


def build_prompt(num_questions: int = 3) -> str:
    """Build French prompt for FFE."""
    return f"""Tu es un arbitre d'echecs FFE experimente.
Genere {num_questions} questions REALISTES basees sur le texte fourni.

CATEGORIES (varier):
1. ARBITRE TERRAIN - cas particuliers en competition
2. ARBITRE ORGANISATEUR - organisation tournoi, formation
3. JOUEUR - questions orales (langage familier)

REGLES:
- FRANCAIS uniquement
- La reponse DOIT etre dans le texte
- Style naturel, pas academique

JSON UNIQUEMENT:
{{"questions": [{{"question": "...", "category": "arbitre_terrain|arbitre_organisateur|question_joueur", "difficulty": "easy|medium|hard"}}]}}"""


def generate_questions(client: Cerebras, chunk_text: str, chunk_id: str) -> list[dict]:
    """Generate questions for a single chunk."""
    system_prompt = build_prompt(3)
    user_prompt = f'Texte:\n"""\n{chunk_text[:2500]}\n"""\n\nGenere 3 questions.'

    for attempt in range(3):
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
                q["corpus"] = "ffe"

            return questions

        except Exception as e:
            if attempt == 2:
                print(f"  Error {chunk_id}: {e}")
                return []
            time.sleep(1)

    return []


def main():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("Error: CEREBRAS_API_KEY environment variable required")
        return
    client = Cerebras(api_key=api_key)

    # Load remaining chunks
    remaining = json.load(open(OUTPUT_DIR / "remaining_chunks.json", encoding="utf-8"))

    # Load existing questions
    existing = json.load(
        open(OUTPUT_DIR / "synthetic_questions_ffe.json", encoding="utf-8")
    )

    print(f"{'=' * 60}")
    print("GENERATION CHUNKS RESTANTS - FFE")
    print(f"{'=' * 60}")
    print(f"Questions existantes: {len(existing)}")
    print(f"Chunks restants: {len(remaining)}")
    print(f"{'=' * 60}\n")

    all_questions = existing.copy()
    errors = 0
    start_time = time.time()

    for i, chunk in enumerate(remaining):
        chunk_id = chunk.get("id", f"chunk-{i}")
        chunk_text = chunk.get("text", "")

        if i % 10 == 0:
            elapsed = time.time() - start_time
            new_q = len(all_questions) - len(existing)
            rate = new_q / elapsed * 60 if elapsed > 0 else 0
            print(f"[{i:4d}/{len(remaining)}] +{new_q:4d} Q ({rate:.0f}/min)")

        questions = generate_questions(client, chunk_text, chunk_id)

        if questions:
            all_questions.extend(questions)
        else:
            errors += 1

        # Checkpoint every 50
        if (i + 1) % 50 == 0:
            with open(
                OUTPUT_DIR / "synthetic_questions_ffe.json", "w", encoding="utf-8"
            ) as f:
                json.dump(all_questions, f, ensure_ascii=False, indent=2)
            print(f"  >> Checkpoint: {len(all_questions)} questions")

    # Final save
    elapsed = time.time() - start_time
    with open(OUTPUT_DIR / "synthetic_questions_ffe.json", "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    new_questions = len(all_questions) - len(existing)
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"Nouvelles questions: {new_questions}")
    print(f"Total questions: {len(all_questions)}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
