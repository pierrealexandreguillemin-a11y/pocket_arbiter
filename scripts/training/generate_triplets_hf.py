"""
Generate synthetic questions for triplet training using HuggingFace Inference API.

Genere des questions realistes pour l'entrainement d'un modele de retrieval:
- Corpus FFE: questions en francais (arbitres terrain, organisateurs, joueurs)
- Corpus FIDE: questions en anglais (multilingual)

100% gratuit via HF Inference API.
~50 samples/min rate limit.

Usage:
    # Corpus FFE (francais)
    python scripts/pipeline/generate_triplets_hf.py --corpus ffe --max-chunks 10

    # Corpus FIDE (anglais)
    python scripts/pipeline/generate_triplets_hf.py --corpus fide --chunks corpus/processed/chunks_fide.json

    # Avec HF token (optionnel, pour rate limit plus eleve)
    HF_TOKEN=hf_xxx python scripts/pipeline/generate_triplets_hf.py --corpus ffe
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import InferenceClient

# Configuration
MODEL = "HuggingFaceH4/zephyr-7b-beta"  # Gratuit sur HF Inference
OUTPUT_DIR = Path("data/synthetic_triplets")
QUESTIONS_PER_CHUNK = 3
RATE_LIMIT_DELAY = 1.5  # secondes entre requetes (safe pour free tier)

# Categories de questions par corpus
QUESTION_CATEGORIES = {
    "ffe": {
        "arbitre_terrain": {
            "description": "Questions d'un arbitre experimente sur le terrain (cas particuliers)",
            "examples": [
                "Quel sera l'Elo d'un joueur apres ce tournoi avec ces resultats?",
                "Quels sont les affichages obligatoires en salle de jeu?",
                "Un portable sonne pendant la partie, quelle sanction?",
                "Le joueur arrive 10 minutes apres le debut, est-il forfait?",
                "Comment gerer une reclamation sur un mat illegal?",
                "Quelle est la procedure si un joueur refuse de signer la feuille?",
            ],
        },
        "arbitre_organisateur": {
            "description": "Questions pour l'organisation d'un tournoi ou formation arbitrale",
            "examples": [
                "Quelles sont les conditions pour devenir arbitre AF2?",
                "Quel est le delai d'homologation d'un tournoi FFE?",
                "Quel materiel est obligatoire pour un tournoi homologue?",
                "Comment calculer les droits d'engagement?",
                "Quelle est la procedure pour homologuer un open?",
                "Combien d'arbitres faut-il pour un tournoi de 100 joueurs?",
            ],
        },
        "question_joueur": {
            "description": "Questions posees par des joueurs (langage oral/verbal)",
            "examples": [
                "J'ai le droit de proposer nulle quand exactement?",
                "Il a touche une piece, il doit la jouer non?",
                "C'est quoi le departage au juste?",
                "Je peux aller aux toilettes pendant ma partie?",
                "Mon adversaire ecrit ses coups avant de jouer, c'est legal?",
                "J'ai oublie d'appuyer sur la pendule, qu'est-ce qui se passe?",
            ],
        },
    },
    "fide": {
        "arbiter_field": {
            "description": "Questions from an experienced arbiter during a tournament",
            "examples": [
                "What is the penalty for a mobile phone ringing?",
                "How to handle a claim of threefold repetition?",
                "When can a player claim a draw under the 50-move rule?",
                "What happens if a player makes an illegal move in time trouble?",
                "How to proceed when both flags have fallen?",
            ],
        },
        "arbiter_organizer": {
            "description": "Questions for tournament organization or arbiter certification",
            "examples": [
                "What are the requirements to become a FIDE Arbiter?",
                "What equipment is mandatory for a FIDE-rated tournament?",
                "How to submit results for FIDE rating?",
                "What is the time control for classical FIDE games?",
            ],
        },
        "player_question": {
            "description": "Questions asked by players (natural spoken language)",
            "examples": [
                "Can I offer a draw before making my move?",
                "My opponent touched a piece, does he have to move it?",
                "What's the tiebreak system in this tournament?",
                "Am I allowed to leave the playing hall?",
            ],
        },
    },
}


def setup_client() -> InferenceClient:
    """Configure HuggingFace Inference client."""
    token = os.environ.get("HF_TOKEN")
    return InferenceClient(model=MODEL, token=token)


def _build_prompt_ffe(categories: dict, num_questions: int) -> str:
    """Build system prompt for FFE corpus (French)."""
    cat_terrain = categories["arbitre_terrain"]
    cat_orga = categories["arbitre_organisateur"]
    cat_joueur = categories["question_joueur"]

    examples_terrain = "\n".join(f"  - {ex}" for ex in cat_terrain["examples"][:3])
    examples_orga = "\n".join(f"  - {ex}" for ex in cat_orga["examples"][:3])
    examples_joueur = "\n".join(f"  - {ex}" for ex in cat_joueur["examples"][:3])

    return f"""Tu es un arbitre d'echecs FFE experimente (AF3 minimum).
Tu generes des questions REALISTES que l'on te pose sur le terrain ou en formation.

TROIS CATEGORIES DE QUESTIONS (varier obligatoirement):

1. ARBITRE TERRAIN - Cas particuliers concrets en competition:
{examples_terrain}

2. ARBITRE ORGANISATEUR - Organisation tournoi ou formation arbitrale:
{examples_orga}

3. JOUEUR - Questions orales d'un joueur (langage familier, verbal):
{examples_joueur}

REGLES STRICTES:
- Langue: FRANCAIS uniquement
- La reponse DOIT etre trouvable dans le texte fourni
- Style: questions naturelles, pas academiques
- Jargon FFE: Elo, cadence, forfait, appariement, homologation, departage
- Genere {num_questions} questions: au moins 1 de chaque categorie si possible

FORMAT JSON UNIQUEMENT (pas de texte autour):
{{"questions": [{{"question": "...", "category": "arbitre_terrain|arbitre_organisateur|question_joueur", "difficulty": "easy|medium|hard"}}]}}"""


def _build_prompt_fide(categories: dict, num_questions: int) -> str:
    """Build system prompt for FIDE corpus (English/multilingual)."""
    cat_field = categories["arbiter_field"]
    cat_orga = categories["arbiter_organizer"]
    cat_player = categories["player_question"]

    examples_field = "\n".join(f"  - {ex}" for ex in cat_field["examples"][:3])
    examples_orga = "\n".join(f"  - {ex}" for ex in cat_orga["examples"][:3])
    examples_player = "\n".join(f"  - {ex}" for ex in cat_player["examples"][:3])

    return f"""You are an experienced FIDE arbiter (IA or FA level).
Generate REALISTIC questions that arbiters or players ask during tournaments.

THREE CATEGORIES OF QUESTIONS (must vary):

1. ARBITER ON FIELD - Specific cases during competition:
{examples_field}

2. ARBITER/ORGANIZER - Tournament organization or certification:
{examples_orga}

3. PLAYER - Oral questions from players (casual spoken language):
{examples_player}

STRICT RULES:
- Language: ENGLISH only
- The answer MUST be found in the provided text
- Style: natural questions, not academic
- FIDE terminology: rating, time control, forfeit, pairing, tiebreak
- Generate {num_questions} questions: at least 1 from each category if possible

JSON FORMAT ONLY (no surrounding text):
{{"questions": [{{"question": "...", "category": "arbiter_field|arbiter_organizer|player_question", "difficulty": "easy|medium|hard"}}]}}"""


def generate_questions(
    client: InferenceClient,
    chunk_text: str,
    chunk_id: str,
    corpus: str = "ffe",
    num_questions: int = 3,
) -> list[dict]:
    """
    Generate synthetic questions for a chunk using HuggingFace Inference.

    Args:
        client: HuggingFace InferenceClient
        chunk_text: Text content of the chunk
        chunk_id: Unique identifier for the chunk
        corpus: "ffe" (French) or "fide" (English)
        num_questions: Number of questions to generate

    Returns:
        List of question dicts with keys: question, category, difficulty, chunk_id
    """
    categories = QUESTION_CATEGORIES.get(corpus, QUESTION_CATEGORIES["ffe"])

    if corpus == "fide":
        system_prompt = _build_prompt_fide(categories, num_questions)
        user_prompt = f"""FIDE regulatory text:
\"\"\"
{chunk_text[:2500]}
\"\"\"

Generate exactly {num_questions} varied questions based on this text."""
    else:
        system_prompt = _build_prompt_ffe(categories, num_questions)
        user_prompt = f"""Texte reglementaire FFE:
\"\"\"
{chunk_text[:2500]}
\"\"\"

Genere exactement {num_questions} questions variees basees sur ce texte."""

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
            temperature=0.7,
        )

        # Extract content
        raw_content = response.choices[0].message.content
        if raw_content is None:
            print(f"  Empty response for {chunk_id}")
            return []
        content = raw_content.strip()

        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]

        # Find JSON object or array
        content = content.strip()
        start_obj = content.find("{")
        start_arr = content.find("[")

        if start_obj >= 0 and (start_arr < 0 or start_obj < start_arr):
            # Object format {"questions": [...]}
            content = content[start_obj:]
            # Find matching closing brace
            depth = 0
            for i, c in enumerate(content):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        content = content[: i + 1]
                        break
        elif start_arr >= 0:
            # Array format [...]
            content = content[start_arr:]
            depth = 0
            for i, c in enumerate(content):
                if c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        content = content[: i + 1]
                        break

        data = json.loads(content)

        # Handle both {"questions": [...]} and [...] formats
        if isinstance(data, list):
            questions = data
        else:
            questions = data.get("questions", [])

        # Add chunk_id to each question
        for q in questions:
            q["chunk_id"] = chunk_id

        return questions

    except json.JSONDecodeError as e:
        print(f"  JSON error {chunk_id}: {e}")
        return []
    except Exception as e:
        print(f"  Error {chunk_id}: {type(e).__name__}: {e}")
        return []


def run_generation(
    chunks_path: str,
    output_dir: Path,
    corpus: str = "ffe",
    max_chunks: Optional[int] = None,
    questions_per_chunk: int = 3,
    delay: float = 1.5,
) -> list[dict]:
    """
    Run question generation pipeline.

    Args:
        chunks_path: Path to chunks JSON file
        output_dir: Output directory for results
        corpus: "ffe" (French) or "fide" (English)
        max_chunks: Limit number of chunks (None = all)
        questions_per_chunk: Questions to generate per chunk
        delay: Delay between API calls (rate limiting)

    Returns:
        List of all generated questions
    """
    # Validate corpus
    if corpus not in QUESTION_CATEGORIES:
        raise ValueError(f"Invalid corpus '{corpus}'. Must be 'ffe' or 'fide'.")

    # Setup
    client = setup_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", data)  # Handle both formats
    if max_chunks:
        chunks = chunks[:max_chunks]

    lang = "francais" if corpus == "ffe" else "english"
    print("=== HuggingFace Triplet Generation ===")
    print(f"Model: {MODEL}")
    print(f"Corpus: {corpus.upper()} ({lang})")
    print(f"Chunks: {len(chunks)}")
    print(f"Questions/chunk: {questions_per_chunk}")
    print(f"Estimated total: {len(chunks) * questions_per_chunk}")
    print(f"Rate limit delay: {delay}s")
    print()

    all_questions: list[dict] = []
    errors = 0
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("id", f"chunk-{i}")
        chunk_text = chunk.get("text", "")

        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = len(all_questions) / elapsed * 60 if elapsed > 0 else 0
            print(
                f"[{i}/{len(chunks)}] {len(all_questions)} questions ({rate:.1f}/min)"
            )

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

        # Checkpoint every 100 chunks
        if (i + 1) % 100 == 0:
            checkpoint_path = output_dir / f"checkpoint_{corpus}_{i+1}.json"
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(all_questions, f, ensure_ascii=False, indent=2)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Rate limiting
        time.sleep(delay)

    # Final save with corpus in filename
    elapsed = time.time() - start_time
    output_path = output_dir / f"synthetic_questions_{corpus}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    print()
    print(f"=== DONE ({corpus.upper()}) ===")
    print(f"Total questions: {len(all_questions)}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Rate: {len(all_questions)/elapsed*60:.1f} questions/min")
    print(f"Output: {output_path}")

    return all_questions


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic questions using HuggingFace Inference API"
    )
    parser.add_argument(
        "--corpus",
        choices=["ffe", "fide"],
        default="ffe",
        help="Corpus type: 'ffe' (French) or 'fide' (English). Default: ffe",
    )
    parser.add_argument(
        "--chunks",
        default=None,
        help="Path to chunks JSON file (default: auto-detect based on corpus)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit number of chunks (default: all)",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=3,
        help="Questions per chunk (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay between requests in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic_triplets",
        help="Output directory",
    )

    args = parser.parse_args()

    # Auto-detect chunks path based on corpus if not specified
    if args.chunks is None:
        if args.corpus == "ffe":
            chunks_path = "corpus/processed/chunks_for_embedding_fr.json"
        else:
            chunks_path = "corpus/processed/chunks_fide.json"
    else:
        chunks_path = args.chunks

    run_generation(
        chunks_path=chunks_path,
        output_dir=Path(args.output),
        corpus=args.corpus,
        max_chunks=args.max_chunks,
        questions_per_chunk=args.questions,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
