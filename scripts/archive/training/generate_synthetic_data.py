"""
Generation de donnees synthetiques pour fine-tuning - Pocket Arbiter

Genere des paires (question, chunk) synthetiques via LLM.

ISO Reference: ISO/IEC 42001 A.6.2.2, ISO/IEC 25010 S4.2

Usage:
    python -m scripts.training.generate_synthetic_data \
        --chunks corpus/processed/chunks_fr.json \
        --output data/training/synthetic_pairs.jsonl
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from scripts.pipeline.utils import get_timestamp, load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Tu es un arbitre d'echecs expert. Voici un extrait de reglement:

---
{chunk_text}
---

Genere exactement {num_questions} questions en francais qu'un arbitre pourrait poser.
Reponds UNIQUEMENT avec un tableau JSON: ["question1", "question2"]"""


class LLMClient(Protocol):
    """Protocol pour les clients LLM."""

    def generate(self, prompt: str) -> str: ...


class MockLLMClient:
    """Client LLM mock pour les tests."""

    def generate(self, prompt: str) -> str:
        return '["Question mock 1 ?", "Question mock 2 ?"]'


class AnthropicClient:
    """Client pour l'API Anthropic Claude."""

    def __init__(
        self, api_key: str | None = None, model: str = "claude-3-haiku-20240307"
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        self.model = model

    def generate(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class OpenAIClient:
    """Client pour l'API OpenAI."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.model = model

    def generate(self, prompt: str) -> str:
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


def create_llm_client(provider: str = "mock", **kwargs: Any) -> LLMClient:
    """Cree un client LLM selon le provider specifie."""
    if provider == "mock":
        return MockLLMClient()
    elif provider == "anthropic":
        return AnthropicClient(**{k: v for k, v in kwargs.items() if v})
    elif provider == "openai":
        return OpenAIClient(**{k: v for k, v in kwargs.items() if v})
    raise ValueError(f"Unknown provider: {provider}")


def parse_questions_response(response: str) -> list[str]:
    """Parse la reponse JSON du LLM."""
    response = response.strip()
    start, end = response.find("["), response.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array found: {response[:100]}")
    questions = json.loads(response[start:end])
    if not isinstance(questions, list):
        raise ValueError(f"Expected list, got {type(questions)}")
    return [q.strip() for q in questions if isinstance(q, str) and q.strip()]


def generate_questions_for_chunk(
    chunk_text: str,
    llm: LLMClient,
    num_questions: int = 2,
    max_retries: int = 3,
) -> list[str]:
    """Genere des questions pour un chunk donne."""
    prompt = PROMPT_TEMPLATE.format(
        chunk_text=chunk_text[:2000], num_questions=num_questions
    )
    for attempt in range(max_retries):
        try:
            response = llm.generate(prompt)
            return parse_questions_response(response)[:num_questions]
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed after {max_retries} attempts") from e
            time.sleep(1)
    return []


def generate_synthetic_dataset(
    chunks: list[dict],
    llm: LLMClient,
    num_questions: int = 2,
    max_chunks: int | None = None,
) -> list[dict]:
    """Genere un dataset synthetique de paires (query, chunk)."""
    pairs: list[dict] = []
    chunks_to_process = chunks[:max_chunks] if max_chunks else chunks

    for chunk in tqdm(chunks_to_process, desc="Generating questions"):
        try:
            questions = generate_questions_for_chunk(chunk["text"], llm, num_questions)
            for q in questions:
                pairs.append(
                    {
                        "query": q,
                        "positive": chunk["text"],
                        "chunk_id": chunk["id"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                    }
                )
        except RuntimeError as e:
            logger.warning(f"Failed for {chunk['id']}: {e}")
            continue

    logger.info(f"Generated {len(pairs)} pairs from {len(chunks_to_process)} chunks")
    return pairs


def save_pairs_jsonl(pairs: list[dict], output_path: Path) -> None:
    """Sauvegarde les paires au format JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(pairs)} pairs to {output_path}")


def load_pairs_jsonl(input_path: Path) -> list[dict]:
    """Charge les paires depuis un fichier JSONL."""
    pairs = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def main() -> None:
    """Point d'entree CLI."""
    parser = argparse.ArgumentParser(description="Generation donnees synthetiques")
    parser.add_argument("--chunks", "-c", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--questions-per-chunk", "-q", type=int, default=2)
    parser.add_argument(
        "--provider", "-p", choices=["mock", "anthropic", "openai"], default="mock"
    )
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data = load_json(args.chunks)
    chunks = data.get("chunks", [])
    logger.info(f"Found {len(chunks)} chunks")

    llm = create_llm_client(provider=args.provider, model=args.model)
    pairs = generate_synthetic_dataset(
        chunks, llm, args.questions_per_chunk, args.max_chunks
    )

    save_pairs_jsonl(pairs, args.output)
    report = {
        "total_chunks": len(chunks),
        "total_pairs": len(pairs),
        "questions_per_chunk": args.questions_per_chunk,
        "provider": args.provider,
        "timestamp": get_timestamp(),
    }
    save_json(report, args.output.with_suffix(".report.json"))
    logger.info(f"Generated {len(pairs)} pairs")


if __name__ == "__main__":
    main()
