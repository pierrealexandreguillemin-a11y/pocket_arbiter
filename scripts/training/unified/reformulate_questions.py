#!/usr/bin/env python3
"""
Etape 2: Reformulation Guidee par Chunk (UNIFIED_TRAINING_DATA_SPEC.md)

Reformule les questions du Gold Standard en langage courant.
Validation BY DESIGN: le LLM voit le chunk cible lors de la reformulation.

ISO 42001 A.6.2.2 - Provenance tracable
ISO 25010 - Exactitude fonctionnelle

Usage:
    python -m scripts.training.unified.reformulate_questions \
        --input data/training/unified/gs_with_chunks.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output data/training/unified/gs_reformulated.json \
        --provider gemini \
        --model gemini-2.0-flash
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "data" / "training" / "unified" / "gs_with_chunks.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "corpus" / "processed" / "chunks_mode_b_fr.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "unified" / "gs_reformulated.json"

REFORMULATION_PROMPT = """Tu es un formateur d'arbitres d'echecs FFE.

CONTEXTE (extrait du reglement):
---
{chunk_text}
---

QUESTION OFFICIELLE (examen DNA):
{original_question}

REPONSE OFFICIELLE:
{answer_text}

TACHE: Reformule la question en langage courant, comme un joueur ou arbitre debutant la poserait oralement.

CONTRAINTES:
1. La reponse DOIT toujours etre trouvable dans le CONTEXTE ci-dessus
2. Garde le meme sens que la question originale
3. Style oral, naturel, tutoriement accepte
4. Vocabulaire courant (eviter jargon technique sauf necessaire)
5. Longueur similaire a l'original (1-2 phrases)
6. Ne pas inclure d'elements de reponse dans la question

EXEMPLES:
- Original: "Quand dit-on qu'un joueur a le trait ?"
- Reformule: "C'est a qui de jouer apres que l'adversaire a appuye sur la pendule ?"

- Original: "L'arbitre qui observe une position illegale..."
- Reformule: "Si l'arbitre voit une position incorrecte mais n'a pas vu le coup, que doit-il faire ?"

Reponds UNIQUEMENT avec la question reformulee, sans guillemets ni explication."""


class LLMClient(Protocol):
    """Protocol pour les clients LLM."""

    def generate(self, prompt: str) -> str: ...


class MockLLMClient:
    """Client mock pour tests (retourne question reformulee simulee)."""

    def generate(self, prompt: str) -> str:
        # Extract original question from prompt and add mock prefix
        lines = prompt.split("\n")
        for i, line in enumerate(lines):
            if "QUESTION OFFICIELLE" in line and i + 1 < len(lines):
                original = lines[i + 1].strip()
                return f"[MOCK] {original}"
        return "Question mock reformulee ?"


class GeminiClient:
    """Client pour l'API Google Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required")
        self.model = model
        self._client = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def generate(self, prompt: str) -> str:
        response = self.client.generate_content(prompt)
        return response.text.strip()


class AnthropicClient:
    """Client pour l'API Anthropic Claude."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-haiku-20240307",
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
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()


class OpenAIClient:
    """Client pour l'API OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.model = model

    def generate(self, prompt: str) -> str:
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()


class OllamaClient:
    """Client pour Ollama local."""

    def __init__(
        self,
        model: str = "gemma2:9b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()


def create_llm_client(provider: str = "mock", **kwargs: Any) -> LLMClient:
    """Create LLM client based on provider."""
    providers = {
        "mock": MockLLMClient,
        "gemini": GeminiClient,
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "ollama": OllamaClient,
    }
    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {list(providers.keys())}"
        )
    return providers[provider](**{k: v for k, v in kwargs.items() if v is not None})


def load_json_file(path: Path) -> Any:
    """Load JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_chunk_index(chunks_data: Any) -> dict[str, dict]:
    """Build index of chunks by ID for fast lookup."""
    chunks = (
        chunks_data.get("chunks", chunks_data)
        if isinstance(chunks_data, dict)
        else chunks_data
    )
    return {chunk["id"]: chunk for chunk in chunks}


def reformulate_question(
    question: dict[str, Any],
    chunk: dict[str, Any],
    llm: LLMClient,
    max_retries: int = 3,
) -> str:
    """Reformulate a single question with chunk context (BY DESIGN)."""
    prompt = REFORMULATION_PROMPT.format(
        chunk_text=chunk.get("text", "")[:3000],
        original_question=question.get("question", ""),
        answer_text=question.get("answer_text", "Non specifie"),
    )

    for attempt in range(max_retries):
        try:
            result = llm.generate(prompt)
            # Clean up result
            result = result.strip().strip('"').strip("'")
            if result and len(result) > 10:
                return result
            raise ValueError(f"Result too short: {result}")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to reformulate: {question.get('id')}")
                return question.get("question", "")  # Fallback to original

    return question.get("question", "")


def process_questions(
    gs_data: dict[str, Any],
    chunk_index: dict[str, dict],
    llm: LLMClient,
    limit: int | None = None,
    skip_existing: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Process all questions, reformulating those with chunks."""
    questions = gs_data.get("questions", [])
    model_name = getattr(llm, "model", "unknown")

    stats = {
        "total": len(questions),
        "reformulated": 0,
        "skipped_no_chunk": 0,
        "skipped_existing": 0,
        "failed": 0,
    }

    questions_to_process = questions[:limit] if limit else questions

    for question in tqdm(questions_to_process, desc="Reformulating"):
        qid = question.get("id", "unknown")
        chunk_id = question.get("expected_chunk_id")

        # Skip if no chunk (adversarial)
        if not chunk_id:
            stats["skipped_no_chunk"] += 1
            continue

        # Skip if already reformulated
        if skip_existing and "question_reformulated" in question:
            stats["skipped_existing"] += 1
            continue

        # Get chunk
        chunk = chunk_index.get(chunk_id)
        if not chunk:
            logger.warning(f"Chunk not found: {chunk_id} for {qid}")
            stats["failed"] += 1
            continue

        # Reformulate
        reformulated = reformulate_question(question, chunk, llm)

        if reformulated and reformulated != question.get("question"):
            question["question_reformulated"] = reformulated
            question["original_annales"] = question.get("question")
            question["reformulation_metadata"] = {
                "model": model_name,
                "date": datetime.now().isoformat()[:10],
                "method": "BY_DESIGN",
            }
            stats["reformulated"] += 1
        else:
            stats["failed"] += 1

    report = {
        "statistics": stats,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
    }

    return gs_data, report


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reformulate GS questions with chunk context (BY DESIGN)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input GS with chunks JSON",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        default=DEFAULT_CHUNKS,
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output reformulated GS JSON",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="mock",
        choices=["mock", "gemini", "anthropic", "openai", "ollama"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model name (provider-specific)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of questions to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-reformulate even if already done",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("ETAPE 2: Reformulation BY DESIGN")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading GS: {args.input}")
    gs_data = load_json_file(args.input)

    logger.info(f"Loading chunks: {args.chunks}")
    chunks_data = load_json_file(args.chunks)
    chunk_index = build_chunk_index(chunks_data)
    logger.info(f"  Indexed {len(chunk_index)} chunks")

    # Create LLM client
    logger.info(f"Creating LLM client: {args.provider}")
    llm_kwargs = {}
    if args.model:
        llm_kwargs["model"] = args.model
    llm = create_llm_client(args.provider, **llm_kwargs)
    logger.info(f"  Model: {getattr(llm, 'model', 'default')}")

    # Process
    logger.info("Processing questions...")
    enriched_gs, report = process_questions(
        gs_data,
        chunk_index,
        llm,
        limit=args.limit,
        skip_existing=not args.force,
    )

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(enriched_gs, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to: {args.output}")

    # Save report
    report_path = args.output.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    stats = report["statistics"]
    logger.info("")
    logger.info("SUMMARY:")
    logger.info(f"  Total questions: {stats['total']}")
    logger.info(f"  Reformulated: {stats['reformulated']}")
    logger.info(f"  Skipped (no chunk): {stats['skipped_no_chunk']}")
    logger.info(f"  Skipped (existing): {stats['skipped_existing']}")
    logger.info(f"  Failed: {stats['failed']}")


if __name__ == "__main__":
    main()
