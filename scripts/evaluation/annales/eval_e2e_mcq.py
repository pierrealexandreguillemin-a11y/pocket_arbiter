#!/usr/bin/env python3
"""
Evaluation end-to-end RAG sur questions MCQ (annales FFE).

Bypasse le probleme de chunk_id corrompu dans le GS:
on mesure si le RAG (retrieval + generation) donne la bonne reponse MCQ.

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency (accuracy)
    - ISO/IEC 42001 - AI traceability

Usage:
    python -m scripts.evaluation.annales.eval_e2e_mcq \
        --db corpus/processed/corpus_mode_b_fr.db \
        --provider gemini --model gemini-2.0-flash \
        --top-k 5 --mode hybrid
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[3]
GS_PATH = (
    PROJECT_ROOT / "tests" / "data" / "gold_standard_annales_fr_v8_adversarial.json"
)
DEFAULT_DB = PROJECT_ROOT / "corpus" / "processed" / "corpus_mode_b_fr.db"
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmarks"

# === Prompt MCQ ===

MCQ_PROMPT = """Tu es un arbitre d'echecs expert. Reponds a la question en choisissant UNE SEULE lettre (A, B, C ou D).

CONTEXTE (extraits du reglement):
---
{context}
---

QUESTION:
{question}

CHOIX:
{choices}

Reponds UNIQUEMENT avec la lettre de la bonne reponse (A, B, C ou D), sans explication."""


# === LLM Clients ===


class LLMClient(Protocol):
    """Protocol pour les clients LLM."""

    def generate(self, prompt: str) -> str: ...


class MockLLMClient:
    """Client mock: repond toujours A."""

    def generate(self, prompt: str) -> str:
        return "A"


class OllamaClient:
    """Client Ollama local."""

    def __init__(self, model: str = "gemma2:latest"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        import urllib.request

        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310 — localhost Ollama only
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()


class GeminiClient:
    """Client Google Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required")
        self.model = model
        self._client: Any = None

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


# === Core Functions ===


def load_testable_mcq(gs_path: Path = GS_PATH) -> list[dict]:
    """Charge les questions MCQ testables du GS v8.

    Filtre: answerable + has mcq_answer + not requires_context.

    Returns:
        Liste de dicts avec id, question, choices, mcq_answer, metadata.
    """
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    questions = []
    for q in gs["questions"]:
        content = q.get("content", {})
        mcq = q.get("mcq", {})

        # Skip unanswerable
        if content.get("is_impossible", False):
            continue

        # Skip no MCQ answer
        if not mcq.get("mcq_answer"):
            continue

        # Skip requires_context
        audit = q.get("audit", {})
        history = str(audit.get("history", ""))
        if "requires_context" in history:
            continue

        # Extract classification metadata
        classification = q.get("classification", {})

        questions.append(
            {
                "id": q["id"],
                "question": content["question"],
                "choices": mcq["choices"],
                "mcq_answer": mcq["mcq_answer"],
                "reasoning_class": classification.get("reasoning_class", "unknown"),
                "difficulty": classification.get("difficulty"),
                "source_uuid": q.get("provenance", {}).get("docs", ["?"])[0],
            }
        )

    return questions


def retrieve_context(
    db_path: Path,
    question: str,
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[dict]:
    """Recupere les chunks pertinents pour une question.

    Args:
        db_path: Chemin vers la base SQLite.
        question: Texte de la question.
        top_k: Nombre de chunks a recuperer.
        mode: 'vector' ou 'hybrid'.

    Returns:
        Liste de chunks (id, text, source, page, score).
    """
    from scripts.pipeline.embeddings import embed_query, load_embedding_model
    from scripts.pipeline.export_search import retrieve_hybrid, smart_retrieve

    # Load model (cached after first call via module-level)
    if not hasattr(retrieve_context, "_model"):
        logger.info("Loading embedding model...")
        retrieve_context._model = load_embedding_model()

    model = retrieve_context._model
    query_emb = embed_query(question, model)

    if mode == "hybrid":
        return retrieve_hybrid(db_path, query_emb, question, top_k=top_k)
    else:
        return smart_retrieve(db_path, query_emb, question, top_k=top_k)


def format_context(chunks: list[dict], max_chars: int = 3000) -> str:
    """Formate les chunks en contexte textuel pour le prompt."""
    parts = []
    total = 0
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        source = chunk.get("source", "?")
        page = chunk.get("page", "?")
        header = f"[{source} p.{page}]"
        entry = f"{header}\n{text}"
        if total + len(entry) > max_chars and parts:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


def format_choices(choices: dict[str, str]) -> str:
    """Formate les choix MCQ."""
    return "\n".join(f"{k}: {v}" for k, v in sorted(choices.items()))


def extract_answer_letter(response: str) -> str | None:
    """Extrait la lettre de reponse du LLM.

    Gere: 'A', 'La reponse est B', 'C.', etc.
    """
    cleaned = response.strip()

    # Direct single letter
    if len(cleaned) == 1 and cleaned.upper() in "ABCD":
        return cleaned.upper()

    # Letter at start: "A.", "B:", "C -", "D)"
    match = re.match(r"^([A-Da-d])[\s.:\-)\]]", cleaned)
    if match:
        return match.group(1).upper()

    # "La reponse est X" / "Answer: X"
    match = re.search(
        r"(?:reponse|answer|lettre)\s*(?:est|is|:)\s*([A-Da-d])", cleaned, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Last resort: first UPPERCASE A-D as standalone word (case-sensitive on original)
    match = re.search(r"\b([A-D])\b", cleaned)
    if match:
        return match.group(1)

    return None


def run_eval(
    questions: list[dict],
    llm: LLMClient,
    db_path: Path = DEFAULT_DB,
    top_k: int = 5,
    mode: str = "hybrid",
    delay: float = 0.5,
) -> dict:
    """Execute l'evaluation end-to-end.

    Returns:
        Dict avec accuracy, details par question, segmentation.
    """
    results: list[dict] = []
    correct = 0
    errors = 0

    for q in tqdm(questions, desc="Eval MCQ"):
        # Retrieve
        chunks = retrieve_context(db_path, q["question"], top_k=top_k, mode=mode)
        context = format_context(chunks)
        choices_text = format_choices(q["choices"])

        # Generate
        prompt = MCQ_PROMPT.format(
            context=context,
            question=q["question"],
            choices=choices_text,
        )

        try:
            raw_response = llm.generate(prompt)
            predicted = extract_answer_letter(raw_response)
        except Exception as e:
            logger.warning("LLM error for %s: %s", q["id"], e)
            raw_response = str(e)
            predicted = None
            errors += 1

        expected = q["mcq_answer"]
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        results.append(
            {
                "id": q["id"],
                "question": q["question"][:100],
                "expected": expected,
                "predicted": predicted,
                "raw_response": raw_response[:200],
                "correct": is_correct,
                "reasoning_class": q["reasoning_class"],
                "difficulty": q["difficulty"],
                "source_uuid": q["source_uuid"],
                "retrieved_chunks": [
                    {"id": c["id"], "page": c.get("page", 0)} for c in chunks[:3]
                ],
            }
        )

        if delay > 0:
            time.sleep(delay)

    total = len(questions)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "results": results,
    }


def compute_segments(results: list[dict]) -> dict:
    """Calcule l'accuracy par segment."""

    segments: dict[str, dict] = {}

    for key_field in ["reasoning_class", "source_uuid"]:
        groups: dict[str, list[bool]] = {}
        for r in results:
            val = r.get(key_field, "unknown")
            groups.setdefault(str(val), []).append(r["correct"])

        seg = {}
        for name, bools in sorted(groups.items()):
            seg[name] = {
                "accuracy": sum(bools) / len(bools),
                "correct": sum(bools),
                "total": len(bools),
            }
        segments[key_field] = seg

    return segments


def print_report(eval_result: dict, segments: dict, mode: str, model: str) -> None:
    """Affiche le rapport console."""
    print()
    print(f"=== EVAL E2E MCQ — {datetime.now().strftime('%Y-%m-%d')} ===")
    print(f"{eval_result['total']} questions | {model} | mode={mode}")
    print()
    print(
        f"  Accuracy:  {eval_result['accuracy']:.1%} ({eval_result['correct']}/{eval_result['total']})"
    )
    print(f"  Errors:    {eval_result['errors']}")
    print()

    # Par reasoning_class
    print("--- Par reasoning_class ---")
    for name, stats in segments.get("reasoning_class", {}).items():
        print(
            f"  {name:20s}  {stats['accuracy']:.1%}  ({stats['correct']}/{stats['total']})"
        )
    print()

    # Top-10 documents (pires)
    print("--- Par document (pire accuracy) ---")
    doc_stats = segments.get("source_uuid", {})
    sorted_docs = sorted(doc_stats.items(), key=lambda x: x[1]["accuracy"])
    for name, stats in sorted_docs[:10]:
        print(
            f"  {name:50s}  {stats['accuracy']:.1%}  ({stats['correct']}/{stats['total']})"
        )
    print()

    # Exemples d'erreurs
    wrong = [r for r in eval_result["results"] if not r["correct"]]
    print(f"--- Erreurs ({len(wrong)} questions) ---")
    for r in wrong[:15]:
        print(f"  {r['id']}: expected={r['expected']} got={r['predicted']}")
        print(f"    Q: {r['question']}")
        print()


def export_results(
    eval_result: dict,
    segments: dict,
    mode: str,
    model: str,
    top_k: int,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Exporte les resultats en JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"eval_e2e_mcq_{date_str}.json"
    path = output_dir / filename

    export = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": model,
            "mode": mode,
            "top_k": top_k,
        },
        "accuracy": eval_result["accuracy"],
        "correct": eval_result["correct"],
        "total": eval_result["total"],
        "errors": eval_result["errors"],
        "segments": segments,
        "results": eval_result["results"],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    logger.info("Results exported to %s", path)
    return path


def run_retrieve_phase(
    questions: list[dict],
    db_path: Path,
    top_k: int,
    mode: str,
    output_path: Path,
) -> Path:
    """Phase 1: retrieval seul, sauve prompts dans un JSON.

    Libere la RAM du modele embedding avant la phase generation.
    """
    prompts = []

    for q in tqdm(questions, desc="Retrieval"):
        chunks = retrieve_context(db_path, q["question"], top_k=top_k, mode=mode)
        context = format_context(chunks)
        choices_text = format_choices(q["choices"])
        prompt = MCQ_PROMPT.format(
            context=context,
            question=q["question"],
            choices=choices_text,
        )
        prompts.append(
            {
                "id": q["id"],
                "question": q["question"],
                "mcq_answer": q["mcq_answer"],
                "reasoning_class": q["reasoning_class"],
                "difficulty": q["difficulty"],
                "source_uuid": q["source_uuid"],
                "prompt": prompt,
                "retrieved_chunks": [
                    {"id": c["id"], "page": c.get("page", 0)} for c in chunks[:3]
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d prompts to %s", len(prompts), output_path)
    return output_path


def run_generate_phase(
    prompts_path: Path,
    llm: "LLMClient",
    delay: float = 0.2,
) -> dict:
    """Phase 2: generation depuis prompts pre-calcules (pas d'embedding model)."""
    with open(prompts_path, encoding="utf-8") as f:
        prompts = json.load(f)

    results: list[dict] = []
    correct = 0
    errors = 0

    for p in tqdm(prompts, desc="Generation"):
        try:
            raw_response = llm.generate(p["prompt"])
            predicted = extract_answer_letter(raw_response)
        except Exception as e:
            logger.warning("LLM error for %s: %s", p["id"], e)
            raw_response = str(e)
            predicted = None
            errors += 1

        expected = p["mcq_answer"]
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append(
            {
                "id": p["id"],
                "question": p["question"][:100],
                "expected": expected,
                "predicted": predicted,
                "raw_response": raw_response[:200],
                "correct": is_correct,
                "reasoning_class": p["reasoning_class"],
                "difficulty": p["difficulty"],
                "source_uuid": p["source_uuid"],
                "retrieved_chunks": p["retrieved_chunks"],
            }
        )

        if delay > 0:
            time.sleep(delay)

    total = len(prompts)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "errors": errors,
        "results": results,
    }


PROMPTS_PATH = OUTPUT_DIR / "eval_e2e_prompts.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval E2E MCQ RAG")
    parser.add_argument(
        "--phase",
        choices=["retrieve", "generate", "both"],
        default="both",
        help="Phase: retrieve (embedding only), generate (LLM only), both (OOM risk)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="SQLite database path",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini", "mock"],
        default="ollama",
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        default="gemma2:latest",
        help="Model name",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve")
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Retrieval mode",
    )
    parser.add_argument(
        "--delay", type=float, default=0.2, help="Delay between API calls (s)"
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")

    args = parser.parse_args()

    # Load questions
    questions = load_testable_mcq()
    logger.info("Loaded %d testable MCQ questions", len(questions))

    if args.limit > 0:
        questions = questions[: args.limit]
        logger.info("Limited to %d questions", len(questions))

    # Phase 1: Retrieve
    if args.phase in ("retrieve", "both"):
        run_retrieve_phase(questions, args.db, args.top_k, args.mode, PROMPTS_PATH)
        if args.phase == "retrieve":
            print("Phase 1 done. Run --phase generate to complete.")
            return

    # Phase 2: Generate
    if args.phase in ("generate", "both"):
        if not PROMPTS_PATH.exists():
            logger.error("No prompts file. Run --phase retrieve first.")
            return

        if args.provider == "ollama":
            llm: LLMClient = OllamaClient(model=args.model)
            logger.info("Using Ollama with model %s", args.model)
        elif args.provider == "gemini":
            llm = GeminiClient(model=args.model)
        else:
            llm = MockLLMClient()
            logger.info("Using MOCK LLM (always answers A)")

        eval_result = run_generate_phase(PROMPTS_PATH, llm, delay=args.delay)

        # Segments
        segments = compute_segments(eval_result["results"])

        # Report
        print_report(eval_result, segments, args.mode, args.model)

        # Export
        path = export_results(eval_result, segments, args.mode, args.model, args.top_k)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
