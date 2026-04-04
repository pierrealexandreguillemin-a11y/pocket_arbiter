"""Phase A: Generate questions from corpus chunks with Gemma 3 4B IT teacher.

RAFT (Berkeley 2024, arXiv:2403.10131) Phase A:
- For each chunk, generate 1-2 realistic questions an arbiter could ask
- Questions generated from the chunk ALONE (no distractors)
- This ensures questions are grounded in the oracle text

Output: /kaggle/working/questions_v5.jsonl
Each line: {"chunk_id", "source", "page", "chunk_text", "question"}

Phase B (separate kernel): load questions, add distractors, generate CoT answers.

Input:  pguillemin/pocket-arbiter-eval-data (corpus_v2_fr.db)
        google/gemma-3/transformers/gemma-3-4b-it/1 (teacher model)
Output: /kaggle/working/questions_v5.jsonl
        /kaggle/working/gen_questions_metrics.json
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sqlite3  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
t_start = time.time()

# -- PHASE 0: Environment --------------------------------------------------
logger.info("=== PHASE 0: Environment ===")
assert torch.cuda.is_available(), "FATAL: No GPU detected"
GPU_PROPS = torch.cuda.get_device_properties(0)
GPU_VRAM_MB = GPU_PROPS.total_memory / 1024 / 1024
logger.info(
    "GPU: %s (%.0f MB VRAM, compute %d.%d)",
    torch.cuda.get_device_name(0),
    GPU_VRAM_MB,
    GPU_PROPS.major,
    GPU_PROPS.minor,
)
assert GPU_VRAM_MB >= 14000, f"FATAL: Need >= 14 GB VRAM, got {GPU_VRAM_MB:.0f} MB"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth>=2025.3"])

from unsloth import FastModel  # noqa: E402

# -- Config -----------------------------------------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
QUESTIONS_PER_CHUNK = 2  # Generate 2 questions per chunk
MAX_NEW_TOKENS_Q = 200  # Question generation output
MIN_CHUNK_WORDS = 20  # Skip chunks shorter than this

# Eval data paths
_EVAL_PATHS = [
    "/kaggle/input/pocket-arbiter-eval-data",
    "/kaggle/input/datasets/pguillemin/pocket-arbiter-eval-data",
]
EVAL_DATA_DIR = next((p for p in _EVAL_PATHS if os.path.isdir(p)), None)
assert EVAL_DATA_DIR is not None, f"FATAL: Eval data not found: {_EVAL_PATHS}"
DB_PATH = os.path.join(EVAL_DATA_DIR, "corpus_v2_fr.db")
assert os.path.exists(DB_PATH), f"FATAL: DB not found: {DB_PATH}"

# Teacher prompt for question generation (chunk only, no distractors per RAFT)
QUESTION_GEN_PROMPT = """Tu es un generateur de questions pour l'entrainement d'un assistant \
specialise en arbitrage d'echecs (reglements FFE/FIDE).

Etant donne un extrait de reglement, genere {n} questions qu'un arbitre pourrait poser \
et dont la reponse se trouve dans l'extrait.

REGLES:
- Questions en francais, style naturel (comme un arbitre sur le terrain)
- La reponse DOIT etre dans l'extrait fourni
- Questions courtes et precises (1 phrase, finir par ?)
- Varier les types: factuel, procedural, definitoire
- NE PAS reformuler l'extrait comme question
- Une question par ligne, numerotee (1. 2.)

Extrait du reglement ({source}, p.{page}):
{text}"""

logger.info(
    "Config: %d Q/chunk, min %d words/chunk", QUESTIONS_PER_CHUNK, MIN_CHUNK_WORDS
)

# -- Helpers ----------------------------------------------------------------


def generate(model, tokenizer, prompt, max_tokens, temperature=0.5):
    """Generate text from the teacher model."""
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def parse_questions(raw_text):
    """Extract questions from teacher output."""
    questions = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip()
        if line.endswith("?") and len(line) >= 15:
            questions.append(line)
    return questions


# -- PHASE 1: Load teacher model -------------------------------------------
logger.info("=== PHASE 1: Load teacher model ===")

# Load with Unsloth ? MANDATORY for Gemma 3 on T4 (fp16 CUDA assert with vanilla bitsandbytes)
# Unsloth handles fp16 precision issues, O(N) memory, and Gemma 3 tokenization fixes.
# Source: unsloth.ai/blog/gemma3, 25+ Kaggle notebooks confirm T4 compatibility.
model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-3-4b-it",
    max_seq_length=2048,
    load_in_4bit=True,
)
model.eval()

vram = torch.cuda.memory_allocated() / 1024 / 1024
logger.info("Teacher loaded: %.0f MB VRAM", vram)

# Smoke test
test_resp = generate(
    model, tokenizer, "Qu'est-ce qu'un forfait aux echecs ?", 50, temperature=0.3
)
logger.info("Smoke test: %s", test_resp[:100])
assert len(test_resp.strip()) > 0, "FATAL: Empty smoke test"

# -- PHASE 2: Load corpus chunks -------------------------------------------
logger.info("=== PHASE 2: Load corpus ===")
conn = sqlite3.connect(DB_PATH)
chunks_raw = conn.execute(
    "SELECT id, source, page, text FROM children ORDER BY source, page"
).fetchall()
conn.close()

chunk_list = [
    {"id": c[0], "source": c[1], "page": c[2], "text": c[3]} for c in chunks_raw
]
logger.info("Loaded %d chunks", len(chunk_list))

# -- PHASE 3: Generate questions -------------------------------------------
logger.info("=== PHASE 3: Generate questions ===")

output_path = os.path.join(OUTPUT_DIR, "questions_v5.jsonl")

total_questions = 0
skipped = 0
errors = 0

with open(output_path, "w", encoding="utf-8") as output_file:
    for i, chunk in enumerate(chunk_list):
        if len(chunk["text"].split()) < MIN_CHUNK_WORDS:
            skipped += 1
            continue

        try:
            prompt = QUESTION_GEN_PROMPT.format(
                n=QUESTIONS_PER_CHUNK,
                source=chunk["source"].replace(".pdf", ""),
                page=chunk["page"],
                text=chunk["text"][:1200],
            )
            raw = generate(model, tokenizer, prompt, MAX_NEW_TOKENS_Q)
            questions = parse_questions(raw)

            for q in questions[:QUESTIONS_PER_CHUNK]:
                entry = {
                    "chunk_id": chunk["id"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "chunk_text": chunk["text"],
                    "question": q,
                }
                output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_questions += 1

        except Exception as e:
            logger.warning("  SKIP chunk %s: %s", chunk["id"], str(e)[:80])
            errors += 1

        # Progress + streaming save + checkpoint
        if (i + 1) % 50 == 0:
            output_file.flush()
            elapsed = (time.time() - t_start) / 60
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(chunk_list) - i - 1) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] questions=%d, skipped=%d, errors=%d, %.1f min (ETA %.1f min)",
                i + 1,
                len(chunk_list),
                total_questions,
                skipped,
                errors,
                elapsed,
                eta,
            )

        # Checkpoint every 200 chunks (survives 9h timeout or crash)
        if (i + 1) % 200 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"questions_v5_ckpt_{i+1}.jsonl")
            output_file.flush()
            # Copy current output as checkpoint
            shutil.copy2(output_path, ckpt_path)
            logger.info(
                "  CHECKPOINT: %s (%d questions so far)", ckpt_path, total_questions
            )

# -- PHASE 4: Summary -----------------------------------------------------
elapsed = (time.time() - t_start) / 60
logger.info("=" * 60)
logger.info("QUESTION GENERATION COMPLETE -- %.1f min", elapsed)
logger.info("Total questions: %d from %d chunks", total_questions, len(chunk_list))
logger.info("Skipped (short): %d, Errors: %d", skipped, errors)
logger.info("Output: %s", output_path)

# Visual inspection: 5 random samples
with open(output_path, encoding="utf-8") as f:
    all_qs = [json.loads(line) for line in f]
logger.info("=== SAMPLES ===")
for idx in random.sample(range(len(all_qs)), min(5, len(all_qs))):
    q = all_qs[idx]
    logger.info("  [%s p.%s] %s", q["source"][:30], q["page"], q["question"])

metrics = {
    "total_chunks": len(chunk_list),
    "skipped_short": skipped,
    "errors": errors,
    "total_questions": total_questions,
    "questions_per_chunk_config": QUESTIONS_PER_CHUNK,
    "questions_per_chunk_actual": round(
        total_questions / max(len(chunk_list) - skipped, 1), 2
    ),
    "elapsed_min": round(elapsed, 1),
    "gpu": torch.cuda.get_device_name(0),
}
with open(os.path.join(OUTPUT_DIR, "gen_questions_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

logger.info("=== DONE ===")
