#!/usr/bin/env python3
"""
Extract answer_text from Corrigé Détaillé sections.

The corrigé détaillé contains the official corrector's explanation
(green text in PDF) which is the actual answer_text we need.

Structure:
    ## Question N :
    [Question text]
    - a) choice a
    - b) choice b
    [Article reference]
    [EXPLANATION TEXT]  <- This is answer_text

    ## Question N+1 :
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from scripts.evaluation.annales.session_utils import detect_session_from_filename

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_corrige_sections(markdown: str) -> list[tuple[str, int, int]]:
    """
    Find all Corrigé Détaillé sections in markdown.

    Returns list of (uv_name, start_pos, end_pos)
    """
    sections = []

    # Pattern for corrigé section headers (flexible to handle format variations)
    # Matches: "## UVR session de juin 2025 - Corrigé détaillé"
    #          "## UVR - juin 2021 - corrigé détaillé et commentaires"
    #          "## ...FFE DNA... UVR session de décembre 2024 - Corrigé détaillé"
    pattern = re.compile(
        r'^##[^\n]*(UV[RCOT])\s*[-\u2013\u2014]?\s*'       # Header + any prefix + UV type
        r'(?:session\s+(?:de\s+)?)?'                       # Optional "session de"
        r'[-\u2013\u2014]?\s*'                             # Another optional dash
        r'[a-zA-Z\u00e9\u00e8\u00ea\u00f4\u00fb]+\s*\d{4}'  # Month + year
        r'[^#\n]*?'                                        # Anything until corrigé
        r'[Cc]orrig[\u00e9e]\s*[Dd][\u00e9e]taill[\u00e9e]',
        re.MULTILINE | re.IGNORECASE
    )

    # Find all corrigé headers
    for match in pattern.finditer(markdown):
        uv = match.group(1).upper()
        start = match.end()

        # Find end: next UV section header or "Commentaires du correcteur" or end
        next_section = re.search(
            r'\n##\s*(?:UV[RCOT]|Fin|Commentaires?\s+du)',
            markdown[start:],
            re.IGNORECASE
        )
        end = start + next_section.start() if next_section else len(markdown)

        sections.append((uv, start, end))
        logger.info(f"Found corrigé section: {uv} at {start}-{end}")

    return sections


def extract_question_explanations(corrige_text: str) -> dict[int, str]:
    """
    Extract explanation text for each question in a corrigé section.

    Returns dict of {question_num: explanation_text}
    """
    explanations = {}

    # Split by question headers
    # Pattern: "## QUESTION N :" or "## Question N" or "Question N :"
    question_pattern = re.compile(
        r'^(?:##\s*)?QUESTION\s+(\d+)\s*:?',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(question_pattern.finditer(corrige_text))

    for i, match in enumerate(matches):
        q_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(corrige_text)

        block = corrige_text[start:end].strip()

        # Extract explanation: text after choices and article reference
        explanation = _extract_explanation_from_block(block)

        if explanation:
            explanations[q_num] = explanation

    return explanations


def _extract_explanation_from_block(block: str) -> str:
    """
    Extract the explanation text from a question block.

    The explanation comes after:
    1. Question text
    2. Choices (- a), - b), etc.)
    3. Article reference line (contains "Article" or "Règles" or "LA")
    """
    lines = block.split('\n')

    # Find the last choice line
    last_choice_idx = -1
    for i, line in enumerate(lines):
        # Choice patterns: "- a)", "- A -", "A -", etc.
        if re.match(r'^\s*-?\s*[a-dA-D][\)\-\s]', line.strip()):
            last_choice_idx = i

    if last_choice_idx == -1:
        # No choices found, try to find article reference directly
        last_choice_idx = 0

    # Find article reference line after last choice
    article_idx = -1
    for i in range(last_choice_idx, len(lines)):
        line = lines[i].strip()
        # Article reference patterns
        if re.search(r'(?:Article|Règle|Chapitre|LA\s*[-–]|Annexe|R\d{2})', line, re.IGNORECASE):
            if len(line) > 10:  # Avoid false positives
                article_idx = i
                break

    if article_idx == -1:
        return ""

    # Explanation is everything after article reference line
    explanation_lines = []
    for i in range(article_idx + 1, len(lines)):
        line = lines[i].strip()
        # Stop at next question marker or image
        if re.match(r'^(?:##\s*)?[Qq]uestion\s+\d+', line):
            break
        if '<!-- image -->' in line:
            continue
        if line:
            explanation_lines.append(line)

    explanation = ' '.join(explanation_lines)

    # Clean up
    explanation = re.sub(r'\s+', ' ', explanation).strip()

    # Skip if too short or just numbers
    if len(explanation) < 20 or explanation.replace(' ', '').isdigit():
        return ""

    return explanation


def extract_all_corrige_answers(docling_json_path: Path) -> dict[str, dict[int, str]]:
    """
    Extract all answer explanations from a Docling JSON file.

    Returns dict of {uv: {question_num: explanation}}
    """
    with open(docling_json_path, encoding='utf-8') as f:
        data = json.load(f)

    # Skip non-Docling files (lists, or dicts without markdown)
    if not isinstance(data, dict):
        logger.debug(f"Skipping {docling_json_path.name}: not a dict")
        return {}

    markdown = data.get('markdown', '')

    if not markdown:
        logger.warning(f"No markdown in {docling_json_path}")
        return {}

    results = {}
    sections = find_corrige_sections(markdown)

    for uv, start, end in sections:
        corrige_text = markdown[start:end]
        explanations = extract_question_explanations(corrige_text)

        if explanations:
            results[uv] = explanations
            logger.info(f"{uv}: extracted {len(explanations)} explanations")

    return results


def _derive_choice_text(question: dict[str, Any]) -> str | None:
    """
    Derive answer text from MCQ choices and correct answer letter.

    Args:
        question: Question dict with choices and mcq_answer.

    Returns:
        Choice text if found, None otherwise.
    """
    choices = question.get("choices", {})
    mcq_answer = question.get("mcq_answer", "")

    if not choices or not mcq_answer:
        return None

    texts = []
    for letter in mcq_answer:
        if letter in choices:
            texts.append(choices[letter])

    if texts:
        return " | ".join(texts)
    return None


def update_gold_standard_with_explanations(
    gs_path: Path,
    docling_dirs: list[Path],
    output_path: Path | None = None
) -> dict[str, Any]:
    """
    Update Gold Standard with answer_text and answer_explanation.

    Implements merged source approach (Option C):
    - answer_text: Derived from MCQ choices (for RAG retrieval)
    - answer_explanation: Official corrector's explanation (detailed)
    - answer_source: 'merged', 'corrige_detaille', or 'choice_only'
    """
    with open(gs_path, encoding='utf-8') as f:
        gs = json.load(f)

    # Collect all explanations from all docling files
    all_explanations: dict[str, dict[str, dict[int, str]]] = {}  # session -> uv -> num -> text

    for docling_dir in docling_dirs:
        for json_file in docling_dir.glob('*.json'):
            if 'report' in json_file.name.lower():
                continue

            # Determine session from filename
            session = detect_session_from_filename(json_file.name)

            explanations = extract_all_corrige_answers(json_file)
            if explanations:
                all_explanations[session] = explanations
                logger.info(f"Session {session}: {sum(len(v) for v in explanations.values())} explanations")

    # Update Gold Standard questions
    stats = {
        'total': len(gs['questions']),
        'merged': 0,           # Both choice text and explanation
        'explanation_only': 0,  # Only explanation (no choice derivation)
        'choice_only': 0,       # Only choice text (no explanation)
        'not_found': 0          # Neither found
    }

    for q in gs['questions']:
        source = q.get('annales_source', {})
        session = source.get('session', '')
        uv = source.get('uv', '')
        q_num = source.get('question_num', 0)

        # Derive choice text from MCQ answer
        choice_text = _derive_choice_text(q)

        # Look up explanation from corrigé détaillé
        explanation = all_explanations.get(session, {}).get(uv, {}).get(q_num)

        # Apply merged source logic
        if choice_text and explanation:
            # Both available: use choice for answer_text, explanation for detail
            q['answer_text'] = choice_text
            q['answer_explanation'] = explanation
            q['answer_source'] = 'merged'
            stats['merged'] += 1
        elif explanation:
            # Only explanation: use for both (old behavior)
            q['answer_text'] = explanation
            q['answer_explanation'] = explanation
            q['answer_source'] = 'corrige_detaille'
            stats['explanation_only'] += 1
        elif choice_text:
            # Only choice: use for answer_text
            q['answer_text'] = choice_text
            q['answer_source'] = 'choice_only'
            stats['choice_only'] += 1
        else:
            # Neither found
            stats['not_found'] += 1

    # Update version
    if isinstance(gs.get('version'), dict):
        gs['version']['number'] = '6.7.0'
    else:
        gs['version'] = {'number': '6.7.0'}

    # Save
    output = output_path or gs_path
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(gs, f, ensure_ascii=False, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract answer_text from Corrigé Détaillé sections"
    )
    parser.add_argument(
        '--gs', type=Path,
        default=Path('tests/data/gold_standard_annales_fr.json'),
        help='Gold Standard JSON file'
    )
    parser.add_argument(
        '--docling-dirs', type=Path, nargs='+',
        default=[
            Path('corpus/processed/annales_all'),
            Path('corpus/processed/annales_juin_2025'),
        ],
        help='Directories with Docling JSON files'
    )
    parser.add_argument(
        '--output', type=Path,
        help='Output path (default: overwrite input)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show stats without writing'
    )
    args = parser.parse_args()

    # Filter existing directories
    docling_dirs = [d for d in args.docling_dirs if d.exists()]

    if not docling_dirs:
        logger.error("No valid docling directories found")
        return

    if args.dry_run:
        # Just show what we'd extract
        for d in docling_dirs:
            for f in d.glob('*.json'):
                if 'report' in f.name.lower():
                    continue
                explanations = extract_all_corrige_answers(f)
                total = sum(len(v) for v in explanations.values())
                print(f"{f.name}: {total} explanations")
        return

    stats = update_gold_standard_with_explanations(
        args.gs, docling_dirs, args.output
    )

    print("\n=== Extraction Results ===")
    print(f"Total questions: {stats['total']}")
    print(f"Updated with corrigé: {stats['updated']}")
    print(f"Already had answer_text: {stats['already_had']}")
    print(f"Not found in corrigé: {stats['not_found']}")

    pct = (stats['updated'] + stats['already_had']) / stats['total'] * 100
    print(f"\nCoverage: {pct:.1f}%")


if __name__ == '__main__':
    main()
