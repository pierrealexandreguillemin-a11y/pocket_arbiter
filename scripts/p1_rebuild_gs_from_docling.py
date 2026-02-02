"""
p1_rebuild_gs_from_docling.py — Phase 1: Rebuild GS metadata from Docling sources.

Strategy:
1. docling_mcq_reference.json = primary source for correct_letter, article_reference, success_rate
2. Docling markdown JSON = source for choices text and explanations (commentaires)
3. Cross-validate: choices[correct_letter] must exist, question text must match

ISO 42001 A.6.2.3: Each modification cites its source (file, section, question_num).
Constraint R5: original_answer is NEVER overwritten (saved to metadata.original_answer).

Usage:
    python scripts/p1_rebuild_gs_from_docling.py --check     # Dry-run
    python scripts/p1_rebuild_gs_from_docling.py --apply      # Apply changes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
GS_PATH = ROOT / "tests" / "data" / "gold_standard_annales_fr_v7.json"
REF_PATH = ROOT / "data" / "evaluation" / "annales" / "docling_mcq_reference.json"
CHUNKS_PATH = ROOT / "corpus" / "processed" / "chunks_mode_b_fr.json"
ANNALES_ALL = ROOT / "corpus" / "processed" / "annales_all"
ANNALES_DEC2024 = ROOT / "corpus" / "processed" / "annales_dec_2024"

SESSION_FILES: dict[str, tuple[str, Path]] = {
    "dec2019": ("Annales-Session-decembre-2019-version-2.json", ANNALES_ALL),
    "jun2021": ("Annales-Session-juin-2021.json", ANNALES_ALL),
    "dec2021": ("Annales-Session-decembre-2021.json", ANNALES_ALL),
    "jun2022": ("Annales-session-juin-2022-vers2.json", ANNALES_ALL),
    "dec2022": ("Annales-session-decembre-2022.json", ANNALES_ALL),
    "jun2023": ("Annales-session-Juin-2023.json", ANNALES_ALL),
    "dec2023": ("Annales-decembre2023.json", ANNALES_ALL),
    "jun2024": ("Annales-juin-2024.json", ANNALES_ALL),
    "dec2024": ("Annales-Decembre-2024.json", ANNALES_DEC2024),
    "jun2025": ("Annales-Juin-2025-VF2.json", ANNALES_DEC2024),
}

UV_MAP = {"UVR": "rules", "UVC": "clubs", "UVO": "open", "UVT": "tournament"}
UV_REVERSE = {v: k for k, v in UV_MAP.items()}

# ---------------------------------------------------------------------------
# Known broken questions with manually verified corrections
# Source: Docling markdown JSON, manually cross-validated against
# sujet + grille + commentaires sections.
# These questions have misaligned choices in docling_mcq_reference.json.
# ---------------------------------------------------------------------------
MANUAL_CORRECTIONS: dict[tuple[str, str, int], dict[str, Any]] = {
    # dec2021/clubs/Q1: Team composition in interclubs
    # Source: Annales-Session-decembre-2021.json, UVC sujet Q1 + grille Q1 + commentaires Q1
    ("dec2021", "clubs", 1): {
        "choices": {
            "A": "NON car on ne peut pas jouer 2 fois une ronde 2.",
            "B": "OUI, au 2ème week-end, un joueur ne peut pas encore être grillé donc il peut jouer dans l'équipe 2.",
            "C": "OUI à condition d'avoir joué la ronde 1 avec l'équipe 2, pas avec l'équipe 1.",
            "D": "OUI à condition qu'il n'ait pas du tout joué lors de la première journée des interclubs.",
        },
        "correct_letter": "D",
        "explanation": "Pour avoir le droit de jouer une ronde 2, un joueur ne doit pas avoir déjà joué 2 matchs. S'il joue le samedi, ça lui fait un match. Il ne pourra donc jouer le dimanche en équipe 2 que s'il n'a pas du tout joué avant ce match du samedi. Réponse D.",
    },
    # dec2021/clubs/Q4: Team substitution between divisions
    # Source: Annales-Session-decembre-2021.json, UVC sujet Q4 + grille Q4 + commentaires Q4
    ("dec2021", "clubs", 4): {
        "choices": {
            "A": "OUI, un joueur peut toujours monter en équipe 1.",
            "B": "OUI parce qu'il n'a pas encore fait 3 matchs avec son équipe.",
            "C": "Il peut changer d'équipe seulement si les 2 équipes jouent dans des groupes différents, pas si elles jouent dans le même groupe.",
            "D": "Dans ce cas, ce joueur ne peut pas jouer en équipe 1 même si elle joue dans un autre groupe.",
        },
        "correct_letter": "D",
        "explanation": "Si un club a 2 équipes dans la même division, un joueur qui a joué pour l'une des équipes ne peut plus jouer dans l'autre (article 3.7.d). Réponse D.",
    },
    # dec2021/clubs/Q5: N2 team scoring with forfaits
    # Source: Annales-Session-decembre-2021.json, UVC sujet Q5 + grille Q5 + commentaires Q5
    ("dec2021", "clubs", 5): {
        "choices": {
            "A": "8-0",
            "B": "7-1",
            "C": "6-1",
            "D": "6-2",
            "E": "5-2",
            "F": "5-3",
        },
        "correct_letter": "E",
        "explanation": "L'équipe proposée n'est pas conforme sur 2 points : a) Il n'y a pas de fille → Le score passe de 1 à « -1 », donc 8 - 2 = 6 pour l'équipe fautive et + 1 pour l'autre équipe. b) L'échiquier 6 a plus de 100 points de plus que l'échiquier 4. Perte administrative 1-0 sur cette table. Donc 6 - 1 = 5, pour l'équipe fautive et « 1 + 1 = + 2 » pour l'autre équipe. Score final 5 - 2, réponse E.",
    },
    # dec2021/clubs/Q10: Starting clocks when team is late
    # Source: Annales-Session-decembre-2021.json, UVC sujet Q10 + grille Q10 + commentaires Q10
    ("dec2021", "clubs", 10): {
        "choices": {
            "A": "Vous appuyez sur les pendules des joueurs ayant les noirs de l'équipe absente et vous demandez aux joueurs présents ayant les noirs de lancer leur pendule.",
            "B": "Vous lancez toutes les pendules, en faisant découler le temps des joueurs absents, sans que les joueurs présents ayant les blancs jouent leur premier coup.",
            "C": "Vous ne démarrez pas les pendules, vous les ajusterez avant le début de la ronde.",
        },
        "correct_letter": "C",
        "explanation": "Vous n'avez pas encore la composition de l'équipe, vous ne connaissez pas encore le temps de pénalité à appliquer avant de démarrer les pendules. Comme il faudra de toute façon régler les pendules, quand la composition de l'équipe vous sera remise, inutile de les démarrer pour l'instant. Réponse C.",
    },
    # dec2021/clubs/Q17: AFC arbitrer un tournoi au système Suisse
    # Source: Annales-Session-decembre-2021.json, UVC sujet Q17 + grille Q17 + commentaires Q17
    ("dec2021", "clubs", 17): {
        "choices": {
            "A": "Seulement dans le cas d'un tournoi interne.",
            "B": "Seulement dans le cadre d'un tournoi scolaire.",
            "C": "Seulement dans le cas d'un open homologué FFE, non FIDE.",
            "D": "Seulement en cas de dérogation de son DRA.",
            "E": "Seulement en cas de dérogation du directeur des titres de la DNA.",
            "F": "Jamais",
        },
        "correct_letter": "D",
        "explanation": "La réponse se trouve directement dans l'article 19 du RI DNA. Réponse D.",
    },
    # dec2021/clubs/Q20: Minimum level for N3 jeunes arbiter
    # Source: Annales-Session-decembre-2021.json, UVC sujet Q20 + grille Q20 + commentaires Q20
    ("dec2021", "clubs", 20): {
        "choices": {
            "A": "AFC.",
            "B": "AFJ.",
            "C": "Arbitre stagiaire.",
            "D": "Arbitre candidat.",
            "E": "Licencié A ayant eu l'accord de son DRA.",
        },
        "correct_letter": "E",
        "explanation": "La réponse se trouve directement dans l'article 2.5 du championnat de France des interclubs jeunes. Réponse E.",
    },
    # jun2022/clubs/Q3: Dates de la période de transfert libre
    # Source: Annales-session-juin-2022-vers2.json, UVC sujet Q3 + grille Q3
    ("jun2022", "clubs", 3): {
        "choices": {
            "A": "du 1er juillet au 30 août.",
            "B": "du 1er juillet au 30 septembre.",
            "C": "du 15 juillet au 30 août.",
            "D": "du 15 juillet au 30 septembre.",
        },
        "correct_letter": "D",
        "explanation": None,  # No explanation in jun2022 corrigé
    },
}


# ---------------------------------------------------------------------------
# Docling Markdown Parsing (choices + explanations only)
# ---------------------------------------------------------------------------
def load_docling_markdown(session: str) -> str:
    """Load the markdown text from a Docling JSON for a given session."""
    fname, directory = SESSION_FILES[session]
    path = directory / fname
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["markdown"]


def find_uv_boundaries(md: str) -> dict[str, tuple[int, int]]:
    """Find start/end positions of each UV section in the markdown.

    Returns {uv_code: (start, end)} for UVR, UVC, UVO, UVT.
    Each UV typically appears multiple times (sujet, grille, corrigé).
    We return the sujet section (with questions+choices).
    """
    boundaries: dict[str, list[int]] = {}

    # Find all UV section headers (skip TOC entries)
    for uv_code in ["UVR", "UVC", "UVO", "UVT"]:
        positions: list[int] = []
        for m in re.finditer(
            rf"(?:^|\n)(?:## )?{uv_code}\s*[-–]?\s*(?:session|de\s)", md, re.IGNORECASE
        ):
            # Skip if in TOC (first 5000 chars or contains "Page")
            if m.start() < 5000:
                ctx = md[m.start() : m.start() + 200]
                if "Page" in ctx and "|" in ctx:
                    continue
            positions.append(m.start())

        # Also look for "rage UVC session" pattern (dec2024 style)
        for m in re.finditer(rf"rage\s+{uv_code}\s+session", md, re.IGNORECASE):
            if m.start() > 5000:
                positions.append(m.start())

        if positions:
            boundaries[uv_code] = sorted(set(positions))

    return boundaries  # type: ignore[return-value]


def extract_questions_per_uv(md: str) -> dict[str, dict[int, dict[str, Any]]]:  # noqa: C901
    """Extract questions with choices, organized by UV section.

    Returns {uv_code: {qnum: {question_text, choices}}}.
    Uses UV section boundaries to avoid cross-UV confusion.
    """
    results: dict[str, dict[int, dict[str, Any]]] = {}

    # Find all UV section start positions
    all_uv_positions: list[tuple[int, str]] = []

    for uv_code in ["UVR", "UVC", "UVO", "UVT"]:
        # Find sujet sections
        for m in re.finditer(
            rf"(?:^|\n)(?:## )?(?:{uv_code}|arbitrage\s+{uv_code}|rage\s+{uv_code})"
            rf"\s*[-–]?\s*(?:session|de\s)",
            md,
            re.IGNORECASE,
        ):
            if m.start() < 5000:
                ctx = md[m.start() : m.start() + 200]
                if "Page" in ctx and "|" in ctx:
                    continue
            all_uv_positions.append((m.start(), uv_code))

    # Also find grille/corrigé markers to delimit sujet sections
    grille_positions: list[int] = []
    for m in re.finditer(r"grille des r.ponses", md, re.IGNORECASE):
        if m.start() > 5000:
            grille_positions.append(m.start())

    corrige_positions: list[int] = []
    for m in re.finditer(r"corrig.+ d.taill", md, re.IGNORECASE):
        if m.start() > 5000:
            corrige_positions.append(m.start())

    # Sort all positions to determine section boundaries
    all_uv_positions.sort()

    # For each UV section, extract questions
    for i, (pos, uv_code) in enumerate(all_uv_positions):
        # Find end of this section: next UV section, or grille, or corrigé
        end = len(md)
        # Check next UV section
        if i + 1 < len(all_uv_positions):
            end = min(end, all_uv_positions[i + 1][0])
        # Check grille positions
        for gp in grille_positions:
            if gp > pos and gp < end:
                end = gp
                break
        # Check corrigé positions
        for cp in corrige_positions:
            if cp > pos and cp < end:
                end = min(end, cp)
                break

        section_text = md[pos:end]
        questions = _extract_questions_from_section(section_text)

        if uv_code not in results:
            results[uv_code] = {}

        # Merge: keep version with most choices
        for qnum, data in questions.items():
            if qnum not in results[uv_code] or len(data["choices"]) > len(
                results[uv_code][qnum].get("choices", {})
            ):
                results[uv_code][qnum] = data

    # Also extract from corrigé sections (which repeat questions with choices)
    for uv_code in ["UVR", "UVC", "UVO", "UVT"]:
        for m in re.finditer(rf"(?:{uv_code}).+(?:corrig|Corrig)", md, re.IGNORECASE):
            if m.start() < 5000:
                continue
            # Find end of corrigé section
            end = len(md)
            for _j, (p2, _u2) in enumerate(all_uv_positions):
                if p2 > m.start() + 200:
                    end = min(end, p2)
                    break
            for cp in corrige_positions:
                if cp > m.start() + 200 and cp < end:
                    end = min(end, cp)
                    break

            section_text = md[m.start() : end]
            questions = _extract_questions_from_section(section_text)

            if uv_code not in results:
                results[uv_code] = {}

            for qnum, data in questions.items():
                if qnum not in results[uv_code] or len(data["choices"]) > len(
                    results[uv_code][qnum].get("choices", {})
                ):
                    results[uv_code][qnum] = data

    return results


def _extract_questions_from_section(text: str) -> dict[int, dict[str, Any]]:
    """Extract questions with choices from a section of markdown text."""
    results: dict[int, dict[str, Any]] = {}

    pattern = r"(?:^|\n)(?:## )?(?:Question|QUESTION)\s+(\d+)\s*:?\s*(?:\n|$)"
    parts = re.split(pattern, text)

    i = 1
    while i < len(parts) - 1:
        try:
            qnum = int(parts[i])
        except ValueError:
            i += 2
            continue

        body = parts[i + 1]
        choices = _extract_choices(body)
        question_text = _extract_question_text(body)

        if qnum not in results or len(choices) > len(results[qnum].get("choices", {})):
            results[qnum] = {
                "question_text": question_text,
                "choices": choices,
            }

        i += 2

    return results


def _extract_choices(body: str) -> dict[str, str]:
    """Extract choices from a question body block."""
    choices: dict[str, str] = {}

    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Match choice patterns:
        # "- A. text", "- a) text", "## A. text", "A. text", "a) text"
        # "- A : text", "a. text"
        m = re.match(r"^(?:-\s*|##\s*)?([A-Fa-f])\s*[.):\-]\s*(.+)$", stripped)
        if m:
            letter = m.group(1).upper()
            text = m.group(2).strip()
            # Don't add empty or very short texts
            if len(text) > 2:
                choices[letter] = text

    return choices


def _extract_question_text(body: str) -> str:
    """Extract question text (before choices) from body."""
    lines: list[str] = []
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            if lines:
                lines.append("")
            continue
        # Stop when we hit a choice line
        if re.match(r"^(?:-\s*|##\s*)?[A-Fa-f]\s*[.):\-]\s+", stripped):
            break
        lines.append(stripped)

    text = " ".join(line for line in lines if line).strip()
    text = re.sub(r"##\s*", "", text).strip()
    return text


def extract_commentaires(md: str) -> dict[str, dict[int, str]]:
    """Extract all commentaires sections from the markdown.

    Returns {uv_code: {qnum: explanation}}.
    """
    results: dict[str, dict[int, str]] = {}

    # Find "Commentaires UVX" sections
    for uv_code in ["UVR", "UVC", "UVO", "UVT"]:
        pattern = rf"Commentaires\s+{uv_code}"
        m = re.search(pattern, md, re.IGNORECASE)
        if not m:
            continue

        start = m.start()
        # Find end: next UV section, or "FIN" marker, or next "Commentaires"
        end = len(md)
        for marker_pat in [
            r"\n## (?:Dur|Question|UVR|UVC|UVO|UVT)",
            r"FIN des commentaires",
            r"FIN du corrig",
        ]:
            em = re.search(marker_pat, md[start + 50 :])
            if em:
                candidate = start + 50 + em.start()
                if candidate < end:
                    end = candidate

        section = md[start:end]
        explanations = _parse_commentaire_entries(section)
        results[uv_code] = explanations

    return results


def _parse_commentaire_entries(text: str) -> dict[int, str]:
    """Parse Q1: ..., Q2: ... entries from a commentaires section."""
    results: dict[int, str] = {}

    # Split on "Q<number> :" or "## Q<number> :"
    parts = re.split(r"(?:^|\n)(?:## )?Q(\d+)\s*:\s*", text)

    i = 1
    while i < len(parts) - 1:
        try:
            qnum = int(parts[i])
        except ValueError:
            i += 2
            continue

        body = parts[i + 1].strip()
        # Clean: remove trailing FIN markers, separators
        for stop in ["-----", "FIN des", "FIN du"]:
            idx = body.find(stop)
            if idx > 0:
                body = body[:idx].strip()

        if body:
            results[qnum] = body
        i += 2

    return results


def extract_inline_explanations(md: str) -> dict[str, dict[int, str]]:
    """For sessions where explanations are inline in corrigé (like dec2024),
    extract "Commentaire article..." blocks after each question."""
    results: dict[str, dict[int, str]] = {}

    # Find "Commentaire article" or "Commentaire Article" blocks
    # These typically appear after each question in the corrigé
    for _m in re.finditer(r"## Commentaire\s+(?:article|Article)\s+(.+?)(?:\n|$)", md):
        # This is more complex to map to question numbers
        # Skip for now - handled by the commentaires parser
        pass

    return results


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------
def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="Rebuild GS from Docling sources")
    parser.add_argument("--check", action="store_true", help="Dry-run mode")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    parser.add_argument("--session", type=str, help="Process single session")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.check and not args.apply:
        print("ERROR: specify --check or --apply", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------------
    # 1. Load all data sources
    # ---------------------------------------------------------------
    print("Loading GS...")
    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)
    print(f"  {len(gs['questions'])} questions, version {gs['version']}")

    print("Loading MCQ reference...")
    with open(REF_PATH, encoding="utf-8") as f:
        ref_list = json.load(f)
    # Index by (session, uv, question_num)
    ref_index: dict[tuple[str, str, int], dict] = {}
    for entry in ref_list:
        key = (entry["session"], entry["uv"], entry["question_num"])
        ref_index[key] = entry
    print(f"  {len(ref_index)} reference entries")

    print("Loading chunks...")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks_list = (
        chunks_data.get("chunks", chunks_data)
        if isinstance(chunks_data, dict)
        else chunks_data
    )
    chunks = {c["id"]: c for c in chunks_list}
    print(f"  {len(chunks)} chunks")

    # ---------------------------------------------------------------
    # 2. Parse all Docling markdowns for choices + explanations
    # ---------------------------------------------------------------
    print("\nParsing Docling markdowns...")
    # {session: {uv_code: {qnum: {question_text, choices}}}}
    md_questions_by_uv: dict[str, dict[str, dict[int, dict]]] = {}
    # {session: {uv_code: {qnum: explanation}}}
    md_explanations: dict[str, dict[str, dict[int, str]]] = {}

    sessions = sorted(SESSION_FILES.keys())
    if args.session:
        sessions = [args.session]

    for session in sessions:
        print(f"  {session}...", end=" ")
        md = load_docling_markdown(session)
        per_uv = extract_questions_per_uv(md)
        commentaires = extract_commentaires(md)
        md_questions_by_uv[session] = per_uv
        md_explanations[session] = commentaires

        for uv_code, qs in sorted(per_uv.items()):
            n_with = sum(1 for q in qs.values() if q["choices"])
            print(f"{uv_code}:{len(qs)}({n_with}) ", end="")
        n_expl = sum(len(v) for v in commentaires.values())
        print(f"| {n_expl} explanations")

    # ---------------------------------------------------------------
    # 3. For each GS question, build verified data + update plan
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Building update plan...")

    changes_log: list[dict] = []
    stats = {
        "total_annales": 0,
        "ref_found": 0,
        "choices_updated": 0,
        "mcq_answer_updated": 0,
        "expected_answer_updated": 0,
        "article_ref_updated": 0,
        "explanation_added": 0,
        "success_rate_updated": 0,
        "flags_cleaned": 0,
        "no_ref": 0,
        "alignment_issues": 0,
    }

    for idx, q in enumerate(gs["questions"]):
        meta = q.get("metadata", {})
        annales = meta.get("annales_source")
        if not annales:
            continue  # Skip human questions

        stats["total_annales"] += 1
        session = annales["session"]
        uv = annales["uv"]
        qnum = annales["question_num"]
        ref_key = (session, uv, qnum)
        uv_code = UV_REVERSE.get(uv, "")
        source_file = SESSION_FILES.get(session, ("?", Path()))[0]

        # -- Check for manual correction first --
        manual = MANUAL_CORRECTIONS.get(ref_key)

        # -- Lookup in MCQ reference --
        ref = ref_index.get(ref_key)
        if not ref and not manual:
            stats["no_ref"] += 1
            if args.verbose:
                print(f"  [{idx}] {q['id']} ({session}/{uv}/Q{qnum}): NO REF")
            continue

        if not ref:
            ref = {}  # Will use manual correction only

        stats["ref_found"] += 1

        # -- Determine best choices --
        gs_choices = meta.get("choices", {})

        # Clean GS choices: remove explanation text merged into last choice
        clean_gs_choices = {}
        for letter, text in gs_choices.items():
            if "##" in text:
                text = text.split("##")[0].strip()
            clean_gs_choices[letter] = text

        if manual:
            # Use manually verified data (highest priority)
            best_choices = manual["choices"]
            correct_letter = manual["correct_letter"]
            choices_source = f"MANUAL [{source_file}]"
            manual_expl = manual.get("explanation")
        else:
            manual_expl = None
            ref_choices = {k.upper(): v for k, v in ref.get("choices", {}).items()}

            md_uv_qs = md_questions_by_uv.get(session, {}).get(uv_code, {})
            md_choices = md_uv_qs.get(qnum, {}).get("choices", {})
            md_choices = {k.upper(): v for k, v in md_choices.items()}

            # -- Correct letter --
            ref_letter = ref.get("correct_letter", "").upper().strip()
            gs_mcq = meta.get("mcq_answer", "")
            correct_letter = ref_letter if ref_letter else gs_mcq

            # Priority: REF > MD > GS
            ref_choices_valid = bool(
                ref_choices
                and correct_letter
                and (len(correct_letter) > 1 or correct_letter in ref_choices)
            )

            if ref_choices_valid:
                best_choices = ref_choices
                choices_source = "docling_mcq_reference.json"
            elif (
                md_choices
                and correct_letter
                and (len(correct_letter) > 1 or correct_letter in md_choices)
            ):
                best_choices = md_choices
                choices_source = f"Docling markdown [{source_file}] {uv_code}"
            elif clean_gs_choices:
                best_choices = clean_gs_choices
                choices_source = "GS (cleaned)"
            else:
                best_choices = {}
                choices_source = "NONE"

        # -- Validate alignment --
        if (
            correct_letter
            and len(correct_letter) == 1
            and correct_letter not in best_choices
        ):
            stats["alignment_issues"] += 1
            if args.verbose:
                print(
                    f"  [{idx}] {q['id']} ({session}/{uv}/Q{qnum}): "
                    f"letter {correct_letter} NOT IN choices "
                    f"{list(best_choices.keys())} (source: {choices_source})"
                )

        # -- Build changes for this question --
        q_changes: list[dict] = []

        # Update choices if different
        if best_choices and best_choices != gs_choices:
            q_changes.append(
                {
                    "field": "metadata.choices",
                    "old_keys": sorted(gs_choices.keys()),
                    "new_keys": sorted(best_choices.keys()),
                    "source": choices_source,
                }
            )
            stats["choices_updated"] += 1

        # Update mcq_answer if different
        if correct_letter and correct_letter != gs_mcq:
            q_changes.append(
                {
                    "field": "metadata.mcq_answer",
                    "old": gs_mcq,
                    "new": correct_letter,
                    "source": "docling_mcq_reference.json",
                }
            )
            stats["mcq_answer_updated"] += 1

        # Update expected_answer from choices[correct_letter]
        old_answer = q.get("expected_answer", "")
        if (
            correct_letter
            and len(correct_letter) == 1
            and correct_letter in best_choices
        ):
            new_answer = best_choices[correct_letter]
            if new_answer != old_answer:
                q_changes.append(
                    {
                        "field": "expected_answer",
                        "old_trunc": old_answer[:60],
                        "new_trunc": new_answer[:60],
                        "source": choices_source,
                    }
                )
                stats["expected_answer_updated"] += 1
        elif correct_letter and len(correct_letter) > 1:
            # Multi-letter: concatenate
            texts = [best_choices[ch] for ch in correct_letter if ch in best_choices]
            if texts:
                new_answer = " | ".join(texts)
                if new_answer != old_answer:
                    q_changes.append(
                        {
                            "field": "expected_answer",
                            "old_trunc": old_answer[:60],
                            "new_trunc": new_answer[:60],
                            "source": f"Multi-letter from {choices_source}",
                        }
                    )
                    stats["expected_answer_updated"] += 1

        # Update article_reference
        ref_article = ref.get("article_reference", "")
        gs_article = meta.get("article_reference", "")
        if ref_article and ref_article != gs_article:
            q_changes.append(
                {
                    "field": "metadata.article_reference",
                    "old_trunc": gs_article[:60],
                    "new": ref_article,
                    "source": "docling_mcq_reference.json",
                }
            )
            stats["article_ref_updated"] += 1

        # Update success_rate
        ref_rate = ref.get("success_rate")
        gs_rate = annales.get("success_rate")
        if ref_rate is not None and ref_rate != gs_rate:
            q_changes.append(
                {
                    "field": "metadata.annales_source.success_rate",
                    "old": gs_rate,
                    "new": ref_rate,
                    "source": "docling_mcq_reference.json",
                }
            )
            stats["success_rate_updated"] += 1

        # Add answer_explanation from commentaires or manual
        gs_expl = meta.get("answer_explanation")
        session_expls = md_explanations.get(session, {})
        expl = manual_expl or session_expls.get(uv_code, {}).get(qnum)
        if expl and expl != gs_expl:
            q_changes.append(
                {
                    "field": "metadata.answer_explanation",
                    "old_was_null": gs_expl is None,
                    "new_trunc": expl[:80],
                    "source": f"Commentaires [{source_file}]",
                }
            )
            stats["explanation_added"] += 1

        # Clean extraction_flags
        flags = meta.get("extraction_flags", [])
        if "no_choices" in flags and best_choices:
            q_changes.append(
                {
                    "field": "metadata.extraction_flags",
                    "action": "remove no_choices",
                    "source": "choices now populated",
                }
            )
            stats["flags_cleaned"] += 1

        if q_changes:
            changes_log.append(
                {
                    "index": idx,
                    "id": q["id"],
                    "key": f"{session}/{uv}/Q{qnum}",
                    "changes": q_changes,
                }
            )

        # -- Apply changes if requested --
        if args.apply and q_changes:
            for change in q_changes:
                field = change["field"]

                if field == "metadata.choices":
                    meta["choices"] = best_choices

                elif field == "metadata.mcq_answer":
                    meta["mcq_answer"] = correct_letter

                elif field == "expected_answer":
                    # R5: preserve original
                    if "original_answer" not in meta:
                        meta["original_answer"] = old_answer
                    if (
                        correct_letter
                        and len(correct_letter) == 1
                        and correct_letter in best_choices
                    ):
                        q["expected_answer"] = best_choices[correct_letter]
                    elif correct_letter and len(correct_letter) > 1:
                        texts = [
                            best_choices[ch]
                            for ch in correct_letter
                            if ch in best_choices
                        ]
                        if texts:
                            q["expected_answer"] = " | ".join(texts)

                elif field == "metadata.article_reference":
                    meta["article_reference"] = ref_article

                elif field == "metadata.annales_source.success_rate":
                    annales["success_rate"] = ref_rate

                elif field == "metadata.answer_explanation":
                    meta["answer_explanation"] = expl

                elif field == "metadata.extraction_flags":
                    meta["extraction_flags"] = [f for f in flags if f != "no_choices"]

    # ---------------------------------------------------------------
    # 4. Report
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Total annales questions: {stats['total_annales']}")
    print(f"Reference found:         {stats['ref_found']}")
    print(f"No reference:            {stats['no_ref']}")
    print(f"Alignment issues:        {stats['alignment_issues']}")
    print()
    print("Changes by field:")
    print(f"  choices:          {stats['choices_updated']}")
    print(f"  mcq_answer:       {stats['mcq_answer_updated']}")
    print(f"  expected_answer:  {stats['expected_answer_updated']}")
    print(f"  article_ref:      {stats['article_ref_updated']}")
    print(f"  explanation:      {stats['explanation_added']}")
    print(f"  success_rate:     {stats['success_rate_updated']}")
    print(f"  flags_cleaned:    {stats['flags_cleaned']}")

    total_changes = sum(len(c["changes"]) for c in changes_log)
    print(f"\nTotal field changes: {total_changes}")
    print(f"Questions affected: {len(changes_log)}")

    if args.verbose and changes_log:
        print("\n--- Sample changes (first 10) ---")
        for entry in changes_log[:10]:
            print(f"\n[{entry['index']}] {entry['id']} ({entry['key']})")
            for c in entry["changes"]:
                print(f"  {c['field']}: {c.get('source', '')}")
                for k, v in c.items():
                    if k not in ("field", "source"):
                        print(f"    {k}: {str(v)[:80]}")

    # ---------------------------------------------------------------
    # 5. Write if applying
    # ---------------------------------------------------------------
    if args.apply:
        print(f"\nWriting GS to {GS_PATH}...")
        with open(GS_PATH, "w", encoding="utf-8") as f:
            json.dump(gs, f, ensure_ascii=False, indent=2)
        print("Done.")
    else:
        print("\n[DRY RUN] Use --apply to write changes.")


if __name__ == "__main__":
    main()
