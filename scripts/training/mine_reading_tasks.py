"""AdaptLLM regex mining for French chess regulation reading comprehension.

Cheng et al. ICLR 2024 — 6 task types mined from connector patterns.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

FR_CONNECTORS: dict[str, str] = {
    "nli_consequent": r"(?:Par conséquent|En conséquence|Par consequent|En consequence)",
    "nli_contrast": r"(?:Cependant|Toutefois|Néanmoins|En revanche|Neanmoins)",
    "causal": r"(?:\bcar\b|parce qu|en raison de|du fait de)",
    "conditional": r"(?:sous réserve|à condition qu|dans le cas où|sauf si"
    r"|sous reserve|a condition qu|dans le cas ou)",
    "reference": (
        r"(?:en application de|conformément|au sens de|tel que prévu|en vertu de"
        r"|conformement|tel que prevu)"
    ),
    "addition": r"(?:De plus|En outre|Par ailleurs)",
}

_PROMPTS: dict[str, str] = {
    "nli_consequent": "La phrase suivante est-elle une consequence du passage ?\n\n{passage}\n\nPhrase : {sentence}",
    "nli_contrast": "La phrase suivante est-elle en contraste avec le passage ?\n\n{passage}\n\nPhrase : {sentence}",
    "causal": "Quelle est la cause mentionnee dans ce passage ?\n\n{passage}",
    "conditional": "Sous quelle(s) condition(s) s'applique cette regle ?\n\n{passage}",
    "reference": "A quel texte ce passage fait-il reference ?\n\n{passage}",
    "addition": "Quelle information supplementaire est apportee dans ce passage ?\n\n{passage}",
    "summarization": "Resumez le passage suivant :\n\n{passage}",
    "completion": "Completez la phrase suivante :\n\n{context}\n\n{masked}",
}

_ALL_CONNECTORS = re.compile("|".join(FR_CONNECTORS.values()), re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 20]


def _paragraphs(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n{2,}", text.strip()) if len(b.strip()) > 40]


def format_exercise(
    task_type: str, user_content: str, assistant_content: str, source: str
) -> dict[str, Any]:
    """Return exercise dict with messages, task_type, source."""
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_type": task_type,
        "source": source,
    }


def mine_connectors(text: str, source: str) -> list[dict[str, Any]]:
    """Mine NLI/causal/conditional/reference/addition exercises via FR_CONNECTORS."""
    exercises: list[dict[str, Any]] = []
    for task_type, pattern in FR_CONNECTORS.items():
        compiled = re.compile(pattern, re.IGNORECASE)
        for para in _paragraphs(text):
            if not compiled.search(para):
                continue
            sentences = _sentences(para)
            matched = next((s for s in sentences if compiled.search(s)), None)
            if matched is None or len(matched) < 10:
                continue
            tmpl = _PROMPTS[task_type]
            if task_type in ("nli_consequent", "nli_contrast"):
                passage = para.replace(matched, "").strip() or para
                user = tmpl.format(passage=passage, sentence=matched)
                answer = "Oui. " + matched
            else:
                user = tmpl.format(passage=para)
                answer = matched
            exercises.append(format_exercise(task_type, user, answer, source))
            break  # one per paragraph per connector type
    return exercises


def mine_summarization(text: str, source: str) -> list[dict[str, Any]]:
    """Mine summarization exercises from markdown headings + section content."""
    exercises: list[dict[str, Any]] = []
    heading_re = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start, end = (
            m.end(),
            (matches[i + 1].start() if i + 1 < len(matches) else len(text)),
        )
        raw = text[start:end].strip()
        if len(raw) < 10:
            # Aggregate sub-sections for parent headings with no direct prose
            level = len(m.group(1))
            for j in range(i + 1, len(matches)):
                sub_start = matches[j].end()
                sub_end = matches[j + 1].start() if j + 1 < len(matches) else len(text)
                raw += " " + text[sub_start:sub_end].strip()
                if len(matches[j].group(1)) <= level:
                    break
        clean = re.sub(r"^#{1,4}\s+.+$", "", raw, flags=re.M).strip()
        if len(clean) < 20:
            continue
        user = _PROMPTS["summarization"].format(passage=clean[:1000])
        exercises.append(format_exercise("summarization", user, heading, source))
    return exercises


def mine_completion(text: str, source: str) -> list[dict[str, Any]]:
    """Mine completion exercises from sentences within 2-sentence radius of connectors."""
    exercises: list[dict[str, Any]] = []
    seen: set[str] = set()
    for para in _paragraphs(text):
        if not _ALL_CONNECTORS.search(para):
            continue
        sentences = _sentences(para)
        if len(sentences) < 2:
            continue
        connector_idxs = [
            i for i, s in enumerate(sentences) if _ALL_CONNECTORS.search(s)
        ]
        candidates: set[int] = set()
        for ci in connector_idxs:
            for off in range(-2, 3):
                idx = ci + off
                if 0 <= idx < len(sentences) and idx != ci:
                    candidates.add(idx)
        for idx in sorted(candidates):
            target = sentences[idx]
            if len(target) < 20 or target in seen:
                continue
            seen.add(target)
            ctx = " ".join(s for i, s in enumerate(sentences) if i != idx)[:400]
            user = _PROMPTS["completion"].format(context=ctx, masked="___")
            exercises.append(format_exercise("completion", user, target, source))
    return exercises


def compute_mining_stats(exercises: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute total, by_type, by_source, la_bias_pct, la_bias_warning."""
    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for ex in exercises:
        t, s = ex.get("task_type", "unknown"), ex.get("source", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
        by_source[s] = by_source.get(s, 0) + 1
    total = len(exercises)
    la_count = sum(n for src, n in by_source.items() if src.upper().startswith("LA"))
    la_pct = round(la_count / total * 100, 1) if total else 0.0
    return {
        "total": total,
        "by_type": by_type,
        "by_source": by_source,
        "la_bias_pct": la_pct,
        "la_bias_warning": la_pct > 80,
    }


def mine_document(text: str, source: str) -> list[dict[str, Any]]:
    """Mine all exercise types from one document."""
    out: list[dict[str, Any]] = []
    out.extend(mine_connectors(text, source))
    out.extend(mine_summarization(text, source))
    out.extend(mine_completion(text, source))
    return out


def main() -> None:
    """CLI: mine reading exercises from corpus JSON files.

    Args via CLI:
        --input: directory with docling_v2_fr/*.json
        --output: output JSONL path
        --stats: output stats JSON path
    """
    parser = argparse.ArgumentParser(description="AdaptLLM regex mining.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    args = parser.parse_args()

    output_path, stats_path = Path(args.output), Path(args.stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(Path(args.input).glob("*.json"))
    if not json_files:
        print(f"ERROR: no JSON files in {args.input}", file=sys.stderr)
        sys.exit(1)

    all_exercises: list[dict[str, Any]] = []
    for path in json_files:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        md = doc.get("markdown", "")
        if not md:
            continue
        exs = mine_document(md, path.name)
        all_exercises.extend(exs)
        print(f"  {path.name}: {len(exs)} exercises")

    stats = compute_mining_stats(all_exercises)
    print(f"\nTotal: {stats['total']} | By type: {stats['by_type']}")
    if stats["la_bias_warning"]:
        print(f"WARNING: LA bias {stats['la_bias_pct']}% > 80%")

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_exercises:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    assert stats["total"] >= 500, f"Gate G2 FAIL: {stats['total']} < 500"
    print("Gate G2 PASS: yield >= 500")


if __name__ == "__main__":
    main()
