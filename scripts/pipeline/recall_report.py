"""Recall report generation: JSON and auto-generated Markdown."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(data: dict, path: Path | str) -> None:
    """Write recall results to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_markdown(data: dict, path: Path | str) -> None:
    """Write recall results to Markdown with YAML header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = data["metadata"]
    g = data["global"]

    lines = [
        "---",
        *[f"{k}: {v}" for k, v in meta.items()],
        "---",
        "",
        "# Recall Baseline — Pipeline v2",
        "",
        "## Global",
        "",
        "| Metrique | Score |",
        "|----------|-------|",
        *[f"| {k} | {v:.1%} |" for k, v in g.items() if k.startswith("recall")],
        f"| MRR | {g['mrr']:.3f} |",
        "",
    ]

    for seg_name, seg_data in data["segments"].items():
        lines.append(f"## Par {seg_name}")
        lines.append("")
        lines.append("| Bucket | Count | R@1 | R@3 | R@5 | R@10 |")
        lines.append("|--------|-------|-----|-----|-----|------|")
        for bucket, vals in seg_data.items():
            lines.append(
                f"| {bucket} | {vals['count']} "
                f"| {vals['recall@1']:.1%} | {vals['recall@3']:.1%} "
                f"| {vals['recall@5']:.1%} | {vals['recall@10']:.1%} |"
            )
        lines.append("")

    errors = data.get("errors", [])
    if errors:
        lines.append("## Top echecs (recall@10 = 0)")
        lines.append("")
        lines.append("| # | Question | Expected | Class |")
        lines.append("|---|----------|----------|-------|")
        for i, e in enumerate(errors, 1):
            q_short = e["question"][:50]
            pages = e["expected_pages"]
            lines.append(
                f"| {i} | {q_short} "
                f"| {e['expected_docs'][0]} p{pages} | {e['reasoning_class']} |"
            )
        lines.append("")

    r5 = g["recall@5"]
    if r5 >= 0.8:
        decision = "Prompt engineering suffisant"
    elif r5 >= 0.6:
        decision = "Optimisations retrieval necessaires"
    else:
        decision = "Fine-tuning embeddings justifie"
    lines.extend(["## Decision", "", f"recall@5 = {r5:.1%} → **{decision}**", ""])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
