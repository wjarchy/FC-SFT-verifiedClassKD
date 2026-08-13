#!/usr/bin/env python3
"""Interactive, resumable blind annotation for the Verifier audit packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

console = Console()
LABEL_KEYS = {
    "c": "correct",
    "f": "wrong_function",
    "a": "wrong_arguments",
    "i": "invalid",
    "u": "unknown",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_rows(rows: list[dict], path: Path) -> None:
    """Save after every decision so an interrupted annotation session can resume."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/diagnostics/verifier_human_audit_final.jsonl",
    )
    args = parser.parse_args()
    path = Path(args.input)
    rows = load_rows(path)

    for index, row in enumerate(rows):
        if row.get("human_label"):
            continue
        console.rule(f"{index + 1}/{len(rows)} · {row['audit_id']}")
        console.print(Panel(row["query"], title="Query", border_style="cyan"))
        console.print(
            Syntax(
                json.dumps(row["tools"], ensure_ascii=False, indent=2),
                "json",
                word_wrap=True,
            )
        )
        console.print(
            Panel(
                json.dumps(row["gold_call"], ensure_ascii=False, indent=2),
                title="Gold call",
                border_style="green",
            )
        )
        console.print(Panel(row["candidate"], title="Candidate", border_style="yellow"))
        console.print(
            "[bold]c[/bold]=correct  [bold]f[/bold]=wrong_function  "
            "[bold]a[/bold]=wrong_arguments  [bold]i[/bold]=invalid  "
            "[bold]u[/bold]=unknown  [bold]q[/bold]=save & quit"
        )
        key = Prompt.ask("Label", choices=[*LABEL_KEYS, "q"], show_choices=False)
        if key == "q":
            save_rows(rows, path)
            remaining = sum(not item.get("human_label") for item in rows)
            console.print(f"Saved. Remaining: {remaining}/{len(rows)}")
            return
        row["human_label"] = LABEL_KEYS[key]
        if key in {"a", "u"}:
            row["human_notes"] = Prompt.ask("Optional note", default="")
        save_rows(rows, path)

    console.print("全部标完。请运行：.venv/bin/python scripts/score_verifier_audit.py")


if __name__ == "__main__":
    main()
