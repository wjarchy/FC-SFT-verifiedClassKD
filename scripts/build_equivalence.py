#!/usr/bin/env python3
"""Build expanded equivalence diagnostic JSONL from split records."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from fc_class_kd.config import load_config
from fc_class_kd.data.equivalence import write_equivalence_diagnostic
from fc_class_kd.data.splits import read_split

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="base")
    parser.add_argument("--split", default="equivalence_diagnostic")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_path = Path(cfg.paths.splits) / f"{args.split}.jsonl"
    records = read_split(split_path)
    out = Path(args.out or Path(cfg.paths.diagnostics) / "equivalence_variants.jsonl")
    n = write_equivalence_diagnostic(records, out)
    console.print(f"Wrote {n} equivalence variants → {out}")


if __name__ == "__main__":
    main()
