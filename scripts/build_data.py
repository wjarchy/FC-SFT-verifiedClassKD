#!/usr/bin/env python3
"""Download, filter, and split xLAM + ToolACE for Fast Pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from fc_class_kd.config import load_config
from fc_class_kd.data.adapters import load_toolace_qwen_records, xlam_row_to_record
from fc_class_kd.data.filters import passes_fast_pilot_filter
from fc_class_kd.data.splits import deterministic_split, split_manifest, write_split

console = Console()


def load_xlam_records(limit: int | None = None) -> list:
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    records = []
    for i, row in enumerate(ds):
        rec = xlam_row_to_record(dict(row), i)
        if rec and passes_fast_pilot_filter(rec):
            records.append(rec)
    return records


def load_toolace_records(limit: int | None = None) -> list:
    return load_toolace_qwen_records(limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Fast Pilot data splits")
    parser.add_argument("--config", default="base")
    parser.add_argument("--xlam-limit", type=int, default=None, help="Cap raw xLAM rows scanned")
    parser.add_argument("--toolace-limit", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--internal-test-size", type=int, default=500)
    parser.add_argument("--diagnostic-size", type=int, default=250)
    parser.add_argument("--equivalence-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["xlam", "toolace"],
        choices=["xlam", "toolace"],
    )
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_dir = Path(cfg.paths.splits)
    processed_dir = Path(cfg.paths.processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_download:
        manifest_path = processed_dir / "all_records.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No cached records at {manifest_path}")
        from fc_class_kd.types import FCRecord

        all_records = []
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                all_records.append(FCRecord.model_validate_json(line.strip()))
    else:
        xlam: list = []
        toolace: list = []

        if "xlam" in args.sources:
            console.print("[bold]Downloading xLAM...[/bold]")
            try:
                xlam = load_xlam_records(args.xlam_limit)
                console.print(f"  xLAM passed filter: {len(xlam)}")
            except Exception as exc:
                console.print(f"[yellow]xLAM skipped: {exc}[/yellow]")
                console.print("  Accept terms on HF + run: huggingface-cli login")

        if "toolace" in args.sources:
            console.print("[bold]Downloading ToolACE (Qwen-cleaned)...[/bold]")
            toolace_raw = load_toolace_records(args.toolace_limit)
            toolace = [r for r in toolace_raw if passes_fast_pilot_filter(r)]
            console.print(f"  ToolACE passed filter: {len(toolace)}")

        all_records = xlam + toolace
        seen_q: set[int] = set()
        deduped = []
        for r in all_records:
            h = hash(r.query.strip())
            if h in seen_q:
                continue
            seen_q.add(h)
            deduped.append(r)
        all_records = deduped

        cache_path = processed_dir / "all_records.jsonl"
        with cache_path.open("w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
        console.print(f"Cached {len(all_records)} records → {cache_path}")

    split = deterministic_split(
        all_records,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        internal_test_size=args.internal_test_size,
        diagnostic_size=args.diagnostic_size,
        equivalence_size=args.equivalence_size,
    )
    write_split(split, splits_dir)

    table = Table(title="Split manifest")
    table.add_column("Split")
    table.add_column("Count", justify="right")
    for name, count in split_manifest(splits_dir).items():
        table.add_row(name, str(count))
    console.print(table)
    console.print("[green]Done.[/green]")


if __name__ == "__main__":
    main()
