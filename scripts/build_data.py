#!/usr/bin/env python3
"""Download, filter, and split xLAM + ToolACE for Fast Pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from fc_class_kd.config import load_config
from fc_class_kd.data.adapters import (
    TOOLACE_QWEN_REPO,
    TOOLACE_QWEN_REVISION,
    TOOLACE_SCHEMA_REPO,
    TOOLACE_SCHEMA_REVISION,
    XLAM_REPO,
    XLAM_REVISION,
    load_toolace_qwen_records,
    xlam_row_to_record,
)
from fc_class_kd.data.filters import passes_fast_pilot_filter
from fc_class_kd.data.splits import deterministic_split, split_manifest, write_split

console = Console()
BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"


def load_xlam_records(limit: int | None = None, revision: str = XLAM_REVISION) -> list:
    ds = load_dataset(
        XLAM_REPO,
        split="train",
        revision=revision,
    )
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    records = []
    for i, row in enumerate(ds):
        rec = xlam_row_to_record(dict(row), i)
        if rec and passes_fast_pilot_filter(rec):
            records.append(rec)
    return records


def load_toolace_records(
    limit: int | None = None,
    revision: str = TOOLACE_QWEN_REVISION,
    schema_revision: str | None = TOOLACE_SCHEMA_REVISION,
) -> list:
    return load_toolace_qwen_records(
        limit,
        revision=revision,
        schema_revision=schema_revision,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_leakage_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def load_bfcl_exclusions(path: Path) -> tuple[set[str], set[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"BFCL leakage report not found at {path}; refusing to build xLAM splits"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("bfcl_revision") != BFCL_REVISION:
        raise ValueError(
            "BFCL leakage report revision does not match the pinned experiment revision: "
            f"{report.get('bfcl_revision')} != {BFCL_REVISION}"
        )
    excluded: set[str] = set()
    excluded_queries: set[str] = set()
    for key in ("exact_normalized_overlaps", "high_similarity_overlaps"):
        for row in report.get(key, []):
            if row.get("source") == "xlam" and row.get("source_id"):
                excluded.add(str(row["source_id"]))
                excluded_queries.add(_normalize_leakage_query(str(row["source_query"])))
    if not excluded:
        raise ValueError("BFCL leakage report contains no xLAM exclusions; audit is incomplete")
    return excluded, excluded_queries


def select_processed_records(
    xlam: list,
    toolace: list,
    target_total: int,
    seed: int,
) -> tuple[list, dict[str, Any]]:
    """Prefer all restored ToolACE rows and use xLAM to fill the requested total."""
    seen_queries: set[str] = set()
    unique_toolace = []
    unique_xlam = []
    # ToolACE has authoritative restored schemas, so it wins any cross-source exact tie.
    for source_records, destination in ((toolace, unique_toolace), (xlam, unique_xlam)):
        for record in source_records:
            normalized_query = _normalize_leakage_query(record.query)
            if normalized_query in seen_queries:
                continue
            seen_queries.add(normalized_query)
            destination.append(record)

    available = unique_toolace + unique_xlam
    if len(available) <= target_total:
        selected = available
        strategy = "all_available"
    elif len(unique_toolace) <= target_total:
        rng = random.Random(seed)
        rng.shuffle(unique_xlam)
        selected = unique_toolace + unique_xlam[: target_total - len(unique_toolace)]
        strategy = "all_toolace_plus_seeded_xlam_fill"
    else:
        rng = random.Random(seed)
        rng.shuffle(unique_toolace)
        selected = unique_toolace[:target_total]
        strategy = "seeded_toolace_sample"

    return selected, {
        "strategy": strategy,
        "target_total": target_total,
        "available_after_exact_query_dedup": {
            "toolace": len(unique_toolace),
            "xlam": len(unique_xlam),
        },
        "selected_by_source": dict(Counter(record.source for record in selected)),
    }


def write_reproducibility_manifest(
    processed_path: Path,
    splits_dir: Path,
    args: argparse.Namespace,
    counts: dict[str, Any],
) -> Path:
    """Bind processed data and splits to source revisions, arguments, and hashes."""
    files = [processed_path, *sorted(splits_dir.glob("*.jsonl"))]
    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "source_datasets": {
            "xlam": {
                "repo_id": XLAM_REPO,
                "revision": args.xlam_revision if "xlam" in args.sources else None,
            },
            "toolace": {
                "repo_id": TOOLACE_QWEN_REPO,
                "revision": args.toolace_revision if "toolace" in args.sources else None,
                "schema_repo_id": TOOLACE_SCHEMA_REPO,
                "schema_revision": (
                    args.toolace_schema_revision if "toolace" in args.sources else None
                ),
                "schema_restoration": (
                    "unique query match plus ordered tool-description/parameter-key structure; "
                    "cleaned function names retained"
                ),
            },
        },
        "parameters": {
            "sources": args.sources,
            "xlam_limit": args.xlam_limit,
            "toolace_limit": args.toolace_limit,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "internal_test_size": args.internal_test_size,
            "diagnostic_size": args.diagnostic_size,
            "equivalence_size": args.equivalence_size,
            "seed": args.seed,
            "deduplication": "NFKC lowercase alphanumeric-token query",
            "bfcl_leakage_report": (
                str(args.bfcl_leakage_report) if "xlam" in args.sources else None
            ),
        },
        "counts": counts,
        "files": {
            str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        },
        "benchmark_leakage": {
            "bfcl_included_as_source": False,
            "bfcl_revision": BFCL_REVISION if "xlam" in args.sources else None,
            "report_path": (
                str(args.bfcl_leakage_report) if "xlam" in args.sources else None
            ),
            "report_sha256": (
                _sha256(args.bfcl_leakage_report) if "xlam" in args.sources else None
            ),
            "status": "exact_and_high_similarity_xlam_matches_excluded",
            "limitations": "Text similarity cannot prove absence of all semantic leakage.",
        },
    }
    path = processed_path.parent / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Fast Pilot data splits")
    parser.add_argument("--config", default="base")
    parser.add_argument("--xlam-limit", type=int, default=None, help="Cap raw xLAM rows scanned")
    parser.add_argument("--toolace-limit", type=int, default=None)
    parser.add_argument("--xlam-revision", default=XLAM_REVISION)
    parser.add_argument("--toolace-revision", default=TOOLACE_QWEN_REVISION)
    parser.add_argument("--toolace-schema-revision", default=TOOLACE_SCHEMA_REVISION)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--internal-test-size", type=int, default=500)
    parser.add_argument("--diagnostic-size", type=int, default=250)
    parser.add_argument("--equivalence-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bfcl-leakage-report",
        type=Path,
        default=Path("reports/BFCL数据泄漏检查.json"),
    )
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["xlam", "toolace"],
        choices=["xlam", "toolace"],
    )
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_dir = args.splits_dir or Path(cfg.paths.splits)
    processed_dir = args.processed_dir or Path(cfg.paths.processed)
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
                xlam = load_xlam_records(args.xlam_limit, revision=args.xlam_revision)
                if not xlam:
                    raise ValueError("xLAM loaded but no rows passed the Fast Pilot filter")
                console.print(f"  xLAM passed filter: {len(xlam)}")
            except Exception as exc:
                raise RuntimeError(
                    "xLAM was requested but could not be loaded. Accept the dataset terms, "
                    "log in to Hugging Face, and rerun; refusing to silently build a "
                    "ToolACE-only split."
                ) from exc

            excluded_ids, excluded_queries = load_bfcl_exclusions(args.bfcl_leakage_report)
            before_exclusion = len(xlam)
            xlam = [
                record
                for record in xlam
                if record.id not in excluded_ids
                and _normalize_leakage_query(record.query) not in excluded_queries
            ]
            console.print(
                f"  xLAM BFCL leakage exclusions: {before_exclusion - len(xlam)}"
            )

        if "toolace" in args.sources:
            console.print("[bold]Downloading ToolACE (Qwen-cleaned)...[/bold]")
            toolace_raw = load_toolace_records(
                args.toolace_limit,
                revision=args.toolace_revision,
                schema_revision=args.toolace_schema_revision,
            )
            toolace = [r for r in toolace_raw if passes_fast_pilot_filter(r)]
            if not toolace:
                raise ValueError("ToolACE loaded but no rows passed the Fast Pilot filter")
            console.print(f"  ToolACE passed filter: {len(toolace)}")

        target_total = (
            args.train_size
            + args.val_size
            + args.internal_test_size
            + args.diagnostic_size
            + args.equivalence_size
        )
        all_records, selection = select_processed_records(
            xlam,
            toolace,
            target_total=target_total,
            seed=args.seed,
        )
        console.print(f"  Selection: {selection}")

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

    split_counts = split_manifest(splits_dir)
    table = Table(title="Split manifest")
    table.add_column("Split")
    table.add_column("Count", justify="right")
    for name, count in split_counts.items():
        table.add_row(name, str(count))
    console.print(table)
    manifest_path = write_reproducibility_manifest(
        processed_dir / "all_records.jsonl",
        splits_dir,
        args,
        {
            "processed_deduplicated": len(all_records),
            "processed_by_source": dict(Counter(record.source for record in all_records)),
            "selection": selection if not args.skip_download else "cached_processed_file",
            "split_assigned": sum(split_counts.values()),
            "processed_unassigned_after_proportional_rounding": (
                len(all_records) - sum(split_counts.values())
            ),
            "splits": split_counts,
        },
    )
    console.print(f"Manifest → {manifest_path}")
    console.print("[green]Done.[/green]")


if __name__ == "__main__":
    main()
