#!/usr/bin/env python3
"""Recover the V6/V7 Formal splits without using model outputs.

The old machine's exact row IDs were not pushed.  Recovery therefore uses the pinned
source revisions, the repository's historical filtering/deduplication policy, seed 42,
and the aggregate control totals recorded in the V7 PRD.  No Teacher or Student result
is read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset

from fc_class_kd.data.adapters import (
    TOOLACE_QWEN_REVISION,
    TOOLACE_SCHEMA_REVISION,
    XLAM_REPO,
    XLAM_REVISION,
    load_toolace_qwen_records,
    xlam_row_to_record,
)
from fc_class_kd.data.filters import passes_fast_pilot_filter
from fc_class_kd.types import FCRecord
from fc_class_kd.v7.equivalence import build_strict_equivalent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
CALIBRATION_POOL_SIZE = 4650
TRAIN_SIZE = 3000
VALIDATION_SIZE = 300
INTERNAL_SIZE = 500
INTERNAL_EQUIV_SIZE = 248
RESERVE_SIZE = 14229
BALANCED_PER_CATEGORY = 112


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def read_bfcl_exclusions(path: Path) -> tuple[set[str], set[str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("bfcl_revision") != BFCL_REVISION:
        raise ValueError("BFCL report revision does not match the V7 recovery definition")
    ids: set[str] = set()
    queries: set[str] = set()
    for key in ("exact_normalized_overlaps", "high_similarity_overlaps"):
        for row in report.get(key, []):
            if row.get("source") != "xlam":
                continue
            ids.add(str(row["source_id"]))
            queries.add(normalized_query(str(row["source_query"])))
    return ids, queries


def safe_xlam_defaults(record: FCRecord, raw_row: dict[str, Any]) -> dict[str, Any]:
    """Accept only non-empty, exactly typed defaults equal to the observed Gold value.

    This is narrower than trusting xLAM defaults generally.  The observed Gold already
    passed schema validation, so exact type/value agreement also validates the retained
    default.  Other xLAM defaults remain disabled.
    """
    tools = raw_row.get("tools")
    if isinstance(tools, str):
        tools = json.loads(tools)
    if not isinstance(tools, list):
        return {}
    raw_tool = next(
        (
            tool
            for tool in tools
            if isinstance(tool, dict) and tool.get("name") == record.gold_call.name
        ),
        None,
    )
    if raw_tool is None:
        return {}
    accepted: dict[str, Any] = {}
    parameters = raw_tool.get("parameters", {})
    if not isinstance(parameters, dict):
        return accepted
    for key, spec in parameters.items():
        if not isinstance(spec, dict) or key not in record.gold_call.arguments:
            continue
        if "default" not in spec:
            continue
        default = spec["default"]
        gold = record.gold_call.arguments[key]
        if default in ("", None, [], {}):
            continue
        if type(default) is type(gold) and default == gold:
            accepted[key] = default
    return accepted


def load_source_universe(
    bfcl_report: Path,
) -> tuple[list[FCRecord], list[FCRecord], dict[str, int]]:
    excluded_ids, excluded_queries = read_bfcl_exclusions(bfcl_report)
    dataset = load_dataset(XLAM_REPO, split="train", revision=XLAM_REVISION)
    xlam: list[FCRecord] = []
    seen_xlam: set[str] = set()
    xlam_filter_passed = 0
    for index, raw in enumerate(dataset):
        row = dict(raw)
        record = xlam_row_to_record(row, index)
        if record is None or not passes_fast_pilot_filter(record):
            continue
        xlam_filter_passed += 1
        normalized = normalized_query(record.query)
        if record.id in excluded_ids or normalized in excluded_queries or normalized in seen_xlam:
            continue
        seen_xlam.add(normalized)
        defaults = safe_xlam_defaults(record, row)
        record = record.model_copy(
            update={
                "metadata": {
                    **record.metadata,
                    "v7_safe_defaults": defaults,
                    "v7_safe_default_policy": "nonempty_exact_type_and_gold_value",
                }
            }
        )
        xlam.append(record)

    toolace = [
        record
        for record in load_toolace_qwen_records(
            revision=TOOLACE_QWEN_REVISION,
            schema_revision=TOOLACE_SCHEMA_REVISION,
        )
        if passes_fast_pilot_filter(record)
    ]

    # Retain the historical source precedence: restored ToolACE wins a normalized
    # cross-source query tie.  No tie is expected in the pinned audit, but enforce it.
    toolace_queries = {normalized_query(record.query) for record in toolace}
    xlam = [record for record in xlam if normalized_query(record.query) not in toolace_queries]
    counts = {
        "xlam_filter_passed": xlam_filter_passed,
        "xlam_after_bfcl_and_normalized_dedup": len(xlam),
        "toolace_restored": len(toolace),
        "universe": len(xlam) + len(toolace),
        "bfcl_excluded_ids": len(excluded_ids),
        "bfcl_excluded_normalized_queries": len(excluded_queries),
    }
    return toolace, xlam, counts


def category(record: FCRecord) -> str | None:
    variant = build_strict_equivalent(record)
    return variant.category if variant is not None else None


def source_counts(records: list[FCRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.source for record in records).items()))


def category_counts(records: list[FCRecord]) -> dict[str, int]:
    counts = Counter(category(record) or "singleton" for record in records)
    return {key: counts.get(key, 0) for key in ("A", "B", "C", "singleton")}


def write_jsonl(path: Path, records: list[FCRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
    temporary.replace(path)


def select_splits(
    toolace: list[FCRecord],
    xlam: list[FCRecord],
    seed: int,
) -> dict[str, list[FCRecord]]:
    if CALIBRATION_POOL_SIZE <= len(toolace):
        raise ValueError("calibration pool must include xLAM")
    xlam_order = list(range(len(xlam)))
    random.Random(seed).shuffle(xlam_order)
    calibration = toolace + [
        xlam[index] for index in xlam_order[: CALIBRATION_POOL_SIZE - len(toolace)]
    ]
    calibration_order = list(range(len(calibration)))
    random.Random(seed).shuffle(calibration_order)
    train = [calibration[index] for index in calibration_order[:TRAIN_SIZE]]
    validation = [
        calibration[index] for index in calibration_order[TRAIN_SIZE : TRAIN_SIZE + VALIDATION_SIZE]
    ]

    # Draw Internal from the same calibrated source pool as Train/Validation.
    # Drawing it from the full remainder would make it almost entirely xLAM because
    # xLAM dominates the raw universe, creating an avoidable source-distribution shift.
    calibration_remainder = [
        calibration[index] for index in calibration_order[TRAIN_SIZE + VALIDATION_SIZE :]
    ]
    random.Random(seed + 1).shuffle(calibration_remainder)
    equiv = [record for record in calibration_remainder if category(record) is not None]
    singleton = [record for record in calibration_remainder if category(record) is None]
    if len(equiv) < INTERNAL_EQUIV_SIZE or len(singleton) < INTERNAL_SIZE - INTERNAL_EQUIV_SIZE:
        raise ValueError("not enough rows for the preregistered Internal composition")
    internal = equiv[:INTERNAL_EQUIV_SIZE] + singleton[: INTERNAL_SIZE - INTERNAL_EQUIV_SIZE]
    random.Random(seed + 2).shuffle(internal)

    used = {record.id for record in train + validation + internal}
    reserve_candidates = [record for record in toolace + xlam if record.id not in used]
    random.Random(seed + 3).shuffle(reserve_candidates)
    if len(reserve_candidates) < RESERVE_SIZE:
        raise ValueError("not enough rows for the recorded reserve size")
    reserve = reserve_candidates[:RESERVE_SIZE]
    unused = reserve_candidates[RESERVE_SIZE:]

    train_equiv = [record for record in train if category(record) is not None]
    by_category = {
        label: [record for record in train_equiv if category(record) == label]
        for label in ("A", "B", "C")
    }
    balanced: list[FCRecord] = []
    rng = random.Random(seed)
    for label in ("A", "B", "C"):
        rows = list(by_category[label])
        rng.shuffle(rows)
        if len(rows) < BALANCED_PER_CATEGORY:
            raise ValueError(f"category {label} has {len(rows)} rows; need {BALANCED_PER_CATEGORY}")
        balanced.extend(rows[:BALANCED_PER_CATEGORY])
    rng.shuffle(balanced)

    return {
        "train": train,
        "validation": validation,
        "internal": internal,
        "internal_equiv": [record for record in internal if category(record) is not None],
        "internal_singleton": [record for record in internal if category(record) is None],
        "equiv_all": train_equiv,
        "balanced_336": balanced,
        "reserve": reserve,
        "unused": unused,
    }


def validate_splits(splits: dict[str, list[FCRecord]]) -> None:
    expected = {
        "train": TRAIN_SIZE,
        "validation": VALIDATION_SIZE,
        "internal": INTERNAL_SIZE,
        "internal_equiv": INTERNAL_EQUIV_SIZE,
        "internal_singleton": INTERNAL_SIZE - INTERNAL_EQUIV_SIZE,
        "balanced_336": BALANCED_PER_CATEGORY * 3,
        "reserve": RESERVE_SIZE,
    }
    for name, count in expected.items():
        if len(splits[name]) != count:
            raise ValueError(f"{name}: expected {count}, got {len(splits[name])}")
    ids = {
        name: {record.id for record in splits[name]} for name in ("train", "validation", "internal")
    }
    for left, right in (("train", "validation"), ("train", "internal"), ("validation", "internal")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise ValueError(f"{left}/{right}: {len(overlap)} overlapping IDs")
    balanced_counts = category_counts(splits["balanced_336"])
    for label in ("A", "B", "C"):
        if balanced_counts[label] != BALANCED_PER_CATEGORY:
            raise ValueError(f"balanced category {label}: {balanced_counts[label]}")
    for name in ("equiv_all", "internal_equiv", "balanced_336"):
        for record in splits[name]:
            variant = build_strict_equivalent(record)
            if variant is None:
                raise ValueError(f"{name}/{record.id}: missing equivalent after selection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="恢复 V7 Formal 数据与等价子集")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v7"),
    )
    parser.add_argument(
        "--bfcl-report",
        type=Path,
        default=Path("reports/BFCL数据泄漏检查.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    bfcl_report = args.bfcl_report
    if not bfcl_report.is_absolute():
        bfcl_report = PROJECT_ROOT / bfcl_report
    output_dir.mkdir(parents=True, exist_ok=True)

    toolace, xlam, universe_counts = load_source_universe(bfcl_report)
    splits = select_splits(toolace, xlam, args.seed)
    validate_splits(splits)

    output_names = (
        "train",
        "validation",
        "internal",
        "internal_equiv",
        "internal_singleton",
        "equiv_all",
        "balanced_336",
        "reserve",
    )
    paths: dict[str, Path] = {}
    for name in output_names:
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, splits[name])
        paths[name] = path

    train_categories = category_counts(splits["train"])
    historical = {"A": 1094, "B": 112, "C": 185, "equiv_all": 1391}
    observed_equiv = len(splits["equiv_all"])
    deviations = {
        "A": train_categories["A"] - historical["A"],
        "B": train_categories["B"] - historical["B"],
        "C": train_categories["C"] - historical["C"],
        "equiv_all": observed_equiv - historical["equiv_all"],
    }
    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "definition": {
            "sources": {
                "xlam": {"repo": XLAM_REPO, "revision": XLAM_REVISION},
                "toolace_cleaned_revision": TOOLACE_QWEN_REVISION,
                "toolace_schema_revision": TOOLACE_SCHEMA_REVISION,
                "bfcl_revision": BFCL_REVISION,
            },
            "calibration_pool_size": CALIBRATION_POOL_SIZE,
            "calibration_note": (
                "Chosen before model runs by minimizing disagreement with the V7 PRD's "
                "historical Train-3000 A/B/C control totals under the recovered rules."
            ),
            "split_sizes": {
                "train": TRAIN_SIZE,
                "validation": VALIDATION_SIZE,
                "internal": INTERNAL_SIZE,
                "internal_equiv": INTERNAL_EQUIV_SIZE,
                "reserve": RESERVE_SIZE,
            },
            "equivalence_precedence": ["C", "B", "A"],
            "safe_xlam_default_policy": "nonempty_exact_type_and_gold_value",
            "balanced_per_category": BALANCED_PER_CATEGORY,
        },
        "universe_counts": universe_counts,
        "counts": {
            name: {
                "rows": len(splits[name]),
                "sources": source_counts(splits[name]),
                "categories": category_counts(splits[name]),
            }
            for name in output_names
        },
        "historical_control_totals": historical,
        "deviations_from_historical": deviations,
        "unused_after_reserve": len(splits["unused"]),
        "files": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")
    print("V7_FORMAL_DATA_OK=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
