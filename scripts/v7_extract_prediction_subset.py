#!/usr/bin/env python3
"""Deterministically derive a prediction artifact for an exact split subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from fc_class_kd.data.splits import read_split
from fc_class_kd.evaluation.predictions import summarize_prediction_rows, validate_prediction_rows
from fc_class_kd.training.formal_sft import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def derive_prediction_subset(
    source_records: Sequence[Any],
    subset_records: Sequence[Any],
    source_rows: Sequence[dict[str, Any]],
    subset_name: str,
) -> list[dict[str, Any]]:
    """Select source rows in subset order and relabel only the split field."""
    source_by_id = {record.id: record for record in source_records}
    row_by_id = {str(row["sample_id"]): row for row in source_rows}
    if len(source_by_id) != len(source_records):
        raise ValueError("source split contains duplicate IDs")
    if len(row_by_id) != len(source_rows):
        raise ValueError("source predictions contain duplicate IDs")
    derived: list[dict[str, Any]] = []
    for record in subset_records:
        if record.id not in source_by_id:
            raise ValueError(f"subset ID is absent from source split: {record.id}")
        if source_by_id[record.id].model_dump(mode="json") != record.model_dump(mode="json"):
            raise ValueError(f"subset record differs from source record: {record.id}")
        if record.id not in row_by_id:
            raise ValueError(f"source prediction missing subset ID: {record.id}")
        row = dict(row_by_id[record.id])
        row["split"] = subset_name
        derived.append(row)
    return derived


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从完整 split 预测确定性派生子集预测")
    parser.add_argument("--source-name", default="internal")
    parser.add_argument("--subset-name", default="internal_equiv")
    parser.add_argument("--source-split", default="data/processed/v7/internal.jsonl")
    parser.add_argument("--subset-split", default="data/processed/v7/internal_equiv.jsonl")
    parser.add_argument("--source-predictions", default="data/candidates/v7/s0_predictions/s0_internal_predictions.jsonl")
    parser.add_argument("--source-manifest", default="data/candidates/v7/s0_predictions/s0_internal_predictions.manifest.json")
    parser.add_argument("--output-dir", default="data/candidates/v7/s0_predictions")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    source_split = resolve(args.source_split)
    subset_split = resolve(args.subset_split)
    source_predictions = resolve(args.source_predictions)
    source_manifest_path = resolve(args.source_manifest)
    for path in (source_split, subset_split, source_predictions, source_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_records = read_split(source_split)
    subset_records = read_split(subset_split)
    source_rows = read_jsonl(source_predictions)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise ValueError("source prediction manifest is not complete")
    if source_manifest.get("predictions_sha256") != sha256_file(source_predictions):
        raise ValueError("source prediction hash does not match its manifest")
    validate_prediction_rows(source_records, source_rows, args.source_name, require_complete=True)
    rows = derive_prediction_subset(source_records, subset_records, source_rows, args.subset_name)
    validate_prediction_rows(subset_records, rows, args.subset_name, require_complete=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    output_dir = resolve(args.output_dir)
    output_path = output_dir / f"s0_{args.subset_name}_predictions.jsonl"
    manifest_path = output_dir / f"s0_{args.subset_name}_predictions.manifest.json"
    if output_path.exists() and output_path.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"refusing to replace different artifact: {output_path}")
    if not output_path.exists():
        atomic_write_text(output_path, payload)
    definition = {
        "kind": "deterministic_prediction_subset",
        "source_name": args.source_name,
        "subset_name": args.subset_name,
        "source_split": str(source_split.relative_to(PROJECT_ROOT)),
        "source_split_sha256": sha256_file(source_split),
        "subset_split": str(subset_split.relative_to(PROJECT_ROOT)),
        "subset_split_sha256": sha256_file(subset_split),
        "source_predictions": str(source_predictions.relative_to(PROJECT_ROOT)),
        "source_predictions_sha256": sha256_file(source_predictions),
        "source_manifest_sha256": sha256_file(source_manifest_path),
    }
    signature = hashlib.sha256(
        json.dumps(definition, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "manifest_version": 1,
        "created_at_utc": utc_now(),
        "run_signature": signature,
        "definition": definition,
        "status": "complete",
        "completed_count": len(rows),
        "predictions_sha256": sha256_file(output_path),
        "summary": summarize_prediction_rows(subset_records, rows),
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("run_signature") != signature:
            raise FileExistsError(f"manifest belongs to another definition: {manifest_path}")
        manifest["created_at_utc"] = old["created_at_utc"]
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("PREDICTION_SUBSET_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
