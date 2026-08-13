#!/usr/bin/env python3
"""Refresh saved S0 labels with the current deterministic Verifier."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fc_class_kd.data.splits import read_split
from fc_class_kd.evaluation.predictions import (
    candidate_surface_from_prediction,
    summarize_prediction_rows,
    validate_prediction_rows,
)
from fc_class_kd.training.formal_sft import sha256_file
from fc_class_kd.verifier.classifier import classify_call

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用当前验证器重新核算 S0 推理标签")
    parser.add_argument("--candidate-dir", default="data/candidates")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "diagnostic"],
        default=["train", "diagnostic"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_dir = PROJECT_ROOT / args.candidate_dir
    for split in args.splits:
        records = read_split(PROJECT_ROOT / "data/splits" / f"{split}.jsonl")
        prediction_path = candidate_dir / f"s0_{split}_predictions.jsonl"
        manifest_path = candidate_dir / f"s0_{split}_predictions.manifest.json"
        rows = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validate_prediction_rows(records, rows, split, require_complete=True)
        changes = 0
        for record, row in zip(records, rows, strict=True):
            prediction = str(row["prediction"])
            result = classify_call(prediction, record.tools, record.gold_call)
            refreshed = {
                "candidate_surface": candidate_surface_from_prediction(prediction),
                "verifier_class": result.class_label.value,
                "verifier_sub_label": result.sub_label,
                "parse_ok": result.parse_ok,
                "verifier_details": result.details,
            }
            changes += any(row.get(key) != value for key, value in refreshed.items())
            row.update(refreshed)
        atomic_write_jsonl(prediction_path, rows)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"{split}: inference manifest is not complete")
        manifest["predictions_sha256"] = sha256_file(prediction_path)
        manifest["summary"] = summarize_prediction_rows(records, rows)
        manifest["revalidation"] = {
            "at_utc": utc_now(),
            "label_change_count": changes,
            "classifier_sha256": sha256_file(
                PROJECT_ROOT / "src/fc_class_kd/verifier/classifier.py"
            ),
            "parser_sha256": sha256_file(
                PROJECT_ROOT / "src/fc_class_kd/verifier/parser.py"
            ),
            "normalizer_sha256": sha256_file(
                PROJECT_ROOT / "src/fc_class_kd/verifier/normalizer.py"
            ),
            "schema_checker_sha256": sha256_file(
                PROJECT_ROOT / "src/fc_class_kd/verifier/schema_checker.py"
            ),
        }
        atomic_write_json(manifest_path, manifest)
        print(
            json.dumps(
                {split: {"label_change_count": changes, **manifest["summary"]}},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    print("S0_PREDICTION_REVALIDATION_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
