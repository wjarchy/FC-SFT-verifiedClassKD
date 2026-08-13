#!/usr/bin/env python3
"""Recompute saved S0 Validation labels after a deterministic verifier change."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fc_class_kd.data.splits import read_split
from fc_class_kd.evaluation.metrics import evaluate_predictions
from fc_class_kd.evaluation.predictions import candidate_surface_from_prediction
from fc_class_kd.training.formal_sft import select_s0_by_validation_ast, sha256_file
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
    parser = argparse.ArgumentParser(description="重新核算公共 S0 的 Validation 结果")
    parser.add_argument("--s0-dir", default="outputs/s0_qwen3_1.7b_seed42")
    parser.add_argument("--validation", default="data/splits/val.jsonl")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    s0_dir = resolve(args.s0_dir)
    records = read_split(resolve(args.validation))
    by_id = {record.id: record for record in records}
    selected_path = s0_dir / "selected_s0.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    evaluations: list[dict[str, Any]] = []
    verifier_hash = sha256_file(PROJECT_ROOT / "src/fc_class_kd/verifier/parser.py")
    timestamp = utc_now()
    for old_evaluation in selected["evaluations"]:
        checkpoint = str(old_evaluation["checkpoint"])
        result_dir = s0_dir / "validation_by_checkpoint" / checkpoint
        predictions_path = result_dir / "validation_predictions.jsonl"
        rows = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [row["sample_id"] for row in rows] != [record.id for record in records]:
            raise ValueError(f"{checkpoint}: Validation IDs or order changed")
        for row in rows:
            record = by_id[row["sample_id"]]
            prediction = str(row["prediction"])
            result = classify_call(prediction, record.tools, record.gold_call)
            row.update(
                candidate_surface=candidate_surface_from_prediction(prediction),
                verifier_class=result.class_label.value,
                verifier_sub_label=result.sub_label,
                parse_ok=result.parse_ok,
                verifier_details=result.details,
            )
        atomic_write_jsonl(predictions_path, rows)
        metrics = evaluate_predictions(
            records,
            [str(row["prediction"]) for row in rows],
        ).rates()
        predictions_hash = sha256_file(predictions_path)
        metrics_path = result_dir / "validation_generation_metrics.json"
        result_file = json.loads(metrics_path.read_text(encoding="utf-8"))
        result_file.update(
            metrics=metrics,
            predictions_sha256=predictions_hash,
            revalidated_at_utc=timestamp,
            verifier_parser_sha256=verifier_hash,
        )
        atomic_write_json(metrics_path, result_file)
        evaluation = dict(old_evaluation)
        evaluation.update(
            ast_accuracy=metrics["ast_accuracy"],
            generation_metrics=metrics,
            predictions_sha256=predictions_hash,
        )
        evaluations.append(evaluation)

    winner = select_s0_by_validation_ast(evaluations)
    selection = dict(selected)
    selection.update(
        selected_checkpoint=winner["checkpoint"],
        selected_epoch=winner["epoch"],
        selected_ast_accuracy=winner["ast_accuracy"],
        selected_validation_loss=winner["validation_loss"],
        evaluations=evaluations,
        revalidated_at_utc=timestamp,
        verifier_parser_sha256=verifier_hash,
    )
    atomic_write_json(selected_path, selection)
    atomic_write_json(
        s0_dir / "best_checkpoint.json",
        {
            "checkpoint": winner["checkpoint"],
            "selection_file": "selected_s0.json",
            "selected_at_utc": selection["selected_at_utc"],
            "revalidated_at_utc": timestamp,
        },
    )
    summary_path = s0_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["s0_selection"] = selection
    summary["validation_revalidated_at_utc"] = timestamp
    atomic_write_json(summary_path, summary)
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    print("S0_REVALIDATION_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
