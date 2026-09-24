#!/usr/bin/env python3
"""Collect both directions of V7 Candidate/Class discordant cases."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fc_class_kd.data.splits import read_split
from fc_class_kd.training.formal_sft import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def prediction_rows(path: Path, expected_ids: list[str]) -> list[dict[str, Any]]:
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete predictions: {path}")
    if manifest.get("predictions_sha256") != sha256_file(path):
        raise ValueError(f"prediction hash mismatch: {path}")
    rows = read_jsonl(path)
    if [str(row["sample_id"]) for row in rows] != expected_ids:
        raise ValueError(f"prediction IDs/order mismatch: {path}")
    return rows


def cache_rows(path: Path, manifest_path: Path, split: str, expected_ids: list[str]) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest.get("artifacts", {}).get(split)
    if manifest.get("status") != "complete" or artifact.get("status") != "complete":
        raise ValueError(f"incomplete cache: {path}")
    if artifact.get("output_sha256") != sha256_file(path):
        raise ValueError(f"cache hash mismatch: {path}")
    rows = read_jsonl(path)
    if [str(row["sample_id"]) for row in rows] != expected_ids:
        raise ValueError(f"cache IDs/order mismatch: {path}")
    return rows


def parse_pair(value: str) -> tuple[str, Path, Path]:
    if "=" not in value or "," not in value:
        raise ValueError("pair must be LABEL=CANDIDATE_JSONL,CLASS_JSONL")
    label, paths = value.split("=", 1)
    candidate, class_kd = paths.split(",", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"unsafe pair label: {label}")
    return label, resolve(candidate), resolve(class_kd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 V7 discordant cases")
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--internal", default="data/processed/v7/internal.jsonl")
    parser.add_argument("--internal-equiv", default="data/processed/v7/internal_equiv.jsonl")
    parser.add_argument("--internal-pool", default="data/candidates/v7/internal_equiv_k4.jsonl")
    parser.add_argument("--s0-predictions", default="data/candidates/v7/s0_predictions/s0_internal_predictions.jsonl")
    parser.add_argument("--s0-cache", default="data/cache/v7/s0/s0_internal_equiv_scores.jsonl")
    parser.add_argument("--s0-cache-manifest", default="data/cache/v7/s0/manifest.json")
    parser.add_argument("--teacher-cache", default="data/cache/v7/teacher/teacher_internal_equiv_scores.jsonl")
    parser.add_argument("--teacher-cache-manifest", default="data/cache/v7/teacher/manifest.json")
    parser.add_argument("--output-jsonl", default="reports/v7_discordant_cases.jsonl")
    parser.add_argument("--output-report", default="reports/v7_discordant_cases.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    internal = read_split(resolve(args.internal))
    equiv = read_split(resolve(args.internal_equiv))
    internal_ids = [record.id for record in internal]
    equiv_ids = [record.id for record in equiv]
    record_by_id = {record.id: record for record in equiv}
    pool = read_jsonl(resolve(args.internal_pool))
    if [str(row["sample_id"]) for row in pool] != equiv_ids:
        raise ValueError("Internal-Equiv pool is misaligned")
    pool_by_id = {str(row["sample_id"]): row for row in pool}
    s0 = prediction_rows(resolve(args.s0_predictions), internal_ids)
    s0_by_id = {str(row["sample_id"]): row for row in s0}
    s0_cache = cache_rows(
        resolve(args.s0_cache), resolve(args.s0_cache_manifest), "internal_equiv", equiv_ids
    )
    teacher = cache_rows(
        resolve(args.teacher_cache),
        resolve(args.teacher_cache_manifest),
        "internal_equiv",
        equiv_ids,
    )
    s0_cache_by_id = {str(row["sample_id"]): row for row in s0_cache}
    teacher_by_id = {str(row["sample_id"]): row for row in teacher}

    cases: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    for pair_value in args.pair:
        label, candidate_path, class_path = parse_pair(pair_value)
        candidate = prediction_rows(candidate_path, internal_ids)
        class_kd = prediction_rows(class_path, internal_ids)
        candidate_by_id = {str(row["sample_id"]): row for row in candidate}
        class_by_id = {str(row["sample_id"]): row for row in class_kd}
        source_hashes[label] = {
            "candidate": sha256_file(candidate_path),
            "class": sha256_file(class_path),
        }
        for sample_id in equiv_ids:
            cand_row = candidate_by_id[sample_id]
            class_row = class_by_id[sample_id]
            cand_correct = cand_row["verifier_class"] == "correct"
            class_correct = class_row["verifier_class"] == "correct"
            if cand_correct == class_correct:
                continue
            record = record_by_id[sample_id]
            pool_row = pool_by_id[sample_id]
            teacher_row = teacher_by_id[sample_id]
            cases.append(
                {
                    "setting": label,
                    "direction": (
                        "candidate_correct_class_wrong"
                        if cand_correct
                        else "candidate_wrong_class_correct"
                    ),
                    "sample_id": sample_id,
                    "equivalence_type": pool_row["equivalence_type"],
                    "query": record.query,
                    "schema": [tool.model_dump(mode="json") for tool in record.tools],
                    "gold": record.gold_call.model_dump(mode="json"),
                    "equivalent_surface": pool_row["candidates"][1]["serialized_call"],
                    "teacher_candidate_probabilities": teacher_row["candidate_probabilities"],
                    "teacher_class_probabilities": teacher_row["class_probabilities"],
                    "teacher_normalized_intra_class_entropy": teacher_row[
                        "normalized_intra_class_entropy"
                    ],
                    "s0_output": s0_by_id[sample_id]["prediction"],
                    "s0_verifier_class": s0_by_id[sample_id]["verifier_class"],
                    "s0_correct_class_confidence": s0_cache_by_id[sample_id][
                        "class_probabilities"
                    ][0],
                    "candidate_output": cand_row["prediction"],
                    "candidate_error_type": cand_row["verifier_class"],
                    "class_output": class_row["prediction"],
                    "class_error_type": class_row["verifier_class"],
                }
            )

    output_path = resolve(args.output_jsonl)
    atomic_write(
        output_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
    )
    counts = Counter((row["setting"], row["direction"]) for row in cases)
    pair_labels = [parse_pair(value)[0] for value in args.pair]
    count_lines = [
        "| Setting | Candidate correct / Class wrong | Candidate wrong / Class correct | Total |",
        "|---|---:|---:|---:|",
    ]
    for label in pair_labels:
        cand_win = counts[(label, "candidate_correct_class_wrong")]
        class_win = counts[(label, "candidate_wrong_class_correct")]
        count_lines.append(f"| {label} | {cand_win} | {class_win} | {cand_win + class_win} |")
    example_lines = [
        "| Setting | Direction | ID | A/B/C | S0 class | S0 conf | Teacher entropy |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in cases[:40]:
        example_lines.append(
            f"| {row['setting']} | {row['direction']} | {row['sample_id']} | "
            f"{row['equivalence_type']} | {row['s0_verifier_class']} | "
            f"{float(row['s0_correct_class_confidence']):.4f} | "
            f"{float(row['teacher_normalized_intra_class_entropy']):.4f} |"
        )
    report = f"""# V7 Discordant Cases

## 两侧 wins

{chr(10).join(count_lines)}

JSONL 同时保存 Candidate win 与 Class win，不筛掉任何一侧。逐 case 包含 query、schema、Gold、Equivalent、Teacher 概率/entropy、S0 输出/confidence、两模型输出和 error type。

## 前 40 个 case 索引

{chr(10).join(example_lines)}

## 资产

- Cases: [v7_discordant_cases.jsonl](v7_discordant_cases.jsonl)
- Rows: {len(cases)}
- SHA256: {sha256_file(output_path)}
- Prediction hashes: {json.dumps(source_hashes, ensure_ascii=False, sort_keys=True)}
"""
    atomic_write(resolve(args.output_report), report)
    print(json.dumps({"rows": len(cases), "counts": {"|".join(key): value for key, value in counts.items()}}, ensure_ascii=False, indent=2))
    print("V7_DISCORDANT_CASES_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
