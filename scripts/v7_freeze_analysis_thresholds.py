#!/usr/bin/env python3
"""Freeze V7 confidence/entropy tertiles using Equiv-All only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def linear_quantile(values: Sequence[float], quantile: float) -> float:
    """NumPy-compatible linear quantile, implemented without hidden defaults."""
    if not values:
        raise ValueError("cannot compute quantile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bucket_counts(values: Sequence[float], low_upper: float, mid_upper: float) -> dict[str, int]:
    if low_upper > mid_upper:
        raise ValueError("tertile cut points are reversed")
    result = {"low": 0, "mid": 0, "high": 0}
    for raw in values:
        value = float(raw)
        if value <= low_upper:
            result["low"] += 1
        elif value <= mid_upper:
            result["mid"] += 1
        else:
            result["high"] += 1
    return result


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def complete_manifest(path: Path, artifact: Path, split: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"cache manifest is incomplete: {path}")
    entry = manifest.get("artifacts", {}).get(split)
    if not isinstance(entry, dict) or entry.get("status") != "complete":
        raise ValueError(f"cache split is incomplete: {split}")
    if entry.get("output_sha256") != sha256_file(artifact):
        raise ValueError(f"cache artifact hash mismatch: {artifact}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结 V7 Equiv-All 分析三分位阈值")
    parser.add_argument("--s0-cache", default="data/cache/v7/s0/s0_equiv_all_scores.jsonl")
    parser.add_argument("--s0-manifest", default="data/cache/v7/s0/manifest.json")
    parser.add_argument("--teacher-cache", default="data/cache/v7/teacher/teacher_equiv_all_scores.jsonl")
    parser.add_argument("--teacher-manifest", default="data/cache/v7/teacher/manifest.json")
    parser.add_argument("--output", default="configs/v7_analysis_thresholds.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    s0_cache = resolve(args.s0_cache)
    s0_manifest_path = resolve(args.s0_manifest)
    teacher_cache = resolve(args.teacher_cache)
    teacher_manifest_path = resolve(args.teacher_manifest)
    output = resolve(args.output)
    for path in (s0_cache, s0_manifest_path, teacher_cache, teacher_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    complete_manifest(s0_manifest_path, s0_cache, "equiv_all")
    complete_manifest(teacher_manifest_path, teacher_cache, "equiv_all")
    s0_rows = read_jsonl(s0_cache)
    teacher_rows = read_jsonl(teacher_cache)
    s0_ids = [str(row["sample_id"]) for row in s0_rows]
    teacher_ids = [str(row["sample_id"]) for row in teacher_rows]
    if s0_ids != teacher_ids or len(set(s0_ids)) != len(s0_ids):
        raise ValueError("S0 and Teacher Equiv-All caches are not exactly aligned")
    s0_values = [float(row["class_probabilities"][0]) for row in s0_rows]
    entropy_values = [float(row["normalized_intra_class_entropy"]) for row in teacher_rows]
    confidence_cut = [linear_quantile(s0_values, 1 / 3), linear_quantile(s0_values, 2 / 3)]
    entropy_cut = [
        linear_quantile(entropy_values, 1 / 3),
        linear_quantile(entropy_values, 2 / 3),
    ]
    payload = {
        "manifest_version": 1,
        "kind": "v7_preregistered_analysis_thresholds",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "analysis_pool": "Equiv-All-1393",
        "analysis_pool_rows": len(s0_rows),
        "quantile_method": "linear: index=(n-1)q; interpolate adjacent ordered values",
        "bucket_rule": "low <= q1; q1 < mid <= q2; high > q2",
        "sources": {
            "s0_cache": str(s0_cache.relative_to(PROJECT_ROOT)),
            "s0_cache_sha256": sha256_file(s0_cache),
            "s0_manifest_sha256": sha256_file(s0_manifest_path),
            "teacher_cache": str(teacher_cache.relative_to(PROJECT_ROOT)),
            "teacher_cache_sha256": sha256_file(teacher_cache),
            "teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        },
        "s0_correct_class_probability": {
            "q1": confidence_cut[0],
            "q2": confidence_cut[1],
            "analysis_pool_bucket_counts": bucket_counts(
                s0_values, confidence_cut[0], confidence_cut[1]
            ),
        },
        "teacher_normalized_intra_class_entropy": {
            "q1": entropy_cut[0],
            "q2": entropy_cut[1],
            "analysis_pool_bucket_counts": bucket_counts(
                entropy_values, entropy_cut[0], entropy_cut[1]
            ),
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"analysis thresholds are already frozen: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    print(serialized, end="")
    print("V7_ANALYSIS_THRESHOLDS_FROZEN=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
