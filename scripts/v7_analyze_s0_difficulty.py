#!/usr/bin/env python3
"""Build the preregistered V7 S0 difficulty analysis from paired predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from fc_class_kd.data.splits import read_split
from fc_class_kd.evaluation.predictions import validate_prediction_rows
from fc_class_kd.training.formal_sft import sha256_file
from fc_class_kd.v7.analysis import paired_summary, tertile_bucket

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


def prediction_manifest_path(path: Path) -> Path:
    if path.suffix != ".jsonl":
        raise ValueError(f"prediction file must end in .jsonl: {path}")
    return path.with_suffix(".manifest.json")


def verified_prediction_rows(path: Path, records: Sequence[Any], split: str) -> list[dict[str, Any]]:
    manifest_path = prediction_manifest_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"prediction artifact is incomplete: {path}")
    if manifest.get("predictions_sha256") != sha256_file(path):
        raise ValueError(f"prediction hash mismatch: {path}")
    rows = read_jsonl(path)
    validate_prediction_rows(records, rows, split, require_complete=True)
    return rows


def verified_cache_rows(
    path: Path,
    manifest_path: Path,
    split: str,
    expected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("artifacts", {}).get(split)
    if manifest.get("status") != "complete" or not isinstance(entry, dict) or entry.get("status") != "complete":
        raise ValueError(f"cache is incomplete: {path}")
    if entry.get("output_sha256") != sha256_file(path):
        raise ValueError(f"cache hash mismatch: {path}")
    rows = read_jsonl(path)
    if [str(row["sample_id"]) for row in rows] != list(expected_ids):
        raise ValueError(f"cache IDs/order mismatch: {path}")
    return rows


def summarize_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return paired_summary(
        [bool(row["candidate_correct"]) for row in samples],
        [bool(row["class_correct"]) for row in samples],
        bootstrap_samples=10_000,
        seed=42,
    )


def grouped(
    samples: Sequence[dict[str, Any]],
    labels: Sequence[str],
    key: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    return {
        label: summarize_samples([row for row in samples if key(row) == label])
        for label in labels
    }


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def table(groups: dict[str, dict[str, Any]], label_title: str) -> str:
    lines = [
        f"| {label_title} | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in groups.items():
        low, high = stats["delta_bootstrap_95ci"]
        interval = "NA" if low is None else f"[{pct(low)}, {pct(high)}]"
        lines.append(
            f"| {label} | {stats['n']} | {stats['candidate_correct']}/{stats['n']} "
            f"({pct(stats['candidate_accuracy'])}) | {stats['class_correct']}/{stats['n']} "
            f"({pct(stats['class_accuracy'])}) | {pct(stats['delta_class_minus_candidate'])} | "
            f"{interval} | {stats['candidate_only_correct']} / {stats['class_only_correct']} | "
            f"{stats['mcnemar_exact_p']:.4f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 V7 S0 Difficulty 分析")
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--class-predictions", required=True)
    parser.add_argument("--s0-predictions", default="data/candidates/v7/s0_predictions/s0_internal_predictions.jsonl")
    parser.add_argument("--internal", default="data/processed/v7/internal.jsonl")
    parser.add_argument("--internal-equiv", default="data/processed/v7/internal_equiv.jsonl")
    parser.add_argument("--internal-pool", default="data/candidates/v7/internal_equiv_k4.jsonl")
    parser.add_argument("--s0-cache", default="data/cache/v7/s0/s0_internal_equiv_scores.jsonl")
    parser.add_argument("--s0-cache-manifest", default="data/cache/v7/s0/manifest.json")
    parser.add_argument("--teacher-cache", default="data/cache/v7/teacher/teacher_internal_equiv_scores.jsonl")
    parser.add_argument("--teacher-cache-manifest", default="data/cache/v7/teacher/manifest.json")
    parser.add_argument("--thresholds", default="configs/v7_analysis_thresholds.json")
    parser.add_argument("--report", default="reports/v7_s0_difficulty_analysis.md")
    parser.add_argument("--summary", default="reports/v7_s0_difficulty_summary.json")
    parser.add_argument("--samples", default="reports/v7_s0_difficulty_samples.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    excluded = {"report", "summary", "samples"}
    paths = {key: resolve(value) for key, value in vars(args).items() if key not in excluded}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    internal = read_split(paths["internal"])
    internal_equiv = read_split(paths["internal_equiv"])
    equiv_ids = [record.id for record in internal_equiv]
    internal_ids = [record.id for record in internal]
    if not set(equiv_ids).issubset(internal_ids):
        raise ValueError("Internal-Equiv is not a subset of Internal")
    candidate_rows = verified_prediction_rows(paths["candidate_predictions"], internal, "internal")
    class_rows = verified_prediction_rows(paths["class_predictions"], internal, "internal")
    s0_rows = verified_prediction_rows(paths["s0_predictions"], internal, "internal")
    candidate_by_id = {str(row["sample_id"]): row for row in candidate_rows}
    class_by_id = {str(row["sample_id"]): row for row in class_rows}
    s0_by_id = {str(row["sample_id"]): row for row in s0_rows}
    pool_rows = read_jsonl(paths["internal_pool"])
    if [str(row["sample_id"]) for row in pool_rows] != equiv_ids:
        raise ValueError("Internal-Equiv K=4 pool is misaligned")
    pool_by_id = {str(row["sample_id"]): row for row in pool_rows}
    s0_cache = verified_cache_rows(
        paths["s0_cache"], paths["s0_cache_manifest"], "internal_equiv", equiv_ids
    )
    teacher_cache = verified_cache_rows(
        paths["teacher_cache"], paths["teacher_cache_manifest"], "internal_equiv", equiv_ids
    )
    s0_cache_by_id = {str(row["sample_id"]): row for row in s0_cache}
    teacher_cache_by_id = {str(row["sample_id"]): row for row in teacher_cache}
    thresholds = json.loads(paths["thresholds"].read_text(encoding="utf-8"))
    source_hashes = thresholds["sources"]
    if source_hashes["s0_cache_sha256"] != sha256_file(resolve(source_hashes["s0_cache"])):
        raise ValueError("frozen S0 threshold source has drifted")
    if source_hashes["teacher_cache_sha256"] != sha256_file(resolve(source_hashes["teacher_cache"])):
        raise ValueError("frozen Teacher threshold source has drifted")
    confidence_thresholds = thresholds["s0_correct_class_probability"]
    entropy_thresholds = thresholds["teacher_normalized_intra_class_entropy"]

    samples: list[dict[str, Any]] = []
    for record in internal_equiv:
        sample_id = record.id
        confidence = float(s0_cache_by_id[sample_id]["class_probabilities"][0])
        entropy = float(teacher_cache_by_id[sample_id]["normalized_intra_class_entropy"])
        samples.append(
            {
                "sample_id": sample_id,
                "equivalence_type": pool_by_id[sample_id]["equivalence_type"],
                "s0_correct": s0_by_id[sample_id]["verifier_class"] == "correct",
                "s0_verifier_class": s0_by_id[sample_id]["verifier_class"],
                "s0_confidence": confidence,
                "s0_confidence_bucket": tertile_bucket(
                    confidence, confidence_thresholds["q1"], confidence_thresholds["q2"]
                ),
                "teacher_normalized_intra_class_entropy": entropy,
                "teacher_entropy_bucket": tertile_bucket(
                    entropy, entropy_thresholds["q1"], entropy_thresholds["q2"]
                ),
                "candidate_correct": candidate_by_id[sample_id]["verifier_class"] == "correct",
                "candidate_verifier_class": candidate_by_id[sample_id]["verifier_class"],
                "class_correct": class_by_id[sample_id]["verifier_class"] == "correct",
                "class_verifier_class": class_by_id[sample_id]["verifier_class"],
            }
        )

    equiv_set = set(equiv_ids)
    full_samples = [
        {
            "candidate_correct": candidate_by_id[sample_id]["verifier_class"] == "correct",
            "class_correct": class_by_id[sample_id]["verifier_class"] == "correct",
            "subset": "equiv" if sample_id in equiv_set else "singleton",
        }
        for sample_id in internal_ids
    ]
    results = {
        "full_internal": summarize_samples(full_samples),
        "internal_equiv": summarize_samples(samples),
        "internal_singleton": summarize_samples(
            [row for row in full_samples if row["subset"] == "singleton"]
        ),
        "s0_correctness": grouped(
            samples, ["correct", "wrong"], lambda row: "correct" if row["s0_correct"] else "wrong"
        ),
        "s0_confidence": grouped(
            samples, ["low", "mid", "high"], lambda row: str(row["s0_confidence_bucket"])
        ),
        "equivalence_type": grouped(
            samples, ["A", "B", "C"], lambda row: str(row["equivalence_type"])
        ),
        "teacher_entropy": grouped(
            samples, ["low", "mid", "high"], lambda row: str(row["teacher_entropy_bucket"])
        ),
    }
    confidence_deltas = [
        results["s0_confidence"][label]["delta_class_minus_candidate"]
        for label in ("high", "mid", "low")
    ]
    results["difficulty_trend"] = {
        "confidence_delta_order_high_mid_low": confidence_deltas,
        "nondecreasing_with_difficulty": all(
            left <= right for left, right in zip(confidence_deltas, confidence_deltas[1:])
        ),
        "s0_wrong_minus_correct_delta": (
            results["s0_correctness"]["wrong"]["delta_class_minus_candidate"]
            - results["s0_correctness"]["correct"]["delta_class_minus_candidate"]
        ),
    }
    summary = {
        "kind": "v7_s0_difficulty_analysis",
        "bootstrap": {"samples": 10_000, "seed": 42, "interval": "paired percentile 95%"},
        "thresholds": thresholds,
        "artifact_hashes": {
            "candidate_predictions": sha256_file(paths["candidate_predictions"]),
            "class_predictions": sha256_file(paths["class_predictions"]),
            "s0_predictions": sha256_file(paths["s0_predictions"]),
            "s0_internal_cache": sha256_file(paths["s0_cache"]),
            "teacher_internal_cache": sha256_file(paths["teacher_cache"]),
            "thresholds": sha256_file(paths["thresholds"]),
        },
        "results": results,
    }
    samples_path = resolve(args.samples)
    atomic_write(
        samples_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
    )
    summary["samples_path"] = str(samples_path.relative_to(PROJECT_ROOT))
    summary["samples_sha256"] = sha256_file(samples_path)
    atomic_write(resolve(args.summary), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    overview = {
        "Full Internal": results["full_internal"],
        "Internal-Equiv": results["internal_equiv"],
        "Internal-Singleton": results["internal_singleton"],
    }
    trend = results["difficulty_trend"]
    report = f"""# V7 S0 Difficulty Analysis

## 结论摘要

- 置信度桶按训练侧 Equiv-All 的 S0 正确功能类概率预先冻结，不使用 Internal outcome 定阈值。
- 从高置信度到低置信度的 Class - Candidate 依次为：{", ".join(pct(value) for value in confidence_deltas)}。
- 难度上升时 delta 单调不减：{str(trend["nondecreasing_with_difficulty"]).lower()}。
- S0 Wrong 与 S0 Correct 的相对 delta 差：{pct(trend["s0_wrong_minus_correct_delta"])}。

## 总体恢复基线

{table(overview, "Subset")}

## View A：S0 Correct vs Wrong

{table(results["s0_correctness"], "S0 状态")}

## View B：S0 fixed-support confidence

阈值：q1={confidence_thresholds["q1"]:.8f}，q2={confidence_thresholds["q2"]:.8f}；分桶规则为 low ≤ q1，q1 < mid ≤ q2，high > q2。

{table(results["s0_confidence"], "置信度")}

## A/B/C

{table(results["equivalence_type"], "等价类型")}

## Teacher 类内 entropy（同步保存，主分析之后解释）

阈值：q1={entropy_thresholds["q1"]:.8f}，q2={entropy_thresholds["q2"]:.8f}；由训练侧 Equiv-All Teacher cache 预先冻结。

{table(results["teacher_entropy"], "Teacher entropy")}

## 统计口径

- 所有比较均为同一样本上的配对比较；Delta 为 Class accuracy - Candidate accuracy。
- 区间是 seed 42、10,000 次的 paired percentile bootstrap 95% CI；McNemar p 是 discordant pair 的双侧精确二项检验。
- 单 seed、多个小子组的区间较宽；子组结果用于机制判断，不把未校正的 p 值当作独立确证。
- 样本级表：[v7_s0_difficulty_samples.jsonl](v7_s0_difficulty_samples.jsonl)，SHA256 {summary["samples_sha256"]}。
"""
    atomic_write(resolve(args.report), report)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("V7_S0_DIFFICULTY_ANALYSIS_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
