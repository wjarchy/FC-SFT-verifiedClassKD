#!/usr/bin/env python3
"""Combine preregistered pair summaries into V7 scaling and entropy reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from fc_class_kd.training.formal_sft import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != "v7_s0_difficulty_analysis":
        raise ValueError(f"wrong summary kind: {path}")
    return value


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def stat_cell(stats: dict[str, Any]) -> str:
    low, high = stats["delta_bootstrap_95ci"]
    return (
        f"{stats['candidate_correct']}/{stats['n']} ({pct(stats['candidate_accuracy'])}) | "
        f"{stats['class_correct']}/{stats['n']} ({pct(stats['class_accuracy'])}) | "
        f"{pct(stats['delta_class_minus_candidate'])} | [{pct(low)}, {pct(high)}]"
    )


def outcome_table(summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Setting | Subset | N | Candidate | Class | Delta | Paired bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    subset_labels = [
        ("full_internal", "Full Internal"),
        ("internal_equiv", "Internal-Equiv"),
        ("internal_singleton", "Internal-Singleton"),
    ]
    for setting, summary in summaries.items():
        for key, label in subset_labels:
            stats = summary["results"][key]
            low, high = stats["delta_bootstrap_95ci"]
            lines.append(
                f"| {setting} | {label} | {stats['n']} | "
                f"{stats['candidate_correct']}/{stats['n']} ({pct(stats['candidate_accuracy'])}) | "
                f"{stats['class_correct']}/{stats['n']} ({pct(stats['class_accuracy'])}) | "
                f"{pct(stats['delta_class_minus_candidate'])} | [{pct(low)}, {pct(high)}] |"
            )
    return "\n".join(lines)


def grouped_table(
    summaries: dict[str, dict[str, Any]],
    result_key: str,
    labels: list[str],
    group_title: str,
) -> str:
    lines = [
        f"| Setting | {group_title} | N | Candidate | Class | Delta | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for setting, summary in summaries.items():
        for label in labels:
            stats = summary["results"][result_key][label]
            low, high = stats["delta_bootstrap_95ci"]
            lines.append(
                f"| {setting} | {label} | {stats['n']} | "
                f"{pct(stats['candidate_accuracy'])} | {pct(stats['class_accuracy'])} | "
                f"{pct(stats['delta_class_minus_candidate'])} | [{pct(low)}, {pct(high)}] |"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 V7 scaling 与 Teacher entropy")
    parser.add_argument("--baseline-summary", default="reports/v7_s0_difficulty_summary.json")
    parser.add_argument("--balanced4-summary", default="reports/v7_balanced336_e4_summary.json")
    parser.add_argument("--equiv-all-summary", default="reports/v7_equiv_all_e1_summary.json")
    parser.add_argument("--output-summary", default="reports/v7_scaling_summary.json")
    parser.add_argument("--scaling-report", default="reports/v7_scaling_report.md")
    parser.add_argument("--entropy-report", default="reports/v7_teacher_entropy_analysis.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_paths = {
        "Balanced-336 × 1": resolve(args.baseline_summary),
        "Balanced-336 × 4": resolve(args.balanced4_summary),
        "Equiv-All-1393 × 1": resolve(args.equiv_all_summary),
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    summaries = {label: load_summary(path) for label, path in source_paths.items()}
    threshold_sources = [
        json.dumps(value["thresholds"]["sources"], sort_keys=True)
        for value in summaries.values()
    ]
    if len(set(threshold_sources)) != 1:
        raise ValueError("pair summaries do not use the same frozen threshold sources")
    metadata = {
        "Balanced-336 × 1": {"unique": 336, "exposures": 336, "steps": 21},
        "Balanced-336 × 4": {"unique": 336, "exposures": 1344, "steps": 84},
        "Equiv-All-1393 × 1": {"unique": 1393, "exposures": 1393, "steps": 88},
    }
    equiv_deltas = {
        label: summary["results"]["internal_equiv"]["delta_class_minus_candidate"]
        for label, summary in summaries.items()
    }
    derived = {
        "balanced4_minus_baseline_delta": (
            equiv_deltas["Balanced-336 × 4"] - equiv_deltas["Balanced-336 × 1"]
        ),
        "equiv_all_minus_balanced4_delta": (
            equiv_deltas["Equiv-All-1393 × 1"] - equiv_deltas["Balanced-336 × 4"]
        ),
    }
    output = {
        "kind": "v7_scaling_and_entropy_analysis",
        "source_summaries": {
            label: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
            for label, path in source_paths.items()
        },
        "setting_metadata": metadata,
        "internal_equiv_deltas": equiv_deltas,
        "derived_delta_contrasts": derived,
        "results": {label: summary["results"] for label, summary in summaries.items()},
    }
    summary_path = resolve(args.output_summary)
    atomic_write(summary_path, json.dumps(output, ensure_ascii=False, indent=2) + "\n")

    scaling = f"""# V7 Data Scaling / Diversity Rescue

## 固定比较

| Setting | Unique samples | Sample exposures | Optimizer steps |
|---|---:|---:|---:|
| Balanced-336 × 1 | 336 | 336 | 21 |
| Balanced-336 × 4 | 336 | 1,344 | 84 |
| Equiv-All-1393 × 1 | 1,393 | 1,393 | 88 |

后两项 exposure 接近；前者重复同一小集合，后者增加 unique query 并保留自然 A/B/C 分布。

## 主结果

{outcome_table(summaries)}

## S0 Correct / Wrong

{grouped_table(summaries, "s0_correctness", ["correct", "wrong"], "S0 状态")}

## S0 confidence tertile

{grouped_table(summaries, "s0_confidence", ["low", "mid", "high"], "置信度")}

## A/B/C

{grouped_table(summaries, "equivalence_type", ["A", "B", "C"], "等价类型")}

## 科研判断

- Balanced-336 从 1 轮增加到 4 轮后，Internal-Equiv 的 Class-Candidate delta 从 -2.02pt 改善到 -0.40pt，但仍未转为 Class 优势；近似相同 exposure 下，Equiv-All-1393×1 的 delta 为 +3.63pt。
- Equiv-All 的目标 subset 提升为 9/248，paired bootstrap 95% CI [-0.81pt, 8.06pt]，McNemar exact p=0.1496；Full Internal 同时为 +2.40pt，95% CI [-0.20pt, 5.20pt]，没有观察到明显整体退化。
- 证据方向支持“更多 unique query 与自然 A/B/C 分布”的组合条件比重复小集合更重要，不支持“只要增加 exposure 就足够”。由于 Equiv-All 同时改变 unique diversity 和类别分布，本实验不能进一步把两者因果拆开。
- 当前仅单 seed，两个主区间均跨 0；因此这是有方向一致性的 rescue 信号，不是已确证的 universal superiority。
- S0 confidence 在 Equiv-All 下呈 high 0.00pt、mid +4.35pt、low +6.67pt，但 S0 free-generation Wrong 组为 -2.78pt；difficulty 机制只得到部分支持，不满足 Weak-S0 的明确触发条件。

## 直接对比

- Balanced-336 × 4 相对 Balanced-336 × 1 的 Internal-Equiv delta 变化：{pct(derived["balanced4_minus_baseline_delta"])}。
- Equiv-All-1393 × 1 相对 Balanced-336 × 4 的 Internal-Equiv delta 变化：{pct(derived["equiv_all_minus_balanced4_delta"])}。
- 所有区间均来自同一样本配对的 seed 42、10,000 次 percentile bootstrap。单 seed 下不把区间重叠解释成等价性证明。
"""
    atomic_write(resolve(args.scaling_report), scaling)

    entropy = f"""# V7 Teacher Entropy Analysis

Teacher 正确功能类内部的 normalized entropy 三分位阈值由训练侧 Equiv-All Teacher cache 预先冻结；Internal outcome 没有参与阈值选择。

{grouped_table(summaries, "teacher_entropy", ["low", "mid", "high"], "Teacher entropy")}

## 科研判断

- Equiv-All-1393×1 的 low/mid/high entropy delta 分别为 +4.12pt、+4.48pt、+2.38pt，没有随 entropy 升高而增强。
- Balanced-336×4 虽从 low -3.09pt 变到 high +2.38pt，但该 setting 总体没有 Class 优势；三种训练设置之间方向不一致。
- 因而 Teacher 类内 entropy 不能解释本轮 Equiv-All rescue，最多保留为未确证的探索性异质性。

## 解释边界

- 这里检验 Class - Candidate 是否随 Teacher 类内偏好不稳定性上升而改善。
- 分桶是机制分析，不更换 Full Internal 或 Internal-Equiv 主指标。
- 多个子组共享样本和模型，未做多重比较校正；结论以方向一致性、效应量和区间共同判断。
"""
    atomic_write(resolve(args.entropy_report), entropy)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("V7_SCALING_REPORT_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
