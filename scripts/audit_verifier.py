#!/usr/bin/env python3
"""Automated Verifier sanity check on gold calls and rule candidates.

This script does not replace the human-labelled PRD §4.4 audit gate.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fc_class_kd.candidates.rules import build_rule_candidates
from fc_class_kd.config import load_config
from fc_class_kd.data.splits import read_split
from fc_class_kd.types import VerifierClass
from fc_class_kd.verifier.classifier import classify_function_call

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="base")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_path = Path(cfg.paths.splits) / f"{args.split}.jsonl"
    records = read_split(split_path)
    rng = random.Random(args.seed)
    if len(records) > args.sample_size:
        records = rng.sample(records, args.sample_size)

    class_counts: Counter = Counter()
    gold_correct = 0
    gold_total = 0
    candidate_audit = []

    for rec in records:
        gold_total += 1
        gold_res = classify_function_call(rec.gold_call, rec.tools, rec.gold_call)
        if gold_res.class_label == VerifierClass.CORRECT:
            gold_correct += 1

        for cand in build_rule_candidates(rec, seed=args.seed):
            res = classify_function_call(cand, rec.tools, rec.gold_call)
            class_counts[res.class_label.value] += 1
            candidate_audit.append(
                {
                    "id": rec.id,
                    "candidate_name": cand.name,
                    "class": res.class_label.value,
                    "sub": res.sub_label,
                    "details": res.details,
                }
            )

    gold_self_check_rate = gold_correct / gold_total if gold_total else 0.0

    table = Table(title="Verifier Audit")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Gold samples", str(gold_total))
    table.add_row("Gold self-check rate", f"{gold_self_check_rate:.4f}")
    for label, count in sorted(class_counts.items()):
        table.add_row(f"Candidate class: {label}", str(count))
    console.print(table)

    automated_sanity_passed = (
        gold_self_check_rate >= cfg.verifier.audit.correct_precision_min
    )
    console.print(f"Automated sanity passed: {automated_sanity_passed}")
    console.print("PRD human-labelled audit gate: NOT EVALUATED")

    out_path = args.out or str(Path(cfg.paths.reports) / "verifier_audit.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "audit_kind": "automated_sanity_only",
                "limitations": (
                    "Candidate labels were produced by the verifier itself; "
                    "human-labelled precision and overall accuracy were not measured."
                ),
                "gold_self_check_rate": gold_self_check_rate,
                "gold_total": gold_total,
                "class_counts": dict(class_counts),
                "automated_sanity_passed": automated_sanity_passed,
                "prd_manual_gate_passed": None,
                "samples": candidate_audit[:50],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    console.print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
