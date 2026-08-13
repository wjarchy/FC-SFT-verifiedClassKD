#!/usr/bin/env python3
"""Audit a downloaded xLAM JSON file before it can enter experiment splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from fc_class_kd.data.adapters import xlam_row_to_record
from fc_class_kd.data.filters import passes_fast_pilot_filter
from fc_class_kd.verifier.classifier import classify_function_call
from fc_class_kd.verifier.schema_checker import normalize_param_type

KNOWN_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _load_existing_queries(path: Path | None) -> set[str]:
    if path is None:
        return set()
    queries: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query = str(row.get("query", "")).strip()
            if query:
                queries.add(query)
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local xLAM data before split building")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--toolace", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/xLAM数据审计.json"))
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=str(args.input), split="train")
    existing_queries = _load_existing_queries(args.toolace)

    answer_counts: Counter[int] = Counter()
    tool_counts: Counter[int] = Counter()
    parameter_types: Counter[str] = Counter()
    normalized_types: Counter[str] = Counter()
    unknown_types: Counter[str] = Counter()
    raw_required: Counter[str] = Counter()
    raw_defaults: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    gold_issue_counts: Counter[str] = Counter()
    accepted_queries: Counter[str] = Counter()
    adapter_accepted = 0
    filter_accepted = 0

    for index, raw_row in enumerate(dataset):
        row = dict(raw_row)
        tools = _decode(row.get("tools"))
        answers = _decode(row.get("answers"))
        row_has_unknown_type = False

        if isinstance(answers, list):
            answer_counts[len(answers)] += 1
        else:
            rejection_reasons["answers_unreadable"] += 1

        if isinstance(tools, list):
            tool_counts[len(tools)] += 1
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                parameters = tool.get("parameters", {})
                if not isinstance(parameters, dict):
                    continue
                for spec in parameters.values():
                    if not isinstance(spec, dict):
                        continue
                    raw_type = str(spec.get("type", "string"))
                    normalized = normalize_param_type(raw_type)
                    parameter_types[raw_type] += 1
                    normalized_types[normalized] += 1
                    if normalized not in KNOWN_TYPES:
                        unknown_types[raw_type] += 1
                        row_has_unknown_type = True
                    if "required" not in spec:
                        raw_required["missing"] += 1
                    elif bool(spec["required"]):
                        raw_required["true"] += 1
                    else:
                        raw_required["false"] += 1
                    if "default" not in spec:
                        raw_defaults["missing"] += 1
                    elif spec["default"] == "":
                        raw_defaults["empty_string"] += 1
                    else:
                        raw_defaults["explicit_nonempty"] += 1
        else:
            rejection_reasons["tools_unreadable"] += 1

        if not row.get("query"):
            rejection_reasons["missing_query"] += 1
        elif not isinstance(answers, list) or len(answers) != 1:
            rejection_reasons["not_single_call"] += 1
        elif not isinstance(tools, list) or len(tools) < 2:
            rejection_reasons["fewer_than_two_tools"] += 1
        elif row_has_unknown_type:
            rejection_reasons["unsupported_parameter_type"] += 1

        record = xlam_row_to_record(row, index)
        if record is None:
            continue
        adapter_accepted += 1
        if not passes_fast_pilot_filter(record):
            result = classify_function_call(record.gold_call, record.tools, record.gold_call)
            if result.details:
                for detail in result.details:
                    gold_issue_counts[detail.split(":", 1)[0]] += 1
            else:
                gold_issue_counts[result.class_label.value] += 1
            continue
        filter_accepted += 1
        accepted_queries[record.query.strip()] += 1

    exact_duplicate_rows = sum(count - 1 for count in accepted_queries.values() if count > 1)
    toolace_overlap = sum(1 for query in accepted_queries if query in existing_queries)
    report = {
        "input": str(args.input),
        "input_bytes": args.input.stat().st_size,
        "input_sha256": _sha256(args.input),
        "raw_rows": len(dataset),
        "answer_count_distribution": dict(sorted(answer_counts.items())),
        "tool_count_distribution": dict(sorted(tool_counts.items())),
        "adapter_accepted": adapter_accepted,
        "fast_pilot_filter_accepted": filter_accepted,
        "accepted_unique_exact_queries": len(accepted_queries),
        "accepted_exact_duplicate_rows": exact_duplicate_rows,
        "exact_query_overlap_with_toolace": toolace_overlap,
        "rejection_reasons": dict(rejection_reasons),
        "gold_issue_counts": dict(gold_issue_counts),
        "schema": {
            "parameter_types": dict(parameter_types),
            "normalized_types": dict(normalized_types),
            "unknown_types": dict(unknown_types),
            "required_field": dict(raw_required),
            "default_field": dict(raw_defaults),
        },
        "limitations": [
            "This audit checks structure and schema consistency, not semantic correctness.",
            "BFCL exact and near-duplicate leakage is audited separately.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
