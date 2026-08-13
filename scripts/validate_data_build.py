#!/usr/bin/env python3
"""Validate a processed FC dataset and its fixed splits before promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from fc_class_kd.candidates.rules import build_rule_candidates
from fc_class_kd.data.filters import passes_fast_pilot_filter
from fc_class_kd.types import FCRecord, VerifierClass
from fc_class_kd.verifier.classifier import classify_call
from fc_class_kd.verifier.schema_checker import normalize_param_type

KNOWN_TYPES = {"string", "integer", "number", "boolean", "array", "object", "null"}
EXPECTED_SPLITS = {
    "train": 2000,
    "val": 300,
    "internal_test": 500,
    "diagnostic": 250,
    "equivalence_diagnostic": 200,
}


def _read_records(path: Path) -> list[FCRecord]:
    records: list[FCRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(FCRecord.model_validate_json(line))
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _load_leakage_queries(path: Path) -> set[str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    queries: set[str] = set()
    for key in ("exact_normalized_overlaps", "high_similarity_overlaps"):
        for row in report.get(key, []):
            if row.get("source") == "xlam":
                queries.add(_normalize_query(str(row["source_query"])))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed data and fixed splits")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--leakage-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    processed_path = args.processed_dir / "all_records.jsonl"
    manifest_path = args.processed_dir / "manifest.json"
    processed = _read_records(processed_path)
    splits = {
        name: _read_records(args.splits_dir / f"{name}.jsonl") for name in EXPECTED_SPLITS
    }

    failures: list[str] = []
    warnings: list[str] = []
    processed_ids = [record.id for record in processed]
    processed_queries = [_normalize_query(record.query) for record in processed]
    if len(processed) != sum(EXPECTED_SPLITS.values()):
        failures.append("processed_count_mismatch")
    if len(set(processed_ids)) != len(processed_ids):
        failures.append("duplicate_record_id")
    if len(set(processed_queries)) != len(processed_queries):
        failures.append("duplicate_normalized_query")

    split_id_sets: dict[str, set[str]] = {}
    for name, expected_count in EXPECTED_SPLITS.items():
        records = splits[name]
        if len(records) != expected_count:
            failures.append(f"split_count_mismatch:{name}")
        ids = {record.id for record in records}
        if len(ids) != len(records):
            failures.append(f"duplicate_id_within_split:{name}")
        split_id_sets[name] = ids

    split_names = list(split_id_sets)
    overlap_counts: dict[str, int] = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = len(split_id_sets[left] & split_id_sets[right])
            overlap_counts[f"{left}__{right}"] = overlap
            if overlap:
                failures.append(f"split_overlap:{left}:{right}")
    assigned_ids = set().union(*split_id_sets.values())
    if assigned_ids != set(processed_ids):
        failures.append("split_union_does_not_equal_processed")

    gold_invalid = 0
    parameter_types: Counter[str] = Counter()
    required_counts: Counter[str] = Counter()
    default_counts: Counter[str] = Counter()
    metadata_failures: Counter[str] = Counter()
    for record in processed:
        if not passes_fast_pilot_filter(record):
            gold_invalid += 1
        for tool in record.tools:
            for parameter in tool.parameters.values():
                normalized_type = normalize_param_type(parameter.type)
                parameter_types[normalized_type] += 1
                if any(member not in KNOWN_TYPES for member in normalized_type.split("|")):
                    failures.append(f"unsupported_type:{normalized_type}")
                required_counts["required" if parameter.required else "optional"] += 1
                default_counts["present" if parameter.default is not None else "absent"] += 1
                if record.source == "xlam" and parameter.default is not None:
                    failures.append("xlam_untrusted_default_retained")
        if record.source == "xlam" and record.metadata.get("schema_policy") != (
            "xlam_infer_required_ignore_untrusted_defaults"
        ):
            metadata_failures["xlam_schema_policy"] += 1
        if record.source == "toolace" and record.metadata.get("schema_match") != (
            "unique_query_and_ordered_structure"
        ):
            metadata_failures["toolace_schema_match"] += 1
    if gold_invalid:
        failures.append("gold_invalid")
    if metadata_failures:
        failures.append("schema_provenance_metadata_invalid")

    excluded_queries = _load_leakage_queries(args.leakage_report)
    leakage_residual = [
        record.id
        for record in processed
        if record.source == "xlam" and _normalize_query(record.query) in excluded_queries
    ]
    if leakage_residual:
        failures.append("bfcl_leakage_residual")

    diagnostic = splits["diagnostic"]
    candidate_sizes: Counter[int] = Counter()
    diagnostic_with_error = 0
    diagnostic_with_two_correct_surfaces = 0
    for record in diagnostic:
        candidates = build_rule_candidates(record, seed=42)
        candidate_sizes[len(candidates)] += 1
        labels = [
            classify_call(candidate.serialized_call, record.tools, record.gold_call).class_label
            for candidate in candidates
        ]
        if any(
            label in {VerifierClass.WRONG_FUNCTION, VerifierClass.WRONG_ARGUMENTS}
            for label in labels
        ):
            diagnostic_with_error += 1
        if sum(label == VerifierClass.CORRECT for label in labels) >= 2:
            diagnostic_with_two_correct_surfaces += 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash_failures: list[str] = []
    for raw_path, expected in manifest.get("files", {}).items():
        path = Path(raw_path)
        if not path.exists() or _sha256(path) != expected.get("sha256"):
            manifest_hash_failures.append(raw_path)
    if manifest_hash_failures:
        failures.append("manifest_hash_mismatch")

    source_counts = Counter(record.source for record in processed)
    source_by_split = {
        name: dict(Counter(record.source for record in records))
        for name, records in splits.items()
    }
    warnings.extend(
        [
            "xLAM required flags are inferred conservatively from optional/default markers.",
            "xLAM defaults are ignored for equivalence because the upstream values are noisy.",
            "Text similarity cannot prove absence of all semantic overlap with BFCL.",
            "Gold validity is structural; semantic correctness still requires sampled review.",
        ]
    )
    report: dict[str, Any] = {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "warnings": warnings,
        "processed_count": len(processed),
        "processed_sha256": _sha256(processed_path),
        "source_counts": dict(source_counts),
        "source_shares": {
            source: count / len(processed) for source, count in source_counts.items()
        },
        "split_counts": {name: len(records) for name, records in splits.items()},
        "source_by_split": source_by_split,
        "split_overlap_counts": overlap_counts,
        "unique_id_count": len(set(processed_ids)),
        "unique_normalized_query_count": len(set(processed_queries)),
        "gold_invalid_count": gold_invalid,
        "schema": {
            "parameter_types": dict(parameter_types),
            "required_counts": dict(required_counts),
            "default_counts": dict(default_counts),
            "metadata_failures": dict(metadata_failures),
        },
        "bfcl_excluded_normalized_queries": len(excluded_queries),
        "bfcl_leakage_residual_count": len(leakage_residual),
        "diagnostic_rule_candidates": {
            "candidate_size_distribution": dict(candidate_sizes),
            "with_structured_error": diagnostic_with_error,
            "with_two_correct_surfaces": diagnostic_with_two_correct_surfaces,
        },
        "manifest_hash_failures": manifest_hash_failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
