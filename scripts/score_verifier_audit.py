#!/usr/bin/env python3
"""Score a completed human Verifier audit packet against the current implementation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fc_class_kd.training.formal_sft import sha256_file
from fc_class_kd.verifier.audit import audit_content_sha256, score_human_audit


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/diagnostics/verifier_human_audit_final.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="data/diagnostics/verifier_human_audit_final.manifest.json",
    )
    parser.add_argument("--out", default="reports/验证器正式人工审核结果.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if manifest.get("visible_content_sha256") != audit_content_sha256(rows):
        raise ValueError("immutable audit content changed after it was prepared")
    key_path = Path(manifest["hidden_key"])
    if manifest.get("hidden_key_sha256") != sha256_file(key_path):
        raise ValueError("hidden audit key changed after it was prepared")
    with key_path.open(encoding="utf-8") as handle:
        key_rows = [json.loads(line) for line in handle if line.strip()]
    if [row["audit_id"] for row in rows] != [row["audit_id"] for row in key_rows]:
        raise ValueError("visible audit packet and hidden key are not aligned")
    report = score_human_audit(rows)
    report["visible_packet_sha256"] = sha256_file(input_path)
    report["candidate_pool_sha256"] = manifest["candidate_pool_sha256"]
    subgroup: dict[str, dict[str, Any]] = {}
    for row, key in zip(rows, key_rows, strict=True):
        name = str(key["audit_stratum"])
        stats = subgroup.setdefault(name, {"count": 0, "matching_label_count": 0})
        stats["count"] += 1
        stats["matching_label_count"] += row["human_label"] == key["verifier_label"]
    for stats in subgroup.values():
        stats["accuracy"] = stats["matching_label_count"] / stats["count"]
    report["audit_strata"] = subgroup
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, report)
    manifest["status"] = "passed" if report["passed"] else "failed"
    manifest["scored_visible_packet_sha256"] = sha256_file(input_path)
    manifest["report"] = str(out_path)
    manifest["report_sha256"] = sha256_file(out_path)
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
