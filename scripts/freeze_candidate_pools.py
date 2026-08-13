#!/usr/bin/env python3
"""Freeze exact K=6 pools only after the human Verifier gate passes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fc_class_kd.candidates.pool import validate_k6_pool
from fc_class_kd.data.splits import read_split
from fc_class_kd.training.formal_sft import sha256_file
from fc_class_kd.types import CandidatePoolRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人工审核通过后冻结两套 K=6 候选池")
    parser.add_argument("--candidate-dir", default="data/candidates")
    parser.add_argument(
        "--audit-manifest",
        default="data/diagnostics/verifier_human_audit_final.manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_dir = PROJECT_ROOT / args.candidate_dir
    audit_manifest_path = PROJECT_ROOT / args.audit_manifest
    audit = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("human Verifier audit has not passed; candidate pools stay unfrozen")
    report_path = Path(audit["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ValueError("human audit report says the gate failed")
    if audit.get("report_sha256") != sha256_file(report_path):
        raise ValueError("human audit report hash mismatch")

    pool_hashes: dict[str, str] = {}
    pool_counts: dict[str, int] = {}
    sources: dict[str, Path] = {}
    destinations: dict[str, Path] = {}
    for split in ("train", "diagnostic"):
        source = candidate_dir / f"{split}_unfrozen.jsonl"
        source_manifest_path = candidate_dir / f"{split}_unfrozen.manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        current_hash = sha256_file(source)
        if source_manifest.get("candidate_pool_sha256") != current_hash:
            raise ValueError(f"{split}: unfrozen pool hash mismatch")
        if audit["candidate_pool_sha256"].get(split) != current_hash:
            raise ValueError(f"{split}: pool changed after human audit sampling")
        records = read_split(PROJECT_ROOT / "data/splits" / f"{split}.jsonl")
        expected_ids = [record.id for record in records]
        pools: list[CandidatePoolRecord] = []
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    pool = CandidatePoolRecord.model_validate_json(line)
                    validate_k6_pool(pool)
                    pools.append(pool)
        if [pool.sample_id for pool in pools] != expected_ids:
            raise ValueError(f"{split}: pool IDs or order do not match the frozen split")
        destination = candidate_dir / f"{split}.jsonl"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite frozen pool: {destination}")
        sources[split] = source
        destinations[split] = destination
        pool_hashes[split] = current_hash
        pool_counts[split] = len(pools)

    frozen_manifest_path = candidate_dir / "frozen_manifest.json"
    if frozen_manifest_path.exists():
        raise FileExistsError("refusing to overwrite frozen_manifest.json")
    for split in ("train", "diagnostic"):
        temporary = destinations[split].with_name(
            f".{destinations[split].name}.tmp-{os.getpid()}"
        )
        shutil.copyfile(sources[split], temporary)
        temporary.replace(destinations[split])
        if sha256_file(destinations[split]) != pool_hashes[split]:
            raise RuntimeError(f"{split}: frozen copy hash mismatch")
    atomic_write_json(
        frozen_manifest_path,
        {
            "manifest_version": 1,
            "status": "frozen",
            "frozen_at_utc": utc_now(),
            "k": 6,
            "pool_count": pool_counts,
            "candidate_pool_sha256": pool_hashes,
            "human_audit_manifest": str(audit_manifest_path.relative_to(PROJECT_ROOT)),
            "human_audit_manifest_sha256": sha256_file(audit_manifest_path),
            "human_audit_report": str(report_path),
            "human_audit_report_sha256": sha256_file(report_path),
            "correct_equivalent_precision": report["correct_equivalent_precision"],
            "overall_accuracy": report["overall_accuracy"],
        },
    )
    print(json.dumps(json.loads(frozen_manifest_path.read_text()), ensure_ascii=False, indent=2))
    print("CANDIDATE_POOLS_FROZEN=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
