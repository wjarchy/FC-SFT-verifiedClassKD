#!/usr/bin/env python3
"""Build and freeze the deterministic V7 K=4 candidate supports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fc_class_kd.data.splits import read_split
from fc_class_kd.types import FCRecord
from fc_class_kd.v7.candidates import build_k4_pool, candidate_class_ids
from fc_class_kd.v7.equivalence import build_strict_equivalent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ("equiv_all", "balanced_336", "internal_equiv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def build_rows(records: list[FCRecord], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        variant = build_strict_equivalent(record)
        if variant is None:
            raise ValueError(f"{split}/{record.id}: missing strict equivalent")
        candidates = build_k4_pool(record)
        class_ids = candidate_class_ids(record, candidates)
        if not (
            len(candidates) == 4
            and len({candidate.serialized_call for candidate in candidates}) == 4
            and class_ids[0] == class_ids[1]
            and len(set(class_ids)) == 3
        ):
            raise ValueError(f"{split}/{record.id}: invalid K=4 functional support")
        rows.append(
            {
                "sample_id": record.id,
                "split": split,
                "equivalence_type": variant.category,
                "equivalence_rule": variant.rule,
                "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                "class_ids": class_ids,
                "class_indices": [0, 0, 1, 2],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造并冻结 V7 K=4 candidate pool")
    parser.add_argument("--data-dir", default="data/processed/v7")
    parser.add_argument("--output-dir", default="data/candidates/v7")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--formal-manifest", default="data/processed/v7/manifest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = resolve(args.data_dir)
    output_dir = resolve(args.output_dir)
    formal_manifest = resolve(args.formal_manifest)
    if not formal_manifest.is_file():
        raise FileNotFoundError(formal_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Any] = {}
    seen_by_split: dict[str, set[str]] = {}
    for split in args.splits:
        input_path = data_dir / f"{split}.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        records = read_split(input_path)
        rows = build_rows(records, split)
        output_path = output_dir / f"{split}_k4.jsonl"
        write_jsonl(output_path, rows)
        seen_by_split[split] = {record.id for record in records}
        artifacts[split] = {
            "input_path": relative(input_path),
            "input_sha256": sha256_file(input_path),
            "output_path": relative(output_path),
            "output_sha256": sha256_file(output_path),
            "rows": len(rows),
            "candidates": len(rows) * 4,
        }

    if (
        "balanced_336" in seen_by_split
        and "equiv_all" in seen_by_split
        and not seen_by_split["balanced_336"] <= seen_by_split["equiv_all"]
    ):
        raise ValueError("Balanced-336 must be an ID subset of Equiv-All")
    if "internal_equiv" in seen_by_split and "equiv_all" in seen_by_split:
        overlap = seen_by_split["internal_equiv"] & seen_by_split["equiv_all"]
        if overlap:
            raise ValueError(f"Internal-Equiv leaks into Train Equiv-All: {len(overlap)}")

    manifest = {
        "manifest_version": 1,
        "kind": "v7_k4_functional_candidate_pools",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": ["scripts/v7_build_k4.py", *sys.argv[1:]],
        "definition": {
            "order": ["gold", "equivalent", "wrong_argument", "wrong_function"],
            "class_indices": [0, 0, 1, 2],
            "formal_manifest": relative(formal_manifest),
            "formal_manifest_sha256": sha256_file(formal_manifest),
        },
        "git": git_state(),
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("V7_K4_OK=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
