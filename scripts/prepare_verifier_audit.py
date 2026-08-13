#!/usr/bin/env python3
"""Prepare a shuffled final K=6 audit packet without exposed Verifier labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fc_class_kd.config import load_config
from fc_class_kd.data.splits import read_split
from fc_class_kd.training.formal_sft import sha256_file
from fc_class_kd.types import CandidatePoolRecord
from fc_class_kd.verifier.audit import (
    audit_content_sha256,
    build_final_blind_audit_rows,
    write_audit_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_pools(path: Path) -> list[CandidatePoolRecord]:
    with path.open(encoding="utf-8") as handle:
        return [CandidatePoolRecord.model_validate_json(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="从两套未冻结 K=6 候选池准备正式盲审包")
    parser.add_argument("--config", default="rtx3090")
    parser.add_argument("--candidate-dir", default="data/candidates")
    parser.add_argument("--per-stratum", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="data/diagnostics/verifier_human_audit_final.jsonl",
    )
    parser.add_argument(
        "--key-out",
        default="data/diagnostics/verifier_human_audit_final_key.jsonl",
    )
    parser.add_argument(
        "--manifest-out",
        default="data/diagnostics/verifier_human_audit_final.manifest.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    candidate_dir = PROJECT_ROOT / args.candidate_dir
    splits = ("train", "diagnostic")
    records_by_split = {
        split: read_split(Path(cfg.paths.splits) / f"{split}.jsonl") for split in splits
    }
    pools_by_split: dict[str, list[CandidatePoolRecord]] = {}
    pool_hashes: dict[str, str] = {}
    pool_manifest_hashes: dict[str, str] = {}
    for split in splits:
        pool_path = candidate_dir / f"{split}_unfrozen.jsonl"
        pool_manifest_path = candidate_dir / f"{split}_unfrozen.manifest.json"
        manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "built_but_unfrozen":
            raise ValueError(f"{split}: expected a built but unfrozen candidate pool")
        if manifest.get("candidate_pool_sha256") != sha256_file(pool_path):
            raise ValueError(f"{split}: candidate pool hash mismatch")
        pools_by_split[split] = read_pools(pool_path)
        pool_hashes[split] = sha256_file(pool_path)
        pool_manifest_hashes[split] = sha256_file(pool_manifest_path)

    rows, key_rows = build_final_blind_audit_rows(
        records_by_split,
        pools_by_split,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    out_path = PROJECT_ROOT / args.out
    key_path = PROJECT_ROOT / args.key_out
    manifest_path = PROJECT_ROOT / args.manifest_out
    existing = [path for path in (out_path, key_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite an audit already in progress: {existing}")
    write_audit_rows(rows, out_path)
    write_audit_rows(key_rows, key_path)
    atomic_write_json(
        manifest_path,
        {
            "manifest_version": 1,
            "audit_kind": "final_k6_pool_blind_human_audit",
            "seed": args.seed,
            "per_stratum": args.per_stratum,
            "row_count": len(rows),
            "visible_packet": str(out_path.relative_to(PROJECT_ROOT)),
            "visible_packet_sha256": sha256_file(out_path),
            "visible_content_sha256": audit_content_sha256(rows),
            "hidden_key": str(key_path.relative_to(PROJECT_ROOT)),
            "hidden_key_sha256": sha256_file(key_path),
            "candidate_pool_sha256": pool_hashes,
            "candidate_pool_manifest_sha256": pool_manifest_hashes,
            "status": "awaiting_human_labels",
        },
    )
    print(f"已写入 {len(rows)} 条正式盲审样本 → {args.out}")
    print("标注时不要打开 key 文件；填完 human_label 后运行 score_verifier_audit.py")
    print("VERIFIER_HUMAN_GATE=NOT_YET_EVALUATED")


if __name__ == "__main__":
    main()
