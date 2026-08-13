#!/usr/bin/env python3
"""Build exact, labeled, but not-yet-frozen K=6 candidate pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fc_class_kd.candidates.pool import build_k6_pool, validate_k6_pool
from fc_class_kd.config import load_config
from fc_class_kd.data.splits import read_split
from fc_class_kd.evaluation.predictions import validate_prediction_rows
from fc_class_kd.prompting.templates import encode_candidate_pair
from fc_class_kd.training.formal_sft import sha256_file
from fc_class_kd.training.model_utils import load_tokenizer
from fc_class_kd.verifier.classifier import classify_call

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def signature(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造 Train/Diagnostic 严格 K=6 候选池草稿")
    parser.add_argument("--config", default="rtx3090")
    parser.add_argument("--candidate-dir", default="data/candidates")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "diagnostic"],
        default=["diagnostic", "train"],
    )
    parser.add_argument("--max-samples", type=int, default=None, help="仅供冒烟测试")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    cfg = load_config(args.config)
    candidate_dir = resolve(args.candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(str(cfg.models.get("student_local", cfg.models.student)))
    max_length = int(cfg.training.max_length)

    for split in args.splits:
        split_path = PROJECT_ROOT / "data/splits" / f"{split}.jsonl"
        records = read_split(split_path)
        if args.max_samples is not None:
            records = records[: args.max_samples]
        prediction_path = candidate_dir / f"s0_{split}_predictions.jsonl"
        prediction_manifest_path = candidate_dir / f"s0_{split}_predictions.manifest.json"
        prediction_manifest = json.loads(
            prediction_manifest_path.read_text(encoding="utf-8")
        )
        predictions = read_jsonl(prediction_path)
        validate_prediction_rows(records, predictions, split, require_complete=True)
        if prediction_manifest.get("status") != "complete":
            raise ValueError(f"{split}: S0 predictions are not marked complete")
        if prediction_manifest["definition"].get("formal_run") != (args.max_samples is None):
            raise ValueError(f"{split}: formal/debug prediction status does not match")
        if prediction_manifest.get("predictions_sha256") != sha256_file(prediction_path):
            raise ValueError(f"{split}: S0 prediction hash mismatch")

        for record, prediction in zip(records, predictions, strict=True):
            fresh = classify_call(
                str(prediction["prediction"]), record.tools, record.gold_call
            )
            if fresh.class_label.value != prediction.get("verifier_class"):
                raise ValueError(f"{split}/{record.id}: stale verifier label")

        output_path = candidate_dir / f"{split}_unfrozen.jsonl"
        manifest_path = candidate_dir / f"{split}_unfrozen.manifest.json"
        definition = {
            "kind": "unfrozen_exact_k6_candidate_pool",
            "split": split,
            "formal_run": args.max_samples is None,
            "record_count": len(records),
            "k": int(cfg.candidates.K),
            "split_sha256": sha256_file(split_path),
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
            "rules_sha256": sha256_file(
                PROJECT_ROOT / "src/fc_class_kd/candidates/pool.py"
            ),
            "verifier_sha256": sha256_file(
                PROJECT_ROOT / "src/fc_class_kd/verifier/classifier.py"
            ),
            "parser_sha256": sha256_file(
                PROJECT_ROOT / "src/fc_class_kd/verifier/parser.py"
            ),
            "max_length": max_length,
        }
        if definition["k"] != 6:
            raise ValueError("formal candidate builder requires K=6")
        run_signature = signature(definition)
        if manifest_path.exists() or output_path.exists():
            if not (manifest_path.exists() and output_path.exists()):
                raise FileExistsError(f"{split}: partial candidate output already exists")
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                old.get("run_signature") == run_signature
                and old.get("candidate_pool_sha256") == sha256_file(output_path)
            ):
                print(f"{split}: 已存在完全一致的候选池草稿，跳过", flush=True)
                continue
            raise FileExistsError(f"{split}: refusing to overwrite a different pool")

        temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        class_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        multi_correct = 0
        student_errors = 0
        student_errors_included = 0
        max_encoded = 0
        with temporary.open("w", encoding="utf-8") as handle:
            for record, prediction in zip(records, predictions, strict=True):
                if prediction["verifier_class"] != "correct":
                    student_errors += 1
                pool = build_k6_pool(
                    record,
                    split,
                    str(prediction["candidate_surface"]),
                )
                validate_k6_pool(pool)
                correct_in_pool = 0
                for candidate in pool.candidates:
                    class_counts[candidate.verifier.class_label.value] += 1
                    source_counts[candidate.source] += 1
                    correct_in_pool += candidate.verifier.class_label.value == "correct"
                    student_errors_included += candidate.source == "s0_real_error"
                    ids, _ = encode_candidate_pair(
                        tokenizer,
                        record,
                        candidate.serialized_call,
                        max_length=max_length,
                    )
                    max_encoded = max(max_encoded, len(ids))
                multi_correct += correct_in_pool >= 2
                handle.write(json.dumps(pool.model_dump(mode="json"), ensure_ascii=False) + "\n")
        temporary.replace(output_path)
        if student_errors != student_errors_included:
            raise ValueError(
                f"{split}: included {student_errors_included}/{student_errors} S0 errors"
            )
        summary = {
            "pool_count": len(records),
            "candidate_count": len(records) * 6,
            "class_counts": dict(class_counts),
            "source_counts": dict(source_counts),
            "multi_correct_pool_count": multi_correct,
            "multi_correct_pool_rate": multi_correct / len(records),
            "s0_real_error_count": student_errors,
            "s0_real_error_included_count": student_errors_included,
            "max_encoded_tokens": max_encoded,
            "requires_human_audit": True,
            "frozen": False,
        }
        manifest = {
            "manifest_version": 1,
            "created_at_utc": utc_now(),
            "run_signature": run_signature,
            "definition": definition,
            "status": "built_but_unfrozen",
            "candidate_pool_sha256": sha256_file(output_path),
            "summary": summary,
        }
        atomic_write_json(manifest_path, manifest)
        print(json.dumps({split: summary}, ensure_ascii=False, indent=2), flush=True)

    print("UNFROZEN_K6_POOLS_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
