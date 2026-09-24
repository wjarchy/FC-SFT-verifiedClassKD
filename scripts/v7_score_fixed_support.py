#!/usr/bin/env python3
"""Resumably score frozen V7 K=4 supports with Teacher or Student S0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from fc_class_kd.config import load_config
from fc_class_kd.data.splits import read_split
from fc_class_kd.scoring.aggregation import candidate_probabilities
from fc_class_kd.training.candidate_scoring import score_candidate_sequences
from fc_class_kd.training.model_utils import load_student_model, max_memory_mb
from fc_class_kd.types import CandidateRecord, FCRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在冻结 V7 K=4 support 上生成分数 cache")
    parser.add_argument("--config", default="v7_rtx3090")
    parser.add_argument("--kind", choices=["teacher", "s0"], default="teacher")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--pool-manifest", default="data/candidates/v7/manifest.json")
    parser.add_argument("--splits", nargs="+", default=["equiv_all", "internal_equiv"])
    parser.add_argument("--output-dir", default="data/cache/v7/teacher")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--resume", choices=["auto", "none"], default="auto")
    return parser.parse_args()


def validate_pool_rows(
    records: list[FCRecord],
    pool_rows: list[dict[str, Any]],
    split: str,
) -> None:
    expected = [record.id for record in records]
    actual = [str(row.get("sample_id")) for row in pool_rows]
    if actual != expected:
        raise ValueError(f"{split}: K=4 pool IDs/order do not match its source data")
    for row in pool_rows:
        candidates = [CandidateRecord.model_validate(item) for item in row["candidates"]]
        if len(candidates) != 4 or row.get("class_indices") != [0, 0, 1, 2]:
            raise ValueError(f"{split}/{row.get('sample_id')}: invalid frozen K=4 row")


def validate_existing(
    expected_ids: list[str],
    rows: list[dict[str, Any]],
    split: str,
) -> None:
    actual = [str(row.get("sample_id")) for row in rows]
    if actual != expected_ids[: len(rows)] or len(set(actual)) != len(actual):
        raise ValueError(f"{split}: cache resume rows are not the exact unique prefix")
    for row in rows:
        for key, expected_length in (
            ("candidate_scores", 4),
            ("candidate_probabilities", 4),
            ("class_probabilities", 3),
            ("intra_class_probabilities", 2),
        ):
            values = row.get(key)
            if not isinstance(values, list) or len(values) != expected_length:
                raise ValueError(f"{split}/{row.get('sample_id')}: invalid {key}")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"{split}/{row.get('sample_id')}: non-finite {key}")


@torch.inference_mode()
def score_one(
    model: Any,
    tokenizer: Any,
    record: FCRecord,
    candidates: list[CandidateRecord],
    max_length: int,
    alpha: float,
) -> list[float]:
    try:
        scores = score_candidate_sequences(
            model,
            tokenizer,
            record,
            candidates,
            max_length=max_length,
            alpha=alpha,
        )
        return [float(value) for value in scores.detach().cpu()]
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        values: list[float] = []
        for candidate in candidates:
            score = score_candidate_sequences(
                model,
                tokenizer,
                record,
                [candidate],
                max_length=max_length,
                alpha=alpha,
            )
            values.append(float(score.detach().cpu()[0]))
        return values


def cache_row(
    model: Any,
    tokenizer: Any,
    record: FCRecord,
    pool_row: dict[str, Any],
    max_length: int,
    alpha: float,
    temperature: float,
) -> dict[str, Any]:
    candidates = [CandidateRecord.model_validate(item) for item in pool_row["candidates"]]
    scores = score_one(model, tokenizer, record, candidates, max_length, alpha)
    probabilities = candidate_probabilities(scores, temperature=temperature)
    correct_mass = probabilities[0] + probabilities[1]
    intra = [probabilities[0] / correct_mass, probabilities[1] / correct_mass]
    entropy = -sum(value * math.log(value) for value in intra if value > 0)
    return {
        "sample_id": record.id,
        "split": pool_row["split"],
        "equivalence_type": pool_row["equivalence_type"],
        "candidate_scores": scores,
        "candidate_probabilities": probabilities,
        "class_ids": pool_row["class_ids"],
        "class_indices": [0, 0, 1, 2],
        "class_probabilities": [correct_mass, probabilities[2], probabilities[3]],
        "intra_class_probabilities": intra,
        "intra_class_entropy": entropy,
        "normalized_intra_class_entropy": entropy / math.log(2.0),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("candidate scoring requires CUDA")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(args.alpha) or args.alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    cfg = load_config(args.config)
    model_path = resolve(
        args.model
        or str(cfg.models.teacher_local if args.kind == "teacher" else cfg.models.student_local)
    )
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    model_revision = args.model_revision or str(
        cfg.models.teacher_revision if args.kind == "teacher" else cfg.models.student_revision
    )
    adapter = resolve(args.adapter) if args.adapter else None
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json missing under {adapter}")
    pool_manifest_path = resolve(args.pool_manifest)
    pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_length = args.max_length or int(cfg.training.max_length)

    split_work: list[dict[str, Any]] = []
    for split in args.splits:
        artifact = pool_manifest["artifacts"].get(split)
        if artifact is None:
            raise KeyError(f"split missing from pool manifest: {split}")
        data_path = resolve(artifact["input_path"])
        pool_path = resolve(artifact["output_path"])
        if sha256_file(data_path) != artifact["input_sha256"]:
            raise ValueError(f"{split}: source data hash drift")
        if sha256_file(pool_path) != artifact["output_sha256"]:
            raise ValueError(f"{split}: K=4 pool hash drift")
        records = read_split(data_path)
        pool_rows = read_jsonl(pool_path)
        validate_pool_rows(records, pool_rows, split)
        output_path = output_dir / f"{args.kind}_{split}_scores.jsonl"
        existing = read_jsonl(output_path)
        if args.resume == "none" and existing:
            raise FileExistsError(output_path)
        validate_existing([record.id for record in records], existing, split)
        split_work.append(
            {
                "split": split,
                "records": records,
                "pool_rows": pool_rows,
                "pool_path": pool_path,
                "data_path": data_path,
                "output_path": output_path,
                "existing": existing,
            }
        )

    definition = {
        "kind": args.kind,
        "model_path": str(model_path),
        "model_revision": model_revision,
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": (sha256_file(adapter / "adapter_model.safetensors") if adapter else None),
        "load_in_4bit": args.load_in_4bit,
        "pool_manifest": relative(pool_manifest_path),
        "pool_manifest_sha256": sha256_file(pool_manifest_path),
        "splits": args.splits,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "max_length": max_length,
    }
    run_signature = signature(definition)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_signature") != run_signature:
            raise ValueError("output directory belongs to a different scoring definition")
    else:
        manifest = {
            "manifest_version": 1,
            "created_at_utc": utc_now(),
            "status": "in_progress",
            "run_signature": run_signature,
            "definition": definition,
            "command": ["scripts/v7_score_fixed_support.py", *sys.argv[1:]],
            "git": git_state(),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "artifacts": {},
        }
        atomic_write_json(manifest_path, manifest)

    torch.cuda.reset_peak_memory_stats()
    base_model, tokenizer = load_student_model(
        str(model_path),
        load_in_4bit=args.load_in_4bit,
        bf16=True,
        fp16=False,
    )
    model = (
        PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
        if adapter is not None
        else base_model
    )
    model.config.use_cache = False
    model.eval()
    for work in split_work:
        rows = list(work["existing"])
        for index in range(len(rows), len(work["records"])):
            row = cache_row(
                model,
                tokenizer,
                work["records"][index],
                work["pool_rows"][index],
                max_length,
                args.alpha,
                args.temperature,
            )
            append_jsonl(work["output_path"], row)
            rows.append(row)
            if len(rows) % 10 == 0 or len(rows) == len(work["records"]):
                manifest["artifacts"][work["split"]] = {
                    "status": "in_progress",
                    "completed_rows": len(rows),
                    "expected_rows": len(work["records"]),
                    "output_path": relative(work["output_path"]),
                }
                manifest["peak_gpu_memory_mb"] = max_memory_mb()
                manifest["updated_at_utc"] = utc_now()
                atomic_write_json(manifest_path, manifest)
                print(f"{work['split']}: {len(rows)}/{len(work['records'])}", flush=True)
        validate_existing([record.id for record in work["records"]], rows, work["split"])
        manifest["artifacts"][work["split"]] = {
            "status": "complete",
            "rows": len(rows),
            "data_path": relative(work["data_path"]),
            "data_sha256": sha256_file(work["data_path"]),
            "pool_path": relative(work["pool_path"]),
            "pool_sha256": sha256_file(work["pool_path"]),
            "output_path": relative(work["output_path"]),
            "output_sha256": sha256_file(work["output_path"]),
        }
        atomic_write_json(manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["peak_gpu_memory_mb"] = max_memory_mb()
    manifest["completed_at_utc"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("V7_FIXED_SUPPORT_SCORING_OK=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
