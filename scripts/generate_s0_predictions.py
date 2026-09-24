#!/usr/bin/env python3
"""Run the selected public S0 on Train and Diagnostic with resumable outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from fc_class_kd.config import load_config
from fc_class_kd.data.splits import read_split
from fc_class_kd.evaluation.predictions import (
    make_prediction_row,
    summarize_prediction_rows,
    validate_prediction_rows,
)
from fc_class_kd.prompting.templates import apply_chat_template
from fc_class_kd.training.formal_sft import sha256_file
from fc_class_kd.training.model_utils import load_student_model, load_tokenizer
from fc_class_kd.training.sft import generate_function_call

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


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
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def signature(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="公共 S0 在 Train/Diagnostic 上自由生成")
    parser.add_argument("--config", default="rtx3090")
    parser.add_argument("--s0-dir", default="outputs/s0_qwen3_1.7b_seed42")
    parser.add_argument("--output-dir", default="data/candidates")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["diagnostic", "train"],
        help="data/splits 下的划分名；V7 推荐改用 --split-paths",
    )
    parser.add_argument(
        "--split-paths",
        nargs="+",
        default=None,
        metavar="NAME=JSONL",
        help="显式命名输入，例如 internal=data/processed/v7/internal.jsonl；设置后替代 --splits",
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-samples", type=int, default=None, help="仅供冒烟测试")
    parser.add_argument("--resume", choices=["auto", "none"], default="auto")
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def resolve_named_splits(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.split_paths is None:
        pairs = [(name, f"data/splits/{name}.jsonl") for name in args.splits]
    else:
        pairs = []
        for item in args.split_paths:
            if "=" not in item:
                raise ValueError(f"split path must use NAME=JSONL: {item}")
            name, raw_path = item.split("=", 1)
            if not name or not name.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"invalid split name: {name!r}")
            pairs.append((name, raw_path))
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError("split names must be unique")
    resolved = [(name, resolve(raw_path)) for name, raw_path in pairs]
    missing = [str(path) for _, path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"split JSONL missing: {missing}")
    return resolved


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S0 inference requires CUDA")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")

    cfg = load_config(args.config)
    model_id = str(cfg.models.get("student_local", cfg.models.student))
    if not Path(model_id).is_dir():
        raise FileNotFoundError("formal S0 inference requires the pinned local base model")
    s0_dir = resolve(args.s0_dir)
    selected_path = s0_dir / "selected_s0.json"
    pointer_path = s0_dir / "best_checkpoint.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    checkpoint_name = str(selected["selected_checkpoint"])
    if pointer.get("checkpoint") != checkpoint_name:
        raise ValueError("selected_s0.json and best_checkpoint.json disagree")
    checkpoint = s0_dir / checkpoint_name
    adapter = checkpoint / "adapter" / "adapter_model.safetensors"
    if not adapter.exists():
        raise FileNotFoundError(f"selected adapter missing: {adapter}")

    tokenizer = load_tokenizer(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work: list[tuple[str, list[Any], Path, Path, dict[str, Any]]] = []
    for split, split_path in resolve_named_splits(args):
        records = read_split(split_path)
        if args.max_samples is not None:
            records = records[: args.max_samples]
        prompt_lengths = [
            len(
                tokenizer(
                    apply_chat_template(tokenizer, record, add_generation_prompt=True),
                    add_special_tokens=False,
                )["input_ids"]
            )
            for record in records
        ]
        if max(prompt_lengths, default=0) + args.max_new_tokens > int(cfg.training.max_length):
            raise ValueError(f"{split}: prompt plus generation budget exceeds max_length")
        prediction_path = output_dir / f"s0_{split}_predictions.jsonl"
        manifest_path = output_dir / f"s0_{split}_predictions.manifest.json"
        definition = {
            "kind": "selected_public_s0_free_generation",
            "split": split,
            "formal_run": args.max_samples is None,
            "record_count": len(records),
            "split_path": (
                str(split_path.relative_to(PROJECT_ROOT))
                if split_path.is_relative_to(PROJECT_ROOT)
                else str(split_path)
            ),
            "split_sha256": sha256_file(split_path),
            "model_path": model_id,
            "model_revision": cfg.models.get("student_revision"),
            "s0_checkpoint": checkpoint_name,
            "selected_s0_sha256": sha256_file(selected_path),
            "adapter_sha256": sha256_file(adapter),
            "max_new_tokens": args.max_new_tokens,
            "decoding": "deterministic_greedy",
            "max_prompt_tokens": max(prompt_lengths, default=0),
        }
        run_signature = signature(definition)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("run_signature") != run_signature:
                raise ValueError(f"{split}: output belongs to a different definition")
        else:
            if prediction_path.exists():
                raise FileExistsError(f"{split}: predictions exist without a manifest")
            manifest = {
                "manifest_version": 1,
                "created_at_utc": utc_now(),
                "run_signature": run_signature,
                "definition": definition,
                "status": "in_progress",
                "completed_count": 0,
            }
            atomic_write_json(manifest_path, manifest)
        rows = read_jsonl(prediction_path)
        if args.resume == "none" and rows:
            raise FileExistsError(f"{split}: --resume none refuses existing predictions")
        validate_prediction_rows(records, rows, split, require_complete=False)
        manifest["completed_count"] = len(rows)
        atomic_write_json(manifest_path, manifest)
        if len(rows) == len(records):
            if manifest.get("status") != "complete":
                summary = summarize_prediction_rows(records, rows)
                manifest.update(
                    status="complete",
                    completed_count=len(rows),
                    predictions_sha256=sha256_file(prediction_path),
                    summary=summary,
                    completed_at_utc=utc_now(),
                )
                atomic_write_json(manifest_path, manifest)
            print(f"{split}: 已完整生成 {len(rows)} 条，跳过", flush=True)
            continue
        work.append((split, records, prediction_path, manifest_path, manifest))

    if not work:
        print("S0_PREDICTIONS_OK=true")
        return 0

    base_model, model_tokenizer = load_student_model(
        model_id,
        load_in_4bit=True,
        fp16=bool(cfg.hardware.get("fp16", False)),
        bf16=bool(cfg.hardware.get("bf16", True)),
    )
    model = PeftModel.from_pretrained(base_model, checkpoint / "adapter", is_trainable=False)
    model.config.use_cache = True
    model.eval()
    for split, records, prediction_path, manifest_path, manifest in work:
        rows = read_jsonl(prediction_path)
        for index in range(len(rows), len(records)):
            record = records[index]
            prompt = apply_chat_template(model_tokenizer, record, add_generation_prompt=True)
            prompt_tokens = len(model_tokenizer(prompt, add_special_tokens=False)["input_ids"])
            prediction = generate_function_call(
                model,
                model_tokenizer,
                record,
                max_new_tokens=args.max_new_tokens,
            )
            generated_tokens = len(
                model_tokenizer(prediction, add_special_tokens=False)["input_ids"]
            )
            row = make_prediction_row(
                record,
                split,
                prediction,
                prompt_tokens,
                generated_tokens,
            )
            append_jsonl(prediction_path, row)
            rows.append(row)
            if len(rows) % 10 == 0 or len(rows) == len(records):
                manifest["completed_count"] = len(rows)
                manifest["updated_at_utc"] = utc_now()
                atomic_write_json(manifest_path, manifest)
                print(f"{split}: {len(rows)}/{len(records)}", flush=True)
        validate_prediction_rows(records, rows, split, require_complete=True)
        manifest.update(
            status="complete",
            completed_count=len(rows),
            predictions_sha256=sha256_file(prediction_path),
            summary=summarize_prediction_rows(records, rows),
            completed_at_utc=utc_now(),
        )
        atomic_write_json(manifest_path, manifest)
        print(json.dumps({split: manifest["summary"]}, ensure_ascii=False, indent=2), flush=True)

    print("S0_PREDICTIONS_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
