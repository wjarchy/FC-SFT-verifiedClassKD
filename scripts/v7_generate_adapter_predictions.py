#!/usr/bin/env python3
"""Batched, resumable greedy evaluation for a frozen V7 adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

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
from fc_class_kd.training.model_utils import load_student_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl_batch(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def signature(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def trim_generated_ids(token_ids: Sequence[int], eos_token_id: int | None) -> list[int]:
    """Remove batch padding after the first EOS while retaining that EOS."""
    values = [int(value) for value in token_ids]
    if eos_token_id is None:
        return values
    try:
        end = values.index(int(eos_token_id))
    except ValueError:
        return values
    return values[: end + 1]


@torch.inference_mode()
def generate_batch(
    model: Any,
    tokenizer: Any,
    records: Sequence[Any],
    max_new_tokens: int,
) -> tuple[list[str], list[int], list[int]]:
    if not records:
        return [], [], []
    prompts = [
        apply_chat_template(tokenizer, record, add_generation_prompt=True)
        for record in records
    ]
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_padding_side
    prompt_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    padded_prompt_length = int(encoded["input_ids"].shape[1])
    generation_config = deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generated = model.generate(
        **encoded,
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    predictions: list[str] = []
    generated_lengths: list[int] = []
    for tensor in generated[:, padded_prompt_length:]:
        token_ids = trim_generated_ids(tensor.tolist(), tokenizer.eos_token_id)
        predictions.append(tokenizer.decode(token_ids, skip_special_tokens=False))
        generated_lengths.append(len(token_ids))
    return predictions, prompt_lengths, generated_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量评测 V7 adapter")
    parser.add_argument("--config", default="v7_rtx3090")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split-name", default="internal")
    parser.add_argument("--split-path", default="data/processed/v7/internal.jsonl")
    parser.add_argument("--output-dir", default="outputs/v7_predictions")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--resume", choices=["auto", "none"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise ValueError("run-name contains unsafe characters")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.split_name):
        raise ValueError("split-name contains unsafe characters")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch-size and max-new-tokens must be positive")
    adapter = resolve(args.adapter)
    adapter_weights = adapter / "adapter_model.safetensors"
    split_path = resolve(args.split_path)
    if not adapter_weights.is_file():
        raise FileNotFoundError(adapter_weights)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    cfg = load_config(args.config)
    model_path = Path(str(cfg.models.student_local))
    if not model_path.is_dir():
        raise FileNotFoundError("evaluation requires the pinned local Student model")
    records = read_split(split_path)
    output_dir = resolve(args.output_dir)
    output_path = output_dir / f"{args.run_name}_{args.split_name}_predictions.jsonl"
    manifest_path = output_dir / f"{args.run_name}_{args.split_name}_predictions.manifest.json"
    definition = {
        "kind": "v7_adapter_free_generation",
        "run_name": args.run_name,
        "split_name": args.split_name,
        "split_path": relative(split_path),
        "split_sha256": sha256_file(split_path),
        "record_count": len(records),
        "base_model": str(model_path),
        "base_revision": str(cfg.models.student_revision),
        "adapter": relative(adapter),
        "adapter_sha256": sha256_file(adapter_weights),
        "decoding": "deterministic_greedy_left_padded_batch",
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    run_signature = signature(definition)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_signature") != run_signature:
            raise ValueError("output manifest belongs to another evaluation definition")
    else:
        if output_path.exists():
            raise FileExistsError("predictions exist without a manifest")
        manifest = {
            "manifest_version": 1,
            "created_at_utc": utc_now(),
            "run_signature": run_signature,
            "definition": definition,
            "status": "in_progress",
            "completed_count": 0,
        }
        atomic_write_json(manifest_path, manifest)
    rows = read_jsonl(output_path)
    if args.resume == "none" and rows:
        raise FileExistsError("--resume none refuses existing predictions")
    validate_prediction_rows(records, rows, args.split_name, require_complete=False)
    if len(rows) == len(records):
        manifest.update(
            status="complete",
            completed_count=len(rows),
            predictions_sha256=sha256_file(output_path),
            summary=summarize_prediction_rows(records, rows),
            completed_at_utc=manifest.get("completed_at_utc", utc_now()),
        )
        atomic_write_json(manifest_path, manifest)
        print("V7_ADAPTER_PREDICTIONS_OK=true")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("adapter evaluation requires CUDA")

    torch.cuda.reset_peak_memory_stats()
    base_model, tokenizer = load_student_model(
        str(model_path),
        load_in_4bit=True,
        bf16=bool(cfg.hardware.bf16),
        fp16=bool(cfg.hardware.fp16),
    )
    model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    for start in range(len(rows), len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        predictions, prompt_lengths, generated_lengths = generate_batch(
            model, tokenizer, batch_records, args.max_new_tokens
        )
        batch_rows = [
            make_prediction_row(record, args.split_name, prediction, prompt_tokens, generated_tokens)
            for record, prediction, prompt_tokens, generated_tokens in zip(
                batch_records,
                predictions,
                prompt_lengths,
                generated_lengths,
                strict=True,
            )
        ]
        append_jsonl_batch(output_path, batch_rows)
        rows.extend(batch_rows)
        manifest["completed_count"] = len(rows)
        manifest["updated_at_utc"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        if len(rows) % 20 == 0 or len(rows) == len(records):
            print(f"{args.run_name}/{args.split_name}: {len(rows)}/{len(records)}", flush=True)
    validate_prediction_rows(records, rows, args.split_name, require_complete=True)
    manifest.update(
        status="complete",
        completed_count=len(rows),
        predictions_sha256=sha256_file(output_path),
        summary=summarize_prediction_rows(records, rows),
        completed_at_utc=utc_now(),
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / (1024**2),
    )
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print("V7_ADAPTER_PREDICTIONS_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
