#!/usr/bin/env python3
"""Train the common Qwen3-1.7B public S0 on Train Gold only.

The runner is deliberately single-GPU and explicit.  It fails on truncation, saves
optimizer/scheduler/RNG state at optimizer-step boundaries, selects S0 by Validation
AST accuracy, and can resume Validation free generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import get_cosine_schedule_with_warmup

from fc_class_kd.config import load_config
from fc_class_kd.data.splits import read_split
from fc_class_kd.evaluation.metrics import evaluate_predictions
from fc_class_kd.training.formal_sft import (
    build_epoch_groups,
    checkpoint_name,
    compute_sft_budget,
    select_best_checkpoint,
    select_s0_by_validation_ast,
    sha256_file,
    supervised_token_count,
)
from fc_class_kd.training.model_utils import (
    attach_lora,
    load_student_model,
    load_tokenizer,
    max_memory_mb,
)
from fc_class_kd.training.sft import (
    EncodedSFTExample,
    collate_sft_examples,
    encode_sft_records,
    generate_function_call,
)
from fc_class_kd.types import FCRecord
from fc_class_kd.verifier.classifier import classify_call

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def write_canonical_training_curve(log_path: Path, output_path: Path) -> str:
    """Deduplicate replayed optimizer steps after checkpoint recovery."""
    rows_by_step: dict[int, dict[str, Any]] = {}
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_step[int(row["global_optimizer_step"])] = row
    rows = [rows_by_step[step] for step in sorted(rows_by_step)]
    atomic_write_json(output_path, rows)
    return sha256_file(output_path)


def git_state() -> dict[str, Any]:
    try:
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
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def length_summary(examples: Sequence[EncodedSFTExample]) -> dict[str, int]:
    lengths = sorted(len(example.input_ids) for example in examples)
    if not lengths:
        raise ValueError("cannot summarize an empty encoded split")

    def percentile(fraction: float) -> int:
        index = round(fraction * (len(lengths) - 1))
        return lengths[index]

    return {
        "count": len(lengths),
        "min": lengths[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": lengths[-1],
        "supervised_tokens": supervised_token_count(examples),
    }


def batch_examples(
    examples: Sequence[EncodedSFTExample],
    indices: Sequence[int],
) -> list[EncodedSFTExample]:
    return [examples[index] for index in indices]


@torch.inference_mode()
def validation_loss(
    model: Any,
    examples: Sequence[EncodedSFTExample],
    pad_token_id: int,
    micro_batch_size: int,
) -> tuple[float, int]:
    """Return token-weighted assistant-only cross-entropy over the full split."""
    model.eval()
    numerator = 0.0
    token_count = 0
    for offset in range(0, len(examples), micro_batch_size):
        current = list(examples[offset : offset + micro_batch_size])
        current_tokens = supervised_token_count(current)
        batch = collate_sft_examples(current, pad_token_id, device=model.device)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Validation loss became NaN or Inf")
        numerator += float(loss.detach().cpu()) * current_tokens
        token_count += current_tokens
    model.train()
    return numerator / token_count, token_count


def save_rng_state(path: Path) -> None:
    payload: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def restore_rng_state(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_rng_state(payload["cpu"])
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state_all(payload["cuda"])


def save_checkpoint(
    output_dir: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: dict[str, Any],
) -> Path:
    name = checkpoint_name(int(state["global_optimizer_step"]))
    destination = output_dir / name
    temporary = output_dir / f".{name}.tmp-{os.getpid()}"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "adapter")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    save_rng_state(temporary / "rng_state.pt")
    atomic_write_json(temporary / "trainer_state.json", state)
    temporary.rename(destination)
    atomic_write_json(
        output_dir / "latest_checkpoint.json",
        {
            "checkpoint": destination.name,
            "global_optimizer_step": state["global_optimizer_step"],
            "saved_at_utc": utc_now(),
        },
    )
    return destination


def resolve_resume_checkpoint(output_dir: Path, resume: str) -> Path | None:
    if resume == "none":
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"--resume none refuses non-empty output directory: {output_dir}"
            )
        return None
    if resume == "auto":
        pointer = output_dir / "latest_checkpoint.json"
        if not pointer.exists():
            return None
        value = json.loads(pointer.read_text(encoding="utf-8"))
        checkpoint = output_dir / str(value["checkpoint"])
    else:
        checkpoint = Path(resume)
    required = [
        checkpoint / "adapter" / "adapter_config.json",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        checkpoint / "rng_state.pt",
        checkpoint / "trainer_state.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"resume checkpoint is incomplete: {missing}")
    return checkpoint


def model_file_inventory(model_path: str) -> list[dict[str, Any]]:
    root = Path(model_path)
    if not root.is_dir():
        return []
    wanted = [
        *sorted(root.glob("*.safetensors")),
        root / "model.safetensors.index.json",
        root / "config.json",
        root / "tokenizer.json",
    ]
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in wanted
        if path.exists()
    ]


def run_signature(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    import hashlib

    return hashlib.sha256(canonical).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练公共 Qwen3-1.7B S0")
    parser.add_argument("--config", default="rtx3090")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="允许 Hugging Face 仓库名；默认只允许本地模型，避免脚本意外联网",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output", default="outputs/s0_qwen3_1.7b_seed42")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto、none 或一个完整检查点目录",
    )
    parser.add_argument(
        "--skip-validation-generation",
        action="store_true",
        help="只按 Validation loss 选检查点，不做最终自由生成",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只检查数据编码、长度和训练预算，不加载模型或写训练输出",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="仅供工程调试；使用后不能作为正式 S0",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="仅供工程调试；使用后不能作为正式 S0",
    )
    return parser.parse_args()


def limit_records(records: list[FCRecord], limit: int | None, label: str) -> list[FCRecord]:
    if limit is None:
        return records
    if limit <= 0:
        raise ValueError(f"{label} limit must be positive")
    return records[:limit]


def resolved_hyperparameters(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    training = cfg.training
    return {
        "epochs": args.epochs if args.epochs is not None else int(training.epochs),
        "micro_batch_size": (
            args.micro_batch_size
            if args.micro_batch_size is not None
            else int(training.micro_batch_size)
        ),
        "gradient_accumulation_steps": (
            args.gradient_accumulation
            if args.gradient_accumulation is not None
            else int(training.gradient_accumulation_steps)
        ),
        "learning_rate": (
            args.learning_rate
            if args.learning_rate is not None
            else float(training.learning_rate)
        ),
        "weight_decay": (
            args.weight_decay
            if args.weight_decay is not None
            else float(training.weight_decay)
        ),
        "warmup_ratio": (
            args.warmup_ratio
            if args.warmup_ratio is not None
            else float(training.warmup_ratio)
        ),
        "max_grad_norm": (
            args.max_grad_norm
            if args.max_grad_norm is not None
            else float(training.max_grad_norm)
        ),
        "max_length": (
            args.max_length if args.max_length is not None else int(training.max_length)
        ),
        "save_every_optimizer_steps": (
            args.save_every_steps
            if args.save_every_steps is not None
            else int(training.checkpoint_every_optimizer_steps)
        ),
    }


def validate_hyperparameters(hparams: dict[str, Any]) -> None:
    for name in (
        "epochs",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "save_every_optimizer_steps",
    ):
        if int(hparams[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(float(hparams[name])) or float(hparams[name]) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 <= float(hparams["warmup_ratio"]) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if float(hparams["weight_decay"]) < 0:
        raise ValueError("weight_decay must be non-negative")


def prepare_inputs(
    args: argparse.Namespace,
    cfg: Any,
    model_id: str,
    hparams: dict[str, Any],
) -> tuple[
    list[FCRecord],
    list[FCRecord],
    list[EncodedSFTExample],
    list[EncodedSFTExample],
    Any,
    Path,
    Path,
]:
    splits_dir = Path(cfg.paths.splits)
    if not splits_dir.is_absolute():
        splits_dir = PROJECT_ROOT / splits_dir
    train_path = splits_dir / f"{args.train_split}.jsonl"
    val_path = splits_dir / f"{args.val_split}.jsonl"
    train_records = limit_records(
        read_split(train_path), args.max_train_samples, "train"
    )
    val_records = limit_records(read_split(val_path), args.max_val_samples, "validation")
    train_ids = {record.id for record in train_records}
    val_ids = {record.id for record in val_records}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Train and Validation overlap: {sorted(overlap)[:5]}")
    tokenizer = load_tokenizer(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must provide a pad_token_id")
    train_examples = encode_sft_records(
        tokenizer,
        train_records,
        max_length=int(hparams["max_length"]),
    )
    val_examples = encode_sft_records(
        tokenizer,
        val_records,
        max_length=int(hparams["max_length"]),
    )
    return (
        train_records,
        val_records,
        train_examples,
        val_examples,
        tokenizer,
        train_path,
        val_path,
    )


def load_or_create_model(
    model_id: str,
    cfg: Any,
    checkpoint: Path | None,
) -> tuple[Any, Any]:
    base_model, tokenizer = load_student_model(
        model_id,
        load_in_4bit=True,
        fp16=bool(cfg.hardware.get("fp16", False)),
        bf16=bool(cfg.hardware.get("bf16", True)),
    )
    if checkpoint is None:
        model = attach_lora(
            base_model,
            r=int(cfg.models.qlora.r),
            lora_alpha=int(cfg.models.qlora.lora_alpha),
            lora_dropout=float(cfg.models.qlora.lora_dropout),
            target_modules=list(cfg.models.qlora.target_modules),
        )
    else:
        if getattr(base_model, "is_loaded_in_4bit", False):
            base_model = prepare_model_for_kbit_training(
                base_model,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint / "adapter",
            is_trainable=True,
        )
        model.config.use_cache = False
    return model, tokenizer


def generate_validation_predictions(
    result_dir: Path,
    model_id: str,
    cfg: Any,
    best_checkpoint: Path,
    records: Sequence[FCRecord],
    max_new_tokens: int,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = result_dir / "validation_generation_state.json"
    predictions_path = result_dir / "validation_predictions.jsonl"
    expected_checkpoint = best_checkpoint.name
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("best_checkpoint") != expected_checkpoint:
            raise ValueError(
                "existing Validation predictions belong to a different best checkpoint"
            )
    else:
        metadata = {
            "best_checkpoint": expected_checkpoint,
            "started_at_utc": utc_now(),
            "completed_ids": 0,
        }
        atomic_write_json(metadata_path, metadata)

    existing: dict[str, str] = {}
    if predictions_path.exists():
        with predictions_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                if sample_id in existing:
                    raise ValueError(f"duplicate Validation prediction for {sample_id}")
                existing[sample_id] = str(row["prediction"])

    base_model, tokenizer = load_student_model(
        model_id,
        load_in_4bit=True,
        fp16=bool(cfg.hardware.get("fp16", False)),
        bf16=bool(cfg.hardware.get("bf16", True)),
    )
    model = PeftModel.from_pretrained(
        base_model,
        best_checkpoint / "adapter",
        is_trainable=False,
    )
    model.eval()
    for index, record in enumerate(records, start=1):
        if record.id in existing:
            continue
        prediction = generate_function_call(
            model,
            tokenizer,
            record,
            max_new_tokens=max_new_tokens,
        )
        result = classify_call(prediction, record.tools, record.gold_call)
        append_jsonl(
            predictions_path,
            {
                "sample_id": record.id,
                "prediction": prediction,
                "verifier_class": result.class_label.value,
                "verifier_sub_label": result.sub_label,
                "parse_ok": result.parse_ok,
            },
        )
        existing[record.id] = prediction
        if index % 10 == 0 or index == len(records):
            metadata["completed_ids"] = len(existing)
            metadata["updated_at_utc"] = utc_now()
            atomic_write_json(metadata_path, metadata)

    missing = [record.id for record in records if record.id not in existing]
    if missing:
        raise RuntimeError(f"Validation generation is incomplete: {missing[:5]}")
    ordered_predictions = [existing[record.id] for record in records]
    metrics = evaluate_predictions(list(records), ordered_predictions).rates()
    result = {
        "best_checkpoint": expected_checkpoint,
        "count": len(records),
        "max_new_tokens": max_new_tokens,
        "metrics": metrics,
        "predictions_sha256": sha256_file(predictions_path),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(result_dir / "validation_generation_metrics.json", result)
    metadata["completed_ids"] = len(records)
    metadata["completed_at_utc"] = result["completed_at_utc"]
    atomic_write_json(metadata_path, metadata)
    del model, base_model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() and not args.prepare_only:
        print("SFT_FAIL: CUDA not available")
        return 1
    cfg = load_config(args.config)
    model_id = args.model or str(cfg.models.get("student_local", cfg.models.student))
    configured_model = str(cfg.models.get("student_local", cfg.models.student))
    model_revision = args.model_revision
    if model_revision is None and model_id == configured_model:
        model_revision = cfg.models.get("student_revision")
    if not args.allow_remote_model and not Path(model_id).is_dir():
        raise ValueError(
            "Formal training requires a local model directory. Download it with the proxy-safe "
            "command first, or explicitly pass --allow-remote-model."
        )
    hparams = resolved_hyperparameters(args, cfg)
    validate_hyperparameters(hparams)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    (
        _train_records,
        val_records,
        train_examples,
        val_examples,
        tokenizer,
        train_path,
        val_path,
    ) = prepare_inputs(args, cfg, model_id, hparams)
    budget = compute_sft_budget(
        len(train_examples),
        int(hparams["micro_batch_size"]),
        int(hparams["gradient_accumulation_steps"]),
        int(hparams["epochs"]),
    )
    preparation = {
        "model": model_id,
        "model_revision": model_revision,
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(val_path),
        "train_lengths": length_summary(train_examples),
        "validation_lengths": length_summary(val_examples),
        "hyperparameters": hparams,
        "budget": budget.__dict__,
        "formal_run": args.max_train_samples is None and args.max_val_samples is None,
    }
    print(json.dumps(preparation, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        print("SFT_PREPARE_OK=true")
        return 0

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    checkpoint = resolve_resume_checkpoint(output_dir, args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature_payload = {
        "model": model_id,
        "model_revision": model_revision,
        "train_sha256": preparation["train_sha256"],
        "validation_sha256": preparation["validation_sha256"],
        "train_count": len(train_examples),
        "validation_count": len(val_examples),
        "hyperparameters": hparams,
        "seed": args.seed,
        "qlora": {
            "r": int(cfg.models.qlora.r),
            "lora_alpha": int(cfg.models.qlora.lora_alpha),
            "lora_dropout": float(cfg.models.qlora.lora_dropout),
            "target_modules": list(cfg.models.qlora.target_modules),
            "bits": int(cfg.models.qlora.bits),
        },
    }
    signature = run_signature(signature_payload)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("run_signature") != signature:
            raise ValueError("output directory belongs to a different training definition")
    else:
        atomic_write_json(
            manifest_path,
            {
                "manifest_version": 1,
                "kind": "public_s0_gold_only_sft",
                "created_at_utc": utc_now(),
                "run_signature": signature,
                "definition": signature_payload,
                "preparation": preparation,
                "data_manifest": {
                    "path": "data/processed/manifest.json",
                    "sha256": sha256_file(
                        PROJECT_ROOT / "data/processed/manifest.json"
                    ),
                },
                "model_files": model_file_inventory(model_id),
                "command": ["scripts/train_sft.py", *sys.argv[1:]],
                "git": git_state(),
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                },
            },
        )

    torch.cuda.reset_peak_memory_stats()
    model, model_tokenizer = load_or_create_model(model_id, cfg, checkpoint)
    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token
    if model_tokenizer.pad_token_id != tokenizer.pad_token_id:
        raise ValueError("tokenizer changed between preparation and model loading")
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(hparams["learning_rate"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    warmup_steps = round(budget.total_optimizer_steps * float(hparams["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=budget.total_optimizer_steps,
    )

    if checkpoint is None:
        state: dict[str, Any] = {
            "run_signature": signature,
            "global_optimizer_step": 0,
            "next_epoch_index": 0,
            "next_group_index": 0,
            "current_epoch_loss_numerator": 0.0,
            "current_epoch_token_count": 0,
            "best_validation_loss": None,
            "best_loss_checkpoint": None,
            "history": [],
        }
        checkpoint = save_checkpoint(output_dir, model, optimizer, scheduler, state)
        print(f"初始检查点：{checkpoint}", flush=True)
    else:
        state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        if state.get("run_signature") != signature:
            raise ValueError("checkpoint belongs to a different training definition")
        optimizer.load_state_dict(
            torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
        )
        scheduler.load_state_dict(
            torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
        )
        restore_rng_state(checkpoint / "rng_state.pt")
        print(f"恢复检查点：{checkpoint}", flush=True)

    log_path = output_dir / "training_log.jsonl"
    start_time = time.time()
    invocation_id = utc_now()
    for epoch_index in range(
        int(state["next_epoch_index"]), int(hparams["epochs"])
    ):
        groups = build_epoch_groups(
            len(train_examples),
            int(hparams["micro_batch_size"]),
            int(hparams["gradient_accumulation_steps"]),
            args.seed,
            epoch_index,
        )
        first_group = (
            int(state["next_group_index"])
            if epoch_index == int(state["next_epoch_index"])
            else 0
        )
        if first_group > len(groups):
            raise ValueError("checkpoint next_group_index exceeds the epoch plan")
        for group_index in range(first_group, len(groups)):
            group = groups[group_index]
            group_token_count = sum(
                supervised_token_count(batch_examples(train_examples, indices))
                for indices in group
            )
            group_loss_numerator = 0.0
            optimizer.zero_grad(set_to_none=True)
            for indices in group:
                current = batch_examples(train_examples, indices)
                current_tokens = supervised_token_count(current)
                batch = collate_sft_examples(
                    current,
                    model_tokenizer.pad_token_id,
                    device=model.device,
                )
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"training loss became NaN or Inf at epoch {epoch_index + 1}, "
                        f"group {group_index + 1}"
                    )
                weighted_loss = loss * (current_tokens / group_token_count)
                weighted_loss.backward()
                value = float(loss.detach().cpu())
                group_loss_numerator += value * current_tokens
                state["current_epoch_loss_numerator"] += value * current_tokens
                state["current_epoch_token_count"] += current_tokens
            trainable = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                float(hparams["max_grad_norm"]),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("gradient norm became NaN or Inf")
            optimizer.step()
            scheduler.step()
            state["global_optimizer_step"] += 1
            state["next_epoch_index"] = epoch_index
            state["next_group_index"] = group_index + 1
            log_row = {
                "time_utc": utc_now(),
                "invocation_id": invocation_id,
                "epoch": epoch_index + 1,
                "group": group_index + 1,
                "groups_in_epoch": len(groups),
                "global_optimizer_step": state["global_optimizer_step"],
                "train_loss": group_loss_numerator / group_token_count,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "supervised_tokens": group_token_count,
            }
            append_jsonl(log_path, log_row)
            if (
                state["global_optimizer_step"]
                % int(hparams["save_every_optimizer_steps"])
                == 0
                and group_index + 1 < len(groups)
            ):
                saved = save_checkpoint(output_dir, model, optimizer, scheduler, state)
                print(
                    f"step={state['global_optimizer_step']}/{budget.total_optimizer_steps} "
                    f"loss={log_row['train_loss']:.6f} checkpoint={saved.name}",
                    flush=True,
                )

        epoch_train_loss = (
            state["current_epoch_loss_numerator"]
            / state["current_epoch_token_count"]
        )
        val_loss, val_tokens = validation_loss(
            model,
            val_examples,
            model_tokenizer.pad_token_id,
            int(hparams["micro_batch_size"]),
        )
        candidate_name = checkpoint_name(int(state["global_optimizer_step"]))
        best_loss, best_path, improved = select_best_checkpoint(
            state["best_validation_loss"],
            state["best_loss_checkpoint"],
            val_loss,
            candidate_name,
        )
        epoch_result = {
            "epoch": epoch_index + 1,
            "global_optimizer_step": state["global_optimizer_step"],
            "train_loss": epoch_train_loss,
            "validation_loss": val_loss,
            "validation_perplexity": math.exp(min(val_loss, 20.0)),
            "validation_supervised_tokens": val_tokens,
            "best_so_far": improved,
            "peak_gpu_memory_mb": max_memory_mb(),
            "checkpoint": candidate_name,
        }
        state["history"].append(epoch_result)
        state["best_validation_loss"] = best_loss
        state["best_loss_checkpoint"] = best_path
        state["next_epoch_index"] = epoch_index + 1
        state["next_group_index"] = 0
        state["current_epoch_loss_numerator"] = 0.0
        state["current_epoch_token_count"] = 0
        saved = save_checkpoint(output_dir, model, optimizer, scheduler, state)
        if improved:
            atomic_write_json(
                output_dir / "best_loss_checkpoint.json",
                {
                    "checkpoint": best_path,
                    "validation_loss": best_loss,
                    "selected_at_utc": utc_now(),
                },
            )
        print(
            f"epoch={epoch_index + 1}/{hparams['epochs']} "
            f"train_loss={epoch_train_loss:.6f} val_loss={val_loss:.6f} "
            f"checkpoint={saved.name} best={best_path}",
            flush=True,
        )

    if not state.get("best_loss_checkpoint"):
        raise RuntimeError("training finished without a selected Validation checkpoint")
    best_loss_checkpoint = output_dir / str(state["best_loss_checkpoint"])
    summary = {
        "kind": "public_s0_gold_only_sft",
        "completed_at_utc": utc_now(),
        "elapsed_seconds_this_invocation": time.time() - start_time,
        "best_loss_checkpoint": best_loss_checkpoint.name,
        "best_validation_loss": state["best_validation_loss"],
        "global_optimizer_steps": state["global_optimizer_step"],
        "history": state["history"],
        "peak_gpu_memory_mb": max_memory_mb(),
        "formal_run": preparation["formal_run"],
    }
    summary["training_curve_sha256"] = write_canonical_training_curve(
        log_path,
        output_dir / "training_curve.json",
    )
    atomic_write_json(output_dir / "training_summary.json", summary)
    model_tokenizer.save_pretrained(output_dir / "tokenizer")

    if not args.skip_validation_generation:
        del optimizer, scheduler, model
        torch.cuda.empty_cache()
        evaluations: list[dict[str, Any]] = []
        for epoch_result in state["history"]:
            checkpoint_path = output_dir / str(epoch_result["checkpoint"])
            generation = generate_validation_predictions(
                output_dir / "validation_by_checkpoint" / checkpoint_path.name,
                model_id,
                cfg,
                checkpoint_path,
                val_records,
                args.max_new_tokens,
            )
            evaluations.append(
                {
                    "checkpoint": checkpoint_path.name,
                    "epoch": epoch_result["epoch"],
                    "ast_accuracy": generation["metrics"].get("ast_accuracy", 0.0),
                    "validation_loss": epoch_result["validation_loss"],
                    "generation_metrics": generation["metrics"],
                    "predictions_sha256": generation["predictions_sha256"],
                }
            )
        selected = select_s0_by_validation_ast(evaluations)
        selection = {
            "selection_rule": (
                "highest Validation AST; ties use lower Validation loss; "
                "remaining ties use earlier epoch"
            ),
            "selected_checkpoint": selected["checkpoint"],
            "selected_epoch": selected["epoch"],
            "selected_ast_accuracy": selected["ast_accuracy"],
            "selected_validation_loss": selected["validation_loss"],
            "evaluations": evaluations,
            "selected_at_utc": utc_now(),
        }
        atomic_write_json(output_dir / "selected_s0.json", selection)
        atomic_write_json(
            output_dir / "best_checkpoint.json",
            {
                "checkpoint": selected["checkpoint"],
                "selection_file": "selected_s0.json",
                "selected_at_utc": selection["selected_at_utc"],
            },
        )
        summary["s0_selection"] = selection
        atomic_write_json(output_dir / "training_summary.json", summary)
    else:
        summary["s0_selection"] = {
            "status": "not_finalized",
            "reason": "Validation free generation was explicitly skipped",
            "provisional_loss_checkpoint": best_loss_checkpoint.name,
        }
        atomic_write_json(output_dir / "training_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("SFT_TRAINING_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
