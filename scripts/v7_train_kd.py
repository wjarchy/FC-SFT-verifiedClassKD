#!/usr/bin/env python3
"""Train a V7 Candidate or Class KD branch from the exact same Formal S0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel, prepare_model_for_kbit_training

from fc_class_kd.config import load_config
from fc_class_kd.data.splits import read_split
from fc_class_kd.training.candidate_scoring import score_candidate_sequences
from fc_class_kd.training.formal_sft import build_epoch_groups, sha256_file
from fc_class_kd.training.model_utils import load_student_model, max_memory_mb
from fc_class_kd.types import CandidateRecord, FCRecord
from fc_class_kd.v7.kd import fixed_support_kd_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def signature(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def save_rng_state(path: Path) -> None:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def restore_rng_state(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["cpu"])
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state_all(payload["cuda"])


def checkpoint_name(step: int) -> str:
    return f"checkpoint-step-{step:06d}"


def save_checkpoint(
    output_dir: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> Path:
    destination = output_dir / checkpoint_name(int(state["global_optimizer_step"]))
    temporary = output_dir / f".{destination.name}.tmp-{os.getpid()}"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "adapter")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
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


def resolve_resume(output_dir: Path, resume: str) -> Path | None:
    if resume == "none":
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"--resume none refuses non-empty output: {output_dir}")
        return None
    if resume == "auto":
        pointer = output_dir / "latest_checkpoint.json"
        if not pointer.exists():
            return None
        checkpoint = output_dir / json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
    else:
        checkpoint = resolve(resume)
    required = (
        checkpoint / "adapter" / "adapter_config.json",
        checkpoint / "optimizer.pt",
        checkpoint / "rng_state.pt",
        checkpoint / "trainer_state.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete KD checkpoint: {missing}")
    return checkpoint


def selected_s0(s0_dir: Path) -> tuple[Path, dict[str, Any]]:
    selection_path = s0_dir / "selected_s0.json"
    pointer_path = s0_dir / "best_checkpoint.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    name = str(selection["selected_checkpoint"])
    if pointer.get("checkpoint") != name:
        raise ValueError("Formal S0 selection files disagree")
    adapter = s0_dir / name / "adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"selected S0 adapter missing: {adapter}")
    return adapter, selection


def validate_inputs(
    records: list[FCRecord],
    pool_rows: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    ids = [record.id for record in records]
    if [str(row.get("sample_id")) for row in pool_rows] != ids:
        raise ValueError("training pool is not aligned exactly with the data rows")
    for row in pool_rows:
        candidates = [CandidateRecord.model_validate(item) for item in row["candidates"]]
        if len(candidates) != 4 or row.get("class_indices") != [0, 0, 1, 2]:
            raise ValueError(f"{row.get('sample_id')}: invalid K=4 training pool")
    cache_by_id: dict[str, dict[str, Any]] = {}
    for row in cache_rows:
        sample_id = str(row.get("sample_id"))
        if sample_id in cache_by_id:
            raise ValueError(f"duplicate Teacher cache ID: {sample_id}")
        probabilities = row.get("candidate_probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != 4:
            raise ValueError(f"{sample_id}: invalid Teacher probabilities")
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in probabilities):
            raise ValueError(f"{sample_id}: non-finite Teacher probabilities")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise ValueError(f"{sample_id}: Teacher probabilities do not sum to one")
        cache_by_id[sample_id] = row
    missing = [sample_id for sample_id in ids if sample_id not in cache_by_id]
    if missing:
        raise ValueError(f"Teacher cache is missing training IDs: {missing[:5]}")
    return cache_by_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 V7 Candidate/Class KD")
    parser.add_argument("--config", default="v7_rtx3090")
    parser.add_argument("--objective", choices=["candidate", "class"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--pool-path", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--s0-dir", default="outputs/v7_formal_s0_seed42")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--loss-weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="auto", help="auto、none 或 checkpoint path")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for value, name in (
        (args.epochs, "epochs"),
        (args.batch_size, "batch_size"),
        (args.save_every_steps, "save_every_steps"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for value, name in (
        (args.learning_rate, "learning_rate"),
        (args.max_grad_norm, "max_grad_norm"),
        (args.temperature, "temperature"),
        (args.loss_weight, "loss_weight"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.alpha) or args.alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")

    cfg = load_config(args.config)
    data_path = resolve(args.data_path)
    pool_path = resolve(args.pool_path)
    teacher_cache_path = resolve(args.teacher_cache)
    s0_dir = resolve(args.s0_dir)
    output_dir = resolve(args.output)
    max_length = args.max_length or int(cfg.training.max_length)
    records = read_split(data_path)
    pool_rows = read_jsonl(pool_path)
    cache_rows = read_jsonl(teacher_cache_path)
    cache_by_id = validate_inputs(records, pool_rows, cache_rows)
    s0_adapter, s0_selection = selected_s0(s0_dir)
    steps_per_epoch = math.ceil(len(records) / args.batch_size)
    budget = {
        "examples": len(records),
        "epochs": args.epochs,
        "sample_exposures": len(records) * args.epochs,
        "batch_size": args.batch_size,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": steps_per_epoch * args.epochs,
    }
    preparation = {
        "objective": args.objective,
        "data_path": relative(data_path),
        "data_sha256": sha256_file(data_path),
        "pool_path": relative(pool_path),
        "pool_sha256": sha256_file(pool_path),
        "teacher_cache": relative(teacher_cache_path),
        "teacher_cache_sha256": sha256_file(teacher_cache_path),
        "s0_dir": relative(s0_dir),
        "s0_checkpoint": s0_adapter.parent.name,
        "s0_selection_sha256": sha256_file(s0_dir / "selected_s0.json"),
        "s0_adapter_sha256": sha256_file(s0_adapter / "adapter_model.safetensors"),
        "student_model": str(cfg.models.student_local),
        "student_revision": str(cfg.models.student_revision),
        "temperature": args.temperature,
        "loss_weight": args.loss_weight,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "max_length": max_length,
        "seed": args.seed,
        "budget": budget,
    }
    print(json.dumps(preparation, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        print("V7_KD_PREPARE_OK=true")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("KD training requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_resume(output_dir, args.resume)
    run_signature = signature(preparation)
    run_manifest_path = output_dir / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("run_signature") != run_signature:
            raise ValueError("output directory belongs to another KD definition")
    else:
        run_manifest = {
            "manifest_version": 1,
            "kind": "v7_fixed_support_kd",
            "created_at_utc": utc_now(),
            "run_signature": run_signature,
            "definition": preparation,
            "s0_selection": s0_selection,
            "command": ["scripts/v7_train_kd.py", *sys.argv[1:]],
            "git": git_state(),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        }
        atomic_write_json(run_manifest_path, run_manifest)

    torch.cuda.reset_peak_memory_stats()
    base_model, tokenizer = load_student_model(
        str(cfg.models.student_local),
        load_in_4bit=True,
        bf16=bool(cfg.hardware.bf16),
        fp16=bool(cfg.hardware.fp16),
    )
    base_model = prepare_model_for_kbit_training(
        base_model,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    adapter_path = checkpoint / "adapter" if checkpoint is not None else s0_adapter
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    model.config.use_cache = False
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if checkpoint is None:
        state: dict[str, Any] = {
            "run_signature": run_signature,
            "global_optimizer_step": 0,
            "next_epoch_index": 0,
            "next_group_index": 0,
            "current_epoch_loss_sum": 0.0,
            "current_epoch_examples": 0,
            "history": [],
        }
        checkpoint = save_checkpoint(output_dir, model, optimizer, state)
        print(f"初始检查点：{checkpoint}", flush=True)
    else:
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        if state.get("run_signature") != run_signature:
            raise ValueError("resume checkpoint belongs to another KD definition")
        optimizer.load_state_dict(
            torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
        )
        restore_rng_state(checkpoint / "rng_state.pt")
        print(f"恢复检查点：{checkpoint}", flush=True)

    log_path = output_dir / "training_log.jsonl"
    start_time = time.time()
    for epoch_index in range(int(state["next_epoch_index"]), args.epochs):
        groups = build_epoch_groups(len(records), 1, args.batch_size, args.seed, epoch_index)
        first_group = (
            int(state["next_group_index"]) if epoch_index == int(state["next_epoch_index"]) else 0
        )
        epoch_loss_sum = float(state.get("current_epoch_loss_sum", 0.0))
        epoch_examples = int(state.get("current_epoch_examples", 0))
        for group_index in range(first_group, len(groups)):
            indices = [index for microbatch in groups[group_index] for index in microbatch]
            optimizer.zero_grad(set_to_none=True)
            group_loss_sum = 0.0
            for index in indices:
                record = records[index]
                candidates = [
                    CandidateRecord.model_validate(item) for item in pool_rows[index]["candidates"]
                ]
                scores = score_candidate_sequences(
                    model,
                    tokenizer,
                    record,
                    candidates,
                    max_length=max_length,
                    alpha=args.alpha,
                ).unsqueeze(0)
                teacher = torch.tensor(
                    [cache_by_id[record.id]["candidate_probabilities"]],
                    dtype=torch.float32,
                    device=scores.device,
                )
                loss = fixed_support_kd_loss(
                    scores,
                    teacher,
                    args.objective,
                    temperature=args.temperature,
                    loss_weight=args.loss_weight,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite KD loss for {record.id}")
                (loss / len(indices)).backward()
                value = float(loss.detach().cpu())
                group_loss_sum += value
                epoch_loss_sum += value
                epoch_examples += 1
                state["current_epoch_loss_sum"] = epoch_loss_sum
                state["current_epoch_examples"] = epoch_examples
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite KD gradient norm")
            optimizer.step()
            state["global_optimizer_step"] += 1
            state["next_epoch_index"] = epoch_index
            state["next_group_index"] = group_index + 1
            log_row = {
                "time_utc": utc_now(),
                "objective": args.objective,
                "epoch": epoch_index + 1,
                "group": group_index + 1,
                "groups_in_epoch": len(groups),
                "global_optimizer_step": state["global_optimizer_step"],
                "mean_kd_loss": group_loss_sum / len(indices),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "learning_rate": args.learning_rate,
                "samples": len(indices),
            }
            append_jsonl(log_path, log_row)
            if state[
                "global_optimizer_step"
            ] % args.save_every_steps == 0 and group_index + 1 < len(groups):
                saved = save_checkpoint(output_dir, model, optimizer, state)
                print(
                    f"step={state['global_optimizer_step']}/{budget['total_optimizer_steps']} "
                    f"loss={log_row['mean_kd_loss']:.6f} checkpoint={saved.name}",
                    flush=True,
                )

        epoch_result = {
            "epoch": epoch_index + 1,
            "global_optimizer_step": state["global_optimizer_step"],
            "mean_kd_loss": epoch_loss_sum / epoch_examples,
            "examples": epoch_examples,
            "peak_gpu_memory_mb": max_memory_mb(),
        }
        state["history"].append(epoch_result)
        state["next_epoch_index"] = epoch_index + 1
        state["next_group_index"] = 0
        state["current_epoch_loss_sum"] = 0.0
        state["current_epoch_examples"] = 0
        saved = save_checkpoint(output_dir, model, optimizer, state)
        print(json.dumps({**epoch_result, "checkpoint": saved.name}), flush=True)

    final_checkpoint = checkpoint_name(int(state["global_optimizer_step"]))
    final_adapter = output_dir / final_checkpoint / "adapter" / "adapter_model.safetensors"
    summary = {
        "kind": "v7_fixed_support_kd",
        "objective": args.objective,
        "completed_at_utc": utc_now(),
        "elapsed_seconds_this_invocation": time.time() - start_time,
        "global_optimizer_steps": state["global_optimizer_step"],
        "history": state["history"],
        "final_checkpoint": final_checkpoint,
        "final_adapter_sha256": sha256_file(final_adapter),
        "peak_gpu_memory_mb": max_memory_mb(),
    }
    atomic_write_json(output_dir / "training_summary.json", summary)
    atomic_write_json(
        output_dir / "final_checkpoint.json",
        {"checkpoint": final_checkpoint, "adapter_sha256": summary["final_adapter_sha256"]},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("V7_KD_TRAINING_OK=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
