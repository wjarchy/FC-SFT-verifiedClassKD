#!/usr/bin/env python3
"""Measure V7 Candidate/Class gradient geometry at the frozen Formal S0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
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
from fc_class_kd.types import CandidateRecord
from fc_class_kd.v7.gradients import gradient_geometry
from fc_class_kd.v7.kd import fixed_support_kd_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def selected_adapter(s0_dir: Path) -> Path:
    selected = json.loads((s0_dir / "selected_s0.json").read_text(encoding="utf-8"))
    pointer = json.loads((s0_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    checkpoint = str(selected["selected_checkpoint"])
    if pointer.get("checkpoint") != checkpoint:
        raise ValueError("Formal S0 selection files disagree")
    adapter = s0_dir / checkpoint / "adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter)
    return adapter


def aligned_inputs(
    records: list[Any], pool_rows: list[dict[str, Any]], cache_rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    ids = [record.id for record in records]
    if [str(row.get("sample_id")) for row in pool_rows] != ids:
        raise ValueError("pool is not exactly aligned with data")
    cache_by_id = {str(row.get("sample_id")): row for row in cache_rows}
    if len(cache_by_id) != len(cache_rows) or any(sample_id not in cache_by_id for sample_id in ids):
        raise ValueError("Teacher cache IDs are duplicated or incomplete")
    for row in pool_rows:
        if len(row.get("candidates", [])) != 4 or row.get("class_indices") != [0, 0, 1, 2]:
            raise ValueError(f"{row.get('sample_id')}: invalid K=4 definition")
    return cache_by_id


def add_gradients(accumulators: list[torch.Tensor], gradients: tuple[Any, ...], scale: float) -> None:
    for accumulator, gradient in zip(accumulators, gradients, strict=True):
        if gradient is not None:
            accumulator.add_(gradient.detach().float().cpu(), alpha=scale)


def cpu_gradients(gradients: tuple[Any, ...]) -> tuple[Any, ...]:
    """Move diagnostic gradients off GPU without changing their values."""
    return tuple(
        None if gradient is None else gradient.detach().float().cpu() for gradient in gradients
    )


def norm_of(gradients: tuple[Any, ...]) -> float:
    squared = 0.0
    for gradient in gradients:
        if gradient is not None:
            value = gradient.detach().double()
            squared += float(torch.sum(value * value))
    return squared**0.5


def diff_norm(left: tuple[Any, ...], right: tuple[Any, ...]) -> float:
    squared = 0.0
    for first, second in zip(left, right, strict=True):
        if first is None and second is None:
            continue
        if first is None:
            delta = -second.detach().double()
        elif second is None:
            delta = first.detach().double()
        else:
            delta = first.detach().double() - second.detach().double()
        squared += float(torch.sum(delta * delta))
    return squared**0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7 Candidate/Class 梯度机制诊断")
    parser.add_argument("--config", default="v7_rtx3090")
    parser.add_argument("--data", default="data/processed/v7/balanced_336.jsonl")
    parser.add_argument("--pool", default="data/candidates/v7/balanced_336_k4.jsonl")
    parser.add_argument("--teacher-cache", default="data/cache/v7/teacher/teacher_equiv_all_scores.jsonl")
    parser.add_argument("--s0-dir", default="outputs/v7_formal_s0_seed42")
    parser.add_argument("--subset-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--summary", default="reports/v7_gradient_analysis.json")
    parser.add_argument("--report", default="reports/v7_gradient_analysis.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.subset_size <= 0:
        raise ValueError("subset-size must be positive")
    cfg = load_config(args.config)
    data_path, pool_path, cache_path = map(resolve, (args.data, args.pool, args.teacher_cache))
    s0_dir = resolve(args.s0_dir)
    adapter = selected_adapter(s0_dir)
    records = read_split(data_path)
    pool_rows = read_jsonl(pool_path)
    cache_rows = read_jsonl(cache_path)
    cache_by_id = aligned_inputs(records, pool_rows, cache_rows)
    if args.subset_size > len(records):
        raise ValueError("subset-size exceeds data size")
    groups = build_epoch_groups(len(records), 1, args.subset_size, args.seed, 0)
    indices = [index for microbatch in groups[0] for index in microbatch]
    if len(indices) != args.subset_size:
        raise AssertionError("first optimizer group has the wrong size")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    base_model, tokenizer = load_student_model(
        str(cfg.models.student_local),
        load_in_4bit=True,
        bf16=bool(cfg.hardware.bf16),
        fp16=bool(cfg.hardware.fp16),
    )
    base_model = prepare_model_for_kbit_training(
        base_model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = PeftModel.from_pretrained(base_model, adapter, is_trainable=True)
    model.config.use_cache = False
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no trainable LoRA parameters")
    candidate_sum = [torch.zeros_like(parameter, device="cpu", dtype=torch.float32) for parameter in parameters]
    inter_sum = [torch.zeros_like(parameter, device="cpu", dtype=torch.float32) for parameter in parameters]
    intra_sum = [torch.zeros_like(parameter, device="cpu", dtype=torch.float32) for parameter in parameters]
    sample_rows: list[dict[str, Any]] = []
    loss_sums = {"candidate": 0.0, "class": 0.0, "intra": 0.0}
    scale = 1.0 / len(indices)
    for position, index in enumerate(indices, start=1):
        record = records[index]
        candidates = [CandidateRecord.model_validate(item) for item in pool_rows[index]["candidates"]]
        scores = score_candidate_sequences(
            model, tokenizer, record, candidates, max_length=args.max_length, alpha=args.alpha
        ).unsqueeze(0)
        teacher = torch.tensor(
            [cache_by_id[record.id]["candidate_probabilities"]],
            dtype=torch.float32,
            device=scores.device,
        )
        candidate_loss = fixed_support_kd_loss(
            scores, teacher, "candidate", temperature=args.temperature
        )
        class_loss = fixed_support_kd_loss(scores, teacher, "class", temperature=args.temperature)
        intra_loss = candidate_loss - class_loss
        if not all(torch.isfinite(value) for value in (candidate_loss, class_loss, intra_loss)):
            raise FloatingPointError(f"non-finite loss for {record.id}")
        candidate_device_grads = torch.autograd.grad(
            candidate_loss, parameters, retain_graph=True, allow_unused=True
        )
        candidate_grads = cpu_gradients(candidate_device_grads)
        del candidate_device_grads
        inter_device_grads = torch.autograd.grad(
            class_loss, parameters, retain_graph=True, allow_unused=True
        )
        inter_grads = cpu_gradients(inter_device_grads)
        del inter_device_grads
        intra_device_grads = torch.autograd.grad(intra_loss, parameters, allow_unused=True)
        intra_grads = cpu_gradients(intra_device_grads)
        del intra_device_grads
        add_gradients(candidate_sum, candidate_grads, scale)
        add_gradients(inter_sum, inter_grads, scale)
        add_gradients(intra_sum, intra_grads, scale)
        values = {
            "candidate": float(candidate_loss.detach().cpu()),
            "class": float(class_loss.detach().cpu()),
            "intra": float(intra_loss.detach().cpu()),
        }
        for key, value in values.items():
            loss_sums[key] += value
        sample_rows.append(
            {
                "sample_id": record.id,
                "equivalence_type": pool_rows[index]["equivalence_type"],
                "candidate_loss": values["candidate"],
                "class_loss": values["class"],
                "intra_loss": values["intra"],
                "grad_candidate_norm": norm_of(candidate_grads),
                "grad_class_norm": norm_of(inter_grads),
                "grad_intra_norm": norm_of(intra_grads),
                "gradient_decomposition_residual_norm": diff_norm(
                    candidate_grads,
                    tuple(
                        None if a is None and b is None else (
                            b if a is None else a if b is None else a + b
                        )
                        for a, b in zip(inter_grads, intra_grads, strict=True)
                    ),
                ),
            }
        )
        del (
            scores,
            teacher,
            candidate_loss,
            class_loss,
            intra_loss,
            candidate_grads,
            inter_grads,
            intra_grads,
        )
        torch.cuda.empty_cache()
        print(f"gradient diagnostic: {position}/{len(indices)} {record.id}", flush=True)

    geometry = gradient_geometry(candidate_sum, inter_sum, intra_sum)
    sample_norm_means = {
        key: statistics.mean(float(row[key]) for row in sample_rows)
        for key in ("grad_candidate_norm", "grad_class_norm", "grad_intra_norm")
    }
    definition = {
        "kind": "v7_gradient_geometry_at_formal_s0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection": "first optimizer group from build_epoch_groups(epoch=0)",
        "model_mode": "eval (dropout disabled); diagnostic geometry, not an optimizer trace",
        "data": relative(data_path),
        "data_sha256": sha256_file(data_path),
        "pool": relative(pool_path),
        "pool_sha256": sha256_file(pool_path),
        "teacher_cache": relative(cache_path),
        "teacher_cache_sha256": sha256_file(cache_path),
        "s0_adapter": relative(adapter),
        "s0_adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "student_revision": str(cfg.models.student_revision),
        "seed": args.seed,
        "subset_size": args.subset_size,
        "sample_ids": [records[index].id for index in indices],
        "equivalence_type_counts": {
            label: sum(pool_rows[index]["equivalence_type"] == label for index in indices)
            for label in ("A", "B", "C")
        },
        "temperature": args.temperature,
        "alpha": args.alpha,
        "max_length": args.max_length,
        "git": git_state(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    output = {
        "definition": definition,
        "mean_losses": {key: value / len(indices) for key, value in loss_sums.items()},
        "aggregate_mean_gradient_geometry": geometry,
        "mean_per_sample_gradient_norms": sample_norm_means,
        "max_per_sample_decomposition_residual_norm": max(
            row["gradient_decomposition_residual_norm"] for row in sample_rows
        ),
        "peak_gpu_memory_mb": max_memory_mb(),
        "samples": sample_rows,
    }
    summary_path = resolve(args.summary)
    atomic_write(summary_path, json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    report = f"""# V7 Gradient Analysis

## 固定诊断定义

- 触发条件：情况 A。Balanced-336×1 的 Class 明显退化，×4 后 Full 差距从 -10.80pt 缩至 -1.60pt；需要检查 Class 是否缺少类内项而信号更弱。
- 模型：冻结的 Formal S0；数据：Balanced-336 的 seed 42、epoch 0 第一个实际 optimizer group，共 {args.subset_size} 条。
- 模型置于 eval mode 关闭 dropout，测量确定性的 objective geometry；这不是某次随机训练 step 的梯度轨迹。
- 精确分解：`L_candidate = L_inter + L_intra`，且 `L_class = L_inter`。三组梯度从同一次 forward graph 独立求导。

## 聚合 mean-loss 梯度

| Quantity | Value |
|---|---:|
| `||grad_candidate||` | {geometry['grad_candidate_norm']:.8f} |
| `||grad_class||` | {geometry['grad_class_norm']:.8f} |
| `||grad_inter||` | {geometry['grad_inter_norm']:.8f} |
| `||grad_intra||` | {geometry['grad_intra_norm']:.8f} |
| Candidate / Class norm | {geometry['candidate_over_class']:.4f} |
| Intra / Inter norm | {geometry['intra_over_inter']:.4f} |
| cosine(grad_inter, grad_intra) | {geometry['cosine_inter_intra']:.6f} |
| decomposition residual norm | {geometry['decomposition_residual_norm']:.3e} |

## 解释边界

- Class 与 inter 的梯度按定义相同；Candidate 的额外信号来自 intra。
- 范数只能解释 signal strength / direction，不能替代 free-generation evaluator，也不能单独证明 scaling 因果机制。
- 本诊断仅一个预定义 optimizer group，不把小样本梯度统计外推成总体显著性结论。

机器可读结果：[v7_gradient_analysis.json](v7_gradient_analysis.json)。
"""
    atomic_write(resolve(args.report), report)
    print(json.dumps({"geometry": geometry, "peak_gpu_memory_mb": output["peak_gpu_memory_mb"]}, indent=2))
    print("V7_GRADIENT_ANALYSIS_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
