#!/usr/bin/env python3
"""Measure V7 Candidate/Class gradient geometry at the frozen Formal S0."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
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
    if len(cache_by_id) != len(cache_rows) or any(
        sample_id not in cache_by_id for sample_id in ids
    ):
        raise ValueError("Teacher cache IDs are duplicated or incomplete")
    for row in pool_rows:
        if len(row.get("candidates", [])) != 4 or row.get("class_indices") != [0, 0, 1, 2]:
            raise ValueError(f"{row.get('sample_id')}: invalid K=4 definition")
    return cache_by_id


def add_gradients(
    accumulators: list[torch.Tensor], gradients: tuple[Any, ...], scale: float
) -> None:
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


def load_diagnostic_model(
    cfg: Any, adapter: Path
) -> tuple[Any, Any, list[tuple[str, torch.Tensor]]]:
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
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError("no trainable LoRA parameters")
    return tokenizer, model, named_parameters


def parse_smoke_positions(value: str, subset_size: int) -> list[int]:
    """Parse one-based smoke positions and return zero-based unique positions."""
    positions: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        position = int(item)
        if not 1 <= position <= subset_size:
            raise ValueError(f"smoke position {position} is outside 1..{subset_size}")
        zero_based = position - 1
        if zero_based not in positions:
            positions.append(zero_based)
    if not positions:
        raise ValueError("memory-smoke-positions must contain at least one position")
    return positions


def worker_command(args: argparse.Namespace, position: int, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        args.config,
        "--data",
        args.data,
        "--pool",
        args.pool,
        "--teacher-cache",
        args.teacher_cache,
        "--s0-dir",
        args.s0_dir,
        "--subset-size",
        str(args.subset_size),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--alpha",
        str(args.alpha),
        "--max-length",
        str(args.max_length),
        "--worker-position",
        str(position),
        "--worker-output",
        str(output),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7 Candidate/Class 梯度机制诊断")
    parser.add_argument("--config", default="v7_rtx3090")
    parser.add_argument("--data", default="data/processed/v7/balanced_336.jsonl")
    parser.add_argument("--pool", default="data/candidates/v7/balanced_336_k4.jsonl")
    parser.add_argument(
        "--teacher-cache", default="data/cache/v7/teacher/teacher_equiv_all_scores.jsonl"
    )
    parser.add_argument("--s0-dir", default="outputs/v7_formal_s0_seed42")
    parser.add_argument("--subset-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--summary", default="reports/v7_gradient_analysis.json")
    parser.add_argument("--report", default="reports/v7_gradient_analysis.md")
    parser.add_argument(
        "--memory-smoke-positions",
        default="",
        help="只检查指定位置能否逐条释放显存，例如 1,8；不生成正式结果",
    )
    parser.add_argument("--worker-position", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_worker(
    args: argparse.Namespace,
    cfg: Any,
    adapter: Path,
    records: list[Any],
    pool_rows: list[dict[str, Any]],
    cache_by_id: dict[str, dict[str, Any]],
    indices: list[int],
) -> int:
    if args.worker_position is None or not args.worker_output:
        raise ValueError("worker mode requires worker-position and worker-output")
    if not 0 <= args.worker_position < len(indices):
        raise ValueError("worker-position is outside the frozen subset")
    output_path = Path(args.worker_output)
    if output_path.exists():
        raise FileExistsError(output_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model, named_parameters = load_diagnostic_model(cfg, adapter)
    parameters = [parameter for _, parameter in named_parameters]
    index = indices[args.worker_position]
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

    candidate_grads = cpu_gradients(
        torch.autograd.grad(candidate_loss, parameters, retain_graph=True, allow_unused=True)
    )
    inter_grads = cpu_gradients(
        torch.autograd.grad(class_loss, parameters, retain_graph=True, allow_unused=True)
    )
    intra_grads = cpu_gradients(torch.autograd.grad(intra_loss, parameters, allow_unused=True))
    for gradients in (candidate_grads, inter_grads, intra_grads):
        if any(
            gradient is not None and not torch.isfinite(gradient).all() for gradient in gradients
        ):
            raise FloatingPointError(f"non-finite gradient for {record.id}")
    values = {
        "candidate": float(candidate_loss.detach().cpu()),
        "class": float(class_loss.detach().cpu()),
        "intra": float(intra_loss.detach().cpu()),
    }
    sample = {
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
                None if a is None and b is None else b if a is None else a if b is None else a + b
                for a, b in zip(inter_grads, intra_grads, strict=True)
            ),
        ),
    }
    torch.cuda.synchronize()
    payload = {
        "position": args.worker_position,
        "sample_id": record.id,
        "parameter_metadata": [
            {"name": name, "shape": list(parameter.shape)} for name, parameter in named_parameters
        ],
        "candidate_gradients": candidate_grads,
        "inter_gradients": inter_grads,
        "intra_gradients": intra_grads,
        "losses": values,
        "sample": sample,
        "peak_gpu_memory_mb": max_memory_mb(),
    }
    torch.save(payload, output_path)
    print(f"单条训练信号完成：{args.worker_position + 1}/{len(indices)} {record.id}", flush=True)
    return 0


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

    if args.worker_position is not None or args.worker_output:
        return run_worker(args, cfg, adapter, records, pool_rows, cache_by_id, indices)

    smoke_positions = (
        parse_smoke_positions(args.memory_smoke_positions, args.subset_size)
        if args.memory_smoke_positions
        else []
    )
    positions = smoke_positions or list(range(len(indices)))
    parameter_metadata: list[dict[str, Any]] | None = None
    candidate_sum: list[torch.Tensor] = []
    inter_sum: list[torch.Tensor] = []
    intra_sum: list[torch.Tensor] = []
    sample_rows: list[dict[str, Any]] = []
    loss_sums = {"candidate": 0.0, "class": 0.0, "intra": 0.0}
    peak_worker_memory_mb = 0.0
    scale = 1.0 / len(indices)

    with tempfile.TemporaryDirectory(prefix="v7-gradient-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for sequence, position in enumerate(positions, start=1):
            worker_output = temporary_root / f"sample-{position:03d}.pt"
            subprocess.run(
                worker_command(args, position, worker_output), cwd=PROJECT_ROOT, check=True
            )
            if not worker_output.is_file():
                raise FileNotFoundError(worker_output)
            if smoke_positions:
                worker_output.unlink()
                print(
                    f"显存检查完成：{sequence}/{len(positions)}，固定位置 {position + 1}",
                    flush=True,
                )
                continue

            payload = torch.load(worker_output, map_location="cpu", weights_only=False)
            worker_output.unlink()
            expected_id = records[indices[position]].id
            if payload["position"] != position or payload["sample_id"] != expected_id:
                raise ValueError("worker result does not match the frozen sample order")
            if parameter_metadata is None:
                parameter_metadata = payload["parameter_metadata"]
                shapes = [tuple(item["shape"]) for item in parameter_metadata]
                candidate_sum = [torch.zeros(shape, dtype=torch.float32) for shape in shapes]
                inter_sum = [torch.zeros(shape, dtype=torch.float32) for shape in shapes]
                intra_sum = [torch.zeros(shape, dtype=torch.float32) for shape in shapes]
            elif payload["parameter_metadata"] != parameter_metadata:
                raise ValueError("trainable parameter order changed across workers")
            add_gradients(candidate_sum, payload["candidate_gradients"], scale)
            add_gradients(inter_sum, payload["inter_gradients"], scale)
            add_gradients(intra_sum, payload["intra_gradients"], scale)
            for key in loss_sums:
                loss_sums[key] += float(payload["losses"][key])
            sample_rows.append(payload["sample"])
            peak_worker_memory_mb = max(peak_worker_memory_mb, float(payload["peak_gpu_memory_mb"]))
            del payload
            gc.collect()
            print(f"训练信号汇总：{sequence}/{len(positions)}", flush=True)

    if smoke_positions:
        print("TRAIN_SIGNAL_MEMORY_SMOKE_OK=true")
        return 0
    if len(sample_rows) != len(indices) or parameter_metadata is None:
        raise AssertionError("official gradient aggregation is incomplete")

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
        "sample_isolation": "one fresh subprocess and model load per sample; CPU aggregation",
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
        "peak_gpu_memory_mb": peak_worker_memory_mb,
        "samples": sample_rows,
    }
    summary_path = resolve(args.summary)
    atomic_write(summary_path, json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    report = f"""# V7 训练信号分析

## 先说结果怎么看

这里比较两种训练方法推动模型参数变化的力量。数值越大，说明这一步推动得越强；方向接近，说明两种信号在推动相近的变化，方向相反则会互相抵消。

完整候选蒸馏（`Candidate KD`）同时学习不同功能类别和同类写法偏好。功能类别蒸馏（`Class KD`）只学习不同功能类别。

## 固定计算范围

- 触发原因：Balanced-336 训练 1 轮时，`Class KD` 明显较差；训练 4 轮后差距缩小，因此检查它是否只是训练信号较弱。
- 模型固定为正式的初始学生模型。样本固定为 Balanced-336 在随机种子 42、首轮训练的第一组，共 {args.subset_size} 条。
- 每条样本都在全新进程中单独加载模型，结束后释放显存；最后只在普通内存中汇总。这样只解决显存累积，不改变样本、模型或计算公式。
- 模型关闭随机丢弃，因此同一输入可重复计算。这里分析的是训练目标本身，不代表一次真实训练更新。
- 公式为 `L_candidate = L_inter + L_intra`，`L_class = L_inter`。其中 `inter` 是不同功能类别之间的信号，`intra` 是同一功能类别内部不同写法之间的信号。

## 16 条样本的平均训练信号

| 衡量内容 | 数值 |
|---|---:|
| 完整候选蒸馏的信号长度 | {geometry["grad_candidate_norm"]:.8f} |
| 功能类别蒸馏的信号长度 | {geometry["grad_class_norm"]:.8f} |
| 不同功能类别之间的信号长度 | {geometry["grad_inter_norm"]:.8f} |
| 同类不同写法的信号长度 | {geometry["grad_intra_norm"]:.8f} |
| 完整候选 / 功能类别信号长度 | {geometry["candidate_over_class"]:.4f} |
| 同类写法 / 不同功能类别信号长度 | {geometry["intra_over_inter"]:.4f} |
| 两部分方向相似度（1 同向，-1 反向） | {geometry["cosine_inter_intra"]:.6f} |
| 分解误差（越接近 0 越好） | {geometry["decomposition_residual_norm"]:.3e} |

## 结论边界

- `Class KD` 的信号按定义就是不同功能类别之间的信号；`Candidate KD` 还多了同类不同写法的信号。
- 信号长度和方向只能帮助解释现象，不能代替正式生成评测，也不能单独证明扩大数据后的变化由它造成。
- 本分析只有一组预先固定的 16 条样本，不能把它当成全部训练数据上的统计定论。

详细计算结果见 [v7_gradient_analysis.json](v7_gradient_analysis.json)。
"""
    atomic_write(resolve(args.report), report)
    print(
        json.dumps(
            {"geometry": geometry, "peak_gpu_memory_mb": output["peak_gpu_memory_mb"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    print("V7_GRADIENT_ANALYSIS_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
