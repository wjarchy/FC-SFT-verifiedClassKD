"""Pure planning and bookkeeping helpers for the formal public-S0 run."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fc_class_kd.training.sft import EncodedSFTExample


@dataclass(frozen=True)
class SFTBudget:
    examples: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    micro_batches_per_epoch: int
    optimizer_steps_per_epoch: int
    total_optimizer_steps: int


def compute_sft_budget(
    example_count: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> SFTBudget:
    """Compute the exact number of micro-batches and optimizer updates."""
    for value, name in (
        (example_count, "example_count"),
        (micro_batch_size, "micro_batch_size"),
        (gradient_accumulation_steps, "gradient_accumulation_steps"),
        (epochs, "epochs"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    micro_batches = math.ceil(example_count / micro_batch_size)
    optimizer_steps = math.ceil(micro_batches / gradient_accumulation_steps)
    return SFTBudget(
        examples=example_count,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        micro_batches_per_epoch=micro_batches,
        optimizer_steps_per_epoch=optimizer_steps,
        total_optimizer_steps=optimizer_steps * epochs,
    )


def build_epoch_groups(
    example_count: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    seed: int,
    epoch_index: int,
) -> list[list[list[int]]]:
    """Return deterministic optimizer groups containing one or more micro-batches."""
    compute_sft_budget(
        example_count,
        micro_batch_size,
        gradient_accumulation_steps,
        epochs=1,
    )
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    order = list(range(example_count))
    random.Random(seed + epoch_index).shuffle(order)
    micro_batches = [
        order[offset : offset + micro_batch_size]
        for offset in range(0, len(order), micro_batch_size)
    ]
    return [
        micro_batches[offset : offset + gradient_accumulation_steps]
        for offset in range(0, len(micro_batches), gradient_accumulation_steps)
    ]


def supervised_token_count(examples: Sequence[EncodedSFTExample]) -> int:
    """Count assistant tokens that contribute to cross-entropy."""
    count = sum(label != -100 for example in examples for label in example.labels)
    if count <= 0:
        raise ValueError("SFT examples contain no supervised assistant tokens")
    return count


def select_best_checkpoint(
    current_best_loss: float | None,
    current_best_path: str | None,
    candidate_loss: float,
    candidate_path: str,
) -> tuple[float, str, bool]:
    """Select strictly lower validation loss; ties keep the earlier checkpoint."""
    if not math.isfinite(candidate_loss):
        raise ValueError("candidate validation loss must be finite")
    if not candidate_path:
        raise ValueError("candidate checkpoint path must be non-empty")
    if current_best_loss is None or candidate_loss < current_best_loss:
        return candidate_loss, candidate_path, True
    if current_best_path is None:
        raise ValueError("current_best_path is required when current_best_loss is set")
    return current_best_loss, current_best_path, False


def select_s0_by_validation_ast(
    evaluations: Sequence[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    """Select highest AST, then lower loss, then the earlier epoch."""
    if not evaluations:
        raise ValueError("at least one Validation evaluation is required")
    checked: list[dict[str, float | int | str]] = []
    for item in evaluations:
        checkpoint = str(item.get("checkpoint", ""))
        ast_accuracy = float(item.get("ast_accuracy", float("nan")))
        validation_loss = float(item.get("validation_loss", float("nan")))
        epoch = int(item.get("epoch", 0))
        if not checkpoint:
            raise ValueError("each Validation evaluation needs a checkpoint")
        if not math.isfinite(ast_accuracy) or not 0 <= ast_accuracy <= 1:
            raise ValueError("ast_accuracy must be finite and in [0, 1]")
        if not math.isfinite(validation_loss):
            raise ValueError("validation_loss must be finite")
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        checked.append(dict(item))
    return min(
        checked,
        key=lambda item: (
            -float(item["ast_accuracy"]),
            float(item["validation_loss"]),
            int(item["epoch"]),
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_name(global_optimizer_step: int) -> str:
    if global_optimizer_step < 0:
        raise ValueError("global_optimizer_step must be non-negative")
    return f"checkpoint-step-{global_optimizer_step:06d}"
