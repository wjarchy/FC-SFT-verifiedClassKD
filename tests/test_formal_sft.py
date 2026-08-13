"""Tests for formal S0 planning and checkpoint selection."""

from pathlib import Path

import pytest

from fc_class_kd.training.formal_sft import (
    build_epoch_groups,
    checkpoint_name,
    compute_sft_budget,
    select_best_checkpoint,
    select_s0_by_validation_ast,
    sha256_file,
    supervised_token_count,
)
from fc_class_kd.training.sft import EncodedSFTExample


def test_formal_budget_matches_2000_example_plan():
    budget = compute_sft_budget(2000, 1, 16, 3)
    assert budget.micro_batches_per_epoch == 2000
    assert budget.optimizer_steps_per_epoch == 125
    assert budget.total_optimizer_steps == 375


def test_epoch_groups_are_deterministic_and_cover_every_example_once():
    first = build_epoch_groups(19, 2, 3, seed=42, epoch_index=0)
    repeated = build_epoch_groups(19, 2, 3, seed=42, epoch_index=0)
    next_epoch = build_epoch_groups(19, 2, 3, seed=42, epoch_index=1)
    assert first == repeated
    assert first != next_epoch
    flattened = [index for group in first for batch in group for index in batch]
    assert sorted(flattened) == list(range(19))
    assert len(flattened) == len(set(flattened))


def test_supervised_token_count_only_counts_unmasked_labels():
    examples = [
        EncodedSFTExample("a", [1, 2, 3], [-100, 2, 3]),
        EncodedSFTExample("b", [4, 5], [-100, 5]),
    ]
    assert supervised_token_count(examples) == 3
    with pytest.raises(ValueError, match="no supervised"):
        supervised_token_count([EncodedSFTExample("c", [1], [-100])])


def test_best_checkpoint_requires_strictly_lower_validation_loss():
    assert select_best_checkpoint(None, None, 0.5, "checkpoint-a") == (
        0.5,
        "checkpoint-a",
        True,
    )
    assert select_best_checkpoint(0.5, "checkpoint-a", 0.5, "checkpoint-b") == (
        0.5,
        "checkpoint-a",
        False,
    )
    assert select_best_checkpoint(0.5, "checkpoint-a", 0.4, "checkpoint-b") == (
        0.4,
        "checkpoint-b",
        True,
    )


def test_s0_selection_prefers_ast_then_loss_then_earlier_epoch():
    evaluations = [
        {"checkpoint": "a", "epoch": 1, "ast_accuracy": 0.7, "validation_loss": 0.2},
        {"checkpoint": "b", "epoch": 2, "ast_accuracy": 0.8, "validation_loss": 0.3},
        {"checkpoint": "c", "epoch": 3, "ast_accuracy": 0.8, "validation_loss": 0.1},
    ]
    assert select_s0_by_validation_ast(evaluations)["checkpoint"] == "c"
    tied = [
        {"checkpoint": "a", "epoch": 1, "ast_accuracy": 0.8, "validation_loss": 0.1},
        {"checkpoint": "b", "epoch": 2, "ast_accuracy": 0.8, "validation_loss": 0.1},
    ]
    assert select_s0_by_validation_ast(tied)["checkpoint"] == "a"


def test_checkpoint_name_and_file_hash(tmp_path: Path):
    assert checkpoint_name(0) == "checkpoint-step-000000"
    assert checkpoint_name(7) == "checkpoint-step-000007"
    path = tmp_path / "value.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
