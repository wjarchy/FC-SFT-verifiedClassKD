"""Regression tests for explicit V7 data binding in legacy runners."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sft_explicit_input_path_requires_a_real_file(tmp_path: Path):
    module = load_script("v7_train_sft", "scripts/train_sft.py")
    data = tmp_path / "formal.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    assert module.resolve_input_path(str(data), tmp_path / "unused") == data
    with pytest.raises(FileNotFoundError, match="experiment input not found"):
        module.resolve_input_path(str(tmp_path / "missing.jsonl"), data)


def test_s0_prediction_named_splits_bind_exact_paths(tmp_path: Path):
    module = load_script("v7_generate_s0", "scripts/generate_s0_predictions.py")
    internal = tmp_path / "internal.jsonl"
    equiv = tmp_path / "internal_equiv.jsonl"
    internal.write_text("{}\n", encoding="utf-8")
    equiv.write_text("{}\n", encoding="utf-8")
    args = argparse.Namespace(
        splits=[],
        split_paths=[f"internal={internal}", f"internal_equiv={equiv}"],
    )
    assert module.resolve_named_splits(args) == [
        ("internal", internal),
        ("internal_equiv", equiv),
    ]


def test_s0_prediction_named_splits_reject_duplicate_names(tmp_path: Path):
    module = load_script("v7_generate_s0_duplicate", "scripts/generate_s0_predictions.py")
    data = tmp_path / "rows.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    args = argparse.Namespace(splits=[], split_paths=[f"internal={data}", f"internal={data}"])
    with pytest.raises(ValueError, match="must be unique"):
        module.resolve_named_splits(args)
