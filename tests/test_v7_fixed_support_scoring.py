"""Regression tests for explicit fixed-support candidate micro-batching."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def load_script():
    path = Path("scripts/v7_score_fixed_support.py")
    spec = importlib.util.spec_from_file_location("v7_fixed_support_scoring", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_one_preserves_candidate_order_across_microbatches(monkeypatch):
    module = load_script()
    observed: list[list[int]] = []

    def fake_score(model, tokenizer, record, candidates, max_length, alpha):
        values = [int(candidate) for candidate in candidates]
        observed.append(values)
        return torch.tensor(values, dtype=torch.float32)

    monkeypatch.setattr(module, "score_candidate_sequences", fake_score)
    result = module.score_one(None, None, None, [0, 1, 2, 3], 10, 1.0, 2)
    assert observed == [[0, 1], [2, 3]]
    assert result == [0.0, 1.0, 2.0, 3.0]
