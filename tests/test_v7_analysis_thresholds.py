"""Tests for preregistered V7 tertile threshold logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_script():
    path = Path("scripts/v7_freeze_analysis_thresholds.py")
    spec = importlib.util.spec_from_file_location("v7_analysis_thresholds", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linear_quantile_and_bucket_boundary_rules():
    module = load_script()
    values = [0.0, 1.0, 2.0, 3.0]
    assert module.linear_quantile(values, 1 / 3) == pytest.approx(1.0)
    assert module.linear_quantile(values, 2 / 3) == pytest.approx(2.0)
    assert module.bucket_counts(values, 1.0, 2.0) == {
        "low": 2,
        "mid": 1,
        "high": 1,
    }


def test_linear_quantile_rejects_bad_inputs():
    module = load_script()
    with pytest.raises(ValueError, match="empty"):
        module.linear_quantile([], 0.5)
    with pytest.raises(ValueError, match="finite"):
        module.linear_quantile([0.0, float("nan")], 0.5)
