"""Tests for exact subset derivation of already generated predictions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from fc_class_kd.types import FCRecord, FunctionCall, ToolSchema


def load_script():
    path = Path("scripts/v7_extract_prediction_subset.py")
    spec = importlib.util.spec_from_file_location("v7_prediction_subset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(sample_id: str, query: str = "q") -> FCRecord:
    return FCRecord(
        id=sample_id,
        source="fixture",
        query=query,
        tools=[ToolSchema(name="f")],
        gold_call=FunctionCall(name="f", arguments={}),
    )


def test_subset_preserves_requested_order_and_relabels_split():
    module = load_script()
    source = [record("a"), record("b"), record("c")]
    rows = [
        {"sample_id": "a", "split": "internal", "prediction": "A"},
        {"sample_id": "b", "split": "internal", "prediction": "B"},
        {"sample_id": "c", "split": "internal", "prediction": "C"},
    ]
    derived = module.derive_prediction_subset(
        source, [source[2], source[0]], rows, "internal_equiv"
    )
    assert [row["sample_id"] for row in derived] == ["c", "a"]
    assert [row["prediction"] for row in derived] == ["C", "A"]
    assert {row["split"] for row in derived} == {"internal_equiv"}


def test_subset_rejects_same_id_with_different_record():
    module = load_script()
    source = [record("a", query="original")]
    rows = [{"sample_id": "a", "split": "internal", "prediction": "A"}]
    with pytest.raises(ValueError, match="differs"):
        module.derive_prediction_subset(
            source, [record("a", query="changed")], rows, "internal_equiv"
        )
