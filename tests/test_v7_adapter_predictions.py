"""Unit tests for batched V7 adapter prediction bookkeeping."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_script():
    path = Path("scripts/v7_generate_adapter_predictions.py")
    spec = importlib.util.spec_from_file_location("v7_adapter_predictions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trim_generated_ids_keeps_first_eos_only():
    module = load_script()
    assert module.trim_generated_ids([10, 11, 2, 2, 2], 2) == [10, 11, 2]
    assert module.trim_generated_ids([10, 11], 2) == [10, 11]
    assert module.trim_generated_ids([10, 11], None) == [10, 11]


def test_signature_is_key_order_independent():
    module = load_script()
    assert module.signature({"a": 1, "b": 2}) == module.signature({"b": 2, "a": 1})
