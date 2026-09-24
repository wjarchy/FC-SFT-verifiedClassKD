"""Integrity checks for resumable V7 fixed-support caches."""

from __future__ import annotations

import importlib.util
import math

import pytest


def load_scoring_script():
    spec = importlib.util.spec_from_file_location(
        "v7_score_fixed_support", "scripts/v7_score_fixed_support.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_row(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "candidate_scores": [-1.0, -2.0, -3.0, -4.0],
        "candidate_probabilities": [0.4, 0.3, 0.2, 0.1],
        "class_probabilities": [0.7, 0.2, 0.1],
        "intra_class_probabilities": [4 / 7, 3 / 7],
    }


def test_fixed_support_cache_accepts_exact_resume_prefix():
    module = load_scoring_script()
    module.validate_existing(["one", "two"], [valid_row("one")], "equiv")


def test_fixed_support_cache_rejects_reordered_resume_rows():
    module = load_scoring_script()
    with pytest.raises(ValueError, match="exact unique prefix"):
        module.validate_existing(["one", "two"], [valid_row("two"), valid_row("one")], "equiv")


def test_fixed_support_cache_rejects_nonfinite_or_wrong_shape():
    module = load_scoring_script()
    row = valid_row("one")
    row["candidate_scores"][0] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        module.validate_existing(["one"], [row], "equiv")
    row = valid_row("one")
    row["class_probabilities"] = [0.7, 0.3]
    with pytest.raises(ValueError, match="invalid class_probabilities"):
        module.validate_existing(["one"], [row], "equiv")
