"""Tests for deterministic V7 paired statistical helpers."""

from __future__ import annotations

import pytest

from fc_class_kd.v7.analysis import (
    exact_two_sided_binomial_p,
    paired_summary,
    tertile_bucket,
)


def test_tertile_bucket_uses_frozen_boundary_rule():
    assert tertile_bucket(0.2, 0.2, 0.8) == "low"
    assert tertile_bucket(0.5, 0.2, 0.8) == "mid"
    assert tertile_bucket(0.8, 0.2, 0.8) == "mid"
    assert tertile_bucket(0.9, 0.2, 0.8) == "high"


def test_paired_summary_counts_discordance_and_is_deterministic():
    candidate = [True, True, False, False]
    class_kd = [True, False, True, False]
    first = paired_summary(candidate, class_kd, bootstrap_samples=500, seed=7)
    second = paired_summary(candidate, class_kd, bootstrap_samples=500, seed=7)
    assert first == second
    assert first["candidate_accuracy"] == pytest.approx(0.5)
    assert first["class_accuracy"] == pytest.approx(0.5)
    assert first["delta_class_minus_candidate"] == pytest.approx(0.0)
    assert first["candidate_only_correct"] == 1
    assert first["class_only_correct"] == 1
    assert first["mcnemar_exact_p"] == pytest.approx(1.0)


def test_exact_binomial_handles_one_sided_discordance():
    assert exact_two_sided_binomial_p(0, 5) == pytest.approx(0.0625)
    assert exact_two_sided_binomial_p(5, 5) == pytest.approx(0.0625)
    assert paired_summary([], [], bootstrap_samples=10)["n"] == 0
