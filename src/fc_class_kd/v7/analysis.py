"""Small, deterministic statistical helpers for V7 paired analyses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def tertile_bucket(value: float, q1: float, q2: float) -> str:
    """Apply the frozen low <= q1 < mid <= q2 < high rule."""
    if not all(math.isfinite(item) for item in (value, q1, q2)):
        raise ValueError("bucket inputs must be finite")
    if q1 > q2:
        raise ValueError("tertile cut points are reversed")
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def exact_two_sided_binomial_p(successes: int, trials: int) -> float:
    """Two-sided exact sign-test p-value for p=0.5."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    probability = sum(math.comb(trials, index) for index in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def paired_summary(
    candidate_correct: Sequence[bool],
    class_correct: Sequence[bool],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize paired accuracy delta with deterministic percentile bootstrap."""
    if len(candidate_correct) != len(class_correct):
        raise ValueError("paired outcomes must have equal length")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    candidate = np.asarray(candidate_correct, dtype=np.int8)
    class_kd = np.asarray(class_correct, dtype=np.int8)
    n = int(candidate.size)
    candidate_wins = int(np.sum((candidate == 1) & (class_kd == 0)))
    class_wins = int(np.sum((candidate == 0) & (class_kd == 1)))
    if n == 0:
        return {
            "n": 0,
            "candidate_correct": 0,
            "class_correct": 0,
            "candidate_accuracy": None,
            "class_accuracy": None,
            "delta_class_minus_candidate": None,
            "delta_bootstrap_95ci": [None, None],
            "candidate_only_correct": 0,
            "class_only_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "mcnemar_exact_p": 1.0,
        }
    deltas = class_kd - candidate
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(bootstrap_samples, n))
    bootstrap = deltas[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    discordant = candidate_wins + class_wins
    return {
        "n": n,
        "candidate_correct": int(candidate.sum()),
        "class_correct": int(class_kd.sum()),
        "candidate_accuracy": float(candidate.mean()),
        "class_accuracy": float(class_kd.mean()),
        "delta_class_minus_candidate": float(deltas.mean()),
        "delta_bootstrap_95ci": [float(low), float(high)],
        "candidate_only_correct": candidate_wins,
        "class_only_correct": class_wins,
        "both_correct": int(np.sum((candidate == 1) & (class_kd == 1))),
        "both_wrong": int(np.sum((candidate == 0) & (class_kd == 0))),
        "mcnemar_exact_p": exact_two_sided_binomial_p(class_wins, discordant),
    }
