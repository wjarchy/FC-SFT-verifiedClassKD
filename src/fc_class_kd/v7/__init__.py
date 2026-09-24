"""V7 recovery utilities with functional-call equivalence classes."""

from fc_class_kd.v7.candidates import build_k4_pool, candidate_class_ids
from fc_class_kd.v7.equivalence import (
    EquivalentVariant,
    build_strict_equivalent,
    canonical_signature,
)
from fc_class_kd.v7.kd import fixed_support_class_probs, fixed_support_kd_loss

__all__ = [
    "EquivalentVariant",
    "build_k4_pool",
    "build_strict_equivalent",
    "candidate_class_ids",
    "canonical_signature",
    "fixed_support_class_probs",
    "fixed_support_kd_loss",
]
