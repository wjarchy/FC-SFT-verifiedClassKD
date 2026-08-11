"""PRD §11.2–11.4 math unit tests."""

import math

import pytest
import torch

from fc_class_kd.losses.kd_losses import candidate_kd_loss, class_kd_loss
from fc_class_kd.scoring.aggregation import (
    aggregate_class_probabilities,
    candidate_probabilities,
    class_distribution_tensor,
    decompose_candidate_kl,
    sequence_score,
)
from fc_class_kd.types import VerifierClass


def test_sequence_score():
    logps = [math.log(0.5), math.log(0.5), math.log(0.5)]
    s = sequence_score(logps, alpha=1.0)
    expected = sum(logps) / 3
    assert abs(s - expected) < 1e-9


def test_candidate_probabilities_sum_to_one():
    scores = [1.0, 2.0, 0.5, 3.0]
    probs = candidate_probabilities(scores, temperature=1.0)
    assert abs(sum(probs) - 1.0) < 1e-9


def test_candidate_probabilities_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        candidate_probabilities([1.0, 2.0], temperature=0.0)


def test_class_aggregation_prd_example():
    """PRD §11.3: [0.30,0.20,0.10,0.40] + classes → [0.60, 0.40]."""
    probs = [0.30, 0.20, 0.10, 0.40]
    labels = [
        VerifierClass.CORRECT,
        VerifierClass.CORRECT,
        VerifierClass.CORRECT,
        VerifierClass.WRONG_FUNCTION,
    ]
    classes = (VerifierClass.CORRECT, VerifierClass.WRONG_FUNCTION)
    agg = aggregate_class_probabilities(probs, labels, classes)
    assert abs(agg[VerifierClass.CORRECT] - 0.60) < 1e-9
    assert abs(agg[VerifierClass.WRONG_FUNCTION] - 0.40) < 1e-9


def test_class_aggregation_excludes_unknown_and_renormalizes():
    probs = [0.4, 0.3, 0.3]
    labels = [
        VerifierClass.CORRECT,
        VerifierClass.WRONG_FUNCTION,
        VerifierClass.UNKNOWN,
    ]
    classes = (VerifierClass.CORRECT, VerifierClass.WRONG_FUNCTION)
    agg = aggregate_class_probabilities(probs, labels, classes)
    assert abs(sum(agg.values()) - 1.0) < 1e-9
    assert abs(agg[VerifierClass.CORRECT] - 4 / 7) < 1e-9
    assert abs(agg[VerifierClass.WRONG_FUNCTION] - 3 / 7) < 1e-9


def test_class_aggregation_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        aggregate_class_probabilities(
            [0.5, 0.5],
            [VerifierClass.CORRECT],
        )


def test_class_distribution_tensor_renormalizes_after_masking_unknown():
    probs = torch.tensor([[0.4, 0.3, 0.3]], requires_grad=True)
    indices = torch.tensor([[0, 1, -1]])
    classes = class_distribution_tensor(probs, indices, num_classes=2)
    assert torch.allclose(classes.sum(dim=-1), torch.ones(1))
    assert torch.allclose(classes, torch.tensor([[4 / 7, 3 / 7]]), atol=1e-6)
    classes[:, 0].sum().backward()
    assert probs.grad is not None


def test_kl_decomposition():
    teacher = [0.4, 0.3, 0.2, 0.1]
    student = [0.35, 0.25, 0.25, 0.15]
    labels = [
        VerifierClass.CORRECT,
        VerifierClass.CORRECT,
        VerifierClass.WRONG_FUNCTION,
        VerifierClass.INVALID,
    ]
    kl_cand, kl_inter, kl_intra = decompose_candidate_kl(teacher, student, labels)
    assert abs(kl_cand - (kl_inter + kl_intra)) < 1e-6


def test_kl_decomposition_with_unknown_uses_known_candidate_distribution():
    teacher = [0.4, 0.3, 0.2, 0.1]
    student = [0.3, 0.25, 0.25, 0.2]
    labels = [
        VerifierClass.CORRECT,
        VerifierClass.CORRECT,
        VerifierClass.WRONG_FUNCTION,
        VerifierClass.UNKNOWN,
    ]
    kl_cand, kl_inter, kl_intra = decompose_candidate_kl(teacher, student, labels)
    assert abs(kl_cand - (kl_inter + kl_intra)) < 1e-6


def test_torch_kd_losses_finite():
    t_probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    s_log = torch.log(t_probs + 1e-8)
    loss_c = candidate_kd_loss(s_log, t_probs, temperature=1.0)
    assert loss_c.item() >= 0

    t_class = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float32)
    s_class_log = torch.log(t_class + 1e-8)
    loss_cls = class_kd_loss(s_class_log, t_class, temperature=1.0)
    assert abs(loss_cls.item()) < 1e-5
