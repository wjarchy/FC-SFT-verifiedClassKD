"""Tests for differentiable complete-candidate sequence scores."""

import math

import torch

from fc_class_kd.training.candidate_scoring import sequence_scores_from_logits


def test_sequence_scores_from_logits_matches_manual_and_backpropagates():
    logits = torch.zeros((1, 4, 3), dtype=torch.float64, requires_grad=True)
    input_ids = torch.tensor([[0, 1, 2, 0]])
    output_mask = torch.tensor([[False, False, True, True]])
    scores = sequence_scores_from_logits(logits, input_ids, output_mask, alpha=1.0)
    assert scores.shape == (1,)
    assert abs(scores.item() + math.log(3.0)) < 1e-6
    scores.sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_sequence_scores_from_logits_rejects_empty_output_mask():
    logits = torch.zeros((1, 3, 2))
    input_ids = torch.zeros((1, 3), dtype=torch.long)
    output_mask = torch.zeros((1, 3), dtype=torch.bool)
    try:
        sequence_scores_from_logits(logits, input_ids, output_mask)
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty output mask should fail")
