"""Scientific invariants for the V7 Candidate/Class branch difference."""

import torch

from fc_class_kd.v7.kd import fixed_support_class_probs, fixed_support_kd_loss


def test_fixed_support_aggregation_has_three_functional_classes():
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    classes = fixed_support_class_probs(probs)
    assert torch.allclose(classes, torch.tensor([[0.7, 0.2, 0.1]]))


def test_class_loss_removes_only_intra_correct_class_preference():
    scores = torch.tensor([[0.1, -0.2, 0.3, -0.4]], requires_grad=True)
    teacher_a = torch.tensor([[0.50, 0.20, 0.20, 0.10]])
    teacher_b = torch.tensor([[0.35, 0.35, 0.20, 0.10]])
    class_a = fixed_support_kd_loss(scores, teacher_a, "class")
    class_b = fixed_support_kd_loss(scores, teacher_b, "class")
    candidate_a = fixed_support_kd_loss(scores, teacher_a, "candidate")
    candidate_b = fixed_support_kd_loss(scores, teacher_b, "candidate")
    assert torch.allclose(class_a, class_b, atol=1e-7)
    assert not torch.allclose(candidate_a, candidate_b, atol=1e-7)


def test_both_v7_objectives_backpropagate_finite_gradients():
    for objective in ("candidate", "class"):
        scores = torch.tensor([[0.1, -0.2, 0.3, -0.4]], requires_grad=True)
        teacher = torch.tensor([[0.50, 0.20, 0.20, 0.10]])
        loss = fixed_support_kd_loss(scores, teacher, objective)
        loss.backward()
        assert torch.isfinite(loss)
        assert scores.grad is not None and torch.isfinite(scores.grad).all()
        assert scores.grad.abs().sum() > 0


def test_class_objective_preserves_inter_class_supervision():
    scores = torch.tensor([[0.1, -0.2, 0.3, -0.4]])
    teacher_a = torch.tensor([[0.50, 0.20, 0.20, 0.10]])
    teacher_b = torch.tensor([[0.40, 0.20, 0.30, 0.10]])
    loss_a = fixed_support_kd_loss(scores, teacher_a, "class")
    loss_b = fixed_support_kd_loss(scores, teacher_b, "class")
    assert not torch.allclose(loss_a, loss_b, atol=1e-7)
