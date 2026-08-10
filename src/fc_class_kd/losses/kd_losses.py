"""Training loss functions for Candidate KD and Class KD."""

from __future__ import annotations

import torch

from fc_class_kd.scoring.aggregation import torch_kl_div


def candidate_kd_loss(
    student_log_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float = 1.0,
    lambda_cand: float = 1.0,
) -> torch.Tensor:
    """L = lambda * tau^2 * KL(pi_T || pi_S). Shapes: [B, K]."""
    tau2 = temperature ** 2
    return lambda_cand * tau2 * torch_kl_div(student_log_probs, teacher_probs)


def class_kd_loss(
    student_class_log_probs: torch.Tensor,
    teacher_class_probs: torch.Tensor,
    temperature: float = 1.0,
    lambda_class: float = 1.0,
) -> torch.Tensor:
    """L = lambda * tau^2 * KL(q_T || q_S). Shapes: [B, C]."""
    tau2 = temperature ** 2
    return lambda_class * tau2 * torch_kl_div(student_class_log_probs, teacher_class_probs)


def combined_sft_kd_loss(
    sft_loss: torch.Tensor,
    kd_loss: torch.Tensor,
) -> torch.Tensor:
    return sft_loss + kd_loss


def masked_mean(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return losses.sum() * 0.0
    return (losses * mask).sum() / mask.sum()
