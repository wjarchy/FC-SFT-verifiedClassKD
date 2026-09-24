"""V7 fixed-support Candidate/Class KD math."""

from __future__ import annotations

import math
from typing import Literal

import torch

from fc_class_kd.losses.kd_losses import candidate_kd_loss, class_kd_loss
from fc_class_kd.scoring.aggregation import class_distribution_tensor

Objective = Literal["candidate", "class"]
FIXED_CLASS_INDICES = (0, 0, 1, 2)


def fixed_support_class_probs(candidate_probs: torch.Tensor) -> torch.Tensor:
    """Aggregate [Gold, Equivalent, WrongArg, WrongFn] into three functional classes."""
    if candidate_probs.ndim != 2 or candidate_probs.shape[1] != 4:
        raise ValueError("V7 candidate probabilities must have shape [B, 4]")
    indices = torch.tensor(
        [FIXED_CLASS_INDICES],
        device=candidate_probs.device,
        dtype=torch.long,
    ).expand(candidate_probs.shape[0], -1)
    return class_distribution_tensor(candidate_probs, indices, num_classes=3)


def fixed_support_kd_loss(
    student_scores: torch.Tensor,
    teacher_candidate_probs: torch.Tensor,
    objective: Objective,
    temperature: float = 2.0,
    loss_weight: float = 1.0,
) -> torch.Tensor:
    """Compute the only intended V7 branch difference on the same K=4 support."""
    if student_scores.ndim != 2 or student_scores.shape[1] != 4:
        raise ValueError("student scores must have shape [B, 4]")
    if teacher_candidate_probs.shape != student_scores.shape:
        raise ValueError("teacher probabilities must align with student scores")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(loss_weight) or loss_weight < 0:
        raise ValueError("loss_weight must be finite and non-negative")
    if not torch.isfinite(teacher_candidate_probs).all() or torch.any(teacher_candidate_probs < 0):
        raise ValueError("teacher probabilities must be finite and non-negative")
    if not torch.allclose(
        teacher_candidate_probs.sum(dim=-1),
        torch.ones(student_scores.shape[0], device=student_scores.device),
        atol=1e-5,
    ):
        raise ValueError("teacher probabilities must sum to one")

    student_log_probs = torch.log_softmax(student_scores / temperature, dim=-1)
    if objective == "candidate":
        return candidate_kd_loss(
            student_log_probs,
            teacher_candidate_probs,
            temperature=temperature,
            lambda_cand=loss_weight,
        )
    if objective != "class":
        raise ValueError(f"unknown V7 KD objective: {objective}")
    student_probs = student_log_probs.exp()
    student_class_probs = fixed_support_class_probs(student_probs)
    teacher_class_probs = fixed_support_class_probs(teacher_candidate_probs)
    return class_kd_loss(
        torch.log(student_class_probs.clamp_min(1e-12)),
        teacher_class_probs,
        temperature=temperature,
        lambda_class=loss_weight,
    )
