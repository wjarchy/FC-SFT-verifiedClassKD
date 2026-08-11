"""Candidate sequence scoring and class aggregation (PRD §1.3–1.4)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from fc_class_kd.types import PRIMARY_CLASSES, VerifierClass


def sequence_score(
    token_logprobs: Sequence[float],
    alpha: float = 1.0,
) -> float:
    """s(y|x) = (1/|y|^alpha) * sum_t log p(y_t|...)."""
    if len(token_logprobs) == 0:
        raise ValueError("token_logprobs must contain at least one scored output token")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    length = len(token_logprobs)
    return sum(token_logprobs) / (length ** alpha)


def candidate_probabilities(
    scores: Sequence[float],
    temperature: float = 1.0,
) -> List[float]:
    """Softmax over candidate sequence scores."""
    if len(scores) == 0:
        return []
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("scores must all be finite")
    t = temperature
    scaled = [s / t for s in scores]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    if total == 0:
        n = len(scores)
        return [1.0 / n] * n
    return [e / total for e in exps]


def aggregate_class_probabilities(
    candidate_probs: Sequence[float],
    class_labels: Sequence[VerifierClass],
    classes: Sequence[VerifierClass] = PRIMARY_CLASSES,
) -> Dict[VerifierClass, float]:
    """
    q(e|x) = sum_{y_i: g(y_i)=e} pi(y_i|x).

    Labels outside `classes` (e.g. unknown) are excluded from aggregation.
    """
    if len(candidate_probs) != len(class_labels):
        raise ValueError("candidate_probs and class_labels must have the same length")
    allowed = set(classes)
    totals = {c: 0.0 for c in classes}
    for prob, label in zip(candidate_probs, class_labels):
        if not math.isfinite(prob) or prob < 0:
            raise ValueError("candidate probabilities must be finite and non-negative")
        if label not in allowed:
            continue
        totals[label] += prob
    retained_mass = sum(totals.values())
    if retained_mass <= 0:
        raise ValueError("no probability mass remains after excluded classes are masked")
    return {label: mass / retained_mass for label, mass in totals.items()}


def class_distribution_tensor(
    candidate_probs: torch.Tensor,
    class_indices: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Batched class aggregation: candidate_probs [B,K], class_indices [B,K]."""
    if candidate_probs.ndim != 2 or class_indices.shape != candidate_probs.shape:
        raise ValueError("candidate_probs and class_indices must have the same [B, K] shape")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if torch.any(class_indices < -1) or torch.any(class_indices >= num_classes):
        raise ValueError("class_indices must be -1 (excluded) or a valid class index")
    if not torch.isfinite(candidate_probs).all() or torch.any(candidate_probs < 0):
        raise ValueError("candidate probabilities must be finite and non-negative")

    valid = class_indices >= 0
    safe_indices = class_indices.masked_fill(~valid, 0).long()
    retained = candidate_probs * valid.to(candidate_probs.dtype)
    out = torch.zeros(
        candidate_probs.shape[0],
        num_classes,
        device=candidate_probs.device,
        dtype=candidate_probs.dtype,
    )
    out.scatter_add_(1, safe_indices, retained)
    retained_mass = out.sum(dim=-1, keepdim=True)
    if torch.any(retained_mass <= 0):
        raise ValueError("at least one row has no mass after excluded classes are masked")
    return out / retained_mass


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """KL(P || Q) for discrete distributions."""
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    if len(p) == 0:
        raise ValueError("p and q must be non-empty")
    total = 0.0
    for pi, qi in zip(p, q):
        if not math.isfinite(pi) or not math.isfinite(qi) or pi < 0 or qi < 0:
            raise ValueError("KL inputs must be finite and non-negative")
        if pi == 0:
            continue
        qi = max(qi, eps)
        total += pi * math.log(pi / qi)
    return total


def decompose_candidate_kl(
    candidate_probs_teacher: Sequence[float],
    candidate_probs_student: Sequence[float],
    class_labels: Sequence[VerifierClass],
    classes: Sequence[VerifierClass] = PRIMARY_CLASSES,
) -> Tuple[float, float, float]:
    """
    Decompose KL(pi_T || pi_S) into inter-class + intra-class (PRD §1.7).

    Returns (kl_candidate, kl_inter, kl_intra).
    """
    if not (
        len(candidate_probs_teacher) == len(candidate_probs_student) == len(class_labels)
    ):
        raise ValueError("teacher probs, student probs, and labels must have the same length")

    allowed = set(classes)
    known = [index for index, label in enumerate(class_labels) if label in allowed]
    if not known:
        raise ValueError("cannot decompose KL when all candidates are excluded")
    teacher_mass = sum(candidate_probs_teacher[index] for index in known)
    student_mass = sum(candidate_probs_student[index] for index in known)
    if teacher_mass <= 0 or student_mass <= 0:
        raise ValueError("teacher and student must assign positive mass to retained candidates")

    teacher = [candidate_probs_teacher[index] / teacher_mass for index in known]
    student = [candidate_probs_student[index] / student_mass for index in known]
    known_labels = [class_labels[index] for index in known]

    q_t = aggregate_class_probabilities(teacher, known_labels, classes)
    q_s = aggregate_class_probabilities(student, known_labels, classes)

    p_t = [q_t[c] for c in classes]
    p_s = [q_s[c] for c in classes]
    kl_inter = kl_divergence(p_t, p_s)

    kl_intra = 0.0
    for cls in classes:
        q_t_e = q_t[cls]
        if q_t_e <= 1e-12:
            continue
        teacher_cond: List[float] = []
        student_cond: List[float] = []
        for prob_t, prob_s, label in zip(teacher, student, known_labels):
            if label != cls:
                continue
            teacher_cond.append(prob_t)
            student_cond.append(prob_s)
        if not teacher_cond:
            continue
        t_sum = sum(teacher_cond)
        s_sum = sum(student_cond)
        if t_sum <= 1e-12:
            continue
        t_norm = [x / t_sum for x in teacher_cond]
        s_norm = [x / s_sum if s_sum > 1e-12 else 1.0 / len(student_cond) for x in student_cond]
        kl_intra += q_t_e * kl_divergence(t_norm, s_norm)

    kl_cand = kl_divergence(
        teacher,
        student,
    )
    return kl_cand, kl_inter, kl_intra


def torch_kl_div(
    log_probs_student: torch.Tensor,
    probs_teacher: torch.Tensor,
) -> torch.Tensor:
    """KL(teacher || student) with teacher probs and student log-probs."""
    return F.kl_div(log_probs_student, probs_teacher, reduction="batchmean")
