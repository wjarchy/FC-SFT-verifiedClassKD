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
        return 0.0
    length = len(token_logprobs)
    return sum(token_logprobs) / (length ** alpha)


def candidate_probabilities(
    scores: Sequence[float],
    temperature: float = 1.0,
) -> List[float]:
    """Softmax over candidate sequence scores."""
    if len(scores) == 0:
        return []
    t = max(temperature, 1e-8)
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
    allowed = set(classes)
    totals = {c: 0.0 for c in classes}
    for prob, label in zip(candidate_probs, class_labels):
        if label not in allowed:
            continue
        totals[label] += prob
    return totals


def class_distribution_tensor(
    candidate_probs: torch.Tensor,
    class_indices: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Batched class aggregation: candidate_probs [B,K], class_indices [B,K]."""
    batch_size = candidate_probs.shape[0]
    out = torch.zeros(batch_size, num_classes, device=candidate_probs.device)
    for b in range(batch_size):
        for k in range(candidate_probs.shape[1]):
            idx = int(class_indices[b, k].item())
            if idx < 0:
                continue
            out[b, idx] += candidate_probs[b, k]
    return out


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """KL(P || Q) for discrete distributions."""
    total = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
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
    q_t = aggregate_class_probabilities(candidate_probs_teacher, class_labels, classes)
    q_s = aggregate_class_probabilities(candidate_probs_student, class_labels, classes)

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
        for prob_t, prob_s, label in zip(
            candidate_probs_teacher, candidate_probs_student, class_labels
        ):
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
        list(candidate_probs_teacher),
        list(candidate_probs_student),
    )
    return kl_cand, kl_inter, kl_intra


def torch_kl_div(
    log_probs_student: torch.Tensor,
    probs_teacher: torch.Tensor,
) -> torch.Tensor:
    """KL(teacher || student) with teacher probs and student log-probs."""
    return F.kl_div(log_probs_student, probs_teacher, reduction="batchmean")
