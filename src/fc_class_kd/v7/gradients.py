"""Gradient geometry helpers for the V7 mechanism diagnostic."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch


def gradient_geometry(
    candidate: Sequence[torch.Tensor],
    inter: Sequence[torch.Tensor],
    intra: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """Summarize norms, decomposition residual, and inter/intra cosine."""
    if not candidate or len(candidate) != len(inter) or len(candidate) != len(intra):
        raise ValueError("gradient lists must be non-empty and aligned")
    totals = {
        "candidate_sq": 0.0,
        "inter_sq": 0.0,
        "intra_sq": 0.0,
        "residual_sq": 0.0,
        "inter_intra_dot": 0.0,
    }
    for cand, between, within in zip(candidate, inter, intra, strict=True):
        if cand.shape != between.shape or cand.shape != within.shape:
            raise ValueError("aligned gradient tensors must have identical shapes")
        values = (cand, between, within)
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError("gradient tensors must be finite")
        cand64 = cand.detach().double()
        inter64 = between.detach().double()
        intra64 = within.detach().double()
        totals["candidate_sq"] += float(torch.sum(cand64 * cand64))
        totals["inter_sq"] += float(torch.sum(inter64 * inter64))
        totals["intra_sq"] += float(torch.sum(intra64 * intra64))
        residual = cand64 - inter64 - intra64
        totals["residual_sq"] += float(torch.sum(residual * residual))
        totals["inter_intra_dot"] += float(torch.sum(inter64 * intra64))
    candidate_norm = math.sqrt(totals["candidate_sq"])
    inter_norm = math.sqrt(totals["inter_sq"])
    intra_norm = math.sqrt(totals["intra_sq"])
    denominator = inter_norm * intra_norm
    cosine = None if denominator == 0 else totals["inter_intra_dot"] / denominator
    return {
        "grad_candidate_norm": candidate_norm,
        "grad_class_norm": inter_norm,
        "grad_inter_norm": inter_norm,
        "grad_intra_norm": intra_norm,
        "candidate_over_class": None if inter_norm == 0 else candidate_norm / inter_norm,
        "intra_over_inter": None if inter_norm == 0 else intra_norm / inter_norm,
        "cosine_inter_intra": cosine,
        "decomposition_residual_norm": math.sqrt(totals["residual_sq"]),
    }
