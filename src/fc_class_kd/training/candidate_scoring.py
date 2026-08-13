"""Differentiable candidate sequence scoring for Candidate/Class KD."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from fc_class_kd.prompting.templates import encode_candidate_pair
from fc_class_kd.training.sft import EncodedSFTExample, collate_sft_examples
from fc_class_kd.types import CandidateRecord, FCRecord


def sequence_scores_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    output_mask: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Score output tokens under a causal LM while preserving the gradient graph."""
    if logits.ndim != 3 or input_ids.ndim != 2 or output_mask.shape != input_ids.shape:
        raise ValueError("expected logits [B,L,V] and input_ids/output_mask [B,L]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logit and token sequence shapes must agree")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")

    shifted_logits = logits[:, :-1, :].float()
    shifted_targets = input_ids[:, 1:]
    shifted_mask = output_mask[:, 1:].to(shifted_logits.dtype)
    token_counts = shifted_mask.sum(dim=-1)
    if torch.any(token_counts <= 0):
        raise ValueError("every candidate must contain at least one scored output token")
    token_log_probs = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=shifted_targets.unsqueeze(-1),
    ).squeeze(-1)
    return (token_log_probs * shifted_mask).sum(dim=-1) / token_counts.pow(alpha)


def score_candidate_sequences(
    model: Any,
    tokenizer: Any,
    record: FCRecord,
    candidates: Sequence[CandidateRecord],
    max_length: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Return differentiable scores [K] for a frozen candidate pool."""
    if not candidates:
        raise ValueError("cannot score an empty candidate pool")
    encoded: list[EncodedSFTExample] = []
    for index, candidate in enumerate(candidates):
        if candidate.sample_id != record.id:
            raise ValueError(
                f"candidate {index} belongs to {candidate.sample_id}, expected {record.id}"
            )
        input_ids, labels = encode_candidate_pair(
            tokenizer,
            record,
            candidate.serialized_call,
            max_length=max_length,
        )
        encoded.append(
            EncodedSFTExample(
                sample_id=f"{record.id}:candidate:{index}",
                input_ids=input_ids,
                labels=labels,
            )
        )
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must have a pad_token_id")
    batch = collate_sft_examples(
        encoded,
        tokenizer.pad_token_id,
        device=next(model.parameters()).device,
    )
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    return sequence_scores_from_logits(
        output.logits,
        batch["input_ids"],
        batch["labels"] != -100,
        alpha=alpha,
    )
