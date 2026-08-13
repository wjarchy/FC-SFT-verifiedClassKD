"""Small, explicit SFT utilities shared by smoke and Tiny Overfit runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from fc_class_kd.prompting.templates import apply_chat_template, encode_sft_pair
from fc_class_kd.types import FCRecord


@dataclass(frozen=True)
class EncodedSFTExample:
    sample_id: str
    input_ids: list[int]
    labels: list[int]


def encode_sft_records(
    tokenizer: Any,
    records: Sequence[FCRecord],
    max_length: int,
) -> list[EncodedSFTExample]:
    """Encode records without silently dropping or truncating any sample."""
    examples: list[EncodedSFTExample] = []
    for record in records:
        input_ids, labels = encode_sft_pair(tokenizer, record, max_length=max_length)
        examples.append(
            EncodedSFTExample(
                sample_id=record.id,
                input_ids=input_ids,
                labels=labels,
            )
        )
    return examples


def collate_sft_examples(
    examples: Sequence[EncodedSFTExample],
    pad_token_id: int,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad an SFT batch while keeping padding out of the CE loss."""
    if not examples:
        raise ValueError("cannot collate an empty SFT batch")
    max_len = max(len(example.input_ids) for example in examples)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    for example in examples:
        if len(example.input_ids) != len(example.labels):
            raise ValueError(f"input/label length mismatch for {example.sample_id}")
        padding = max_len - len(example.input_ids)
        input_rows.append(example.input_ids + [pad_token_id] * padding)
        label_rows.append(example.labels + [-100] * padding)
        attention_rows.append([1] * len(example.input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
    }


@torch.inference_mode()
def generate_function_call(
    model: Any,
    tokenizer: Any,
    record: FCRecord,
    max_new_tokens: int = 192,
) -> str:
    """Greedy direct generation; return only the newly generated assistant text."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt = apply_chat_template(tokenizer, record, add_generation_prompt=True)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]
    generation_config = deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generated = model.generate(
        **encoded,
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=False)
