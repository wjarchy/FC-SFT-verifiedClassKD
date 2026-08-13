"""Unit tests for reusable SFT batching helpers."""

import pytest
import torch

from fc_class_kd.training.sft import EncodedSFTExample, collate_sft_examples


def test_collate_sft_examples_masks_right_padding():
    examples = [
        EncodedSFTExample("a", [1, 2, 3], [-100, 2, 3]),
        EncodedSFTExample("b", [4, 5], [-100, 5]),
    ]
    batch = collate_sft_examples(examples, pad_token_id=0)
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([[-100, 2, 3], [-100, 5, -100]]))


def test_collate_sft_examples_rejects_bad_shapes_and_empty_batches():
    with pytest.raises(ValueError, match="empty"):
        collate_sft_examples([], pad_token_id=0)
    with pytest.raises(ValueError, match="length mismatch"):
        collate_sft_examples(
            [EncodedSFTExample("bad", [1, 2], [-100])],
            pad_token_id=0,
        )
