"""Tests for prompt/assistant label boundaries."""

import pytest

from fc_class_kd.prompting.templates import encode_candidate_pair, encode_sft_pair
from fc_class_kd.types import FCRecord, FunctionCall, ToolSchema


class _FakeTokenizer:
    def __init__(self, prompt: str = "PROMPT", full: str = "PROMPTANSWER") -> None:
        self.prompt = prompt
        self.full = full
        self.thinking_flags: list[bool] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tools
        assert tokenize is False
        self.thinking_flags.append(enable_thinking)
        return self.prompt if add_generation_prompt else self.full

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


def _record() -> FCRecord:
    return FCRecord(
        id="prompt",
        source="test",
        query="q",
        tools=[ToolSchema(name="f")],
        gold_call=FunctionCall(name="f", arguments={}),
    )


def test_sft_mask_labels_only_assistant_suffix_and_disables_thinking():
    tokenizer = _FakeTokenizer()
    input_ids, labels = encode_sft_pair(tokenizer, _record(), max_length=64)
    prompt_len = len(tokenizer.prompt)
    assert labels[:prompt_len] == [-100] * prompt_len
    assert labels[prompt_len:] == input_ids[prompt_len:]
    assert tokenizer.thinking_flags == [False, False]


def test_sft_mask_rejects_nonprefix_template_boundary():
    tokenizer = _FakeTokenizer(prompt="PROMPT", full="DIFFERENTANSWER")
    with pytest.raises(ValueError, match="not a prefix"):
        encode_sft_pair(tokenizer, _record(), max_length=64)


def test_sft_encoding_rejects_gold_truncation():
    tokenizer = _FakeTokenizer()
    with pytest.raises(ValueError, match="would truncate"):
        encode_sft_pair(tokenizer, _record(), max_length=8)


def test_candidate_encoding_preserves_arbitrary_surface_and_masks_prompt():
    tokenizer = _FakeTokenizer(
        prompt="PROMPT",
        full='PROMPT<tool_call>{"name": "f", "arguments": {}}</tool_call>',
    )
    candidate = '{"arguments": {}, "name": "f"}'
    input_ids, labels = encode_candidate_pair(tokenizer, _record(), candidate, max_length=128)
    rendered = "".join(chr(token) for token in input_ids)
    assert candidate in rendered
    assert labels[: len(tokenizer.prompt)] == [-100] * len(tokenizer.prompt)
    assert labels[len(tokenizer.prompt):] == input_ids[len(tokenizer.prompt):]
