"""Prompting package."""

from fc_class_kd.prompting.templates import (
    apply_chat_template,
    build_training_messages,
    encode_sft_pair,
    record_to_openai_tools,
)

__all__ = [
    "apply_chat_template",
    "build_training_messages",
    "encode_sft_pair",
    "record_to_openai_tools",
]
