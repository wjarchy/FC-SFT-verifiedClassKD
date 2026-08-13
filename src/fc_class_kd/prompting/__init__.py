"""Prompting package."""

from fc_class_kd.prompting.templates import (
    apply_chat_template,
    build_training_messages,
    encode_candidate_pair,
    encode_sft_pair,
    record_to_openai_tools,
    render_candidate_completion,
)

__all__ = [
    "apply_chat_template",
    "build_training_messages",
    "encode_candidate_pair",
    "encode_sft_pair",
    "record_to_openai_tools",
    "render_candidate_completion",
]
