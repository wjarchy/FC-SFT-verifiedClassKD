"""Serialize tools and build chat messages for Qwen3 FC."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from fc_class_kd.types import FCRecord, FunctionCall, ToolSchema


def tool_schema_to_openai(tool: ToolSchema) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in tool.parameters.items():
        prop: Dict[str, Any] = {"type": _map_type(param.type), "description": param.description or ""}
        if param.enum is not None:
            prop["enum"] = param.enum
        if param.default is not None:
            prop["default"] = param.default
        properties[name] = prop
        if param.required:
            required.append(name)
    params: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        params["required"] = required
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": params,
        },
    }


def _map_type(raw: str) -> str:
    mapping = {
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "str": "string",
        "string": "string",
        "list": "array",
        "array": "array",
        "dict": "object",
        "object": "object",
    }
    return mapping.get(raw.lower(), "string")


def record_to_openai_tools(record: FCRecord) -> List[Dict[str, Any]]:
    return [tool_schema_to_openai(t) for t in record.tools]


def build_user_message(record: FCRecord) -> str:
    return record.query.strip()


def build_training_messages(record: FCRecord, gold_call: FunctionCall) -> List[Dict[str, Any]]:
    """Messages for SFT: user query + assistant tool call (OpenAI-style)."""
    return [
        {"role": "user", "content": build_user_message(record)},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": gold_call.name,
                        "arguments": json.dumps(gold_call.arguments, ensure_ascii=False),
                    },
                }
            ],
        },
    ]


def apply_chat_template(
    tokenizer: Any,
    record: FCRecord,
    gold_call: Optional[FunctionCall] = None,
    add_generation_prompt: bool = False,
) -> str:
    tools = record_to_openai_tools(record)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": build_user_message(record)}]
    if gold_call is not None:
        messages = build_training_messages(record, gold_call)
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def encode_sft_pair(
    tokenizer: Any,
    record: FCRecord,
    max_length: int = 1024,
) -> Tuple[List[int], List[int]]:
    """
  Full sequence with labels; only assistant FC tokens have label != -100.
  """
    full_text = apply_chat_template(tokenizer, record, record.gold_call, add_generation_prompt=False)
    prompt_text = apply_chat_template(
        tokenizer, record, add_generation_prompt=True
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        raise ValueError(
            f"encoded sample {record.id} has {len(full_ids)} tokens and would truncate "
            f"the assistant call at max_length={max_length}"
        )
    if len(prompt_ids) >= len(full_ids):
        raise ValueError(f"sample {record.id} has no assistant output tokens to supervise")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"generation prompt tokens are not a prefix of the full training sequence for {record.id}"
        )

    labels = [-100] * len(full_ids)
    for i in range(len(prompt_ids), len(full_ids)):
        labels[i] = full_ids[i]
    return full_ids, labels
