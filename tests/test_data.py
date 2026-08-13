"""Tests for data adapters (synthetic rows, no download)."""

import json

import pytest

from fc_class_kd.data.adapters import (
    load_toolace_qwen_records,
    toolace_openai_row_to_record,
    toolace_original_system_to_schemas,
    xlam_row_to_record,
)
from fc_class_kd.data.filters import passes_fast_pilot_filter


def test_xlam_adapter_single_call():
    tools = [
        {
            "name": "get_weather",
            "description": "weather",
            "parameters": {
                "city": {"type": "string", "required": True, "description": "city"},
            },
        },
        {
            "name": "get_news",
            "description": "news",
            "parameters": {
                "q": {"type": "string", "required": True, "description": "query"},
            },
        },
    ]
    row = {
        "query": "Weather in Chengdu?",
        "tools": json.dumps(tools),
        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Chengdu"}}]),
    }
    rec = xlam_row_to_record(row, 0)
    assert rec is not None
    assert rec.gold_call.name == "get_weather"
    assert passes_fast_pilot_filter(rec)


def test_xlam_rejects_multi_answer():
    row = {
        "query": "q",
        "tools": json.dumps([{"name": "a", "parameters": {}}, {"name": "b", "parameters": {}}]),
        "answers": json.dumps(
            [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]
        ),
    }
    assert xlam_row_to_record(row, 1) is None


def test_xlam_actual_schema_shape_infers_required_and_normalizes_types():
    tools = [
        {
            "name": "calculate",
            "parameters": {
                "values": {"type": "List[int]"},
                "limit": {"type": "int, optional", "default": ""},
                "tags": {"type": "set"},
            },
        },
        {"name": "other", "parameters": {}},
    ]
    row = {
        "query": "Calculate values with tags",
        "tools": json.dumps(tools),
        "answers": json.dumps(
            [{"name": "calculate", "arguments": {"values": [1, 2], "tags": ["a"]}}]
        ),
    }
    record = xlam_row_to_record(row, 2)
    assert record is not None
    params = record.tools[0].parameters
    assert params["values"].type == "array"
    assert params["values"].required is True
    assert params["limit"].type == "integer"
    assert params["limit"].required is False
    assert params["limit"].default is None
    assert params["tags"].unordered is True
    assert passes_fast_pilot_filter(record)


def test_xlam_rejects_unsupported_callable_schema():
    row = {
        "query": "Differentiate a function",
        "tools": json.dumps(
            [
                {
                    "name": "differentiate",
                    "parameters": {"fn": {"type": "Callable[[float], float]"}},
                },
                {"name": "other", "parameters": {}},
            ]
        ),
        "answers": json.dumps(
            [{"name": "differentiate", "arguments": {"fn": "x ** 2"}}]
        ),
    }
    assert xlam_row_to_record(row, 3) is None


def test_toolace_original_schema_restores_required_without_guessing_defaults():
    system = '''prefix
Here is a list of functions in JSON format that you can invoke:
[{"name":"weather","parameters":{"type":"dict","properties":{"city":{"type":"string"},"days":{"type":"int"}},"required":["city"]},"required":null}]
trailing text'''
    tools = toolace_original_system_to_schemas(system)
    assert len(tools) == 1
    assert tools[0].parameters["city"].required is True
    assert tools[0].parameters["days"].required is False
    assert tools[0].parameters["city"].default is None


def test_toolace_original_schema_preserves_union_types():
    system = '''prefix
Here is a list of functions in JSON format that you can invoke:
[{"name":"lookup","parameters":{"type":"dict","properties":{"limit":{"type":["integer","null"]},"key":{"type":["integer","string"]}},"required":["key"]}}]
trailing text'''
    tools = toolace_original_system_to_schemas(system)
    assert tools[0].parameters["limit"].type == "integer|null"
    assert tools[0].parameters["key"].type == "integer|string"


def test_cleaned_toolace_loss_is_present_in_upstream_shaped_row():
    row = {
        "conversations": [
            {"role": "user", "content": "Find nearby places"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_nearby",
                            "arguments": '{"lat": 30.0}',
                        }
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "search_nearby",
                "parameters": {
                    # This reproduces the lossy fields stored in the upstream cleaned
                    # parquet: no required list and an invalid empty-string float default.
                    "lat": {"type": "float", "default": ""},
                },
            },
            {"name": "other_tool", "parameters": {}},
        ],
    }
    record = toolace_openai_row_to_record(row, 0)
    assert record is not None
    assert record.tools[0].parameters["lat"].required is False
    assert record.tools[0].parameters["lat"].default == ""


def test_experiment_loader_refuses_cleaned_toolace_without_schema_restoration():
    with pytest.raises(ValueError, match="lossy required/default"):
        load_toolace_qwen_records(schema_revision=None)
