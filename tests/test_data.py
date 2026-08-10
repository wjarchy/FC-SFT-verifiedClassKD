"""Tests for data adapters (synthetic rows, no download)."""

import json

from fc_class_kd.data.adapters import xlam_row_to_record
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
