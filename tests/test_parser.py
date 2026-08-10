"""Tests for verifier parser."""

import pytest

from fc_class_kd.verifier.parser import parse_function_call, serialize_function_call
from fc_class_kd.types import FunctionCall


def test_parse_plain_json():
    text = '{"name": "get_weather", "arguments": {"city": "Chengdu"}}'
    call, err = parse_function_call(text)
    assert err is None
    assert call.name == "get_weather"
    assert call.arguments["city"] == "Chengdu"


def test_parse_markdown_block():
    text = 'Here is the call:\n```json\n{"name": "f", "arguments": {}}\n```'
    call, err = parse_function_call(text)
    assert err is None
    assert call.name == "f"


def test_parse_invalid():
    call, err = parse_function_call("not json")
    assert call is None
    assert err is not None


def test_serialize_roundtrip():
    call = FunctionCall(name="f", arguments={"a": 1, "b": 2})
    text = serialize_function_call(call)
    parsed, _ = parse_function_call(text)
    assert parsed == call
