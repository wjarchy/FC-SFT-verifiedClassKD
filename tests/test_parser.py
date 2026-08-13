"""Tests for verifier parser."""

from fc_class_kd.types import FunctionCall
from fc_class_kd.verifier.parser import (
    extract_function_call_surface,
    parse_function_call,
    serialize_function_call,
)


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


def test_parse_tool_wrapper_with_braces_inside_string():
    text = (
        '<tool_call>\n{"name":"check","arguments":{"value":"[]{}()"}}\n'
        '</tool_call><|im_end|>'
    )
    call, err = parse_function_call(text)
    assert err is None
    assert call == FunctionCall(name="check", arguments={"value": "[]{}()"})
    assert extract_function_call_surface(text) == (
        '{"name":"check","arguments":{"value":"[]{}()"}}'
    )


def test_parse_tool_wrapper_with_deeply_nested_arguments():
    text = (
        '<tool_call>{"name":"search","arguments":'
        '{"filters":{"country":{"code":"CN"}},"items":[{"id":1}]}}'
        '</tool_call>'
    )
    call, err = parse_function_call(text)
    assert err is None
    assert call is not None
    assert call.arguments["filters"]["country"]["code"] == "CN"


def test_serialize_roundtrip():
    call = FunctionCall(name="f", arguments={"a": 1, "b": 2})
    text = serialize_function_call(call)
    parsed, _ = parse_function_call(text)
    assert parsed == call
