"""Tests for verifier classifier and schema checker."""

from fc_class_kd.types import FunctionCall, ToolParameter, ToolSchema, VerifierClass
from fc_class_kd.verifier.classifier import classify_call, classify_function_call
from fc_class_kd.verifier.normalizer import calls_equivalent, canonicalize_call


def _weather_tools():
    return [
        ToolSchema(
            name="get_weather",
            parameters={
                "city": ToolParameter(type="string", required=True),
                "days": ToolParameter(type="integer", required=False, default=1),
            },
        ),
        ToolSchema(
            name="get_air_quality",
            parameters={"city": ToolParameter(type="string", required=True)},
        ),
    ]


def test_gold_is_correct():
    tools = _weather_tools()
    gold = FunctionCall(name="get_weather", arguments={"city": "Chengdu", "days": 3})
    res = classify_function_call(gold, tools, gold)
    assert res.class_label == VerifierClass.CORRECT


def test_wrong_function():
    tools = _weather_tools()
    gold = FunctionCall(name="get_weather", arguments={"city": "Chengdu"})
    wrong = FunctionCall(name="get_air_quality", arguments={"city": "Chengdu"})
    res = classify_function_call(wrong, tools, gold)
    assert res.class_label == VerifierClass.WRONG_FUNCTION


def test_wrong_function_takes_precedence_over_bad_distractor_arguments():
    tools = [
        ToolSchema(
            name="gold",
            parameters={"city": ToolParameter(type="string", required=True)},
        ),
        ToolSchema(
            name="distractor",
            parameters={"count": ToolParameter(type="integer", required=True)},
        ),
    ]
    res = classify_call(
        '{"name":"distractor","arguments":{"count":"not-an-int"}}',
        tools,
        FunctionCall(name="gold", arguments={"city": "Chengdu"}),
    )
    assert res.class_label == VerifierClass.WRONG_FUNCTION


def test_missing_required():
    tools = _weather_tools()
    gold = FunctionCall(name="get_weather", arguments={"city": "Chengdu"})
    missing = FunctionCall(name="get_weather", arguments={})
    res = classify_function_call(missing, tools, gold)
    assert res.class_label == VerifierClass.WRONG_ARGUMENTS
    assert res.sub_label == "missing_required"


def test_key_reorder_equivalent():
    tools = _weather_tools()
    gold = FunctionCall(name="get_weather", arguments={"city": "Chengdu", "days": 3})
    reordered = FunctionCall(name="get_weather", arguments={"days": 3, "city": "Chengdu"})
    assert calls_equivalent(reordered, gold, tools)
    res = classify_function_call(reordered, tools, gold)
    assert res.class_label == VerifierClass.CORRECT


def test_canonicalization_idempotent():
    tools = _weather_tools()
    call = FunctionCall(name="get_weather", arguments={"city": "Chengdu"})
    c1 = canonicalize_call(call, tools, call)
    c2 = canonicalize_call(c1, tools, call)
    assert c1 == c2


def test_date_looking_strings_are_not_equivalent_without_explicit_format():
    tools = [
        ToolSchema(
            name="lookup",
            parameters={"key": ToolParameter(type="string", required=True)},
        )
    ]
    gold = FunctionCall(name="lookup", arguments={"key": "2026/08/13"})
    changed = FunctionCall(name="lookup", arguments={"key": "2026-08-13"})
    assert not calls_equivalent(changed, gold, tools)


def test_json_schema_union_type_accepts_each_member_and_rejects_others():
    tools = [
        ToolSchema(
            name="lookup",
            parameters={"key": ToolParameter(type="integer|string", required=True)},
        )
    ]
    gold = FunctionCall(name="lookup", arguments={"key": 7})
    assert classify_function_call(gold, tools, gold).class_label == VerifierClass.CORRECT
    string_call = FunctionCall(name="lookup", arguments={"key": "seven"})
    assert classify_function_call(string_call, tools, string_call).class_label == (
        VerifierClass.CORRECT
    )
    wrong = FunctionCall(name="lookup", arguments={"key": [7]})
    assert classify_function_call(wrong, tools, gold).class_label == VerifierClass.WRONG_ARGUMENTS
