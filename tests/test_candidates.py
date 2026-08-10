"""Tests for rule-based candidates."""

from fc_class_kd.candidates.rules import build_rule_candidates
from fc_class_kd.types import FunctionCall, ToolParameter, ToolSchema, VerifierClass
from fc_class_kd.verifier.classifier import classify_function_call


def _record():
    from fc_class_kd.types import FCRecord

    tools = [
        ToolSchema(
            name="get_weather",
            parameters={
                "city": ToolParameter(type="string", required=True),
                "days": ToolParameter(type="integer", required=False),
            },
        ),
        ToolSchema(
            name="get_air",
            parameters={"city": ToolParameter(type="string", required=True)},
        ),
    ]
    gold = FunctionCall(name="get_weather", arguments={"city": "Chengdu", "days": 3})
    return FCRecord(id="t1", source="test", query="q", tools=tools, gold_call=gold)


def test_build_rule_candidates_has_error_types():
    rec = _record()
    cands = build_rule_candidates(rec, seed=42)
    assert len(cands) >= 3
    labels = set()
    for c in cands:
        res = classify_function_call(c, rec.tools, rec.gold_call)
        labels.add(res.class_label)
    assert VerifierClass.CORRECT in labels
    assert VerifierClass.WRONG_FUNCTION in labels or VerifierClass.WRONG_ARGUMENTS in labels
