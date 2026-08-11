"""Tests for equivalence variant builder."""

from fc_class_kd.data.equivalence import build_equiv_variants
from fc_class_kd.types import FCRecord, FunctionCall, ToolParameter, ToolSchema
from fc_class_kd.verifier.normalizer import calls_equivalent


def _rec():
    tools = [
        ToolSchema(
            name="get_weather",
            parameters={
                "city": ToolParameter(type="string", required=True),
                "days": ToolParameter(type="integer", required=False, default=1),
            },
        ),
        ToolSchema(name="other", parameters={"x": ToolParameter(type="string", required=True)}),
    ]
    gold = FunctionCall(name="get_weather", arguments={"city": "Chengdu", "days": 3})
    return FCRecord(id="t1", source="test", query="q", tools=tools, gold_call=gold)


def test_build_equiv_variants_nonempty():
    variants = build_equiv_variants(_rec())
    types = {v["variant_type"] for v in variants}
    assert "key_reorder" in types


def test_non_default_optional_value_is_not_omitted_as_equivalent():
    variants = build_equiv_variants(_rec())
    types = {v["variant_type"] for v in variants}
    assert "implicit_default" not in types


def test_explicit_schema_default_can_be_safely_omitted():
    rec = _rec()
    rec.gold_call.arguments["days"] = 1
    variants = build_equiv_variants(rec)
    implicit = next(v for v in variants if v["variant_type"] == "implicit_default")
    candidate = FunctionCall.model_validate(implicit["gold_call"])
    assert calls_equivalent(candidate, rec.gold_call, rec.tools)
