"""Tests for equivalence variant builder."""

from fc_class_kd.data.equivalence import build_equiv_variants
from fc_class_kd.types import FCRecord, FunctionCall, ToolParameter, ToolSchema


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
