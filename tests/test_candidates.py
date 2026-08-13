"""Tests for rule-based candidates."""

from fc_class_kd.candidates.rules import build_rule_candidates
from fc_class_kd.types import FCRecord, FunctionCall, ToolParameter, ToolSchema, VerifierClass
from fc_class_kd.verifier.classifier import classify_call


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
        res = classify_call(c.serialized_call, rec.tools, rec.gold_call)
        labels.add(res.class_label)
    assert VerifierClass.CORRECT in labels
    assert VerifierClass.WRONG_FUNCTION in labels or VerifierClass.WRONG_ARGUMENTS in labels


def test_key_reorder_is_preserved_as_distinct_surface_candidate():
    rec = _record()
    cands = build_rule_candidates(rec, seed=42)
    gold = next(c for c in cands if c.source == "gold")
    reordered = next(c for c in cands if c.source == "strict_equivalent_key_reorder")

    assert gold.serialized_call != reordered.serialized_call
    assert gold.parsed_call == reordered.parsed_call
    assert classify_call(
        reordered.serialized_call, rec.tools, rec.gold_call
    ).class_label == VerifierClass.CORRECT


def test_nested_arguments_can_be_deduplicated_without_hash_errors():
    tools = [
        ToolSchema(
            name="search",
            parameters={
                "filters": ToolParameter(type="object", required=True),
                "tags": ToolParameter(type="array", required=True, unordered=True),
            },
        ),
        ToolSchema(name="other", parameters={}),
    ]
    rec = FCRecord(
        id="nested",
        source="test",
        query="q",
        tools=tools,
        gold_call=FunctionCall(
            name="search",
            arguments={"filters": {"country": "CN"}, "tags": ["a", "b"]},
        ),
    )

    cands = build_rule_candidates(rec, seed=42)
    surfaces = [candidate.serialized_call for candidate in cands]
    assert len(surfaces) == len(set(surfaces))


def test_wrong_function_required_array_and_object_get_matching_top_level_types():
    rec = FCRecord(
        id="complex_distractor",
        source="test",
        query="q",
        tools=[
            ToolSchema(name="gold"),
            ToolSchema(
                name="other",
                parameters={
                    "items": ToolParameter(type="array", required=True),
                    "filters": ToolParameter(type="object", required=True),
                },
            ),
        ],
        gold_call=FunctionCall(name="gold", arguments={}),
    )
    candidate = next(
        item for item in build_rule_candidates(rec) if item.source == "wrong_function"
    )
    assert candidate.parsed_call is not None
    assert candidate.parsed_call.arguments == {"items": [], "filters": {}}
