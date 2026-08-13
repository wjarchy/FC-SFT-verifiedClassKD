"""Tests for the exact formal K=6 pool constructor."""

from fc_class_kd.candidates.pool import build_k6_pool
from fc_class_kd.types import FCRecord, FunctionCall, ToolParameter, ToolSchema, VerifierClass


def _record(arguments=None) -> FCRecord:
    if arguments is None:
        arguments = {"city": "Chengdu", "days": 3}
    return FCRecord(
        id="sample",
        source="test",
        query="weather",
        tools=[
            ToolSchema(
                name="weather",
                parameters={
                    "city": ToolParameter(type="string", required=True),
                    "days": ToolParameter(type="integer", required=False, default=1),
                },
            ),
            ToolSchema(name="air", parameters={}),
        ],
        gold_call=FunctionCall(name="weather", arguments=arguments),
    )


def test_k6_contains_all_four_classes_and_real_student_error():
    pool = build_k6_pool(
        _record(),
        split="train",
        student_surface='{"name":"weather","arguments":{"city":"Beijing","days":3}}',
    )
    assert len(pool.candidates) == 6
    assert len({item.serialized_call for item in pool.candidates}) == 6
    assert pool.candidates[0].source == "gold"
    assert any(item.source == "s0_real_error" for item in pool.candidates)
    assert {item.verifier.class_label for item in pool.candidates} == set(VerifierClass) - {
        VerifierClass.UNKNOWN
    }


def test_correct_student_output_is_not_mislabeled_as_hard_error():
    pool = build_k6_pool(
        _record(),
        split="diagnostic",
        student_surface='{"name":"weather","arguments":{"city":"Chengdu","days":3}}',
    )
    assert all(item.source != "s0_real_error" for item in pool.candidates)


def test_real_student_error_keeps_provenance_when_equal_to_rule_candidate():
    pool = build_k6_pool(
        _record(),
        split="train",
        student_surface='{"name": "air", "arguments": {}}',
    )
    actual = [item for item in pool.candidates if item.source == "s0_real_error"]
    assert len(actual) == 1
    assert actual[0].verifier.class_label == VerifierClass.WRONG_FUNCTION


def test_zero_argument_gold_still_gets_exact_structured_pool():
    record = FCRecord(
        id="empty",
        source="test",
        query="run",
        tools=[ToolSchema(name="run"), ToolSchema(name="stop")],
        gold_call=FunctionCall(name="run", arguments={}),
    )
    pool = build_k6_pool(record, split="train", student_surface=None)
    assert len(pool.candidates) == 6
    assert VerifierClass.WRONG_ARGUMENTS in {
        item.verifier.class_label for item in pool.candidates
    }
