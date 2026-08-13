"""Tests for persisted S0 prediction rows."""

import pytest

from fc_class_kd.evaluation.predictions import (
    candidate_surface_from_prediction,
    make_prediction_row,
    validate_prediction_rows,
)
from fc_class_kd.types import FCRecord, FunctionCall, ToolSchema


def _record(sample_id: str = "one") -> FCRecord:
    return FCRecord(
        id=sample_id,
        source="test",
        query="q",
        tools=[ToolSchema(name="f"), ToolSchema(name="g")],
        gold_call=FunctionCall(name="f", arguments={}),
    )


def test_candidate_surface_preserves_nested_json_from_tool_wrapper():
    prediction = (
        '<tool_call>\n{"name":"f","arguments":{"x":{"literal":"{}"}}}'
        '\n</tool_call><|im_end|>'
    )
    assert candidate_surface_from_prediction(prediction) == (
        '{"name":"f","arguments":{"x":{"literal":"{}"}}}'
    )


def test_invalid_tool_body_is_kept_without_double_wrapper():
    assert candidate_surface_from_prediction(
        '<tool_call>{"name":</tool_call><|im_end|>'
    ) == '{"name":'


def test_prediction_prefix_check_rejects_reordered_resume_file():
    records = [_record("one"), _record("two")]
    rows = [
        make_prediction_row(records[1], "train", '{"name":"f","arguments":{}}', 3, 4)
    ]
    with pytest.raises(ValueError, match="exact split prefix"):
        validate_prediction_rows(records, rows, "train", require_complete=False)
