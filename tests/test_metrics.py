"""Tests for evaluation fail-fast invariants."""

import pytest

from fc_class_kd.evaluation.metrics import evaluate_predictions
from fc_class_kd.types import FCRecord, FunctionCall, ToolSchema


def test_evaluation_rejects_missing_predictions():
    records = [
        FCRecord(
            id="one",
            source="test",
            query="q",
            tools=[ToolSchema(name="f")],
            gold_call=FunctionCall(name="f", arguments={}),
        )
    ]
    with pytest.raises(ValueError, match="Prediction count mismatch"):
        evaluate_predictions(records, [])
