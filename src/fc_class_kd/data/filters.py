"""Record-level filters for Fast Pilot data criteria."""

from __future__ import annotations

from fc_class_kd.types import FCRecord
from fc_class_kd.verifier.classifier import classify_function_call
from fc_class_kd.verifier.parser import parse_function_call, serialize_function_call


def gold_is_parseable(record: FCRecord) -> bool:
    text = serialize_function_call(record.gold_call)
    call, err = parse_function_call(text)
    return call is not None and err is None


def gold_is_correct(record: FCRecord) -> bool:
    result = classify_function_call(record.gold_call, record.tools, record.gold_call)
    return result.class_label.value == "correct"


def passes_fast_pilot_filter(record: FCRecord, min_tools: int = 2) -> bool:
    if len(record.tools) < min_tools:
        return False
    if not record.query.strip():
        return False
    if not gold_is_parseable(record):
        return False
    if not gold_is_correct(record):
        return False
    return True
