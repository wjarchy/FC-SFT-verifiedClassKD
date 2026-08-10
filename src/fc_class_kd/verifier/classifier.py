"""Verifier classifier: map parsed calls to VerifierClass."""

from __future__ import annotations

from typing import List, Optional

from fc_class_kd.types import FunctionCall, ToolSchema, VerifierClass, VerifierResult
from fc_class_kd.verifier.normalizer import calls_equivalent, canonicalize_call
from fc_class_kd.verifier.parser import parse_function_call
from fc_class_kd.verifier.schema_checker import check_schema


def _sub_label_from_issues(issues: List[str]) -> Optional[str]:
    joined = ",".join(issues)
    if "missing_required" in joined:
        return "missing_required"
    if "unknown_keys" in joined:
        return "wrong_key"
    if "wrong_type" in joined:
        return "wrong_type"
    if "wrong_enum" in joined:
        return "wrong_value"
    return None


def classify_call(
    raw_text: str,
    tools: List[ToolSchema],
    gold_call: FunctionCall,
) -> VerifierResult:
    call, err = parse_function_call(raw_text)
    if call is None:
        return VerifierResult(
            class_label=VerifierClass.INVALID,
            sub_label="parse_error",
            parse_ok=False,
            details=[err or "parse_failed"],
        )

    valid, issues, kind = check_schema(call, tools)
    if not valid:
        if kind == "wrong_function":
            label = VerifierClass.WRONG_FUNCTION
            sub = None
        elif kind == "wrong_arguments":
            label = VerifierClass.WRONG_ARGUMENTS
            sub = _sub_label_from_issues(issues)
        else:
            label = VerifierClass.INVALID
            sub = "schema_invalid"
        return VerifierResult(
            class_label=label,
            sub_label=sub,
            parse_ok=True,
            details=issues,
        )

    canon = canonicalize_call(call, tools, gold_call)
    if call.name != gold_call.name:
        return VerifierResult(
            class_label=VerifierClass.WRONG_FUNCTION,
            canonical_call=canon,
            parse_ok=True,
            details=["function_name_mismatch"],
        )

    if calls_equivalent(call, gold_call, tools):
        return VerifierResult(
            class_label=VerifierClass.CORRECT,
            canonical_call=canon,
            parse_ok=True,
            details=[],
        )

    # Schema-valid but not equivalent to gold — value mismatch
    return VerifierResult(
        class_label=VerifierClass.WRONG_ARGUMENTS,
        sub_label="wrong_value",
        canonical_call=canon,
        parse_ok=True,
        details=["schema_valid_but_not_equivalent"],
    )


def classify_function_call(
    call: FunctionCall,
    tools: List[ToolSchema],
    gold_call: FunctionCall,
) -> VerifierResult:
    from fc_class_kd.verifier.parser import serialize_function_call

    return classify_call(serialize_function_call(call), tools, gold_call)
