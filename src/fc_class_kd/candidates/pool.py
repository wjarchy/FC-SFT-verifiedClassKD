"""Deterministic construction and validation of the formal K=6 pools."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from fc_class_kd.candidates.rules import build_rule_candidates
from fc_class_kd.data.equivalence import build_equiv_variants
from fc_class_kd.types import (
    CandidatePoolRecord,
    CandidateRecord,
    FCRecord,
    FunctionCall,
    LabeledCandidate,
    ToolParameter,
    VerifierClass,
)
from fc_class_kd.verifier.classifier import classify_call
from fc_class_kd.verifier.parser import parse_function_call, serialize_function_call
from fc_class_kd.verifier.schema_checker import normalize_param_type


def _candidate(record: FCRecord, source: str, call: FunctionCall) -> CandidateRecord:
    return CandidateRecord(
        sample_id=record.id,
        source=source,
        serialized_call=serialize_function_call(call, sort_keys=False),
        parsed_call=call,
    )


def _wrong_value(value: Any, spec: ToolParameter) -> Any | None:
    if spec.enum:
        for alternative in spec.enum:
            if alternative != value:
                return copy.deepcopy(alternative)
        return None
    kind = normalize_param_type(spec.type)
    if "|" in spec.type:
        kind = normalize_param_type(spec.type.split("|", 1)[0])
    if kind == "string":
        return f"{value}__changed" if isinstance(value, str) else "__changed"
    if kind == "integer":
        return value + 1 if isinstance(value, int) and not isinstance(value, bool) else 1
    if kind == "number":
        return value + 1.0 if isinstance(value, (int, float)) and not isinstance(value, bool) else 1.0
    if kind == "boolean":
        return not value if isinstance(value, bool) else True
    if kind == "array":
        return [*value, "__changed"] if isinstance(value, list) else ["__changed"]
    if kind == "object":
        changed = dict(value) if isinstance(value, dict) else {}
        changed["__changed"] = True
        return changed
    return None


def _wrong_type(value: Any, spec: ToolParameter) -> Any | None:
    accepted = {normalize_param_type(part) for part in spec.type.split("|")}
    choices: list[tuple[str, Any]] = [
        ("string", "__wrong_type__"),
        ("integer", 7),
        ("boolean", True),
        ("array", []),
        ("object", {}),
    ]
    for kind, alternative in choices:
        if kind not in accepted and type(alternative) is not type(value):
            return alternative
    return None


def _formal_rule_candidates(record: FCRecord) -> list[CandidateRecord]:
    candidates = build_rule_candidates(record, seed=42, include_invalid=False)

    # The shared equivalence builder permits only rules already accepted by the
    # verifier.  Keep at most one surface per exact string; selection happens later.
    for variant in build_equiv_variants(record):
        call = FunctionCall.model_validate(variant["gold_call"])
        candidates.append(_candidate(record, f"strict_equivalent_{variant['variant_type']}", call))

    gold_tool = next(tool for tool in record.tools if tool.name == record.gold_call.name)
    for tool in record.tools:
        if tool.name == record.gold_call.name:
            continue
        candidates.append(
            _candidate(
                record,
                "wrong_function",
                FunctionCall(name=tool.name, arguments={}),
            )
        )

    for key, value in record.gold_call.arguments.items():
        spec = gold_tool.parameters.get(key)
        if spec is None:
            continue
        alternative = _wrong_value(value, spec)
        if alternative is not None:
            args = copy.deepcopy(record.gold_call.arguments)
            args[key] = alternative
            candidates.append(
                _candidate(
                    record,
                    f"wrong_value:{key}",
                    FunctionCall(name=record.gold_call.name, arguments=args),
                )
            )
        alternative = _wrong_type(value, spec)
        if alternative is not None:
            args = copy.deepcopy(record.gold_call.arguments)
            args[key] = alternative
            candidates.append(
                _candidate(
                    record,
                    f"wrong_type:{key}",
                    FunctionCall(name=record.gold_call.name, arguments=args),
                )
            )

    # Deterministic schema errors provide meaningful fallbacks even for zero-argument
    # gold calls; unlike random gibberish, their failure mode is explicit.
    for index in range(1, 9):
        args = copy.deepcopy(record.gold_call.arguments)
        args[f"__unexpected_argument_{index}__"] = None
        candidates.append(
            _candidate(
                record,
                f"wrong_key_fallback:{index}",
                FunctionCall(name=record.gold_call.name, arguments=args),
            )
        )

    candidates.append(
        CandidateRecord(
            sample_id=record.id,
            source="invalid_json",
            serialized_call='{"name":',
            parsed_call=None,
        )
    )
    return candidates


def _deduplicate(candidates: Iterable[CandidateRecord]) -> list[CandidateRecord]:
    seen: set[str] = set()
    unique: list[CandidateRecord] = []
    for candidate in candidates:
        if candidate.serialized_call in seen:
            continue
        seen.add(candidate.serialized_call)
        unique.append(candidate)
    return unique


def _label(record: FCRecord, candidate: CandidateRecord) -> tuple[CandidateRecord, Any]:
    result = classify_call(candidate.serialized_call, record.tools, record.gold_call)
    return candidate, result


def build_k6_pool(
    record: FCRecord,
    split: str,
    student_surface: str | None,
) -> CandidatePoolRecord:
    """Build one exact K=6 pool; only an actual S0 error may fill the student slot."""
    if not split:
        raise ValueError("split must be non-empty")
    rules = _formal_rule_candidates(record)
    raw = list(rules)
    if student_surface:
        parsed, _ = parse_function_call(student_surface)
        student = CandidateRecord(
            sample_id=record.id,
            source="s0_real_error",
            serialized_call=student_surface,
            parsed_call=parsed,
        )
        # Preserve the provenance of a real S0 error when it happens to equal a
        # deterministic rule candidate.  Gold stays first and wins exact duplication.
        raw = [rules[0], student, *rules[1:]]
    labeled = [_label(record, candidate) for candidate in _deduplicate(raw)]

    gold = next((item for item in labeled if item[0].source == "gold"), None)
    if gold is None or gold[1].class_label != VerifierClass.CORRECT:
        raise ValueError(f"{record.id}: missing a verifier-correct gold candidate")
    selected = [gold]

    def take(predicate: Any) -> None:
        if len(selected) >= 6:
            return
        used = {item[0].serialized_call for item in selected}
        match = next(
            (
                item
                for item in labeled
                if item[0].serialized_call not in used and predicate(item)
            ),
            None,
        )
        if match is not None:
            selected.append(match)

    take(
        lambda item: item[0].source.startswith("strict_equivalent")
        and item[1].class_label == VerifierClass.CORRECT
    )
    take(
        lambda item: item[0].source == "s0_real_error"
        and item[1].class_label != VerifierClass.CORRECT
    )
    take(
        lambda item: item[0].source == "wrong_function"
        and item[1].class_label == VerifierClass.WRONG_FUNCTION
    )
    take(
        lambda item: item[1].class_label == VerifierClass.WRONG_ARGUMENTS
        and item[0].source != "s0_real_error"
    )
    take(lambda item: item[1].class_label == VerifierClass.INVALID)

    # Fill absent equivalent/student slots with deterministic structured errors.
    for desired in (
        VerifierClass.WRONG_ARGUMENTS,
        VerifierClass.WRONG_FUNCTION,
        VerifierClass.INVALID,
        VerifierClass.CORRECT,
    ):
        while len(selected) < 6:
            before = len(selected)
            take(lambda item, desired=desired: item[1].class_label == desired)
            if len(selected) == before:
                break
    if len(selected) != 6:
        raise ValueError(f"{record.id}: could construct only {len(selected)} unique candidates")

    output: list[LabeledCandidate] = []
    for index, (candidate, result) in enumerate(selected):
        output.append(
            LabeledCandidate(
                **candidate.model_dump(),
                candidate_id=f"{record.id}__c{index}",
                verifier=result,
            )
        )
    pool = CandidatePoolRecord(sample_id=record.id, split=split, candidates=output)
    validate_k6_pool(pool)
    return pool


def validate_k6_pool(pool: CandidatePoolRecord) -> None:
    if len(pool.candidates) != 6:
        raise ValueError(f"{pool.sample_id}: expected K=6")
    surfaces = [candidate.serialized_call for candidate in pool.candidates]
    if len(set(surfaces)) != 6:
        raise ValueError(f"{pool.sample_id}: candidate surfaces are not unique")
    if pool.candidates[0].source != "gold":
        raise ValueError(f"{pool.sample_id}: gold must be candidate zero")
    labels = {candidate.verifier.class_label for candidate in pool.candidates}
    required = {
        VerifierClass.CORRECT,
        VerifierClass.WRONG_FUNCTION,
        VerifierClass.WRONG_ARGUMENTS,
        VerifierClass.INVALID,
    }
    if not required.issubset(labels):
        missing = sorted(label.value for label in required - labels)
        raise ValueError(f"{pool.sample_id}: missing verifier classes {missing}")
    if any(candidate.verifier.class_label == VerifierClass.UNKNOWN for candidate in pool.candidates):
        raise ValueError(f"{pool.sample_id}: unknown verifier class is forbidden")
