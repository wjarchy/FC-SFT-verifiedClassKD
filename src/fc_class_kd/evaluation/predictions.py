"""Bookkeeping helpers for resumable free-generation artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from fc_class_kd.evaluation.metrics import evaluate_predictions
from fc_class_kd.types import FCRecord
from fc_class_kd.verifier.classifier import classify_call
from fc_class_kd.verifier.parser import extract_function_call_surface


def candidate_surface_from_prediction(prediction: str) -> str:
    """Extract the JSON body for candidate scoring, preserving valid model surface."""
    surface = extract_function_call_surface(prediction)
    if surface is not None:
        return surface
    cleaned = prediction.strip()
    if "<tool_call>" in cleaned:
        cleaned = cleaned.split("<tool_call>", 1)[1]
        if "</tool_call>" in cleaned:
            cleaned = cleaned.split("</tool_call>", 1)[0]
    cleaned = cleaned.replace("<|im_end|>", "").strip()
    return cleaned or prediction.strip() or "<empty_model_output>"


def make_prediction_row(
    record: FCRecord,
    split: str,
    prediction: str,
    prompt_tokens: int,
    generated_tokens: int,
) -> dict[str, Any]:
    result = classify_call(prediction, record.tools, record.gold_call)
    return {
        "sample_id": record.id,
        "split": split,
        "prediction": prediction,
        "candidate_surface": candidate_surface_from_prediction(prediction),
        "verifier_class": result.class_label.value,
        "verifier_sub_label": result.sub_label,
        "parse_ok": result.parse_ok,
        "verifier_details": result.details,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
    }


def validate_prediction_rows(
    records: Sequence[FCRecord],
    rows: Sequence[dict[str, Any]],
    split: str,
    require_complete: bool,
) -> None:
    if len(rows) > len(records):
        raise ValueError(f"{split}: prediction file is longer than its split")
    expected_prefix = [record.id for record in records[: len(rows)]]
    actual = [str(row.get("sample_id", "")) for row in rows]
    if actual != expected_prefix:
        raise ValueError(f"{split}: existing predictions are not the exact split prefix")
    if len(set(actual)) != len(actual):
        raise ValueError(f"{split}: duplicate prediction IDs")
    if any(row.get("split") != split for row in rows):
        raise ValueError(f"{split}: prediction row has a wrong split label")
    if require_complete and len(rows) != len(records):
        raise ValueError(f"{split}: predictions are incomplete")


def summarize_prediction_rows(
    records: Sequence[FCRecord],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if len(records) != len(rows):
        raise ValueError("metrics require one prediction per record")
    predictions = [str(row["prediction"]) for row in rows]
    metrics = evaluate_predictions(list(records), predictions).rates()
    return {
        "count": len(rows),
        "metrics": metrics,
        "verifier_class_counts": dict(Counter(row["verifier_class"] for row in rows)),
        "mean_generated_tokens": (
            sum(int(row["generated_tokens"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "max_generated_tokens": max((int(row["generated_tokens"]) for row in rows), default=0),
    }
