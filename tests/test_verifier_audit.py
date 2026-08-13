"""Tests for reproducible blind Verifier audit preparation and scoring."""

import json
from pathlib import Path

import pytest

from fc_class_kd.candidates.pool import build_k6_pool
from fc_class_kd.data.splits import read_split
from fc_class_kd.verifier.audit import (
    audit_content_sha256,
    build_blind_audit_rows,
    build_final_blind_audit_rows,
    score_human_audit,
)


def test_blind_audit_has_no_verifier_prediction_and_is_reproducible():
    records = read_split(Path("data/fixtures/smoke_4.jsonl"))
    first = build_blind_audit_rows(records, per_class=1, seed=42)
    second = build_blind_audit_rows(records, per_class=1, seed=42)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(first) == 4
    assert all("verifier_label" not in row and "candidate_source" not in row for row in first)


def test_human_audit_requires_completed_valid_labels():
    records = read_split(Path("data/fixtures/smoke_4.jsonl"))
    rows = build_blind_audit_rows(records, per_class=1, seed=42)
    with pytest.raises(ValueError, match="missing/invalid"):
        score_human_audit(rows)


def test_final_audit_is_blind_and_covers_s0_errors_and_four_classes():
    base = read_split(Path("data/fixtures/smoke_4.jsonl"))
    records_by_split = {
        "train": [record.model_copy(update={"id": f"train_{record.id}"}) for record in base],
        "diagnostic": [
            record.model_copy(update={"id": f"diagnostic_{record.id}"}) for record in base
        ],
    }
    pools_by_split = {}
    for split, records in records_by_split.items():
        pools_by_split[split] = [
            build_k6_pool(
                record,
                split,
                '{"name":"not_a_tool","arguments":{}}',
            )
            for record in records
        ]
    visible, key = build_final_blind_audit_rows(
        records_by_split,
        pools_by_split,
        per_stratum=1,
        seed=42,
    )
    assert len(visible) == 5
    assert len(key) == 5
    assert {row["audit_stratum"] for row in key} == {
        "correct",
        "wrong_function",
        "wrong_arguments",
        "invalid",
        "s0_real_error",
    }
    assert all("verifier_label" not in row and "candidate_source" not in row for row in visible)
    assert all(row["audit_stage"] == "final_k6_pool" for row in visible)
    assert len({row["sample_id"] for row in visible}) == len(visible)
    before = audit_content_sha256(visible)
    visible[0]["human_label"] = "correct"
    visible[0]["human_notes"] = "checked"
    assert audit_content_sha256(visible) == before
