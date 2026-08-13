"""Utilities for a human-labelled, reproducible Verifier audit gate."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fc_class_kd.candidates.rules import build_rule_candidates
from fc_class_kd.types import (
    PRIMARY_CLASSES,
    CandidatePoolRecord,
    FCRecord,
    VerifierClass,
)
from fc_class_kd.verifier.classifier import classify_call

HUMAN_LABELS = {label.value for label in (*PRIMARY_CLASSES, VerifierClass.UNKNOWN)}


def audit_content_sha256(rows: Sequence[dict[str, Any]]) -> str:
    """Hash immutable audit content while permitting humans to fill label fields."""
    ignored = {"human_label", "human_sub_label", "human_notes"}
    content = [{key: value for key, value in row.items() if key not in ignored} for row in rows]
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_blind_audit_rows(
    records: Sequence[FCRecord],
    per_class: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Stratify by hidden Verifier prediction, then remove it from annotation rows."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    buckets: dict[VerifierClass, list[dict[str, Any]]] = {
        label: [] for label in PRIMARY_CLASSES
    }
    for record in records:
        for index, candidate in enumerate(build_rule_candidates(record, seed=seed)):
            result = classify_call(candidate.serialized_call, record.tools, record.gold_call)
            if result.class_label not in buckets:
                continue
            buckets[result.class_label].append(
                {
                    "audit_id": f"{record.id}:rule:{index}",
                    "sample_id": record.id,
                    "query": record.query,
                    "tools": [tool.model_dump() for tool in record.tools],
                    "gold_call": record.gold_call.model_dump(),
                    "candidate": candidate.serialized_call,
                    "human_label": None,
                    "human_sub_label": None,
                    "human_notes": "",
                }
            )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in PRIMARY_CLASSES:
        rows = buckets[label]
        if len(rows) < per_class:
            raise ValueError(
                f"only {len(rows)} audit candidates available for hidden class {label.value}; "
                f"requested {per_class}"
            )
        selected.extend(rng.sample(rows, per_class))
    rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["audit_id"] = f"audit_{index:04d}"
    return selected


def _balanced_sample(
    rows: Sequence[dict[str, Any]],
    count: int,
    rng: random.Random,
    excluded_samples: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in rows
        if (str(row["split"]), str(row["sample_id"])) not in excluded_samples
    ]
    if len(rows) < count:
        raise ValueError(
            f"only {len(rows)} unused-sample rows available; requested {count}"
        )
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    splits = sorted({str(row["split"]) for row in rows})
    target_per_split = count // len(splits)
    for split in splits:
        available = [row for row in rows if row["split"] == split]
        take = min(target_per_split, len(available))
        for row in rng.sample(available, take):
            selected.append(row)
            used.add((str(row["split"]), str(row["candidate_id"])))
    remaining = [
        row
        for row in rows
        if (str(row["split"]), str(row["candidate_id"])) not in used
    ]
    selected.extend(rng.sample(remaining, count - len(selected)))
    return selected


def build_final_blind_audit_rows(
    records_by_split: dict[str, Sequence[FCRecord]],
    pools_by_split: dict[str, Sequence[CandidatePoolRecord]],
    per_stratum: int = 40,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample four predicted classes plus a dedicated S0-real-error stratum."""
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    strata: dict[str, list[dict[str, Any]]] = {
        label.value: [] for label in PRIMARY_CLASSES
    }
    strata["s0_real_error"] = []
    record_lookup: dict[tuple[str, str], FCRecord] = {}
    for split, records in records_by_split.items():
        for record in records:
            key = (split, record.id)
            if key in record_lookup:
                raise ValueError(f"duplicate audit source record: {key}")
            record_lookup[key] = record

    for split, pools in pools_by_split.items():
        for pool in pools:
            if pool.split != split:
                raise ValueError(f"{pool.sample_id}: pool split mismatch")
            record = record_lookup.get((split, pool.sample_id))
            if record is None:
                raise ValueError(f"{split}/{pool.sample_id}: no source record")
            for candidate in pool.candidates:
                fresh = classify_call(
                    candidate.serialized_call,
                    record.tools,
                    record.gold_call,
                )
                if fresh.class_label != candidate.verifier.class_label:
                    raise ValueError(f"{candidate.candidate_id}: stale pool verifier label")
                entry = {
                    "split": split,
                    "sample_id": record.id,
                    "candidate_id": candidate.candidate_id,
                    "candidate_source": candidate.source,
                    "verifier_label": fresh.class_label.value,
                    "query": record.query,
                    "tools": [tool.model_dump(mode="json") for tool in record.tools],
                    "gold_call": record.gold_call.model_dump(mode="json"),
                    "candidate": candidate.serialized_call,
                }
                if candidate.source == "s0_real_error":
                    if fresh.class_label == VerifierClass.CORRECT:
                        raise ValueError(f"{candidate.candidate_id}: S0 error is verifier-correct")
                    strata["s0_real_error"].append(entry)
                elif not (
                    fresh.class_label == VerifierClass.CORRECT
                    and candidate.source == "gold"
                ):
                    strata[fresh.class_label.value].append(entry)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_samples: set[tuple[str, str]] = set()
    sampling_order = ("s0_real_error", *[label.value for label in PRIMARY_CLASSES])
    for stratum in sampling_order:
        sampled = _balanced_sample(
            strata[stratum],
            per_stratum,
            rng,
            used_samples,
        )
        for row in sampled:
            selected.append({**row, "audit_stratum": stratum})
            used_samples.add((str(row["split"]), str(row["sample_id"])))
    rng.shuffle(selected)

    visible_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        audit_id = f"final_audit_{index:04d}"
        visible_rows.append(
            {
                "audit_id": audit_id,
                "audit_stage": "final_k6_pool",
                "sample_id": row["sample_id"],
                "query": row["query"],
                "tools": row["tools"],
                "gold_call": row["gold_call"],
                "candidate": row["candidate"],
                "human_label": None,
                "human_sub_label": None,
                "human_notes": "",
            }
        )
        key_rows.append(
            {
                "audit_id": audit_id,
                "split": row["split"],
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "candidate_source": row["candidate_source"],
                "audit_stratum": row["audit_stratum"],
                "verifier_label": row["verifier_label"],
            }
        )
    return visible_rows, key_rows


def write_audit_rows(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_human_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Re-run the current Verifier and compare it with completed human labels."""
    confusion: Counter[tuple[str, str]] = Counter()
    errors: list[dict[str, Any]] = []
    predicted_correct = 0
    true_correct_when_predicted = 0
    for row_index, row in enumerate(rows, start=1):
        human = row.get("human_label")
        if human not in HUMAN_LABELS:
            raise ValueError(
                f"row {row_index} ({row.get('audit_id')}) has missing/invalid "
                f"human_label={human!r}; allowed={sorted(HUMAN_LABELS)}"
            )
        record = FCRecord.model_validate(
            {
                "id": row["sample_id"],
                "source": "human_audit",
                "query": row["query"],
                "tools": row["tools"],
                "gold_call": row["gold_call"],
            }
        )
        predicted = classify_call(
            row["candidate"],
            record.tools,
            record.gold_call,
        ).class_label.value
        confusion[(human, predicted)] += 1
        if predicted == VerifierClass.CORRECT.value:
            predicted_correct += 1
            if human == VerifierClass.CORRECT.value:
                true_correct_when_predicted += 1
        if human != predicted:
            errors.append(
                {
                    "audit_id": row["audit_id"],
                    "human_label": human,
                    "verifier_label": predicted,
                    "human_notes": row.get("human_notes", ""),
                }
            )

    total = len(rows)
    correct_count = sum(count for (human, predicted), count in confusion.items() if human == predicted)
    correct_precision = (
        true_correct_when_predicted / predicted_correct if predicted_correct else 0.0
    )
    overall_accuracy = correct_count / total if total else 0.0
    labels = sorted(HUMAN_LABELS)
    final_k6 = bool(rows) and all(
        row.get("audit_stage") == "final_k6_pool" for row in rows
    )
    return {
        "audit_kind": (
            "human_labelled_blind_final_k6_gate"
            if final_k6
            else "human_labelled_blind_rule_candidate_gate"
        ),
        "total": total,
        "correct_equivalent_precision": correct_precision,
        "overall_accuracy": overall_accuracy,
        "thresholds": {
            "correct_equivalent_precision_min": 0.98,
            "overall_accuracy_min": 0.95,
        },
        "passed": correct_precision >= 0.98 and overall_accuracy >= 0.95,
        "confusion_matrix": {
            human: {predicted: confusion[(human, predicted)] for predicted in labels}
            for human in labels
        },
        "errors": errors,
        "limitations": (
            "The audit samples the complete unfrozen K=6 Train and Diagnostic pools, "
            "including a dedicated S0-real-error stratum; passing permits freezing these "
            "exact files but does not validate a later rebuilt pool."
            if final_k6
            else "This preliminary gate covers deterministic rule candidates. The final "
            "K=6 pool must be audited again after S0 on-policy candidates are added."
        ),
    }
