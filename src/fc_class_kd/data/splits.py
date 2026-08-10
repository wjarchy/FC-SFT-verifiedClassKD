"""Train/val/test split utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from fc_class_kd.types import FCRecord


def deterministic_split(
    records: Sequence[FCRecord],
    seed: int,
    train_size: int,
    val_size: int,
    internal_test_size: int,
    diagnostic_size: int,
    equivalence_size: int,
) -> Dict[str, List[FCRecord]]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    need = train_size + val_size + internal_test_size + diagnostic_size + equivalence_size
    if len(indices) < need:
        # Scale down proportionally when sources are insufficient (e.g. ToolACE-only before xLAM)
        if len(indices) < equivalence_size + diagnostic_size + 100:
            raise ValueError(
                f"Not enough records: have {len(indices)}, need at least "
                f"{equivalence_size + diagnostic_size + 100}."
            )
        ratio = len(indices) / need
        train_size = int(train_size * ratio)
        val_size = int(val_size * ratio)
        internal_test_size = int(internal_test_size * ratio)
        diagnostic_size = int(diagnostic_size * ratio)
        equivalence_size = int(equivalence_size * ratio)
        need = train_size + val_size + internal_test_size + diagnostic_size + equivalence_size
        indices = indices[:need]

    slices = {
        "train": indices[:train_size],
        "val": indices[train_size:train_size + val_size],
        "internal_test": indices[train_size + val_size:train_size + val_size + internal_test_size],
        "diagnostic": indices[
            train_size + val_size + internal_test_size:
            train_size + val_size + internal_test_size + diagnostic_size
        ],
        "equivalence_diagnostic": indices[
            train_size + val_size + internal_test_size + diagnostic_size:
            train_size + val_size + internal_test_size + diagnostic_size + equivalence_size
        ],
    }

    return {name: [records[i] for i in idxs] for name, idxs in slices.items()}


def write_split(split: Dict[str, List[FCRecord]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in split.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")


def read_split(path: Path) -> List[FCRecord]:
    records: List[FCRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(FCRecord.model_validate_json(line))
    return records


def split_manifest(split_dir: Path) -> Dict[str, int]:
    manifest: Dict[str, int] = {}
    for path in sorted(split_dir.glob("*.jsonl")):
        manifest[path.stem] = sum(1 for _ in path.open(encoding="utf-8"))
    return manifest
