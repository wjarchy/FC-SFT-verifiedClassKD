#!/usr/bin/env python3
"""Check training-source queries against a pinned local BFCL checkout."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from fc_class_kd.data.adapters import xlam_row_to_record
from fc_class_kd.data.filters import passes_fast_pilot_filter


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_normalize(text).split())


def _iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    """Read BFCL's mixed JSONL and regular JSON metadata files."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        parsed_lines = [json.loads(line) for line in lines if line.strip()]
    except json.JSONDecodeError:
        payload = json.loads("\n".join(lines))
        if isinstance(payload, list):
            parsed_lines = payload
        elif isinstance(payload, dict) and "question" in payload:
            parsed_lines = [payload]
        else:
            # BFCL format-sensitivity metadata maps categories to case IDs and contains
            # no query text; the referenced cases are already present in other files.
            parsed_lines = []
    for row in parsed_lines:
        if isinstance(row, dict):
            yield row


def _extract_user_texts(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            found.append(value["content"])
        else:
            for child in value.values():
                found.extend(_extract_user_texts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_extract_user_texts(child))
    return found


def _load_bfcl(data_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(data_dir.glob("BFCL_v4_*.json")):
        for row in _iter_json_records(path):
            texts = _extract_user_texts(row.get("question"))
            if not texts and isinstance(row.get("question"), str):
                texts = [row["question"]]
            for turn, text in enumerate(texts):
                if text.strip():
                    rows.append(
                        {
                            "id": f"{row.get('id', path.stem)}:user:{turn}",
                            "category": path.stem,
                            "query": text.strip(),
                        }
                    )
    return rows


def _load_toolace(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _iter_json_records(path):
        rows.append(
            {
                "id": str(row.get("id", "toolace")),
                "source": "toolace",
                "query": str(row["query"]).strip(),
            }
        )
    return rows


def _load_xlam(path: Path) -> list[dict[str, str]]:
    dataset = load_dataset("json", data_files=str(path), split="train")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(dataset):
        record = xlam_row_to_record(dict(raw_row), index)
        if record is None or not passes_fast_pilot_filter(record):
            continue
        query = record.query.strip()
        if query in seen:
            continue
        seen.add(query)
        rows.append({"id": record.id, "source": "xlam", "query": query})
    return rows


def _near_matches(
    sources: list[dict[str, str]],
    bfcl: list[dict[str, str]],
) -> list[dict[str, Any]]:
    bfcl_tokens = [_tokens(row["query"]) for row in bfcl]
    postings: dict[str, set[int]] = defaultdict(set)
    for index, tokens in enumerate(bfcl_tokens):
        for token in tokens:
            postings[token].add(index)
    document_frequency = {token: len(indices) for token, indices in postings.items()}

    suspicious: list[dict[str, Any]] = []
    for source in sources:
        source_norm = _normalize(source["query"])
        source_tokens = _tokens(source["query"])
        if not source_tokens:
            continue
        rarest = sorted(
            (token for token in source_tokens if token in postings),
            key=lambda token: document_frequency[token],
        )[:6]
        candidate_indices: set[int] = set()
        for token in rarest:
            candidate_indices.update(postings[token])
        best: dict[str, Any] | None = None
        for index in candidate_indices:
            target_tokens = bfcl_tokens[index]
            intersection = len(source_tokens & target_tokens)
            union = len(source_tokens | target_tokens)
            jaccard = intersection / union if union else 0.0
            containment = intersection / min(len(source_tokens), len(target_tokens))
            if jaccard < 0.45 and containment < 0.65:
                continue
            target_norm = _normalize(bfcl[index]["query"])
            sequence = SequenceMatcher(None, source_norm, target_norm, autojunk=False).ratio()
            score = max(jaccard, sequence)
            if best is None or score > best["score"]:
                best = {
                    "source": source["source"],
                    "source_id": source["id"],
                    "source_query": source["query"],
                    "bfcl_id": bfcl[index]["id"],
                    "bfcl_category": bfcl[index]["category"],
                    "bfcl_query": bfcl[index]["query"],
                    "token_jaccard": round(jaccard, 6),
                    "token_containment": round(containment, 6),
                    "sequence_ratio": round(sequence, 6),
                    "score": round(score, 6),
                }
        if best is not None and (
            best["token_jaccard"] >= 0.8 or best["sequence_ratio"] >= 0.9
        ):
            suspicious.append(best)
    return sorted(suspicious, key=lambda row: row["score"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ToolACE/xLAM query overlap with BFCL")
    parser.add_argument("--toolace", type=Path, required=True)
    parser.add_argument("--xlam", type=Path, required=True)
    parser.add_argument("--bfcl-data", type=Path, required=True)
    parser.add_argument("--bfcl-revision", required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/BFCL数据泄漏检查.json"))
    args = parser.parse_args()

    source_rows = _load_toolace(args.toolace) + _load_xlam(args.xlam)
    bfcl_rows = _load_bfcl(args.bfcl_data)
    source_by_norm: dict[str, list[dict[str, str]]] = defaultdict(list)
    bfcl_by_norm: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        source_by_norm[_normalize(row["query"])].append(row)
    for row in bfcl_rows:
        bfcl_by_norm[_normalize(row["query"])].append(row)

    exact: list[dict[str, Any]] = []
    for normalized in sorted(source_by_norm.keys() & bfcl_by_norm.keys()):
        for source in source_by_norm[normalized]:
            for target in bfcl_by_norm[normalized]:
                exact.append(
                    {
                        "source": source["source"],
                        "source_id": source["id"],
                        "source_query": source["query"],
                        "bfcl_id": target["id"],
                        "bfcl_category": target["category"],
                    }
                )

    near = _near_matches(source_rows, bfcl_rows)
    report = {
        "bfcl_revision": args.bfcl_revision,
        "bfcl_data_dir": str(args.bfcl_data),
        "bfcl_files": sorted(path.name for path in args.bfcl_data.glob("BFCL_v4_*.json")),
        "bfcl_user_queries_total": len(bfcl_rows),
        "bfcl_unique_normalized_queries": len(bfcl_by_norm),
        "source_query_counts": dict(Counter(row["source"] for row in source_rows)),
        "source_unique_normalized_queries": len(source_by_norm),
        "exact_normalized_overlap_count": len(exact),
        "exact_normalized_overlaps": exact,
        "high_similarity_overlap_count": len(near),
        "high_similarity_overlaps": near,
        "near_thresholds": {"token_jaccard": 0.8, "sequence_ratio": 0.9},
        "limitations": [
            "High-similarity matches require manual semantic review.",
            "No automatic text-similarity method can prove the absence of all semantic leakage.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {
        "exact_normalized_overlaps", "high_similarity_overlaps"
    }}, ensure_ascii=False, indent=2))
    print("Top high-similarity matches:")
    for row in near[:20]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
