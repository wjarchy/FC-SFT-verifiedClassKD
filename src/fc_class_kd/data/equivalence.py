"""Build equivalence diagnostic variants (PRD §7.2)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

from fc_class_kd.types import FCRecord, FunctionCall
from fc_class_kd.verifier.normalizer import apply_schema_defaults, canonicalize_call
from fc_class_kd.verifier.parser import serialize_function_call


def _key_reorder(call: FunctionCall) -> Optional[FunctionCall]:
    if len(call.arguments) < 2:
        return None
    keys = sorted(call.arguments.keys(), reverse=True)
    return FunctionCall(name=call.name, arguments={k: call.arguments[k] for k in keys})


def _implicit_defaults(record: FCRecord) -> Optional[FunctionCall]:
    """Gold with only required args — optional/default omitted."""
    tool = next((t for t in record.tools if t.name == record.gold_call.name), None)
    if tool is None:
        return None
    args = {}
    for k, p in tool.parameters.items():
        if p.required and k in record.gold_call.arguments:
            args[k] = copy.deepcopy(record.gold_call.arguments[k])
    if not args:
        return None
    full = set(record.gold_call.arguments.keys())
    minimal = set(args.keys())
    if full == minimal:
        return None
    return FunctionCall(name=record.gold_call.name, arguments=args)


def build_equiv_variants(record: FCRecord) -> List[Dict[str, object]]:
    """Return list of {variant_type, query, gold_call} for equivalence diagnostic."""
    variants: List[Dict[str, object]] = []
    gold = record.gold_call

    reordered = _key_reorder(gold)
    if reordered and serialize_function_call(reordered, sort_keys=False) != serialize_function_call(
        gold, sort_keys=False
    ):
        variants.append(
            {
                "variant_type": "key_reorder",
                "query": record.query,
                "gold_call": reordered.model_dump(),
                "original_gold": gold.model_dump(),
            }
        )

    minimal = _implicit_defaults(record)
    if minimal:
        variants.append(
            {
                "variant_type": "implicit_default",
                "query": record.query,
                "gold_call": minimal.model_dump(),
                "original_gold": gold.model_dump(),
            }
        )

    canon = canonicalize_call(gold, record.tools, gold)
    canon_text = serialize_function_call(canon)
    gold_text = serialize_function_call(gold)
    if canon_text != gold_text:
        variants.append(
            {
                "variant_type": "canonical_normalization",
                "query": record.query,
                "gold_call": canon.model_dump(),
                "original_gold": gold.model_dump(),
            }
        )

    return variants


def build_equiv_record(record: FCRecord, variant: Dict[str, object]) -> FCRecord:
    """FCRecord where gold_call is equivalent variant (same tools/query)."""
    gold = FunctionCall.model_validate(variant["gold_call"])
    return FCRecord(
        id=f"{record.id}_equiv_{variant['variant_type']}",
        source=record.source,
        query=str(variant.get("query", record.query)),
        tools=record.tools,
        gold_call=gold,
        metadata={
            **record.metadata,
            "equiv_variant": variant["variant_type"],
            "original_id": record.id,
        },
    )


def write_equivalence_diagnostic(
    records: List[FCRecord],
    out_path: Path,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            for var in build_equiv_variants(rec):
                equiv = build_equiv_record(rec, var)
                row = {
                    "record": equiv.model_dump(),
                    "variant_type": var["variant_type"],
                    "original_gold": var["original_gold"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count
