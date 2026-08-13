#!/usr/bin/env python3
"""校验填写完成的 Excel，并把人工标签安全导回正式盲审包。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from fc_class_kd.verifier.audit import HUMAN_LABELS, audit_content_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEADER_ROW = 4

CHINESE_TO_LABEL = {
    "正确": "correct",
    "函数错误": "wrong_function",
    "参数错误": "wrong_arguments",
    "无效输出": "invalid",
    "无法确定": "unknown",
}
CHINESE_TO_SUB_LABEL = {
    "缺少必填参数": "missing_required",
    "参数名错误": "wrong_key",
    "参数类型错误": "wrong_type",
    "参数值错误": "wrong_value",
}
EXPECTED_HEADERS = {
    "审核ID",
    "人工标签（必填）",
    "参数错误细分（可选）",
    "备注",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入 Verifier 人工标注 Excel")
    parser.add_argument(
        "--excel",
        default="data/diagnostics/Verifier人工标注表.xlsx",
    )
    parser.add_argument(
        "--packet",
        default="data/diagnostics/verifier_human_audit_final.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="data/diagnostics/verifier_human_audit_final.manifest.json",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查，不修改 JSONL",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_cell(value: Any) -> str:
    return "" if value is None else str(value).strip()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_metadata(workbook: Any) -> dict[str, Any]:
    if "元数据" not in workbook.sheetnames:
        raise ValueError("Excel 缺少元数据工作表；请使用项目生成的正式标注表")
    sheet = workbook["元数据"]
    return {
        clean_cell(sheet.cell(row, 1).value): sheet.cell(row, 2).value
        for row in range(1, sheet.max_row + 1)
    }


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    excel_path = PROJECT_ROOT / args.excel
    packet_path = PROJECT_ROOT / args.packet
    manifest_path = PROJECT_ROOT / args.manifest

    packet_rows = load_jsonl(packet_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_content_hash = audit_content_sha256(packet_rows)
    if manifest.get("visible_content_sha256") != current_content_hash:
        raise ValueError("当前正式盲审包内容与清单不一致，拒绝导入")

    workbook = load_workbook(excel_path, data_only=False, read_only=False)
    metadata = read_metadata(workbook)
    if clean_cell(metadata.get("可见内容哈希")) != current_content_hash:
        raise ValueError("Excel 对应的盲审包版本与当前版本不同，拒绝导入")
    if int(metadata.get("样本数", -1)) != len(packet_rows):
        raise ValueError("Excel 样本数与当前正式盲审包不一致")
    if "人工标注" not in workbook.sheetnames:
        raise ValueError("Excel 缺少‘人工标注’工作表")

    sheet = workbook["人工标注"]
    header_to_column = {
        clean_cell(sheet.cell(HEADER_ROW, column).value): column
        for column in range(1, sheet.max_column + 1)
    }
    missing_headers = EXPECTED_HEADERS - set(header_to_column)
    if missing_headers:
        raise ValueError(f"Excel 缺少必要列：{sorted(missing_headers)}")

    excel_rows: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    first_data_row = HEADER_ROW + 1
    for row_number in range(first_data_row, first_data_row + len(packet_rows)):
        audit_id = clean_cell(sheet.cell(row_number, header_to_column["审核ID"]).value)
        label_text = clean_cell(
            sheet.cell(row_number, header_to_column["人工标签（必填）"]).value
        )
        sub_label_text = clean_cell(
            sheet.cell(row_number, header_to_column["参数错误细分（可选）"]).value
        )
        notes = clean_cell(sheet.cell(row_number, header_to_column["备注"]).value)
        if not audit_id:
            errors.append(f"Excel 第 {row_number} 行缺少审核ID")
            continue
        if audit_id in excel_rows:
            errors.append(f"Excel 出现重复审核ID：{audit_id}")
            continue
        label = CHINESE_TO_LABEL.get(label_text, label_text)
        sub_label = CHINESE_TO_SUB_LABEL.get(sub_label_text, sub_label_text)
        if label not in HUMAN_LABELS:
            errors.append(
                f"第 {row_number} 行 {audit_id}：人工标签未填或不合法（当前为 {label_text!r}）"
            )
        if sub_label and sub_label not in CHINESE_TO_SUB_LABEL.values():
            errors.append(
                f"第 {row_number} 行 {audit_id}：参数错误细分不合法（当前为 {sub_label_text!r}）"
            )
        if sub_label and label != "wrong_arguments":
            errors.append(
                f"第 {row_number} 行 {audit_id}：只有‘参数错误’才能填写参数错误细分"
            )
        if label == "unknown" and not notes:
            errors.append(
                f"第 {row_number} 行 {audit_id}：选择‘无法确定’时必须在备注中说明原因"
            )
        excel_rows[audit_id] = {
            "human_label": label,
            "human_sub_label": sub_label,
            "human_notes": notes,
        }

    expected_ids = [str(row["audit_id"]) for row in packet_rows]
    if set(excel_rows) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(excel_rows))
        extra = sorted(set(excel_rows) - set(expected_ids))
        if missing:
            errors.append(f"Excel 缺少审核ID：{missing[:10]}")
        if extra:
            errors.append(f"Excel 出现未知审核ID：{extra[:10]}")
    if errors:
        preview = "\n".join(f"- {message}" for message in errors[:30])
        suffix = f"\n另有 {len(errors) - 30} 个问题" if len(errors) > 30 else ""
        raise ValueError(f"Excel 尚不能导入，共 {len(errors)} 个问题：\n{preview}{suffix}")

    updated_rows: list[dict[str, Any]] = []
    for row in packet_rows:
        labels = excel_rows[str(row["audit_id"])]
        updated = dict(row)
        updated["human_label"] = labels["human_label"]
        updated["human_sub_label"] = labels["human_sub_label"] or None
        updated["human_notes"] = labels["human_notes"]
        updated_rows.append(updated)
    if audit_content_sha256(updated_rows) != current_content_hash:
        raise RuntimeError("导入过程改变了盲审题目内容，已停止且未写入")

    counts = {label: 0 for label in sorted(HUMAN_LABELS)}
    for row in updated_rows:
        counts[str(row["human_label"])] += 1
    print(f"完整性检查通过：{len(updated_rows)}/{len(updated_rows)} 条已标注")
    print("人工标签数量：" + json.dumps(counts, ensure_ascii=False, sort_keys=True))
    if args.check_only:
        print("CHECK_ONLY=true；没有修改正式 JSONL")
        return

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup_path = packet_path.with_name(
        f"{packet_path.stem}.导入前备份-{timestamp}{packet_path.suffix}"
    )
    shutil.copy2(packet_path, backup_path)
    atomic_write_jsonl(packet_path, updated_rows)
    if audit_content_sha256(load_jsonl(packet_path)) != current_content_hash:
        raise RuntimeError("写入后的盲审题目内容校验失败")
    print(f"已导回 → {display_path(packet_path)}")
    print(f"导入前备份 → {display_path(backup_path)}")
    print("下一步运行：.venv/bin/python scripts/score_verifier_audit.py")


if __name__ == "__main__":
    main()
