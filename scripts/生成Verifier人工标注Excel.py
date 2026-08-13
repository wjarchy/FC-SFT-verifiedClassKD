#!/usr/bin/env python3
"""把正式 Verifier 盲审包整理成便于人工填写的 Excel。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
from openpyxl.worksheet.datavalidation import DataValidation

from fc_class_kd.verifier.audit import audit_content_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEADER_ROW = 4
FIRST_DATA_ROW = HEADER_ROW + 1

LABEL_TO_CHINESE = {
    "correct": "正确",
    "wrong_function": "函数错误",
    "wrong_arguments": "参数错误",
    "invalid": "无效输出",
    "unknown": "无法确定",
}
SUB_LABEL_TO_CHINESE = {
    "missing_required": "缺少必填参数",
    "wrong_key": "参数名错误",
    "wrong_type": "参数类型错误",
    "wrong_value": "参数值错误",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成正式 Verifier 人工标注 Excel")
    parser.add_argument(
        "--input",
        default="data/diagnostics/verifier_human_audit_final.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="data/diagnostics/verifier_human_audit_final.manifest.json",
    )
    parser.add_argument(
        "--output",
        default="data/diagnostics/Verifier人工标注表.xlsx",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def pretty_candidate(value: str) -> str:
    try:
        return pretty_json(json.loads(value))
    except (json.JSONDecodeError, TypeError):
        return value


def pretty_tools(tools: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for tool in tools:
        lines = [f"【{tool['name']}】{tool.get('description') or '无说明'}"]
        parameters = tool.get("parameters") or {}
        if not parameters:
            lines.append("  参数：无")
        for name, spec in parameters.items():
            required = "必填" if spec.get("required") else "可选"
            details = [str(spec.get("type") or "未说明类型"), required]
            if spec.get("enum") is not None:
                details.append(f"可选值={spec['enum']}")
            if spec.get("default") is not None:
                details.append(f"默认值={spec['default']!r}")
            description = spec.get("description") or "无说明"
            lines.append(f"  - {name}（{'；'.join(details)}）：{description}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def set_literal_text(cell: Any, value: str) -> None:
    """强制作为文本写入，避免数据内容被 Excel 当作公式。"""
    cell.value = value
    cell.data_type = "s"


def add_instruction_sheet(workbook: Workbook, row_count: int) -> None:
    sheet = workbook.active
    sheet.title = "标注说明"
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions["A"].width = 22
    sheet.column_dimensions["B"].width = 95

    title_fill = PatternFill("solid", fgColor="1F4E78")
    section_fill = PatternFill("solid", fgColor="D9EAF7")
    sheet.merge_cells("A1:B1")
    sheet["A1"] = "Verifier 正式人工盲审"
    sheet["A1"].font = Font(size=18, bold=True, color="FFFFFF")
    sheet["A1"].fill = title_fill
    sheet["A1"].alignment = Alignment(horizontal="center", vertical="center")
    sheet.row_dimensions[1].height = 34

    instructions = [
        ("你需要做什么", f"进入“人工标注”工作表，逐行只填写绿色的三列。共 {row_count} 条；“人工标签”为必填，其余两列按需填写。"),
        ("正确", "候选与标准答案语义相同：函数正确，参数名、类型和值正确。仅参数顺序变化、明确默认值省略或项目允许的规范化差异，也算正确。"),
        ("函数错误", "候选调用了错误函数，或者函数名根本不在该行列出的可用工具中。即使参数看起来合理，也选函数错误。"),
        ("参数错误", "函数正确，但参数缺失，或参数名、类型、可选值、具体值错误，或出现工具说明不允许的参数。可在下一列选择更细的原因。"),
        ("无效输出", "候选无法解析成一个完整的 {name, arguments} 工具调用，例如 JSON 截断、语法损坏或包含多个调用。"),
        ("无法确定", "只有工具说明不足、依赖外部知识或运行时状态，导致人工也不能高置信判断时才选；备注中必须写明原因。"),
        ("盲审要求", "不要打开 verifier_human_audit_final_key.jsonl，不要运行 Verifier，也不要根据候选的生成方式猜标签。只看本表给出的用户问题、工具说明、标准答案和待判断候选。"),
        ("保存与交回", "可以随时保存并关闭 Excel。全部填完后告诉 Codex“Excel 已标完”，由 Codex 检查完整性、导回并计算审核结果。"),
    ]
    start_row = 3
    for offset, (name, description) in enumerate(instructions):
        row = start_row + offset
        sheet.cell(row, 1, name)
        sheet.cell(row, 2, description)
        sheet.cell(row, 1).font = Font(bold=True, color="1F1F1F")
        sheet.cell(row, 1).fill = section_fill
        sheet.cell(row, 1).alignment = Alignment(vertical="top", wrap_text=True)
        sheet.cell(row, 2).alignment = Alignment(vertical="top", wrap_text=True)
        sheet.row_dimensions[row].height = 42 if name != "盲审要求" else 54

    sheet["A13"] = "快捷建议"
    sheet["A13"].font = Font(bold=True)
    sheet["A13"].fill = section_fill
    sheet["B13"] = "先看待判断候选能否解析，再看函数名，最后逐项核对参数。表格顶部会自动显示已填写数量。"
    sheet["B13"].alignment = Alignment(wrap_text=True)


def add_annotation_sheet(workbook: Workbook, rows: list[dict[str, Any]]) -> None:
    sheet = workbook.create_sheet("人工标注")
    sheet.sheet_view.showGridLines = False
    sheet.sheet_view.zoomScale = 75

    last_data_row = FIRST_DATA_ROW + len(rows) - 1
    sheet.merge_cells("A1:J1")
    sheet["A1"] = "Verifier 正式人工盲审标注表"
    sheet["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    sheet["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    sheet["A1"].alignment = Alignment(horizontal="center", vertical="center")
    sheet.row_dimensions[1].height = 30

    sheet["A2"] = "标注进度"
    sheet["A2"].font = Font(bold=True)
    sheet["B2"] = f'=COUNTIF(G{FIRST_DATA_ROW}:G{last_data_row},"<>")&"/{len(rows)}"'
    sheet["C2"] = "只填写绿色列：人工标签、参数错误细分、备注。"
    sheet.merge_cells("C2:J2")
    sheet["C2"].font = Font(color="006100", bold=True)
    sheet["C2"].alignment = Alignment(wrap_text=True)
    sheet["A3"] = "判断顺序"
    sheet["A3"].font = Font(bold=True)
    sheet["B3"] = "①候选能否解析 → ②函数是否正确 → ③参数是否正确。拿不准时选“无法确定”并写备注。"
    sheet.merge_cells("B3:J3")
    sheet["B3"].alignment = Alignment(wrap_text=True)

    headers = [
        "序号",
        "审核ID",
        "用户问题",
        "可用工具与参数说明",
        "标准答案",
        "待判断候选",
        "人工标签（必填）",
        "参数错误细分（可选）",
        "备注",
        "完成状态",
    ]
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(HEADER_ROW, column, header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4472C4" if column < 7 else "548235")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.row_dimensions[HEADER_ROW].height = 34
    sheet["G4"].comment = Comment(
        "从下拉菜单选择：正确、函数错误、参数错误、无效输出、无法确定。",
        "FC_ClassKD",
    )
    sheet["H4"].comment = Comment(
        "仅当人工标签为“参数错误”时填写。",
        "FC_ClassKD",
    )

    thin_gray = Side(style="thin", color="D9D9D9")
    border = Border(left=thin_gray, right=thin_gray, top=thin_gray, bottom=thin_gray)
    locked_fill = PatternFill("solid", fgColor="F2F2F2")
    input_fill = PatternFill("solid", fgColor="E2F0D9")
    for index, row in enumerate(rows, start=1):
        excel_row = FIRST_DATA_ROW + index - 1
        values = [
            index,
            str(row["audit_id"]),
            str(row["query"]),
            pretty_tools(row["tools"]),
            pretty_json(row["gold_call"]),
            pretty_candidate(str(row["candidate"])),
            LABEL_TO_CHINESE.get(row.get("human_label"), ""),
            SUB_LABEL_TO_CHINESE.get(row.get("human_sub_label"), ""),
            str(row.get("human_notes") or ""),
        ]
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(excel_row, column)
            if isinstance(value, str):
                set_literal_text(cell, value)
            else:
                cell.value = value
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = border
            cell.fill = input_fill if column in (7, 8, 9) else locked_fill
            cell.protection = Protection(locked=column not in (7, 8, 9))
            if column in (5, 6):
                cell.font = Font(name="Consolas", size=10)
        status_cell = sheet.cell(excel_row, 10)
        status_cell.value = f'=IF(G{excel_row}="","待填写","已填写")'
        status_cell.alignment = Alignment(horizontal="center", vertical="center")
        status_cell.border = border
        status_cell.fill = locked_fill
        sheet.row_dimensions[excel_row].height = 150

    widths = {
        "A": 7,
        "B": 19,
        "C": 46,
        "D": 78,
        "E": 40,
        "F": 40,
        "G": 18,
        "H": 22,
        "I": 32,
        "J": 12,
    }
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = f"C{FIRST_DATA_ROW}"
    sheet.auto_filter.ref = f"A{HEADER_ROW}:J{last_data_row}"

    options = workbook.create_sheet("下拉选项")
    for index, value in enumerate(LABEL_TO_CHINESE.values(), start=1):
        options.cell(index, 1, value)
    for index, value in enumerate(SUB_LABEL_TO_CHINESE.values(), start=1):
        options.cell(index, 2, value)
    options.sheet_state = "hidden"

    label_validation = DataValidation(
        type="list",
        formula1="'下拉选项'!$A$1:$A$5",
        allow_blank=False,
    )
    label_validation.error = "请从下拉菜单选择五种标签之一。"
    label_validation.errorTitle = "标签不合法"
    label_validation.prompt = "请选择人工判断，不要输入 Verifier 的预测。"
    label_validation.promptTitle = "人工标签"
    label_validation.showErrorMessage = True
    label_validation.showInputMessage = True
    sheet.add_data_validation(label_validation)
    label_validation.add(f"G{FIRST_DATA_ROW}:G{last_data_row}")

    sub_label_validation = DataValidation(
        type="list",
        formula1="'下拉选项'!$B$1:$B$4",
        allow_blank=True,
    )
    sub_label_validation.error = "请从下拉菜单选择参数错误细分，或留空。"
    sub_label_validation.errorTitle = "细分标签不合法"
    sub_label_validation.showErrorMessage = True
    sheet.add_data_validation(sub_label_validation)
    sub_label_validation.add(f"H{FIRST_DATA_ROW}:H{last_data_row}")

    colors = {
        "正确": "C6EFCE",
        "函数错误": "FCE4D6",
        "参数错误": "FFF2CC",
        "无效输出": "F4CCCC",
        "无法确定": "D9EAD3",
    }
    for label, color in colors.items():
        sheet.conditional_formatting.add(
            f"G{FIRST_DATA_ROW}:G{last_data_row}",
            FormulaRule(
                formula=[f'G{FIRST_DATA_ROW}="{label}"'],
                fill=PatternFill("solid", fgColor=color),
            ),
        )


def add_metadata_sheet(
    workbook: Workbook,
    source_path: Path,
    manifest_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    sheet = workbook.create_sheet("元数据")
    values = {
        "用途": "FC_ClassKD Verifier 正式人工盲审",
        "源文件": str(source_path.relative_to(PROJECT_ROOT)),
        "清单文件": str(manifest_path.relative_to(PROJECT_ROOT)),
        "可见内容哈希": audit_content_sha256(rows),
        "样本数": len(rows),
        "表头行": HEADER_ROW,
        "数据起始行": FIRST_DATA_ROW,
    }
    for index, (key, value) in enumerate(values.items(), start=1):
        sheet.cell(index, 1, key)
        sheet.cell(index, 2, value)
    sheet.sheet_state = "veryHidden"


def main() -> None:
    args = parse_args()
    source_path = PROJECT_ROOT / args.input
    manifest_path = PROJECT_ROOT / args.manifest
    output_path = PROJECT_ROOT / args.output
    rows = load_jsonl(source_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_content_hash = audit_content_sha256(rows)
    if manifest.get("visible_content_sha256") != actual_content_hash:
        raise ValueError("正式盲审包内容与清单不一致，拒绝生成 Excel")
    if manifest.get("row_count") != len(rows):
        raise ValueError("正式盲审包样本数与清单不一致")

    workbook = Workbook()
    workbook.properties.title = "FC_ClassKD Verifier 正式人工盲审"
    workbook.properties.subject = "只填写人工标签、参数错误细分和备注"
    workbook.properties.creator = "FC_ClassKD"
    add_instruction_sheet(workbook, len(rows))
    add_annotation_sheet(workbook, rows)
    add_metadata_sheet(workbook, source_path, manifest_path, rows)
    workbook.active = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    print(f"已生成 {len(rows)} 条正式盲审样本 → {output_path.relative_to(PROJECT_ROOT)}")
    print("请只填写‘人工标注’工作表中的绿色三列。")


if __name__ == "__main__":
    main()
