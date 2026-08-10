# FC Class KD — Fast Pilot

Verifier-defined Class-Level Distillation for Function Calling (PRD v4).

## Quick start (no GPU)

```bash
cd fc_class_kd
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e .
python scripts/doctor.py
pytest -q
```

## 数据与阶段对照

每个 Step 读哪些文件、写哪些目录，见 **`docs/DATA_MAP.md`**。  
`data/splits/`、`data/processed/` 等子目录里也有简短 `README.md` 标注。

Requires HuggingFace access for `Salesforce/xlam-function-calling-60k` (gated — accept terms + `huggingface-cli login`).

**ToolACE-only fallback** (no gate): `tryumanshow/ToolACE-Qwen-cleaned` (~2700 filtered single-call samples).

```bash
# ToolACE only (works without HF login)
python scripts/build_data.py --sources toolace

# Both sources after HF login
python scripts/build_data.py --sources xlam toolace
```

Processed cache: `data/processed/all_records.jsonl`  
Splits: `data/splits/*.jsonl`

## Verifier audit

```bash
python scripts/audit_verifier.py --split train --sample-size 200
```

## Project layout

See `FC_ClassKD_Fast_Pilot_PRD_v4.md` §10.

## GPU steps (2080 Ti / 4090)

1. `scripts/smoke_test.py` — 0.6B forward/backward
2. `scripts/run_diagnostic.py` — 200–300 pre-diagnostic
3. `scripts/train_sft.py` → S0
4. `scripts/cache_teacher_scores.py`
5. `scripts/train_candidate_kd.py` / `train_class_kd.py`
6. `scripts/evaluate_*.py`

## Sync via git

See **`docs/WORKFLOW_COMPANY_PERSONAL.md`** for 公司 PC ↔ 个人 PC ↔ 实验室 协同流程（bundle、云盘、数据复现）。

Large data under `data/raw/`, `data/processed/`, `outputs/` are gitignored by default.
Copy `data/splits/` + `data/processed/all_records.jsonl` via cloud/USB for exact reproducibility.
