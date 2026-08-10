# Fast Pilot status

## Done (no-GPU phase)

- [x] Project scaffold + configs (`base`, `rtx2080ti`, `rtx4090`)
- [x] Verifier (parser / schema / normalizer / classifier)
- [x] Scoring + KL decomposition unit tests (17/17 pass)
- [x] Data adapters (xLAM + ToolACE-Qwen-cleaned)
- [x] `build_data.py` — ToolACE splits built (2688 records, auto-scaled)
- [x] `doctor.py`, `audit_verifier.py` — gold precision 100% on 200 samples

## Data status

| Source | Status | Filtered count |
|--------|--------|----------------|
| ToolACE-Qwen-cleaned | Downloaded | ~2690 |
| xLAM | Gated — needs `huggingface-cli login` | pending |

Splits in `data/splits/` (auto-scaled to available data).

## Next (PRD §13)

### Step 2 — 2080 Ti（需 GPU）

```bash
python scripts/smoke_test.py --config rtx2080ti
python scripts/tiny_overfit.py --config rtx2080ti   # 待实现
```

### Step 3 — Pre-Diagnostic（待实现 `run_diagnostic.py`）

### 无 GPU 可先做

```bash
python scripts/build_equivalence.py
pytest -q
```
