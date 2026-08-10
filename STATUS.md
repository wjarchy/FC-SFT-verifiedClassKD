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

## Next (GPU @ 2080 Ti)

- [ ] `huggingface-cli login` + re-run with `--sources xlam toolace` for full scale
- [ ] Smoke test (0.6B)
- [ ] Pre-diagnostic 200–300
- [ ] S0 SFT → three branches → evaluation
