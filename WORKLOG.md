# Work Log

## 2026-08-11 — Repository audit and static correctness hardening

### Environment observed

- Conda `wj`: Python 3.11.15。
- CPU test stack: PyTorch 2.5.1+cpu、NumPy 1.26.4、pytest 8.4.2。
- GPU 暂不可用：kernel driver `535.288.01` 与 NVML library `580.173` 失配。
- 根分区剩余约 70GB；未发现独立 `/data` 挂载。

### Code changes

- 新增 `CandidateRecord`，保留候选 surface string 与 parsed AST。
- 修复 key reorder、嵌套参数去重和真实 invalid JSON 候选。
- 修复 default omission 假等价。
- 修复 `unknown` mask 后的概率归一化与 KL 分解。
- 为候选分数、概率、形状和评测条数增加 fail-fast 校验。
- Qwen3 Prompt 强制 non-thinking，验证 mask 边界并禁止静默截断 Gold。
- 为 reference-only equivalence 产物增加不可用于 generation robustness 的机器可读标志与警告。
- 将 Qwen3 的 Transformers 最低版本提升至 4.51.0。

### Verification

- `python -m pytest -q`: `31 passed`。
- `ruff check src scripts tests --select E9,F,I`: passed。
- `python -m compileall -q src scripts tests`: passed。
- GPU Smoke：未运行，原因是 NVIDIA 驱动失配和正式 GPU 依赖尚未安装。

### Remaining blockers

- GPU 驱动/库版本一致性。
- 首轮数据源选择与可复现数据落盘。
- 人工 Verifier Gate。
- paired model-visible Equivalence Diagnostic 协议。
