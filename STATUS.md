# Fast Pilot 当前进度

快照日期：2026-08-10

执行基线：`FC_ClassKD_Fast_Pilot_PRD_v4.md`

当前结论：**工程骨架阶段；尚未通过正式实验前置 Gate，也没有可用于论文结论的模型结果。**

## 证据口径

- **本机已验证**：当前 checkout 中能直接看到或在本机重新运行得到的结果。
- **历史外部产物**：旧机器生成、仓库仅保留摘要，但原始数据未随 Git 同步。
- **未完成**：不能因存在函数、占位目录或模板而记为实验完成。

## PRD Step 1–9 总表

| PRD Step | 目标 | 当前状态 | 可核验证据 / 缺口 |
|---|---|---|---|
| 1 | 环境、数据、Verifier | **部分完成** | 配置、adapter、filter、split、Verifier 已实现；本机仅有 4 条 fixture，无全量 processed/splits；正式人工 Verifier Gate 未做 |
| 2 | 0.6B Smoke / Tiny Overfit | **未通过** | `smoke_test.py` 与 fixture 已有，但本机无 CUDA、未实跑；`tiny_overfit.py` 未实现 |
| 3 | Pre-Diagnostic | **未开始** | `run_diagnostic.py` 未实现，无 Teacher/Student 诊断结果 |
| 4 | 公共 1.7B SFT Student S0 | **未开始** | `train_sft.py` 未实现，无 S0 adapter/metrics/predictions |
| 5 | 冻结 K=6 Candidate Pool | **未完成** | 仅有可变长度 rule-candidate helper；缺 S0 hard error、严格 K=6、来源标签、冻结文件与统计 |
| 6 | 8B Teacher Score Cache | **未开始** | 序列分数公式 helper 已有；模型打分与 cache 脚本未实现 |
| 7 | 三条同预算训练分支 | **未开始** | KD loss helper 已有；三个训练入口及公平性记录未实现 |
| 8 | Internal / Equivalence / BFCL | **未开始** | 内部指标 helper 已有；生成评测脚本和 BFCL adapter 未实现 |
| 9 | Go / Weak-Go / No-Go | **未开始** | 只有结果模板，无实验结果 |

## 本机已验证

- `python -m pytest -q`：**18 passed**（2026-08-10，CPU 最小测试环境）。
- 测试覆盖：parser、Verifier 基本分类、规范化幂等性、xLAM synthetic adapter、
  rule candidate、sequence score、类别聚合、Candidate KL 分解、KD loss 有限性。
- Git 当前历史有 3 个既有提交；两份 PRD 原先未跟踪。
- 目标 GitHub 仓库可通过 SSH 访问，检查时为空仓库。

## 不能高估的历史结果

### ToolACE 数据

旧说明记录 ToolACE-only 共约 2688 条，自动缩放后的划分约为：

| Split | 历史记录条数 |
|---|---:|
| train | 1654 |
| val | 248 |
| internal_test | 413 |
| diagnostic | 206 |
| equivalence_diagnostic | 165 |

这些 JSONL **不在当前 checkout 中**，因此这里只能记为“历史外部产物”，不能记为本机已复现。
精确复现应同步 `data/processed/all_records.jsonl` 与 `data/splits/*.jsonl`，或按固定来源版本、
筛选代码、参数和 seed 重建并记录 manifest/hash。

### Verifier 报告

`reports/verifier_audit.json` 显示 200 条 Gold 的 self-check rate 为 100%，并统计了规则候选的
Verifier 输出类别。这是**自动 sanity check**，不是 PRD §4.4 的正式 Gate：候选类别没有独立人工
真值，因此不能计算 correct/equivalent precision 或 overall audited accuracy。

正式进入训练前仍需：

1. 抽样 100–300 个覆盖四主类的候选并进行人工盲标；
2. 计算 correct/equivalent precision 与 overall accuracy；
3. 同时满足 `precision >= 98%`、`overall accuracy >= 95%`；
4. 保存标注协议、逐条标签、混淆矩阵和错误修订记录。

### Equivalence Diagnostic

当前 `build_equivalence.py` 只生成 gold call 的 key reorder/default/canonical 变体 helper；它没有改变
模型实际看到的 prompt/tool schema，也没有配套生成评测。若直接用同一预测对规范化后的等价 Gold
计分，无法测出模型对表面形式的敏感性。因此 Step 8b 仍记为未实现，正式运行前必须先冻结可识别的
扰动对象、配对方式和 `RobustnessDrop` 计算口径。

## 已知工程注意点

- `smoke_test.py` 已修正为：SFT loss 对 LoRA 做 backward，Class-KD toy loss 单独验证可导性。
  **端到端 Class KD → LoRA 梯度**仍要等候选序列打分接入训练图后验证。
- `build_data.py` 已改为按规范化 query 字符串去重，避免用 Python 进程随机化 hash 作为持久实验逻辑。
- 当前默认 `ruff check` 尚有历史类型注解/格式风格诊断；核心 `E9/F` 问题已清理，完整 lint 尚未设为验收 Gate。
- 本机当前没有完整项目依赖和 CUDA；测试使用的是仓库内最小 `.venv`，该目录不会提交。

## 下一步（严格顺序）

1. **数据落盘与版本化记录**：同步或重建 ToolACE/xLAM 数据，生成 split manifest、来源 revision、
   参数、文件 hash；确认 BFCL 零污染。
2. **正式 Verifier 人工 Gate**：先通过再继续；失败则修 Verifier 并重审。
3. **冻结 Equivalence Diagnostic 协议**：确保 paired perturbation 真正改变模型输入/输出表面表示，
   同时保持语义与难度可控。
4. **GPU Step 2**：运行已修复的 smoke；实现 Tiny Overfit，并满足 loss 明显下降、训练集 AST ≥ 90%。
5. **实现并运行 Pre-Diagnostic**：Teacher Top-1、Teacher–Student gap、有效错误候选比例、等价覆盖率
   全部过 Gate 后，才租 4090 跑 Step 4–9。

首轮不扩展第二 seed、Token KD、Class-Hard、ACEBench、梯度冲突、分层 KD、multi-turn 或 RL；
这些继续遵守 v4 的 Phase 2 边界。
