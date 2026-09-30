# 实验报告目录

本目录保存体积较小、可以复查并适合提交到 Git 的实验汇总。逐样本预测、大型候选池和训练权重应放在 `reports/predictions/`、`data/` 或 `outputs/`，这些路径默认不提交。

当前报告：

- `快速试验结果模板.md`：V5 的 Opportunity、Gap、四分支和最终判定统一模板。
- `v7_execution_ledger.md`：V7 灾后恢复、执行命令、阶段状态与关键决策的连续账本。
- `v7_recovery_smoke.md`：V7 最低限度环境、代码和数值 smoke 结果（完成后生成）。
- `v7_data_recovery.md`：Formal 数据恢复、计数、哈希和偏差解释（完成后生成）。
- `v7_s0_recovery.md`：Formal S0 重训、Validation 选模、Internal 预测与资产哈希。
- `v7_teacher_cache_recovery.md`：固定 K=4 Teacher 概率、entropy 与缓存哈希。
- `v7_analysis_threshold_freeze.md`：在 KD outcome 前冻结的 S0 confidence / Teacher entropy 三分位。
- `v7_s0_difficulty_analysis.md`：Balanced-336×1 的配对 S0 难度、置信度、A/B/C 与 Teacher entropy 分层结果。
- `v7_shutdown_checkpoint.md`：本次关机前的精确完成边界、哈希与恢复入口。
- `v7_balanced336_e4_analysis.md`：Balanced-336×4 Candidate/Class 的主指标与预注册分层。
- `v7_equiv_all_e1_analysis.md`：Equiv-All-1393×1 Candidate/Class 的主指标与预注册分层。
- `v7_scaling_report.md`：336×1、336×4 与 Equiv-All×1 的 exposure / unique diversity 主比较。
- `v7_teacher_entropy_analysis.md`：按训练侧预冻结 Teacher 类内 entropy 三分位的机制分析。
- `v7_discordant_cases.md`：三个 setting 中 Candidate-only / Class-only 正确样本的无筛选汇总。
- `v7_gradient_analysis.md`：固定初始学生模型上预定义 16 条样本的训练信号分析；已完成并独立核对。
- `v7阶段审查检查点.md`：数据规模对比完成、训练信号分析尚未完成时，供用户审查的通俗中文暂停报告。
- `v7_final_decision.md`：V7 最终科研判定；结论为 `V7_CONDITIONAL_RESCUE`，即“有条件保留”。
- 后续新增 `opportunity_report.json/.csv` 与 `opportunity_examples.jsonl`：V5 第一生死 Gate；当前尚未生成。
- `实验环境记录.md`：RTX 3090、关键依赖和 Qwen3-0.6B 模型版本。
- `冒烟与小样本过拟合结果.md`：真实 GPU 冒烟测试与 8 条样本过拟合结果。
- `正式SFT脚本检查结果.md`：公共 S0 训练入口、断点恢复和选模规则的工程检查。
- `公共S0正式训练结果.md`：V4 三轮训练、Validation 选模、独立复核和旧 S0 文件哈希。
- `候选池初步统计.md`：V4 加入 S0 真实错误之前的规则候选覆盖。
- `合并数据质量报告.md`：V4 ToolACE+xLAM 3,250 条混合数据的质量结论；仅作 V5 数据源选择证据。
- `合并数据质量报告.html`：同一结论的便携交互报告，含筛选漏斗、指标卡和精确表格。
- `合并数据质量报告.artifact.json`：生成 HTML 的可复算数据、查询和来源说明。
- `xLAM数据审计.json`、`BFCL数据泄漏检查.json`、`合并数据质量验证.json`：机器可读的筛选与验证证据。
- `verifier_audit.json`：自动自检结果，不是独立人工审核结果。
- `验证器审核进度.md`：V4 四类盲标包与审核状态；不替代 V5 canonicalizer audit。

后续正式报告使用中文文件名，例如 `能力诊断结果.md`、`快速试验正式结果.md` 和 `是否继续实验结论.md`。

每份正式报告必须记录模型版本、数据与划分哈希、配置、随机种子、代码提交、机器和时间。不得把占位符、自动自检或未复现的历史摘要写成论文实验结论。

当前 HTML 报告已通过输入验证、打包验证和结构验证。训练机没有 Chromium，因此没有执行浏览器交互、桌面宽度和窄屏宽度检查；语义化备用内容与报告数据仍已一并嵌入。
