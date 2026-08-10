# Reports

本目录保存可审计、体积小、适合进入 Git 的实验汇总。逐样本预测和大型中间文件应放在
`reports/predictions/`、`data/` 或 `outputs/`，这些路径默认不提交。

当前文件：

- `verifier_audit.json`：历史 ToolACE 样本上的**自动 sanity check**；不是人工标注的 PRD Gate。
- `fast_pilot_results_template.md`：Step 8/9 统一结果表模板，当前没有实验数值。

计划中的正式报告遵循 PRD v4 §12：

- `environment.md`
- `verifier_audit.md`
- `diagnostic.md`
- `candidate_stats.md`
- `fast_pilot_results.md`
- `go_no_go.md`

报告必须同时记录模型 revision、数据/划分 hash、配置、seed、代码 commit、机器与时间，且不得把
占位符、自动 self-check 或未复现的历史摘要写成正式实验结论。
