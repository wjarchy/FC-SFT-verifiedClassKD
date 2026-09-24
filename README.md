# FC Class KD 快速试验

本项目研究单轮、单工具调用场景下，由验证器定义等价类的 Function Calling 蒸馏方法。

当前唯一执行基线是 [`ClassKD_V7_Lean_Execution_PRD.md`](ClassKD_V7_Lean_Execution_PRD.md)。V3–V6 文档只保留为历史设计与恢复依据，不得覆盖 V7 的固定执行顺序和判据。

当前机器为灾后恢复实例。V6/V7 的大文件资产已经丢失，现按 V7 PRD 只恢复 Formal 数据、Formal S0、K=4 候选池、Teacher cache 和必要基线，不重跑已经有历史结论的前置 Gate。实时状态见 [`reports/v7_execution_ledger.md`](reports/v7_execution_ledger.md)。

## 项目导航

- [`ClassKD_V7_Lean_Execution_PRD.md`](ClassKD_V7_Lean_Execution_PRD.md)：当前唯一可执行 PRD。
- [`reports/v7_execution_ledger.md`](reports/v7_execution_ledger.md)：V7 恢复、执行和阶段结果账本。
- [`artifacts_manifest.json`](artifacts_manifest.json)：大文件路径、哈希、命令和模型/数据版本清单。
- [`06_FC_形式不变性_ClassKD_PRD_v6.md`](06_FC_形式不变性_ClassKD_PRD_v6.md)：V6 历史定义与结果背景。
- [`实验实施计划.md`](实验实施计划.md)：V5 P0–P15 的当前执行与验收清单。
- [`实验决策记录.md`](实验决策记录.md)：影响实验定义和公平性的固定决定。
- [`实验工作日志.md`](实验工作日志.md)：逐次执行记录。
- [`docs/正式实验执行流程.md`](docs/正式实验执行流程.md)：V5 Gate 优先的完整流程及数据隔离边界。
- [`docs/公共S0训练说明.md`](docs/公共S0训练说明.md)：V4 S0 的历史训练与复现说明；是否能用于 V5 需重新判定。
- [`docs/验证器人工审核说明.md`](docs/验证器人工审核说明.md)：V4 Excel 审核的历史说明；不替代 V5 canonicalizer audit。
- [`reports/公共S0正式训练结果.md`](reports/公共S0正式训练结果.md)：V4 三轮 S0 结果、选模和独立复核。
- [`docs/ToolACE数据修复与筛选说明.md`](docs/ToolACE数据修复与筛选说明.md)：丢失参数语义的影响、修复方法和 2,103 条筛选过程。
- [`docs/xLAM数据筛选说明.md`](docs/xLAM数据筛选说明.md)：xLAM 实际参数格式、保守处理和 BFCL 泄漏剔除。
- [`docs/HuggingFace授权与数据下载说明.md`](docs/HuggingFace授权与数据下载说明.md)：xLAM gated 数据的网页授权和 AutoDL 登录步骤。
- [`docs/数据与文件流向.md`](docs/数据与文件流向.md)：每个阶段读取和写入哪些文件。
- [`docs/多台机器协作说明.md`](docs/多台机器协作说明.md)：公司、个人和实验机器之间如何同步。
- [`reports/README.md`](reports/README.md)：已经生成的检查报告。

## 当前代码包含什么

- V4 验证器：解析工具调用、检查 schema、生成错误分析标签；四个旧主类不得作为 V5 KD class。
- 数据处理：读取 ToolACE/xLAM、筛选样本、恢复原始 ToolACE 参数说明、固定随机划分。
- 蒸馏计算：候选序列打分、类别聚合、Candidate KD、Class KD 和 KL 分解检查。
- GPU 调试与训练：Qwen3 提示、QLoRA、冒烟测试、小样本过拟合和可恢复的公共 S0 训练入口。
- 自动测试：V4 最近一次完整工程检查为 64 个测试全部通过；V5 新增实现仍需按新 Gate 补测。

大体积 JSONL、模型权重、候选分数和训练输出不提交到 Git。数据版本、构建参数和 SHA256 保存于 `data/processed/manifest.json`。

## 本机快速检查

```bash
.venv/bin/python scripts/doctor.py
.venv/bin/python -m pytest -q
```

重新构建 V4 历史合并数据（**不是 V5 当前正式数据命令**）：

```bash
.venv/bin/python scripts/build_data.py --sources xlam toolace --seed 42
```

## v5 当前执行入口

1. 选定一个主数据源并固定 300 条 `opportunity_probe`。
2. 实现确定性 canonicalizer（规范化器）。
3. 跑完 v5 要求的 14 类 canonicalizer 单元测试。
4. Teacher 对每条 Query 自然采样 8 次。
5. 对采样结果做 surface 去重、合法性验证和 canonicalize。
6. Teacher 对 unique candidates 重新做完整序列打分。
7. 输出 Data Opportunity Report（数据机会报告）。
8. 根据 Opportunity Gate 决定是否继续 S0 和四分支 KD；报告出来前不进入大规模训练。

不要提交 `.env`、访问令牌、SSH 私钥、模型权重或敏感原始数据。Shell 下载不得使用 `127.0.0.1:17890`；该代理仅供 OpenAI/Codex。
