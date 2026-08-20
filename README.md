# FC Class KD 快速试验

本项目研究单轮、单工具调用场景下，由验证器定义等价类的 Function Calling 蒸馏方法。

当前唯一执行基线是 [`05_FC_Equivalence_Class_KD_GoNoGo_Pilot_PRD_v5_FINAL.md`](05_FC_Equivalence_Class_KD_GoNoGo_Pilot_PRD_v5_FINAL.md)。[`FC_ClassKD_Fast_Pilot_PRD_v4.md`](FC_ClassKD_Fast_Pilot_PRD_v4.md) 和 [`FC_Equivalence_Distillation_Pilot_PRD_v3.md`](FC_Equivalence_Distillation_Pilot_PRD_v3.md) 只保留为历史设计背景。

截至 v4 阶段，项目已经完成正式 3,250 条数据划分、公共 Qwen3-1.7B S0 训练，以及 S0 在 Diagnostic 和 Train 上的自由生成；两套 K=6 候选池草稿和人工审核表也已生成。v5 已重设研究问题、等价类定义和 Go/No-Go 顺序，因此这些产物暂记为历史可复用资产，不能直接视为通过 v5 Gate。当前第一步是构造 300 条 `opportunity_probe`，实现确定性 canonicalizer，并输出 Data Opportunity Report。准确状态见 [`实验进度.md`](实验进度.md)。

## 项目导航

- [`05_FC_Equivalence_Class_KD_GoNoGo_Pilot_PRD_v5_FINAL.md`](05_FC_Equivalence_Class_KD_GoNoGo_Pilot_PRD_v5_FINAL.md)：当前唯一可执行 Pilot PRD。
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
