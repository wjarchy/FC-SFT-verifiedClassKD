# FC Class KD 快速试验

本项目研究单轮、单工具调用场景下，由验证器定义等价类的 Function Calling 蒸馏方法。

当前唯一执行基线是 [`FC_ClassKD_Fast_Pilot_PRD_v4.md`](FC_ClassKD_Fast_Pilot_PRD_v4.md)。完整研究设计 [`FC_Equivalence_Distillation_Pilot_PRD_v3.md`](FC_Equivalence_Distillation_Pilot_PRD_v3.md) 只作为背景，不用于扩大首轮范围。

当前已经完成正式 3,250 条数据划分、公共 Qwen3-1.7B S0 训练，以及 S0 在 Diagnostic 和 Train 上的自由生成。两套严格 K=6 候选池草稿和 200 条正式人工盲审表已经生成，现在等待人工审核 Verifier 标签；通过后才能冻结候选池。准确状态见 [`实验进度.md`](实验进度.md)。

## 项目导航

- [`实验实施计划.md`](实验实施计划.md)：从当前进度到三分支评测的阶段安排。
- [`实验决策记录.md`](实验决策记录.md)：影响实验定义和公平性的固定决定。
- [`实验工作日志.md`](实验工作日志.md)：逐次执行记录。
- [`docs/正式实验执行流程.md`](docs/正式实验执行流程.md)：S0 优先的新流程及数据隔离边界。
- [`docs/公共S0训练说明.md`](docs/公共S0训练说明.md)：正式 SFT 的数据、参数、选模和断点恢复说明。
- [`docs/验证器人工审核说明.md`](docs/验证器人工审核说明.md)：Excel 标注方法、标签含义、审核门槛和候选池冻结步骤。
- [`reports/公共S0正式训练结果.md`](reports/公共S0正式训练结果.md)：三轮结果、最终选模和独立复核。
- [`docs/ToolACE数据修复与筛选说明.md`](docs/ToolACE数据修复与筛选说明.md)：丢失参数语义的影响、修复方法和 2,103 条筛选过程。
- [`docs/xLAM数据筛选说明.md`](docs/xLAM数据筛选说明.md)：xLAM 实际参数格式、保守处理和 BFCL 泄漏剔除。
- [`docs/HuggingFace授权与数据下载说明.md`](docs/HuggingFace授权与数据下载说明.md)：xLAM gated 数据的网页授权和 AutoDL 登录步骤。
- [`docs/数据与文件流向.md`](docs/数据与文件流向.md)：每个阶段读取和写入哪些文件。
- [`docs/多台机器协作说明.md`](docs/多台机器协作说明.md)：公司、个人和实验机器之间如何同步。
- [`reports/README.md`](reports/README.md)：已经生成的检查报告。

## 当前代码包含什么

- 验证器：解析工具调用、检查参数说明、保守规范化、划分四个主类。
- 数据处理：读取 ToolACE/xLAM、筛选样本、恢复原始 ToolACE 参数说明、固定随机划分。
- 蒸馏计算：候选序列打分、类别聚合、Candidate KD、Class KD 和 KL 分解检查。
- GPU 调试与训练：Qwen3 提示、QLoRA、冒烟测试、小样本过拟合和可恢复的公共 S0 训练入口。
- 自动测试：当前 64 个测试全部通过。

大体积 JSONL、模型权重、候选分数和训练输出不提交到 Git。数据版本、构建参数和 SHA256 保存于 `data/processed/manifest.json`。

## 本机快速检查

```bash
.venv/bin/python scripts/doctor.py
.venv/bin/python -m pytest -q
```

重新构建当前正式合并数据：

```bash
.venv/bin/python scripts/build_data.py --sources xlam toolace --seed 42
```

## 当前正式执行顺序

1. 训练只学习 Train Gold 的公共 S0。
2. S0 分别在 Train 与 Diagnostic 自由生成，构造互相隔离的两套 K=6 候选池。
3. 人工抽检最终候选池中的验证器标签，通过后冻结候选池。
4. 在 Diagnostic 比较 8B Teacher 与 S0 的自由生成 AST，以及两者选择正确候选类的能力。
5. 诊断通过后缓存完整 Train Teacher Scores。
6. 从同一 S0 训练 SFT-cont、Candidate KD、Class KD。
7. 统一在 Internal Test、Equiv Diagnostic 和 BFCL 上自由生成评测。

不要提交 `.env`、访问令牌、SSH 私钥、模型权重或敏感原始数据。Shell 下载不得使用 `127.0.0.1:17890`；该代理仅供 OpenAI/Codex。
