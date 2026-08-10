# FC Class KD — Fast Pilot

Verifier-defined Class-Level Distillation for single-turn, single-call Function Calling.

当前实验唯一执行基线是 [`FC_ClassKD_Fast_Pilot_PRD_v4.md`](FC_ClassKD_Fast_Pilot_PRD_v4.md)。
[`FC_Equivalence_Distillation_Pilot_PRD_v3.md`](FC_Equivalence_Distillation_Pilot_PRD_v3.md)
只保留为完整研究设计背景，不用于扩大首轮 Fast Pilot 范围。

> 当前阶段：no-GPU 工程骨架已建立，正式实验尚未开始。准确进度、证据边界和下一步见
> [`STATUS.md`](STATUS.md)。

## 当前仓库包含什么

- Verifier：解析、Schema 检查、保守规范化、四主类分类。
- 数据入口：xLAM / ToolACE adapter、筛选、固定 seed 划分。
- 数学组件：候选序列分数、类别聚合、Candidate/Class KL、KL 分解。
- 调试骨架：Qwen3 prompt、QLoRA 模型加载、4 条合成 smoke fixture。
- 18 个 CPU 单元测试；2026-08-10 本机复核为 `18 passed`。

当前 checkout **不包含** 全量 `data/processed/`、`data/splits/`、模型权重或训练输出。
历史机器曾报告 ToolACE-only 2688 条处理结果，但必须同步原文件或重跑后才能在本机复现。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/doctor.py
python -m pytest -q
```

数据构建：

```powershell
# 无需 xLAM gate 的最小路径
python scripts/build_data.py --sources toolace --seed 42

# 获得 xLAM 权限并登录 Hugging Face 后
python scripts/build_data.py --sources xlam toolace --seed 42
```

数据流、各阶段输入输出见 [`docs/DATA_MAP.md`](docs/DATA_MAP.md)。公司 PC、个人 PC、
实验室机器之间的同步方式见
[`docs/WORKFLOW_COMPANY_PERSONAL.md`](docs/WORKFLOW_COMPANY_PERSONAL.md)。

## Fast Pilot 执行顺序

1. 恢复/重建固定数据划分，完成人工标注的 Verifier Gate。
2. 在 GPU 上跑 `smoke_test.py`，再实现并通过 Tiny Overfit。
3. 完成 200–300 条 Pre-Diagnostic，通过后才进入正式训练。
4. 训练公共 S0，冻结 K=6 Candidate Pool，缓存 8B Teacher 分数。
5. 从同一 S0 跑 SFT Continuation、Candidate KD、Class KD。
6. 统一完成 Internal、Equivalence、BFCL 评测并给出 Go/Weak-Go/No-Go。

候选池、Teacher cache、模型输出和逐样本预测默认不进入 Git；紧凑配置、代码和 Markdown/JSON
汇总报告进入 Git。不要提交 `.env`、Token、SSH 私钥、模型权重或敏感原始数据。
