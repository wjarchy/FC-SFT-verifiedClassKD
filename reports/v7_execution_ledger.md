# V7 执行账本

> 当前执行基线：[ClassKD_V7_Lean_Execution_PRD.md](../ClassKD_V7_Lean_Execution_PRD.md)。
> 本账本只记录 V7 恢复与续跑，不把 V4/V5/V6 历史摘要冒充本次结果。

## 2026-09-24：灾后恢复启动

- 起始分支：`main`；起始提交：`07d72d62cb6a880388412382308567398e230256`。
- V7 工作分支：`recovery/v7-rescue`。
- 工作树起始状态：只有用户恢复的 `ClassKD_V7_Lean_Execution_PRD.md` 未跟踪；数据、模型、checkpoint、Teacher cache 和预测均不存在。
- GPU：NVIDIA GeForce RTX 3090 24GB；CUDA 与 BF16 可用。
- 可复用环境：Conda `wj`，Python 3.11.16、PyTorch 2.5.1+cu124、Transformers 4.57.6、PEFT 0.20.0、bitsandbytes 0.50.0。
- 旧代码回归：64 项测试全部通过。
- 固定数据 revision 已可访问：xLAM `26d14ebf...`、ToolACE cleaned `74befe8a...`、ToolACE schema `6bda777c...`。
- 已确认 Hugging Face 登录有效；下载缓存使用 `/root/autodl-tmp/hf_cache`。

### Formal 数据恢复调查

- xLAM 原始 60,000 条；旧严格筛选通过 19,642 条；精确问题去重后 18,251 条。
- 按现有 BFCL 排除和更严格规范化去重后为 17,924 条。
- ToolACE 严格恢复后为 2,103 条。
- 单用 xLAM 的结构比例无法解释历史 `A/B/C ≈ 1094/112/185`；旧构建器的 ToolACE+xLAM 双源抽样能得到接近该比例的自然分布。
- 剩余小差异指向旧机器未提交的 V6 等价构造实现，重点是递归数值表示、嵌套对象换序和安全默认值契约。恢复时只使用 PRD 允许的严格规则，不按模型结果挑样本。

## 2026-09-24：Formal 数据恢复完成

- 生成命令：`HF_HOME=/root/autodl-tmp/hf_cache PYTHONPATH=src /root/miniconda3/envs/wj/bin/python scripts/v7_build_formal_data.py --seed 42`。
- 数据定义与逐文件哈希：[`data/processed/v7/manifest.json`](../data/processed/v7/manifest.json)，manifest SHA256 为 `5c46124daae63779e164a2b21ef12d52d9e1e9cc3b987c32519b47d3c2c94512`。
- Formal Train / Validation / Internal：3,000 / 300 / 500；Internal 中等价样本 248、singleton 252。
- Train 等价分布：A=1,096、B=112、C=185，`Equiv-All=1,393`；`Balanced-336` 为 A/B/C 各 112。
- 相对 V7 PRD 中历史约数的偏差：A 和总等价样本各 +2，B/C 无偏差。旧机器未提交精确行 ID，因此不能声称逐样本复刻；本次恢复数据从此冻结，所有 V7 分支共用，不根据模型结果再调整。
- Internal 从与 Train/Validation 相同的 4,650 条校准池中按 248/252 分层抽取，来源为 ToolACE 245、xLAM 255，避免从完整 xLAM 主导的剩余池抽样造成新的来源漂移。
- 校验：八个文件的行数、SHA256、唯一 ID、来源/类别统计全部匹配 manifest；Train/Validation/Internal 两两无 ID 重叠；`equiv_all`、`internal_equiv`、`balanced_336` 共 1,977 条均能确定性生成 K=4，且 Gold/Equivalent 同类、两个负例分属另外两类。
- 代码回归：70 项测试通过；`KL_candidate = KL_inter + KL_intra` 数值恒等和 Candidate/Class 反向传播均已覆盖。

### 下一步

1. 固定并下载 Student/Teacher revision，完成真实 GPU 端到端 smoke。
2. 使用冻结的 Formal Train/Validation 恢复 Formal S0。
3. 再生成 K=4 Teacher cache 和恢复 Balanced-336×1 两基线。

## 2026-09-24：Phase 0 与 S0 训练前 Gate 完成

- Student 固定为 `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，本地路径 `/root/autodl-tmp/models/Qwen3-1.7B`；两个权重分片 SHA256 已写入根 manifest。
- Teacher 固定为 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`，权重尚待下载。
- 73 项测试通过。独立数值检查：`KL_candidate=0.149475211619`、`KL_inter=0.022730254805`、`KL_intra=0.126744956814`，残差 `2.776e-17`。
- 真实 1.7B 4-bit QLoRA smoke：SFT loss `1.4807`；Class KD loss `0.002017`；SFT 和 Class KD 均产生有限 LoRA 梯度，冻结 base 无梯度；adapter 可保存；峰值显存 4,256.8 MiB。
- Formal 预编码首次用旧上限 2,048 时在 `xlam_4469` 主动失败，证明无静默截断。全量审计显示 Train 仅 2/3,000 条超过 2,048，最大 2,392；Validation 最大 1,893；Internal 最大 1,965。
- 新增 V7 专用配置 `configs/v7_rtx3090.yaml`，把 `max_length` 固定为 2,560。全部 S0 Gold 与 Equiv-All/Balanced/Internal-Equiv K=4 均低于此上限；Internal 最长 prompt+192 生成预算为 2,118。
- 最长 2,392-token 样本真实前反向通过，loss `0.248720`，峰值显存 8,047.3 MiB。Formal 训练预算确认为 3 轮、每轮 188 optimizer steps、总计 564 steps。
- 完整报告：[`v7_recovery_smoke.md`](v7_recovery_smoke.md)。

### 下一步

1. 启动可断点恢复的 Formal S0 训练和三轮 Validation 选模。
2. 保存 selected adapter 与 Internal/Equiv free-generation predictions。
3. 下载固定 Teacher 并生成 K=4 cache。
