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

## 2026-09-24：K=4 Candidate Pool 冻结

- `Equiv-All` 1,393 行、`Balanced-336` 336 行、`Internal-Equiv` 248 行，共 1,977 个 sample / 7,908 个 candidate。
- 每条顺序固定为 Gold、Equivalent、Wrong Argument、Wrong Function；功能类索引固定为 `[0,0,1,2]`，四个 surface 唯一。
- `Balanced-336` ID 严格属于 `Equiv-All`；`Internal-Equiv` 与 `Equiv-All` 无 ID 重叠。
- 机器可读清单：[`data/candidates/v7/manifest.json`](../data/candidates/v7/manifest.json)，SHA256 `4dae3a369f4de6150de29157cbdd7024b51eaab90744a6d08024a3cb6861874d`。
- 大型 JSONL 由 Git 忽略，但路径、输入/输出 SHA256 和生成命令均已保存；Teacher cache 将直接绑定此 manifest。

## 2026-09-24：Teacher 权重恢复并固定

- Teacher 固定为 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`，本地路径 `/root/autodl-tmp/models/Qwen3-8B`。
- 五个权重分片、权重索引、配置和 tokenizer 的 SHA256 已写入根 [`artifacts_manifest.json`](../artifacts_manifest.json)，模型大文件不进入 Git。
- 已在完全离线模式加载配置和 tokenizer：hidden size 4,096、36 层、词表 151,936；下一步只对冻结的 K=4 pool 生成 Teacher cache。

## 2026-09-24：Formal S0 恢复完成

- 使用冻结的 Formal Train-3000 / Validation-300，从固定 Student revision 执行 4-bit QLoRA Gold-only SFT；3 轮共 564 optimizer steps，seed 42，LR 2e-4，有效 batch 16。
- 三轮 Validation AST 为 78.33%、81.33%、82.67%；按预注册的最高 Validation AST 规则选择 epoch 3 / step 564。epoch 2 虽有更低 Validation loss，但不改变选模指标。
- 选中 adapter SHA256 为 6bb07558d42735f32ae1676fc3eaae64c786523f8bb55bb4078a90b9ef06d2d5；训练峰值显存 6,868.4 MiB。
- Internal-500 自由生成 AST 为 402/500 = 80.40%；Internal-Equiv-248 为 176/248 = 70.97%。这些值只用于后续 S0 difficulty 分层，不是 Class KD 效果结论。
- Internal-Equiv 预测从完整 Internal 预测按冻结 ID 顺序确定性派生，没有重复跑推理；两个预测文件及其 manifest 均绑定数据、checkpoint 和 decoding 定义。
- 完整报告：[v7_s0_recovery.md](v7_s0_recovery.md)；大文件路径与 SHA256 已写入根 [artifacts_manifest.json](../artifacts_manifest.json)。

### 下一步

1. 生成 Teacher 与 S0 在 Equiv-All / Internal-Equiv K=4 support 上的固定分数 cache。
2. 在查看恢复基线 outcome 前，用 Equiv-All 分布冻结 confidence / entropy tertile。
3. 恢复 Balanced-336×1 Candidate/Class 模型和逐样本预测。

## 2026-09-24：Teacher K=4 cache 恢复完成

- Qwen3-8B BF16 在冻结 K=4 support 上用 tau=2、alpha=1 评分；Equiv-All 1,393 行与 Internal-Equiv 248 行均完整。
- 首次 candidate batch 4 在第 173 个长样本因完整词表 log-softmax OOM；异常 fallback 因 traceback 持有显存而不能可靠回收。172 行 partial 已移至 /tmp 诊断目录，不进入正式结果。
- scorer 改为显式、写入 run signature 的 candidate micro-batch；90 项测试通过。正式 cache 从零以统一 batch 2 重算，未混用两种数值路径，峰值显存 21,595.4 MiB。
- 概率完整性检查通过：每行 4 个 candidate probability 与 3 个 class probability 均有限且和为 1；normalized intra-class entropy 全部位于 [0,1]。
- sanity：Teacher 正确功能类在 Equiv-All 和 Internal-Equiv 均为 100% class argmax；这里只记录顺手得到的历史 Gate 一致性，不把它设成新阻塞 Gate。
- Equiv-All 正确类概率 min/median/max 为 0.5241/0.6982/0.8618；normalized intra-class entropy median 为 0.9758，仍保留相对三分位差异。
- 完整报告：[v7_teacher_cache_recovery.md](v7_teacher_cache_recovery.md)；manifest 与两个大型 JSONL 的 SHA256 已写入根 [artifacts_manifest.json](../artifacts_manifest.json)。

### 下一步

1. 用 selected S0 对同一 Equiv-All / Internal-Equiv support 评分。
2. 由 Equiv-All 分布冻结 S0 confidence 和 Teacher entropy tertile 后，才评测恢复的 KD 基线 outcome。
