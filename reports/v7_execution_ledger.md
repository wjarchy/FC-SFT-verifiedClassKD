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

## 2026-09-24：S0 fixed-support cache 与分析阈值冻结

- selected S0 adapter 以 4-bit、tau=2、alpha=1 对 Equiv-All 1,393 行和 Internal-Equiv 248 行完成 K=4 打分；正式峰值显存 13,291.5 MiB。
- Equiv-All S0 正确功能类概率 min/median/max 为 0.5150/0.6027/0.7210；Internal-Equiv 为 0.5172/0.6018/0.6884。fixed support class argmax 全对不等同于 free-generation 全对，difficulty 仍使用连续 confidence 与独立 free-generation correctness 两个视角。
- 在任何恢复 KD baseline prediction 生成前，已只用 Equiv-All-1393 冻结三分位阈值。
- S0 confidence q1/q2 = 0.58942356 / 0.61558665；Teacher normalized intra-class entropy q1/q2 = 0.96580232 / 0.98335794。
- 线性 quantile 和边界规则固定为 low ≤ q1，q1 < mid ≤ q2，high > q2；训练侧桶数均为 465/464/464。
- 完整定义：[v7_analysis_threshold_freeze.md](v7_analysis_threshold_freeze.md) 与 [configs/v7_analysis_thresholds.json](../configs/v7_analysis_thresholds.json)。

### 下一步

1. 恢复 Balanced-336×1 Candidate/Class checkpoint 与 Internal predictions。
2. 立即执行优先级最高的 S0 Difficulty Analysis，再进入 scaling。

## 2026-09-24：用户关机前保存点

- 用户要求在合适节点保存并暂停。当前正在运行的 Balanced-336×1 Candidate KD 已完成整个 epoch，不在半个 optimizer step 或未落盘状态中止。
- Candidate 分支完成 336 exposures / 21 optimizer steps；mean KD loss 0.037090，峰值显存 18,363.7 MiB。
- 最终 checkpoint 为 checkpoint-step-000021，adapter SHA256 为 e458945adc6a16c80d11f306f8ab761f1cf812734c26e3708c9704bb4eeea0a2。
- optimizer、RNG、trainer state、run manifest、training summary 和 training log 均已保存并写入根 artifacts_manifest.json；该目录约 267 MiB。
- Candidate 训练已完成但尚未做 Internal free-generation；Balanced-336×1 Class 分支尚未启动，因此当前没有不公平的半对比较，也没有产生 KD outcome。
- 关机前没有后台 GPU 进程需要保留。下一次从 Class 分支开始，然后依次评测 Candidate/Class 并执行 S0 Difficulty。
- 精确恢复说明：[v7_shutdown_checkpoint.md](v7_shutdown_checkpoint.md)。

## 2026-09-29：Balanced-336×1 与 S0 Difficulty 完成

- 从已推送检查点恢复后，本地与远端均为 `dfe86e3`，90 项测试通过；Candidate 未重训。Class 使用同一 S0、数据、K=4、Teacher cache、seed 与超参完成 336 exposures / 21 steps，最终 adapter SHA256 为 `727363c99f26be5b6b38e9971414dd045d0ae727cecca1855ea408feedf8bbb7`。
- 冻结的 batch=4 贪心评测结果：Full Internal 上 Candidate 为 360/500（72.00%），Class 为 306/500（61.20%）；Internal-Equiv 上分别为 156/248（62.90%）与 151/248（60.89%）；Internal-Singleton 上分别为 204/252（80.95%）与 155/252（61.51%）。
- Internal-Equiv 的 Class-Candidate delta 为 -2.02pt，配对 bootstrap 95% CI [-5.24pt, 0.81pt]，McNemar exact p=0.3018。Full 的 -10.80pt 主要来自 singleton 的 -19.44pt；Class 的 invalid 多为 `arguments` 退化为 scalar/list，而非截断或乱码。
- S0 Correct/Wrong delta 分别为 -2.27pt/-1.39pt；S0 confidence 从 high/mid/low 依次为 -2.47pt/-1.09pt/-2.67pt，不满足“难度越高 Class 相对越好”的单调趋势。当前 Weak-S0 条件不受支持，待 scaling 后按 PRD 正式记为 `SKIPPED`。
- 历史 78.23%/77.02% 不能作为逐样本复现目标：灾后 Formal cohort 仅按 revision、规则与聚合控制数重建，旧机器未保存最终 ID。为隔离另一个口径差异，用逐字节相同的 step-0 S0 在 batch=4 上诊断 Internal-Equiv，得到 169/248（68.15%），低于原 batch=1 的 176/248（70.97%）；batching 解释 2.82pt，但不足以解释全部 KD 负迁移。
- 实现审计确认 step-0 adapter 与 selected S0 SHA256 完全一致，pool 顺序固定为 `[Gold, Equivalent, WrongArg, WrongFn]`、class index 为 `[0,0,1,2]`，Class loss 是正确的聚合 KL；未发现可归因于错误权重、cache 错配、随机解码或 parser 的污染。
- 正式报告：[v7_s0_difficulty_analysis.md](v7_s0_difficulty_analysis.md)。训练、预测、诊断与报告路径/哈希已写入根 [artifacts_manifest.json](../artifacts_manifest.json)。下一步严格执行冻结矩阵的 Balanced-336×4 与 Equiv-All-1393×1。

## 2026-09-29：Scaling / Diversity、Entropy 与 Discordant Cases 完成

- 冻结矩阵全部完成：Balanced-336×4 的每支 1,344 exposures / 84 steps；Equiv-All-1393×1 的每支 1,393 exposures / 88 steps。四支均从同一 Formal S0 出发，使用同一 K=4 定义、Teacher cache、seed 42、LR 2e-4、tau 2、batch 16 和 max length 2,560。
- Balanced-336×4：Full Candidate/Class 为 363/500（72.60%）/355/500（71.00%），delta -1.60pt，95% CI [-4.00pt, 0.80pt]；Internal-Equiv 为 160/248（64.52%）/159/248（64.11%），delta -0.40pt，95% CI [-3.63pt, 2.82pt]。
- Equiv-All-1393×1：Full 为 359/500（71.80%）/371/500（74.20%），delta +2.40pt，95% CI [-0.20pt, 5.20pt]，McNemar p=0.1038；Internal-Equiv 为 153/248（61.69%）/162/248（65.32%），delta +3.63pt，95% CI [-0.81pt, 8.06pt]，McNemar p=0.1496。
- Balanced-336 从 1 轮重复到 4 轮仅把目标 delta 从 -2.02pt 改善到 -0.40pt；近似相同 exposure 下，Equiv-All 的 delta 比 Balanced×4 高 4.03pt。结果支持“更多 unique query 与自然 A/B/C 分布”的组合条件优于只重复小集合，但两因素同时变化，不能进一步作因果拆分。
- 独立复核确认六个 prediction 文件各 500 行、ID 唯一且与 Internal 顺序逐项一致；三组 Full/Equiv correct counts 和 candidate-only/class-only 配对数均与报告一致。
- Teacher entropy 在 Equiv-All 的 low/mid/high delta 为 +4.12/+4.48/+2.38pt，没有随 entropy 上升而增强；Balanced×4 方向也不一致。因此 entropy 机制不受支持。
- Equiv-All 的 S0 confidence high/mid/low delta 为 0.00/+4.35/+6.67pt，但 S0 free-generation Correct/Wrong 为 +6.25/-2.78pt。difficulty 仅部分支持，未达到 Weak-S0 的明确触发条件；按 PRD 正式记为 `Weak-S0 SKIPPED`，不为寻找正结果追加训练。
- Discordant case 不做胜例筛选：三个 setting 的 Candidate-only/Class-only 分别为 10/5、9/8、11/20，共保存 63 行完整上下文。
- 单 seed 且 Equiv-All 两个主区间均跨 0，因此当前是条件性、方向一致的 rescue 信号，不是 universal superiority 证据。

## 2026-09-29：Gradient Analysis 定义冻结

- PRD 情况 A 被触发：Balanced-336×1 的 Class Full 明显退化，×4 后差距缩小但仍未胜出；执行最小梯度诊断以检查类内项带来的 signal strength / direction。
- 在查看任何梯度结果前冻结：Formal S0、Balanced-336、seed 42、epoch 0 的第一个真实 optimizer group（16 条）、tau 2、alpha 1、max length 2,560；eval mode 关闭 dropout。
- 同一次 forward graph 分别计算 `L_candidate`、`L_inter=L_class` 与 `L_intra=L_candidate-L_class`，报告三者 mean-loss gradient 的范数、inter/intra cosine 与分解残差。
- 该诊断只解释优化几何，不替代正式 evaluator，也不外推为总体显著性结论；脚本为 `scripts/v7_analyze_gradients.py`，输出为 `v7_gradient_analysis.md/.json`。

### 首次执行的工程修复

- 从已推送的 `bb54bb7` 开始执行；前 6 条只打印进度，没有输出任何梯度数值或正式文件，第 7 条 forward 因 CUDA allocator OOM 退出。
- 失败样本长度 756 tokens，并不比第 1 条的 778 tokens 更长；GPU 进程退出后显存完全释放，定位为跨样本保留三套诊断梯度和 allocator 碎片，而非数据超长或硬件故障。
- 修复仅将每套已算出的梯度立即无损转存 CPU，并在每条样本结束后释放临时对象/CUDA cache；样本选择、loss、forward graph、模型 mode 和报告量均不变。正式结果从第 1 条重新计算，不混用首次 partial。
- 第二次从第 1 条重算后通过原失败点，但在第 8 条（该组最长的 1,200 tokens）forward 前仍有 22.58 GiB 被标记为 allocated，说明仅删除局部变量未回收 Python 引用环；该次同样未写正式输出、未打印任何梯度值。
- 第二次工程修复在相同释放点增加 `gc.collect()` 后再清 CUDA cache，不改变计算图或统计定义。93 项测试继续通过；正式结果仍将从第 1 条统一重算。
- 第三次从第 1 条重算仍在第 8 条 forward 前以同样的 22.58 GiB allocated 状态 OOM，证明显式删除、CPU 转存、GC 和 empty-cache 不足以清理该进程内的滞留；该次仍无正式输出或梯度数值。
- 下一工程路径固定为每条样本重载同一 Formal S0（或每条独立子进程）后在 CPU 聚合，以隔离 CUDA 生命周期；这不改变冻结样本、loss 或聚合定义。

## 2026-09-29：用户审查暂停点

- 用户要求先保存并细致同步后暂停。已生成 [v7阶段审查检查点.md](v7阶段审查检查点.md)，汇总完成边界、主结果、原因分析、统计限制、训练信号分析的工程状态和精确恢复入口。
- `Weak-S0 SKIPPED` 已确定；`v7_final_decision.md` 尚未生成，临时判断倾向 `V7_CONDITIONAL_RESCUE`，但保留与 `V7_NO_GO` 的边界争议供审查，不提前落最终标签。
- 暂停时不存在 `v7_gradient_analysis.md/.json`，没有 partial 混入；GPU 无进程、显存占用 0 MiB。
- 已完成科学结果的基准提交为 `3b7512ab88301c1a680ccabc4cda865d56a995d9`，本地与远端一致。当前文档保存提交将在本条之后生成并推送。

## 2026-09-29：强化中文表达规则

- 用户反馈本次交付包含过多英文和专业词，影响阅读。阶段审查报告已改写为通俗中文，保留的英文只限正式方法名、文件名、代码字段和最终判定标签，并在首次出现时解释。
- 已新建本机全局规则文件 `/root/.codex/AGENTS.md`。今后这台机器上的 Codex 回复、进度、计划、报告、章节标题和表格标题都默认使用通俗、直接、简明的中文。
- 项目根目录 `AGENTS.md` 也已强化同一要求，并增加交付前语言检查：能用中文说清楚的内容，不使用英文缩写或专业黑话。

## 2026-09-30：恢复执行并冻结逐条隔离实现

- 用户明确要求继续工作，项目状态由“等待用户审查”恢复为“执行中”。
- 显存修复已按暂停前约定完成：16 条固定样本各自在全新子进程中加载同一个初始学生模型，只计算一条样本；结果转到普通内存后子进程退出，显存随进程释放；主进程最后计算完全相同的 16 条平均训练信号。
- 这次修复只改变显存生命周期，不改变样本、顺序、模型、随机种子、损失公式、温度、序列长度或汇总方法。显存检查模式只验证指定样本能否完成和释放，不读取、不展示训练信号数值，也不生成正式报告。
- 在查看任何新结果前，脚本已通过格式检查、位置去重与越界保护检查，以及全部 93 项测试。脚本 SHA256 为 `46a5512a3986d8826f8ad0f180949a0ba19beb6d84aded6a7664a5c3025dca73`。
- 下一步先检查固定位置 1 和该组最长的固定位置 8；两条都通过后，才从干净提交正式计算全部 16 条。

### 第一次逐条隔离检查与第二层显存修复

- 第一次逐条隔离检查中，固定位置 1 完成并退出；固定位置 8 在一次同时计算 4 个候选的前向过程中显存不足。失败发生在求导前，说明最长样本即使没有跨样本残留，一次放入 4 个候选仍超过 24GB。
- 该次检查未读取或打印任何训练信号数值，也没有生成 `v7_gradient_analysis.md/.json`。失败进程退出后显存恢复到 0 MiB。
- 第二层修复把每条样本的 4 个候选逐个计算。先求损失对 4 个候选分数各自的影响，再按链式法则逐个求参数信号并相加。数学上与直接对完整损失求导相同，只降低同一时刻保留的计算内容。
- 新增回归测试，用小型可求导例子比较“直接求导”和“逐候选链式求导”，三部分信号均一致。全部测试由 93 项增至 94 项并通过。
- 候选小批量固定为 1，脚本 SHA256 更新为 `ad9c5a52c4239fa5315da7ef8e92ce011cfae14874ca982b8085d02f17239a35`。再次查看结果前先提交并推送，然后重跑固定位置 1 和 8。
- 更新后的显存检查已通过：固定位置 1（`toolace_6797`）和本组最长的固定位置 8（`xlam_30185`）都独立完成，两个子进程均正常退出，显存恢复到 0 MiB。
- 检查结束后再次确认正式报告不存在，说明检查模式没有把局部结果混入正式分析。下一步从已推送的干净提交统一计算全部 16 条。

## 2026-09-30：训练信号分析完成

- 从干净提交 `2e41f9a4a05e81d5ea92ff342be7550fa62560e2` 正式计算全部 16 条固定样本。每条样本单独启动进程，4 个候选逐个求导，全部成功；峰值显存 12,180.9 MiB，结束后显存为 0 MiB。
- 完整候选蒸馏、功能类别蒸馏和同类写法额外信号的长度分别为 0.8178、0.7958、0.0761。完整候选信号只强 2.77%，额外信号约为类别间信号的 9.56%，两部分方向相似度为 0.2495，属于轻微同向。
- 这不支持“Class KD 因为少一项损失而信号大幅变弱”作为小数据退化的主要解释。额外信号可能帮助早期学习，但无法单独解释 1 轮训练时的巨大差距。
- 独立核对确认 16 条样本顺序、输入和模型哈希、全部有限值、损失精确分解、报告数值与显存上限均正确。训练信号分解的相对误差为 0.23%，逐样本最大 0.96%，不改变强弱判断。
- 正式结果为 [v7_gradient_analysis.md](v7_gradient_analysis.md) 和 [v7_gradient_analysis.json](v7_gradient_analysis.json)，SHA256 分别为 `89567b119e9560d47d2098ee55af9029a66d20a45e923fd7419fd014f9d2381a`、`8d863bcad1fea45842bbe8aca350d7b2d16fec72800bfc8284a1926f5edba5a9`。

## 2026-09-30：V7 最终判定

- 最终判定为 `V7_CONDITIONAL_RESCUE`，中文含义是“有条件保留”。纯 `Class KD` 不能作为通用替代方案，只在使用全部 1,393 条不同样本并保留自然分布时出现一致正向结果。
- 没有判为 `V7_RESCUED`：全部数据 +2.40 个百分点、等价子集 +3.63 个百分点的 95% 区间都跨 0，且只有一个随机种子；难度只部分支持，教师不确定程度不支持。
- 没有判为 `V7_NO_GO`：大数据方案在全部、等价和非等价子集上均为正向；固定候选信心从高到低的差值为 0.00、+4.35、+6.67 个百分点，存在预定条件下的一致趋势。
- `Weak-S0 SKIPPED` 保持不变，不为寻找正结果追加弱学生训练。
- 最终报告为 [v7_final_decision.md](v7_final_decision.md)，SHA256 为 `4ca27a20cbb3976c7b3d31733dbe4323263d81ef2bff169f81cdc55a142b9b5a`。
