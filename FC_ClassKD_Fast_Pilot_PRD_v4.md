# Function Calling Class-Level Distillation — Fast Pilot PRD v4（当前修订版）

> **目的**：用尽可能少的训练 Run 和租卡成本，尽快判断“Verifier-defined Class KD”是否值得继续投入。  
> **原则**：Fast Pilot 只回答“有没有可信正向信号”，不在第一轮证明完整论文的所有 RQ。  
> **正式论文的 RQ1–RQ4 仍保留，但 RQ2/RQ3/RQ4 的完整归因放到 Pilot 通过之后。**

> **2026-08-12 修订说明**：本文件现在是当前实验的最高执行基线。主要修订为：先训练公共 S0，
> 再比较 Teacher 与 S0；Train 和 Diagnostic 分别构造 K=6 候选池；正式人工审核从加入 S0
> 真实错误后的完整候选池抽样；ToolACE 必须恢复原始 Team-ACE 参数说明。若本文件与旧工作记录冲突，
> 以本修订版和根目录 `实验决策记录.md` 为准。

---

# 0. Fast Pilot 最终只回答 3 个问题

## Q1. Class KD 是否比 Vanilla SFT 有实际收益？

比较：

- SFT continuation
- Class KD

看：

- Internal AST / Canonical Call Accuracy
- BFCL 单轮兼容子集

---

## Q2. Class KD 是否比 Candidate KD 更适合当前小模型？

这是 Fast Pilot 最关键的 controlled comparison。

两者必须使用：

- 同一个 Teacher
- 同一个 Student 初始化
- 同一个 Candidate Pool
- 同一个 Verifier
- 同一个序列打分函数
- 同一个训练步数
- 同一个优化器和 Batch 设置

唯一核心差异：

```text
Candidate KD = inter-class + intra-class matching
Class KD     = inter-class matching
```

Fast Pilot 不要求立即证明完整的“capacity-limited hypothesis”，只观察：

> 在 Qwen3-1.7B 上，去掉 intra-class matching 后，最终 FC 结构正确率和等价鲁棒性是否至少出现稳定正向趋势。

---

## Q3. Class KD 是否真的降低对等价 / 表面格式的敏感性？

比较：

- Original Internal Test
- Equivalence Diagnostic

核心指标：

\[
\mathrm{RobustnessDrop}
=
\mathrm{Metric}_{original}
-
\mathrm{Metric}_{equiv}
\]

希望观察：

\[
\mathrm{Drop}_{ClassKD}
<
\mathrm{Drop}_{CandidateKD}
\]

这是第一轮最重要的机制证据。

---

# 1. 当前方法

## 1.1 基础任务

给定：

- 用户请求 \(x\)
- 候选工具 Schema 集合 \(\mathcal T\)

模型输出一个单轮、单函数调用：

\[
(x,\mathcal T)\rightarrow y
\]

例如：

```json
{
  "name": "get_weather_forecast",
  "arguments": {
    "city": "Chengdu",
    "days": 3
  }
}
```

测试阶段学生模型必须直接生成 Function Call。

不得使用：

- Teacher 重排
- Candidate Pool 搜索
- Verifier 修复
- LLM Judge

---

## 1.2 Candidate Pool

每条训练样本默认：

\[
K=6
\]

推荐组成：

| 类型 | 数量 | 说明 |
|---|---:|---|
| Gold | 1 | 原始标准调用 |
| Strict Equivalent | 1 | 字段重排、明确默认参数等高置信等价形式 |
| Student Hard Error | 1 | 初始 SFT Student 的高概率真实错误 |
| Wrong Function | 1 | 相似工具中的错误函数 |
| Wrong Argument | 1 | missing / wrong key / wrong type |
| Wrong Value / Invalid | 1 | 优先构造 wrong value |

没有安全等价变体时，不得强行制造。

---

## 1.3 Candidate Sequence Score

Teacher / Student 对每个完整候选调用打分：

\[
s_M(y_i\mid x)
=
\frac{1}{|y_i|^\alpha}
\sum_t
\log p_M(y_{i,t}\mid x,y_{i,<t})
\]

默认：

```yaml
alpha: 1.0
```

候选概率：

\[
\pi_M(y_i\mid x)
=
\frac{\exp(s_M(y_i)/\tau)}
{\sum_j\exp(s_M(y_j)/\tau)}
\]

---

## 1.4 Verifier Class

Fast Pilot 只保留四个主类：

```text
correct
wrong_function
wrong_arguments
invalid
```

其中 `wrong_arguments` 内部可以记录子标签：

```text
missing_required
wrong_key
wrong_type
wrong_value
```

Verifier 映射：

\[
g_x(y_i)=e
\]

类别概率：

\[
q_M(e\mid x)
=
\sum_{y_i:g_x(y_i)=e}
\pi_M(y_i\mid x)
\]

---

## 1.5 Candidate KD

\[
\mathcal L_{Cand}
=
D_{KL}(\pi_T\|\pi_S)
\]

总 Loss：

\[
\mathcal L
=
\mathcal L_{SFT}
+
\lambda_{cand}\tau^2\mathcal L_{Cand}
\]

---

## 1.6 Class KD

\[
\mathcal L_{Class}
=
D_{KL}(q_T\|q_S)
\]

总 Loss：

\[
\mathcal L
=
\mathcal L_{SFT}
+
\lambda_{class}\tau^2\mathcal L_{Class}
\]

---

## 1.7 RQ2 的理论记录（Fast Pilot 不要求完整验证）

Candidate KL 可以精确分解：

\[
D_{KL}(\pi_T\|\pi_S)
=
D_{KL}(q_T\|q_S)
+
\sum_e q_T(e)
D_{KL}
\left(
\pi_T(\cdot|e)
\|
\pi_S(\cdot|e)
\right)
\]

记：

```text
inter-class = D_KL(qT || qS)
intra-class = Σe qT(e) D_KL(πT(.|e) || πS(.|e))
```

所以：

```text
Candidate KD = inter-class + intra-class
Class KD     = inter-class
```

Fast Pilot 只验证：

> 在当前 1.7B Student 上，去掉 intra-class 约束是否出现正向趋势。

以下完整机制实验全部延后：

- 0.6B vs 1.7B Capacity Comparison
- Intra-class Entropy Buckets
- Gradient Conflict Analysis
- Hierarchical KD

---

# 2. 模型与硬件

## 2.1 模型

### 主 Student

```text
Qwen3-1.7B
```

训练方式：

```text
4-bit NF4 QLoRA
Non-thinking mode
Gradient Checkpointing
```

### Teacher

```text
Qwen3-8B
```

用途：

- Candidate Sequence Score
- Class Distribution

Teacher 冻结，不参与 Student 训练反向传播。

### 本地调试模型

```text
Qwen3-0.6B
```

仅用于：

- Smoke Test
- Loss 调试
- Verifier / Candidate Pipeline 调试

**Fast Pilot 不要求把 0.6B 正式跑成一组论文实验。**

---

## 2.2 当前 GPU 环境

当前主实验机器为 AutoDL RTX 3090 24GB，已经确认 CUDA、BF16、4-bit QLoRA 可用。

负责：

- 0.6B 冒烟与小样本过拟合
- 1.7B 公共 S0 和三条训练分支
- 8B 小规模诊断与完整 Train 候选打分
- Internal、Equivalence 和 BFCL 推理

默认：

```yaml
bf16: true
micro_batch_size: 1
gradient_checkpointing: true
max_length: 2048
K: 6
```

实验室 RTX 2080 Ti 11GB 可以承担代码检查、小规模候选构造或较短序列调试，但不再把它写成
正式流程的固定前置机器。若 8B 在 3090 上显存不足，先减小打分批次或序列长度，再决定是否换卡；
不能因此改变模型、数据或候选池定义。

---

# 3. 数据规模

Fast Pilot 不追求完整论文规模。

默认目标：

| Split | 数量 |
|---|---:|
| Train | 1,500–3,000 |
| Validation | 300 |
| Internal Test | 500 |
| Diagnostic | 200–300 |
| Equivalence Diagnostic | 200 左右 |

优先从公开 Function Calling 数据中筛选：

- xLAM Function Calling 数据
- ToolACE

只保留：

- single-turn
- single-call
- 有明确 Tool Schema
- Gold Call 可稳定解析

BFCL 测试数据不得进入训练、候选构造或超参数选择。

## 3.1 当前数据状态

当前已固定 3,250 条合并数据：保留全部 2,103 条恢复原始参数说明的 ToolACE，再从去重、
去 BFCL 泄漏且 Gold 合法的 xLAM 中按 seed=42 补入 1,147 条。正式划分为：

| Split | 当前数量 |
|---|---:|
| Train | 2,000 |
| Validation | 300 |
| Internal Test | 500 |
| Diagnostic | 250 |
| Equivalence Diagnostic | 200 |

3,250 条全部且仅进入一个划分，ID 和规范化问题文本均唯一，五个划分互不重叠。正式全集 SHA256
为 `472b5b8538c...`，完整哈希见 `data/processed/manifest.json`。

## 3.2 ToolACE 参数说明规则

`tryumanshow/ToolACE-Qwen-cleaned` 的上游 Parquet 已经丢失参数必填信息，并给绝大多数参数填入
默认值，其中大量整数、浮点数和布尔参数的默认值还是空字符串。该问题不是本项目适配器制造的，
但旧流程曾错误地直接采用这份参数说明。

当前强制规则：

- cleaned ToolACE 只提供整理后的单轮、单调用内容和函数名。
- `required`、`type`、`enum`、`default` 从固定版本的原始 `Team-ACE/ToolACE` 恢复。
- 只接受用户问题唯一匹配，且工具顺序、工具说明、参数名集合完全一致的数据。
- 恢复后 Gold 不合法的样本必须剔除。
- 正式加载器未提供原始 Team-ACE 版本时直接报错，禁止静默回退。

详细筛选过程见 `docs/ToolACE数据修复与筛选说明.md`。

## 3.3 xLAM 参数说明与泄漏处理

xLAM 固定版本为 `26d14ebf...`。实际 JSON 与数据卡示例不完全一致：参数没有显式 `required`，
可选性写在 `str, optional` 等类型字符串中，而且默认值存在大量类型不匹配。当前采用保守规则：

- 支持并规范化 `str/int/float/bool/List/Tuple/set/Dict` 等顶层类型。
- `Callable` 等无法可靠映射到 JSON 的类型整条剔除。
- 有 `optional` 或默认字段的参数视为可选；其余视为必填。
- xLAM 默认值全部不参与等价规范化，防止制造假等价。
- Gold 在上述参数说明下不合法的数据剔除。
- 与固定 BFCL v4 提交 `6ea5797...` 完全相同或达到高相似阈值的问题全部剔除。

结构审计和筛选数量见 `docs/xLAM数据筛选说明.md` 与 `reports/合并数据质量报告.md`。

---

# 4. Verifier 最小实现

Fast Pilot 不做复杂语义 Judge。

必须支持：

## 4.1 Parse

- JSON Parse
- Function Name 提取
- Arguments 提取

## 4.2 Schema Check

- Function 是否存在
- Required Key
- Unknown Key
- Type
- Enum

## 4.3 Canonicalization

只允许高精度规则：

- JSON Key 排序
- Schema 明确默认参数
- 明确的数值规范化
- 明确的日期规范化
- Schema 明确声明的无序集合排序

不允许：

- LLM 判断“语义差不多”
- 根据一次 API 执行结果猜等价
- 模糊字符串匹配直接判 Correct

---

## 4.4 Verifier 人工审核门槛

公共 S0 只学习 Train Gold，不依赖候选池或 Verifier，因此可以先训练。正式人工审核阻塞的是：

- 候选池冻结
- 完整 Train Teacher Scores 缓存
- Candidate KD 与 Class KD 训练

加入 S0 真实错误并补齐 K=6 后，从最终 Train 和 Diagnostic 候选池分层抽检 100–200 个候选。
必须覆盖四个主类和 S0 真实错误。S0 前的规则候选盲标包只能作为早期检查，不能代替正式审核。

要求：

```text
correct / equivalent precision >= 98%
overall audited accuracy >= 95%
```

若未通过：

```text
STOP
→ 修 Verifier
→ 重建候选池并重新人工审核
→ 不进入 Teacher 完整缓存和蒸馏训练
```

---

# 5. Pre-Diagnostic：公共 S0 和完整诊断候选池之后进行

使用 200–300 条独立 Diagnostic。当前正式划分为 250 条。Pre-Diagnostic 不使用 Train、
Validation、Internal Test、Equivalence Diagnostic 或 BFCL，分成自由生成和候选类选择两部分。

## 5.1 自由生成能力差距

S0 与 8B Teacher 在完全相同的 Diagnostic 输入上直接生成工具调用，报告：

- Parse Success
- Function Name Accuracy
- Argument Accuracy
- AST / Canonical Call Accuracy
- Wrong Function / Wrong Arguments / Invalid

S0 AST 建议处于 40%–80%。高于 85% 说明任务可能过简单；低于 20% 时优先排查 Prompt、
Chat Template、Tool Schema、Label Mask、Truncation 和 Parser。Teacher 应明显强于 S0，
但自由生成部分先如实报告差值，不套用候选类 Top-1 的 80% 门槛。

## 5.2 候选类选择能力差距

S0 与 Teacher 对完全相同的 K=6 Diagnostic 候选池打分，检查 Teacher 是否能把概率集中到
Verifier 定义的正确类，而不是要求 Teacher 每条都选对。

要求优先满足：

```text
Teacher Class Top-1 >= 80%
Teacher - S0 >= 5 percentage points
```

理想差距：

```text
Teacher - S0 >= 8 points
```

## 5.3 Candidate Pool 是否有意义

至少 90% 的样本含有一个有效错误候选。错误不能主要是随机乱码，并且必须包含一定比例的
`wrong_function` 和 `wrong_arguments`。

## 5.4 等价样本是否足够

至少满足以下一项：

- 自然或安全构造的 multi-valid 样本至少占 10%。
- 单独构造约 200 条高精度 Equivalence Diagnostic。

## 5.5 通过后的动作

只有自由生成差距、候选类差距、候选质量和等价覆盖都合理，才对完整 Train K=6 候选池运行
8B Teacher 打分并保存 Teacher Scores。诊断失败时先停止扩大计算，不用 Internal Test 或 BFCL
反复试验来调整门槛。

---

# 6. Fast Pilot 的训练设计

## 6.1 共同初始化

先只用 Train Gold 训练一个公共 Vanilla SFT Student。该阶段不能使用候选、Verifier 标签或
Teacher Scores：

```text
Base Qwen3-1.7B
        ↓
     SFT Stage
        ↓
        S0
```

保存：

```text
S0 LoRA Adapter
S0 validation metrics
S0 predictions
```

当前固定训练方式：Train 2,000 条，训练三轮；4-bit QLoRA，微批次 1，梯度累计 16，学习率
`2e-4`，3% warmup 后余弦下降，最大长度 2,048。每轮在完整 Validation 300 条自由生成，首先
选择 AST 最高的轮次；AST 相同则选择 Validation loss 更低的轮次；仍相同则选更早轮次。
Internal Test、Diagnostic、Equivalence Diagnostic 和 BFCL 不参与选择 S0。

S0 随后分别对 Train 和 Diagnostic 自由生成：

- Train 生成结果用于构造训练候选池。
- Diagnostic 生成结果用于构造诊断候选池。
- 两套候选池禁止交叉使用。
- Internal Test、Equivalence Diagnostic 和 BFCL 不构造训练候选。

---

## 6.2 从同一个 S0 分三条支路

```text
                    ┌→ SFT Continuation
S0 Common Student ──┼→ Candidate KD
                    └→ Class KD
```

三条支路：

- 相同步数
- 相同学习率
- 相同 Batch
- 相同 Optimizer
- 相同 Candidate Pool（候选方法）
- 相同 Seed
- 相同 Validation

### Branch A — SFT Continuation

目的：

> 控制 Candidate / Class KD 比 Vanilla 多训练了一段时间的问题。

只继续相同步数的普通 SFT。

### Branch B — Candidate KD

使用：

```text
SFT + Candidate KL
```

### Branch C — Class KD

使用：

```text
SFT + Class KL
```

---

## 6.3 Fast Pilot 只使用一个 Seed

```yaml
seed: 42
```

Fast Pilot 阶段不补第二 Seed。

只有 Go 后才补：

```yaml
seed: 43
```

---

# 7. Fast Pilot 只做 3 组正式评测

# 7.1 Internal Test

统一输出：

- Parse Success Rate
- Function Name Accuracy
- Argument Accuracy
- AST / Canonical Call Accuracy
- Wrong Function Rate
- Wrong Argument Rate
- Invalid Rate

最关键：

```text
SFT Continuation
Candidate KD
Class KD
```

---

# 7.2 Equivalence Diagnostic

对严格等价的输入 / 输出表示变化进行测试。

优先只做三类：

1. JSON / Arguments Key Reorder
2. Explicit / Implicit Schema Default
3. Safe Canonical Normalization

计算：

\[
Drop
=
AST_{original}
-
AST_{equiv}
\]

核心看：

```text
Class KD Drop
vs
Candidate KD Drop
vs
SFT Drop
```

---

# 7.3 BFCL 单轮兼容子集

Fast Pilot 只接一个外部 Benchmark：

```text
BFCL
```

优先选择：

- single-turn
- 与 single-call / AST Evaluation 兼容部分

目的：

> 防止方法只在自建数据和候选体系中有效。

Fast Pilot 暂不强制 ACEBench。

---

# 8. Fast Pilot Go / No-Go

不要求所有指标都赢。

## 8.1 GO — 明显值得继续

满足以下两类证据中的至少两类：

### A. 主任务收益

```text
Class KD > SFT Continuation
```

在 Internal AST / BFCL 至少一个主要指标上有稳定正向趋势。

### B. 相对 Candidate KD 有优势

```text
Class KD >= Candidate KD
```

并且至少在一个核心指标或 Equivalence Diagnostic 上更好。

### C. RQ1 机制信号

```text
RobustnessDrop(Class KD)
<
RobustnessDrop(Candidate KD)
```

差异足够稳定，不是少量样本偶然造成。

---

## 8.2 WEAK GO — 可以继续，但需要收窄主张

例如：

```text
Class KD ≈ Candidate KD overall
but
Class KD significantly better on equivalence robustness
```

此时可以继续研究：

> Class KD 主要改善鲁棒性，而不是总体 Accuracy。

---

## 8.3 NO-GO

出现任一明显情况：

```text
Class KD consistently < Candidate KD
and
no equivalence robustness gain
```

或：

```text
Class KD only improves self-built diagnostic
but clearly degrades BFCL
```

或：

```text
Teacher / Verifier / Candidate Pool quality fails gates
```

则先停止扩大实验。

---

# 9. Fast Pilot 通过后再做什么

以下内容全部属于 **Phase 2 / Full Validation**，不得阻塞第一轮结果。

## Phase 2-A：正式验证 RQ2

增加：

```text
Qwen3-0.6B
Candidate KD
Class KD
```

比较：

\[
\Delta_{0.6B}
vs
\Delta_{1.7B}
\]

并增加：

- Intra-class Entropy Buckets
- 可选 Gradient Cosine Analysis

---

## Phase 2-B：正式验证 RQ3

增加：

```text
Class-Hard
Class Label Smoothing
Hard Negative
```

核心链：

```text
Class-Hard
→ Label Smoothing
→ Class KD
```

回答：

> Teacher soft class distribution 是否真的提供额外知识？

---

## Phase 2-C：完整论文 Baseline

再补：

- Token KD
- Random Negative
- Hard Negative
- Second Seed
- K = 4 / 6 / 8
- ACEBench
- 更完整 BFCL
- Optional Hierarchical KD

---

# 10. Fast Pilot 最小工程目录

```text
fc_class_kd/
├── configs/
│   ├── base.yaml
│   ├── rtx2080ti.yaml
│   ├── rtx3090.yaml
│   └── rtx4090.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   ├── diagnostics/
│   ├── candidates/
│   └── cache/
├── src/
│   ├── data/
│   ├── verifier/
│   ├── candidates/
│   ├── scoring/
│   ├── losses/
│   ├── training/
│   └── evaluation/
├── scripts/
│   ├── doctor.py
│   ├── smoke_test.py
│   ├── build_data.py
│   ├── build_candidates.py
│   ├── audit_verifier.py
│   ├── run_diagnostic.py
│   ├── cache_teacher_scores.py
│   ├── train_sft.py
│   ├── train_candidate_kd.py
│   ├── train_class_kd.py
│   ├── evaluate_internal.py
│   ├── evaluate_equivalence.py
│   └── evaluate_bfcl.py
├── outputs/
└── reports/
```

Fast Pilot 不要求提前实现：

```text
ACEBench Adapter
Class-Hard
Label Smoothing
Hard Negative
Entropy Bucket Training Logic
Gradient Conflict
Hierarchical KD
Multi-turn
RL
```

---

# 11. 最小测试要求

正式训练 S0 或运行 8B Teacher 前必须通过：

## 11.1 Tiny Overfit

32 条以内：

```text
Qwen3-0.6B / 1.7B
```

要求：

```text
Loss 明显下降
AST >= 90%（训练集）
```

---

## 11.2 Candidate Score Test

人工手算一个 Toy Candidate Set。

要求：

```text
sequence score error < 1e-6
candidate probability sum = 1
```

---

## 11.3 Class Aggregation Test

例如：

```text
candidate probabilities:
[0.30, 0.20, 0.10, 0.40]

class:
[correct, correct, correct, wrong]
```

必须得到：

```text
class probabilities:
[0.60, 0.40]
```

---

## 11.4 Candidate KL Decomposition Test

必须数值验证：

\[
KL_{candidate}
=
KL_{inter}
+
KL_{intra}
\]

误差：

```text
< 1e-6
```

---

## 11.5 Gradient Test

必须确认：

```text
Class KD loss.backward()
```

后：

- LoRA 参数有非零梯度
- Frozen Base 参数无 `.grad`
- Teacher 无梯度
- 无 NaN / Inf

---

# 12. Fast Pilot 输出物

只要求：

```text
reports/
├── 实验环境记录.md
├── 验证器审核进度.md
├── 能力诊断结果.md
├── 候选池正式统计.md
├── 快速试验正式结果.md
└── 是否继续实验结论.md
```

`快速试验正式结果.md` 必须包含一张统一主表：

| Method | Internal AST | BFCL | Equiv AST | Robustness Drop |
|---|---:|---:|---:|---:|
| SFT Continuation | | | | |
| Candidate KD | | | | |
| Class KD | | | | |

另附：

- Wrong Function Rate
- Wrong Argument Rate
- Invalid Rate
- 20 条代表性错误样本

---

# 13. Fast Pilot 的执行顺序

严格按下面顺序：

```text
Step 1
环境 + 数据修复 + 自动检查
        ↓
Step 2
0.6B Smoke / Tiny Overfit
        ↓
Step 3
训练公共 1.7B SFT Student S0
        ↓
Step 4
S0 分别对 Train / Diagnostic 自由生成
构造两套完整 K=6 Candidate Pool
        ↓
Step 5
人工审核最终候选中的 Verifier 标签
通过后冻结候选池和文件哈希
        ↓
Step 6
Diagnostic 上完成 Teacher vs S0：
自由生成 + 候选类 Top-1
        ↓
       Gate
   Fail → Stop
        ↓ Pass
Step 7
离线缓存完整 Train 的 8B Teacher Scores
        ↓
Step 8
从 S0 跑三条同预算分支：
SFT-cont / Candidate KD / Class KD
        ↓
Step 9
Internal + Equivalence + BFCL
        ↓
Step 10
Go / Weak-Go / No-Go
```

---

# 14. Fast Pilot 明确不做的事情

第一轮不得因为“顺手”而加入：

- 第二 Seed
- K=4/8 敏感性
- 0.6B 正式论文结果
- ACEBench
- Token KD
- Class-Hard
- Label Smoothing
- Random / Hard Negative
- Gradient Conflict
- Hierarchical KD
- 4B Student
- 14B Teacher
- RL / DPO / GRPO
- Multi-turn
- Parallel / Multiple Calls
- 动态 On-policy Candidate Refresh

这些都必须等 Fast Pilot 给出正向信号后再决定。

---

# 15. Fast Pilot Definition of Done

Fast Pilot 完成只要求以下项目：

- [ ] 环境和模型可运行
- [ ] 公共 SFT Student S0 完成
- [ ] Train / Diagnostic 两套 Candidate Pool K=6 可复现
- [ ] 最终候选池上的 Verifier 人工审核通过
- [ ] Teacher / S0 自由生成和候选类预诊断通过
- [ ] 完整 Train Teacher Scores 已冻结
- [ ] SFT Continuation 完成
- [ ] Candidate KD 完成
- [ ] Class KD 完成
- [ ] Internal Test 完成
- [ ] Equivalence Diagnostic 完成
- [ ] BFCL 单轮兼容子集完成
- [ ] 输出统一主表
- [ ] 给出 Go / Weak-Go / No-Go

---

# 16. Go 后再恢复完整论文 RQ

Fast Pilot 通过后，再恢复完整论文证据链：

```text
RQ1
等价调用聚合是否降低表面敏感性？
    ↓
RQ2
类内候选完整拟合是否给容量受限学生带来可避免负担？
    ↓
RQ3
Teacher Soft Class Distribution 是否优于 Hard Class Label？
    ↓
RQ4
标准 BFCL / ACEBench 上是否具有外部有效性？
```

因此，本 v4 不是删除 RQ2–RQ4，而是把它们从：

> “第一轮必须全部完成”

改成：

> “只有核心 idea 出现正向信号后才投入资源验证”。

这也是本 Fast Pilot 的核心目标：**尽快得到可信的 Go / No-Go 结论。**
