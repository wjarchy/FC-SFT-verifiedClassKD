# Function Calling Class-Level Distillation — Fast Pilot PRD v4

> **目的**：用尽可能少的训练 Run 和租卡成本，尽快判断“Verifier-defined Class KD”是否值得继续投入。  
> **原则**：Fast Pilot 只回答“有没有可信正向信号”，不在第一轮证明完整论文的所有 RQ。  
> **正式论文的 RQ1–RQ4 仍保留，但 RQ2/RQ3/RQ4 的完整归因放到 Pilot 通过之后。**

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

## 2.2 GPU 分工

### 实验室 RTX 2080 Ti 11GB

负责：

- 环境搭建
- 数据处理
- Verifier
- Candidate 构造
- 0.6B Smoke Test
- 1.7B Tiny Overfit
- 8B INT4 小样本预诊断

默认：

```yaml
fp16: true
bf16: false
micro_batch_size: 1
gradient_checkpointing: true
max_length: 1024
```

### RTX 4090 24GB

只在 Fast Pilot 已经通过本地 Smoke / Diagnostic 后租用。

负责：

- Qwen3-8B 正式 Teacher Cache
- Qwen3-1.7B 三个正式训练分支
- BFCL 推理评测

默认：

```yaml
bf16: true
micro_batch_size: 1
gradient_checkpointing: true
max_length: 1024-1536
K: 6
```

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

## 4.4 Verifier Fast Gate

正式训练前抽检 100–200 个候选。

要求：

```text
correct / equivalent precision >= 98%
overall audited accuracy >= 95%
```

若未通过：

```text
STOP
→ 修 Verifier
→ 不进入正式训练
```

---

# 5. Pre-Diagnostic：正式租卡前必须完成

样本：

```text
200–300 条
```

检查：

## 5.1 Teacher 是否够强

要求优先满足：

```text
Teacher Class Top-1 >= 80%
Teacher - Student >= 5 percentage points
```

理想：

```text
Teacher - Student >= 8 points
```

---

## 5.2 Candidate Pool 是否有意义

要求：

```text
>= 90% 样本至少有一个有效错误候选
```

错误不能主要是随机乱码。

必须包含一定比例：

```text
wrong_function
wrong_arguments
```

---

## 5.3 等价样本是否足够

至少满足之一：

```text
自然 / 安全构造的 multi-valid 样本 >= 10%
```

或者：

```text
单独构造约 200 条高精度 Equivalence Diagnostic
```

---

## 5.4 Student 难度是否合理

Vanilla / Initial Student AST 建议：

```text
40%–80%
```

若：

```text
> 85%
```

任务可能过简单。

若：

```text
< 20%
```

优先排查：

- Prompt
- Chat Template
- Tool Schema
- Label Mask
- Truncation
- Parser

---

# 6. Fast Pilot 的训练设计

## 6.1 共同初始化

先训练一个公共 Vanilla SFT Student：

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
│   └── rtx4090.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── diagnostics/
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

正式租 4090 前必须通过：

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
├── environment.md
├── verifier_audit.md
├── diagnostic.md
├── candidate_stats.md
├── fast_pilot_results.md
└── go_no_go.md
```

`fast_pilot_results.md` 必须只有一张主表：

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
环境 + 数据 + Verifier
        ↓
Step 2
0.6B Smoke / Tiny Overfit
        ↓
Step 3
200–300 样本 Pre-Diagnostic
        ↓
       Gate
   Fail → Stop
        ↓ Pass
Step 4
训练公共 1.7B SFT Student S0
        ↓
Step 5
构造并冻结 K=6 Candidate Pool
        ↓
Step 6
离线缓存 8B Teacher Scores
        ↓
Step 7
从 S0 跑三条同预算分支：
SFT-cont / Candidate KD / Class KD
        ↓
Step 8
Internal + Equivalence + BFCL
        ↓
Step 9
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
- [ ] Verifier 审计通过
- [ ] Candidate Pool K=6 可复现
- [ ] Teacher / Student 预诊断通过
- [ ] 公共 SFT Student S0 完成
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
