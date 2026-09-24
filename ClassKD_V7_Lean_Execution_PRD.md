# Class KD V7 精简续跑执行 PRD

> 目标：在新 AutoDL 3090 上，只恢复 V7 必需的资产，不重新复刻已经有结论的 V6 前置 Gate，然后优先完成最可能给出机制结论的 V7 实验。
>
> 当前策略：**数据恢复 → 必要基线恢复 → S0 Difficulty → 数据规模/多样性 Rescue → 其他机制分析 → 条件式 Weak-S0 → V7 最终判定。**

---

# 0. 核心原则

这是 V7 续跑，不是 V6 重做。

以下历史结论已经接受，不重新作为阻塞 Gate：

- Natural Equivalence 路线已 No-Go；
- Teacher Gate 已通过；
- Student–Teacher Gap Gate 已通过；
- V6 Balanced-336 Pilot 已得到 `MECHANISM_NO_GO`；
- 历史结果只作为参考，不要求逐条复刻。

**不要重新跑以下内容：**

```text
300×8 / 16 次自然采样 Gate
完整 Teacher Gate 重验
完整 Student–Teacher Gap Gate 重验
V5/V6 机会集前置探索
旧 K=6 候选池
旧 V4/V5 S0 实验
V6 全套历史报告重建
```

如果在恢复 Teacher cache 时顺手可以重新计算 Teacher Gate / Gap 指标，可以作为 sanity check 记录，但不要把它们当成新的阻塞阶段，也不要为了精确复现这些历史数字浪费 GPU 时间。

---

# 1. 当前研究问题

Candidate KD：

```text
L_candidate = L_inter + L_intra
```

其中：

- `L_inter`：不同功能类别之间的监督；
- `L_intra`：同一严格功能等价类内部的 surface preference 监督。

Class KD：

```text
L_class = L_inter
```

V6 第一轮 Balanced-336：

```text
Internal-Equiv:
Candidate KD ≈ 78.23%
Class KD ≈ 77.02%

Full Internal:
Candidate KD ≈ 82.80%
Class KD ≈ 83.20%
```

所以 V6 按预注册规则判：

```text
MECHANISM_NO_GO
```

但这只能说明：

> 单 seed、336 条、约 21 updates 的第一轮 Pilot 没有证明 Pure Class KD 优于 Candidate KD。

不能证明 Class KD 永远更差。

---

# 2. V7 现在最重要的两个假设

## H1：S0 Difficulty Hypothesis

Class KD 可能主要帮助：

```text
S0 本身不会 / 不确定 / confidence 低
```

的样本。

而对：

```text
S0 已经很强 / 已经正确 / confidence 高
```

的样本，Candidate KD 多出来的 intra-class supervision 可能反而有帮助。

所以优先问题是：

> **Class KD 的相对收益是否随着 S0 difficulty 增大而上升？**

## H2：Data Diversity / Exposure Hypothesis

V6 只用了：

```text
Balanced-336
A/B/C = 112/112/112
1 epoch
≈21 optimizer steps
```

但 Train-3000 中实际有：

```text
1391 条 non-trivial equivalence
A/B/C = 1094/112/185
```

所以 V6 失败可能来自：

- unique query 太少；
- 总 training exposure 太少；
- 人为平衡破坏自然数据分布。

V7 必须比较：

```text
Balanced-336 × 4 epochs
vs
Equiv-All-1391 × 1 epoch
```

两边 sample exposure / optimizer updates 接近：

```text
336×4 = 1344 samples ≈84 updates
1391×1 = 1391 samples ≈87 updates
```

这样可以把：

```text
重复训练同一小数据
```

和：

```text
更多 unique query / 自然分布
```

尽量拆开。

---

# 3. 当前服务器和项目

工作目录：

```text
/root/FC_ClassKD
```

GPU：

```text
RTX 3090 24GB
```

Python：

```text
3.11
```

PyTorch：

```text
2.5.1+cu124
```

HF cache：

```text
/root/autodl-tmp/hf_cache
```

网络规则遵守 `AGENTS.md`。

当前任务分支建议：

```text
recovery/v7-rescue
```

每完成一个阶段：

```bash
git add ...
git commit -m "v7: ..."
git push
```

不要再次把数天工作只留在 AutoDL 本地。

---

# 4. 历史参考数据

这些数字用于判断恢复是否大体正确，不需要逐 bit 复现。

## Formal Data

```text
Train = 3000
Validation = 300
Internal = 500
Reserve ≈ 14229

Non-trivial equivalence:
1391 / 3000 = 46.37%

A = 1094
B = 112
C = 185

Balanced:
A = 112
B = 112
C = 112
Total = 336
```

## 模型

```text
Teacher = Qwen3-8B
Student = Qwen3-1.7B
```

## Candidate Pool

```text
K=4

Gold
Equivalent
Wrong Argument
Wrong Function
```

## KD

```text
tau = 2
lambda = 1
LR = 2e-4
batch = 16
seed = 42
```

## 历史 V6 结果

```text
Internal-Equiv N=248

Candidate:
194/248 = 78.23%

Class:
191/248 = 77.02%
```

```text
Full Internal N=500

Candidate:
414/500 = 82.80%

Class:
416/500 = 83.20%
```

不要把这些历史数值直接写成本次新运行结果。

---

# 5. Phase 0：最低限度环境和代码检查

不要做大规模 V6 审计。

只做训练前必需检查：

```text
1. 当前 Git SHA
2. Python / torch / CUDA / transformers / peft / bitsandbytes
3. GPU 可用
4. pytest 通过
5. candidate scoring 可运行
6. Class aggregation 可运行
7. Candidate/Class loss 可反向传播
8. canonicalizer / evaluator 关键测试通过
```

需要有一个最小数值检查：

```text
KL_candidate ≈ KL_inter + KL_intra
```

这个只作为防止恢复代码写错的 unit/smoke gate。

输出：

```text
reports/v7_recovery_smoke.md
```

通过后立即进入数据恢复。

---

# 6. Phase 1：优先恢复 V7 真正需要的数据

这是当前第一优先级。

必须恢复：

```text
Train-3000
Validation-300
Internal-500
Internal-Equiv subset
Internal-Singleton subset

Equiv-All-1391
Balanced-336
```

必须区分旧仓库中的 Train-2000 历史数据。

禁止直接用旧 Train-2000 替代 V6/V7 Formal Data。

## 6.1 数据恢复最低 Gate

重点对齐：

```text
Train = 3000

Equiv-All ≈ 1391

A/B/C ≈
1094 / 112 / 185

Balanced =
112 / 112 / 112 = 336
```

以及：

```text
Internal = 500
Internal-Equiv ≈ 248
```

如果数字完全不同，优先排查：

```text
数据 revision
seed
filter
dedup
canonicalizer
equivalence rule
default handling
numeric handling
```

如果只存在非常小、可解释的恢复差异，可以继续，但必须记录。

如果数据定义发生实质变化，则停止并向用户汇报。

## 6.2 输出

```text
reports/v7_data_recovery.md
data/.../manifest.json
```

记录：

```text
数据源 revision
seed
count
A/B/C
hash
生成命令
```

---

# 7. Phase 2：恢复 Formal S0

目标：

```text
Qwen3-1.7B Formal S0
```

不要自动使用旧 V4 Train-2000 的 `checkpoint-step-000375`。

先根据恢复出的 V6/V7 Formal Data 和代码/文档判断：

> V6 当时正式 S0 的训练定义是什么。

如果实体 S0 已经丢失，则按 Formal V6 定义重训。

S0 必须保存：

```text
adapter
selected checkpoint
validation result
Internal free-generation predictions
Internal-Equiv predictions
```

这些 predictions 后面直接用于 S0 Difficulty。

输出：

```text
reports/v7_s0_recovery.md
```

---

# 8. Phase 3：恢复 V7 必需的 K=4 Candidate Pool + Teacher Cache

不重跑 Teacher Gate。

只做后续训练需要的实体资产。

每个 equivalence sample：

```text
1 Gold
1 Equivalent
1 Wrong Argument
1 Wrong Function
```

要求：

```text
Canon(Gold) == Canon(Equivalent)
Canon(Negative) != Canon(Gold)
```

然后用：

```text
Qwen3-8B
tau = 2
```

计算 Teacher sequence scores / probabilities。

保存：

```text
candidate scores
candidate probabilities
class probabilities
intra-class probabilities
intra-class entropy
```

Teacher entropy 后面 V7 直接使用，因此这里一次算好。

输出：

```text
reports/v7_teacher_cache_recovery.md
artifacts_manifest.json
```

---

# 9. Phase 4：最低成本恢复 V6 Candidate / Class 基线

因为旧 Candidate/Class checkpoint 和 sample-level predictions 已丢失，而 S0 Difficulty 分析需要：

```text
同一批样本上
Candidate prediction
Class prediction
S0 difficulty
```

所以必须重新跑一次：

```text
Balanced-336 × 1 epoch
```

但这个不是“重做 V6 Gate”。

只是为了生成 V7 必需的：

```text
Candidate model
Class model
sample-level predictions
```

配置固定：

```text
same S0
same Balanced-336
same K=4
tau=2
lambda=1
LR=2e-4
batch=16
seed=42
≈21 updates
```

评测：

```text
Full Internal
Internal-Equiv
Internal-Singleton
```

历史数值只做 sanity：

```text
Equiv:
78.23 vs 77.02

Full:
82.80 vs 83.20
```

如果结果大体同量级并且两者接近，就直接进入 V7。

不要为了复现到完全相同的 194/191 而反复耗费时间。

如果结果出现极端反转（例如 +10pt），才排查恢复问题。

---

# 10. V7 第一主实验：S0 Difficulty Analysis

这是当前最高优先级。

不要先跑 Gradient。

## 10.1 View A：S0 Correct vs Wrong

对 Internal-Equiv：

按 S0 在 KD 前的 free-generation 分：

```text
Bucket A:
S0 Correct

Bucket B:
S0 Wrong
```

分别统计：

```text
N

Candidate correct
Candidate accuracy

Class correct
Class accuracy

Delta = Class - Candidate
```

这是最直观的 Difficulty 分析。

## 10.2 View B：S0 Candidate Confidence

对 S0 在 K=4 fixed support 上计算：

```text
p_S0(correct functional class)
```

根据训练 / analysis pool 的分布提前确定 tertile：

```text
Low confidence
Mid confidence
High confidence
```

然后应用到 Internal-Equiv。

每组：

```text
N
Candidate Acc
Class Acc
Delta
```

## 10.3 主要判据

重点判断是否出现：

```text
High confidence / S0 Correct:
Candidate >= Class

Low confidence / S0 Wrong:
Class > Candidate
```

或者至少：

```text
Class - Candidate
随着 S0 difficulty 增加而持续上升
```

同时保存 sample-level table：

```text
sample_id
S0 correctness
S0 confidence
Candidate result
Class result
A/B/C
Teacher entropy
```

## 10.4 输出

```text
reports/v7_s0_difficulty_analysis.md
reports/v7_s0_difficulty_samples.jsonl
```

完成后立即 commit + push。

---

# 11. V7 第二主实验：Data Scaling / Diversity Rescue

S0 Difficulty 做完以后马上执行，不需要等待用户确认。

## 11.1 Balanced-336 × 4

```text
336 unique
112/112/112
4 epochs
1344 sample exposures
≈84 updates
```

分别：

```text
Candidate KD
Class KD
```

## 11.2 Equiv-All-1391 × 1

```text
1391 unique
1094/112/185
1 epoch
1391 sample exposures
≈87 updates
```

分别：

```text
Candidate KD
Class KD
```

## 11.3 评测

每个模型：

```text
Full Internal
Internal-Equiv
Internal-Singleton
```

同时分：

```text
A/B/C
S0 Correct/Wrong
S0 confidence tertiles
```

Teacher entropy 分桶可以顺带计算，但不要因为做 entropy 分析而延迟主结果。

## 11.4 重点解释

### 情况 1

```text
336×4 救回 Class
1391×1 未明显救回
```

倾向：

> V6 主要是 training exposure 太低。

### 情况 2

```text
336×4 仍不行
1391×1 Class 明显变好
```

倾向：

> unique query diversity / natural distribution 更重要。

### 情况 3

```text
两种 scaling 都支持 Class
```

倾向：

> V6 主要受 pilot scale 限制。

### 情况 4

```text
两种 scaling 都 Candidate >= Class
```

Pure Class KD 风险很高。

输出：

```text
reports/v7_scaling_report.md
```

---

# 12. V7 第三层：Teacher Entropy Analysis

这一步不需要新训练。

直接使用：

```text
Teacher cache
Candidate/Class predictions
```

即可。

对正确 equivalence class 内：

```text
q_T(candidate | class)
```

计算：

```text
H_intra
normalized H_intra
```

按 tertile：

```text
Low
Mid
High
```

比较：

```text
Class - Candidate
```

重点看：

> Class 是否主要在 Teacher 类内 preference 不稳定 / entropy 高的样本上有优势。

输出：

```text
reports/v7_teacher_entropy_analysis.md
```

---

# 13. V7 第四层：Discordant Cases

同样不需要新训练。

收集：

```text
Candidate correct / Class wrong
Candidate wrong / Class correct
```

每个 case 保存：

```text
query
schema
Gold
Equivalent
Teacher probabilities
Teacher entropy
S0 output
S0 confidence
Candidate output
Class output
A/B/C
error type
```

同时展示两边 wins。

输出：

```text
reports/v7_discordant_cases.md
reports/v7_discordant_cases.jsonl
```

---

# 14. Gradient Analysis 改成“条件性解释实验”

**不要放在 V7 最前面。**

原因：

它只能解释“为什么”，不能先告诉我们“是否有值得解释的现象”。

所以现在改成：

在下面情况之一发生时执行：

### 情况 A

```text
336×1 Class 不好
336×4 明显变好
```

这时 Gradient / signal strength 很有价值，用来解释：

> Class 是否因为 loss 少一项而梯度更弱，需要更多 exposure。

### 情况 B

```text
S0 Difficulty 显示 Hard samples Class 好，Easy samples Candidate 好
```

可以用 Gradient 分析比较不同 difficulty bucket。

### 情况 C

```text
Scaling 和 difficulty 都没明显信号
```

Gradient 只做最小诊断，不需要扩展。

最低要求：

```text
||grad_candidate||
||grad_class||
||grad_inter||
||grad_intra||
```

有条件可加：

```text
cosine(grad_inter, grad_intra)
```

输出：

```text
reports/v7_gradient_analysis.md
```

---

# 15. Weak-S0 仍然是条件实验

只有当 S0 Difficulty 明确提示：

```text
S0 越弱
Class 相对 Candidate 越好
```

才做。

如果没有这个趋势：

```text
Weak-S0 SKIPPED
```

不为了涨点强行构造弱模型。

## Weak-S0 构造

优先：

```text
Formal SFT trajectory 的 earlier checkpoint
```

或：

```text
更少 SFT exposure
```

不能随机破坏模型。

目标大约：

```text
比 Formal S0 低 5–10 个 absolute AST points
```

但不为命中数字反复调。

## Weak-S0 主实验

默认：

```text
Equiv-All-1391 × 1
```

比较：

```text
Candidate KD
Class KD
```

输出：

```text
reports/v7_weak_s0_report.md
```

---

# 16. V7 的执行顺序现在固定为

```text
必要环境 smoke
        ↓
V6/V7 Formal Data 恢复
        ↓
Formal S0 恢复
        ↓
K=4 + Teacher Cache 恢复
        ↓
Balanced-336×1 Candidate / Class
（只为恢复基线模型和 predictions）
        ↓
【优先 1】S0 Difficulty
        ↓
【优先 2】336×4 vs 1391×1 Scaling
        ↓
Teacher Entropy
        ↓
Discordant Cases
        ↓
Gradient（条件性解释）
        ↓
Weak-S0（只有 Difficulty 支持才做）
        ↓
V7 Final Decision
```

---

# 17. V7 完成标准

以下全部完成即可结束 V7：

```text
1. Formal 数据恢复
2. Formal S0 恢复
3. K=4 / Teacher cache 恢复
4. 336×1 baseline predictions 恢复
5. S0 Difficulty 完成
6. 336×4 / 1391×1 完成
7. Teacher Entropy 完成
8. Discordant cases 完成
9. Gradient 按条件执行或明确只做最小诊断
10. Weak-S0 按条件执行或明确 SKIP
11. Final Decision
```

不需要：

```text
重新跑完整 Teacher Gate
重新跑 Gap Gate
重新跑 Natural Equivalence Gate
重新证明 V5/V6 所有历史结论
```

---

# 18. 最终科研判定

## `V7_RESCUED`

如果：

```text
Class KD 在 scaling setting 的目标 Equiv subset 上稳定优于 Candidate
Full 不明显退化
且 S0 Difficulty / Entropy 至少有一个机制结果与提升一致
```

则：

> Pure Class KD 被救回，可以进入正式论文补全阶段。

## `V7_CONDITIONAL_RESCUE`

如果：

```text
总体 Candidate ≈/≥ Class
```

但：

```text
S0 Hard / Low-confidence
或
Teacher High-entropy
```

子群中 Class 明显更好，而且趋势一致，

则：

> Pure Class KD 不是 universal replacement，但“什么时候该删 intra supervision”是有效 RQ。

下一步自然走：

```text
Adaptive / Selective Intra-Class Distillation
```

本轮不要自动实现。

## `V7_NO_GO`

如果：

```text
336×4
1391×1
Difficulty
Entropy
```

都没有提供一致的 Class 优势，

则停止 Pure Class KD 主线。

不要继续 LR/tau/lambda fishing。

---

# 19. 最终报告

生成：

```text
reports/v7_final_decision.md
```

至少包含：

```text
1. 恢复了哪些必要资产
2. 336×1 baseline
3. S0 Difficulty 结果
4. 336×4 vs 1391×1
5. Teacher Entropy
6. Discordant Cases
7. Gradient（若执行）
8. Weak-S0（执行或 SKIP）
9. 统计不确定性
10. 最终：
    V7_RESCUED /
    V7_CONDITIONAL_RESCUE /
    V7_NO_GO
11. 下一阶段科研建议
12. Git SHA + artifact paths/hash
```

---

# 20. 保存纪律

必须维护：

```text
reports/v7_execution_ledger.md
artifacts_manifest.json
```

每个正式实验完成：

```bash
git add code/config/report/manifest
git commit
git push
```

checkpoint/cache/predictions 不直接进 Git，但必须保存：

```text
path
SHA256
生成命令
模型 revision
数据 hash
```

---

# 21. 允许停下来的条件

仅以下情况允许等待用户：

```text
HF gated 权限缺失
GitHub 权限缺失
GPU硬故障
磁盘不可解决
V6 Formal 数据存在两个互斥且无法判断的定义
关键模型 revision 永久不可获取
```

普通：

```text
OOM
依赖问题
代码 bug
路径错误
测试失败
```

自己解决并继续。

---

# 22. 最终执行指令

现在开始执行。

不要重新完整复刻 V6。

先恢复 V7 所需：

```text
Data
S0
K=4
Teacher Cache
336×1 baseline predictions
```

然后优先：

```text
S0 Difficulty
→
336×4 vs 1391×1
```

只要没有硬阻塞，连续工作直到：

```text
reports/v7_final_decision.md
```

完成，并完成最后一次：

```text
git commit
git push
```

然后再向用户汇报最终 V7 结论。
