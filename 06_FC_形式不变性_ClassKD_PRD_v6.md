# FC 形式不变性 Class KD — PRD v6

**版本日期：2026-08-24**  
**当前阶段：训练前的数据覆盖重构与第二轮 Gate**

---

# 0. 当前已经确定的事实

## 0.1 原来的“自然等价类”假设没有通过

ToolACE、Qwen3-8B、严格 canonicalizer 下：

```text
300 × 8 自然生成：
Any-Class 非平凡等价 = 2/300 = 0.67%
Gold-Class 非平凡等价 = 0/300

前 100 条追加到 16 samples/query：
Any-Class = 1/100 = 1%
Gold-Class = 0/100

结论：Natural Equivalence RQ = RESEARCH_NO_GO
```

这个结论保留，不允许删除或改写。

## 0.2 但补救实验给出了明显正信号

对严格可验证的等价写法进行确定性构造，再让同一 Teacher 重新打分：

```text
300 条总体：
Equivalent > Negative = 91.33%
Median Equivalent Alternative Mass = 28.54%

只保留真正 argument-key-order 的 109 条：
Equivalent > Negative = 96.33%
Median Alternative Mass = 30.75%
```

说明：

> Teacher 自然解码虽然几乎只输出一种标准写法，但当严格等价的其他写法被提供出来时，Teacher 通常明显认为它们比真正错误调用更合理，而且给予的相对权重并不小。

因此 Class KD 的研究问题从“自然多解”正式改为：

> **对于任务规则能够严格证明功能等价的多个 Function Call 表示，KD 是否有必要继续让 Student 模仿 Teacher 在这些等价写法内部的偏好？**

---

# 1. v6 最终研究目标

本论文希望同时完成两层贡献：

1. **机制发现**：普通 KD 会学习一些任务并不关心的“等价写法内部偏好”。
2. **训练方法**：提出 Class KD，把多个严格等价调用合并后再蒸馏，让 Student 更关注真正影响功能正确性的差异。

最终方法的目标不是要求所有 FC 数据都必须有丰富等价类，而是：

```text
普通 FC 数据
    ↓
自动检查当前调用是否存在严格可验证的等价变体
    ↓
有等价变体：使用 Class KD
没有等价变体：该调用形成单例，Class KD 自动退化为普通 Candidate KD
```

因此正式训练不能只挑“有等价类的样本”。

---

# 2. 为什么现在不能直接训练

当前 300 条 probe 的实际 gold 结构过于简单：

```text
平均 arguments = 1.36
arguments > 2 = 28/300 = 9.33%
实际 nested object/list = 0
实际 numeric argument = 0
真正 runtime-supported default = 0
```

当前补救正信号几乎只来自 JSON/object 字段换序。

如果现在直接训练，即使 Class KD 最终赢了，也只能非常安全地证明：

> “Class KD 对 object-key-order 这种等价关系有效。”

不足以支撑更一般的“Function Call 功能等价蒸馏”。

因此 v6 在 Student 正式训练前增加一次 **等价类型覆盖重构**。

---

# 3. v6 数据设计原则

## 3.1 不追求“所有语义相似都算等价”

只允许能够由程序和正式协议严格证明的关系。

禁止：

```text
Beijing ↔ 北京
NYC ↔ New York
C ↔ Celsius
日期自然语言改写
单位换算（除非 API 契约明确规定）
不同函数仅凭描述相似就判等价
LLM Judge / embedding similarity
```

## 3.2 至少覆盖三个不同层次

### A. 对象结构不变

包括：

```text
flat argument object 字段换序
nested object 内字段换序
```

注意：root 层 `name/arguments` 换序只作为附加实验，不作为主要类型。

### B. 值的严格等价表示

首选：

```text
number 类型下：20 ↔ 20.0
```

必须先通过 schema validation，不能把非法类型修成合法类型。

### C. 运行规则明确保证的等价

优先寻找以下至少一种：

```text
明确 runtime default：省略默认参数 ↔ 显式填写默认值

或

明确声明顺序无意义的集合/数组：
[A,B] ↔ [B,A]
```

必须有真实函数签名、运行时实现或 benchmark contract 支持。

---

# 4. 数据来源顺序

不要一开始就完全放弃 ToolACE。

按下面顺序执行：

```text
Step A：扫描 ToolACE 全量候选数据
    ↓
优先寻找：
>=2 / >=3 arguments
nested object
实际 numeric arguments
array/set
可验证 runtime default
    ↓
如果三个层次都能获得足够样本：继续只用 ToolACE
    ↓
如果某些类型不足：再增加第二个 FC 数据源补齐
```

第二数据源优先考虑已有公开、Schema 明确、Gold Call 稳定的数据，例如 xLAM/APIGen 一类数据。

选择第二数据源的标准必须提前固定：

- Function schema 完整；
- Gold call 可解析；
- 能程序化验证等价；
- 不是因为“听说 Class KD 在这个数据上效果好”才选择。

---

# 5. 新的“等价覆盖验证集”

这是一套 **训练前 Gate 数据**，不是最终训练数据。

目标至少包含三个主要类型：

```text
类型 A：对象结构换序
类型 B：数值严格等价表示
类型 C：运行规则等价（default 或明确无序集合）
```

推荐：

```yaml
每个主要类型:
  target_queries: 150
  minimum_queries: 100
```

如果 nested object 数据足够，可把 flat / nested 分开报告，但二者仍属于同一大类。

Gate 数据必须与后续正式测试集隔离。

---

# 6. Gate 的 Candidate Support 必须统一

为避免“某类等价变体更多，所以概率总和天然更高”的偏差，主 Gate 每条 Query 固定：

```text
1 × Gold
1 × Strict Equivalent
1 × Wrong Argument
1 × Wrong Function 或高质量 Hard Negative
```

即主 Gate 固定 K=4。

如果某个 Query 可以生成多个 Equivalent，只选一个作为主 Gate 候选；其他 Equivalent 可用于附加的多成员等价类分析，但不能混进主 Gate 后再和别的类型直接比较。

所有候选必须：

```text
parse
→ schema validate
→ canonicalize
```

并满足：

```text
Canon(Gold) == Canon(Equivalent)
Canon(Negative) != Canon(Gold)
```

---

# 7. Teacher Gate 看哪些指标

不要再把 Natural Equivalence Rate 作为主 Gate。

每一种等价类型分别统计下面三个核心问题。

## 7.1 Teacher 是否仍偏爱原写法？

看：

```text
Gold score - Equivalent score
```

如果通常 > 0，说明 Teacher 有“写法偏好”。

## 7.2 Teacher 是否真正认可等价写法？

看：

```text
Equivalent score > Negative score
```

统计比例。

这个指标越高，越说明 Teacher 知道“写法虽然不同，但比真正错误调用合理”。

## 7.3 等价写法的相对权重是否足够大？

在固定 K=4 support 中用当前冻结的 sequence score 和 tau=2 计算相对概率。

看：

```text
Equivalent Alternative Mass
```

注意：这个值只能解释为“在固定候选集合中的相对权重”，不能写成 Teacher 全输出空间里的真实生成概率。

---

# 8. v6 Teacher Gate 判定

## STRONG PASS

要求：

1. 至少覆盖 3 种主要等价类型；
2. 至少有 1 种不是 object/key-order 类型；
3. 每种主要类型 usable queries >= 100；
4. 每种主要类型：

```text
Equivalent > Negative Rate >= 70%
Median Equivalent Alternative Mass >= 10%
```

5. 不允许总体正信号几乎全部由 root-key-order 支撑。

## NARROW PASS

如果只有 object/key-order 类通过，而 numeric/runtime 类不成立：

```text
不再声称一般 Functional Equivalence
研究范围收窄为 Object/Permutation-Invariant KD
```

可以继续做方法实验，但论文 Claim 必须收窄。

## NO-GO

如果多个不同等价类型都出现：

```text
Equivalent > Negative < 60%
或
Equivalent mass < 5%
```

则说明 Teacher 并没有形成一般的“认可等价形式但偏爱某种写法”的结构。

停止一般 Class KD 主张。

---

# 9. Gate 通过后：正式训练数据怎么构造

这是 v6 最重要的实用性约束：

> **正式训练不使用只挑等价样本的特殊训练集。**

正式训练使用完整 FC 训练数据。

每条训练样本自动处理：

```text
Gold Call
    ↓
检查是否存在当前规则允许的严格等价变体
    ↓
有：加入 1–2 个 Equivalent Candidate
无：不强行构造
    ↓
再加入相同的错误候选
```

对于无等价变体的样本：

```text
Class KD == Candidate KD
```

因此方法天然适用于普通数据，只是在等价结构丰富的数据上预计收益更明显。

---

# 10. Teacher Gate 通过后的下一道 Gate：Student 是否真的需要学这些写法偏好？

先训练普通 Gold SFT，得到公共 Student S0。

然后对同一个固定 Candidate Support：

```text
Teacher 打分
Student S0 打分
```

把 Candidate KD 的 Teacher-Student 差异拆成：

```text
真正功能不同造成的差异
+
同一正确功能内部“写法偏好”造成的差异
```

核心问题：

> 普通 Candidate KD 的训练压力里，有多少是在纠正“写法偏好”？

建议判定：

```text
>=10%：Strong Pass
5%–10%：Weak Pass
<5%：No-Go
```

这个 Gate 通过以后才正式训练 Class KD。

---

# 11. 第一轮最小训练

先只跑三个分支：

```text
Common S0
  ├─ SFT Continuation
  ├─ Candidate KD
  └─ Class KD
```

Candidate KD 与 Class KD 必须使用：

- 同一个 S0；
- 同一训练集；
- 同一个 Candidate Pool；
- 同一个 Equivalent Variant；
- 同一个 Negative；
- 同一个 Teacher score；
- 同一优化器和训练步数。

唯一差别：

```text
Candidate KD：保留正确等价类内部的 Teacher 写法偏好
Class KD：只要求正确类总概率正确，不要求复制类内偏好
```

第一轮只判断：

```text
Class KD 是否可信地优于 Candidate KD？
```

---

# 12. 第一轮训练如果为正，再补正式论文 Baseline

顺序：

```text
1. Equivalence-Augmented SFT
   → 排除“只是把等价数据加进训练就够了”

2. STAR top-k FKL / CKD
   → 与当前较强 FC KD 方法比较

3. 第二 Seed
   → 排除随机波动

4. 第二 Student size
   → 检查方法是否只对一个模型成立
```

---

# 13. 泛化实验最后再做

只有 Class KD 在主实验中确实有效后，再做：

```text
第二数据源
更复杂 nested schema
不同等价类型分桶
Schema 扰动 / Schema 替换环境
BFCL 外部测试
```

Schema 鲁棒性是加分项，不在当前阶段预设“Class KD 一定更鲁棒”。

---

# 14. 正式论文最终应回答的五个问题

1. Teacher 是否在多类严格等价写法上存在“偏爱一种，但仍明显认可其他等价写法”的现象？
2. Student 与 Teacher 在这种写法偏好上是否存在明显差异？
3. 删除类内写法偏好后，Class KD 是否比 Candidate KD 更好？
4. 提升是否不能被简单的等价数据增强解释？
5. Class KD 是否能与 STAR 等强 FC KD 方法竞争，并在外部 FC / Schema 变化环境中保持优势？

---

# 15. 现在立刻让代码 Agent 做什么

**现在不要训练 Student。**

按顺序执行：

```text
P1. 扫描 ToolACE 全量数据结构
    - actual argument count
    - nested object
    - numeric actual values
    - array/set
    - default 是否有真实 runtime contract

P2. 按三层等价关系寻找样本
    A 对象结构
    B 数值表示
    C runtime 规则

P3. ToolACE 不足的类型，再从第二数据源补齐

P4. 固定一套新的 Equivalence Coverage Probe
    - 每类目标 150，最低 100
    - 不使用 BFCL

P5. 每条固定 K=4：
    Gold / Equivalent / Wrong-Argument / Wrong-Function-or-Hard-Negative

P6. 冻结当前 canonicalizer 规则
    只允许增加“有正式契约证明”的新规则

P7. 用同一 Qwen3-8B BF16 Teacher 重新打分

P8. 按类型输出：
    - Gold vs Equivalent 分数差
    - Equivalent > Negative Rate
    - Equivalent Alternative Mass

P9. 判定：
    STRONG_PASS / NARROW_PASS / NO_GO

P10. Gate 通过以后，再开始 S0。
```

---

# 16. 当前禁止事项

在新 Gate 完成之前不要：

- 训练正式 Class KD；
- 为了凑样本放宽到自然语言同义词；
- 使用 LLM Judge 判断等价；
- 把 BFCL 当训练/调参数据；
- 用高温/Beam Search 制造自然等价率；
- 只挑最终结果好的等价类型报告；
- 把 root-key-order 当成一般功能等价的主要证据。

---

# 17. v6 的一句话原则

> **先证明 Class KD 面对的是多种真实、严格可验证的 Function Call 等价关系，而不是单一 JSON 换序特例；再证明删除这些等价类内部的 Teacher 写法偏好，确实能训练出更好的小模型 FC 能力。**

---

# 18. 正式评测固定为三层

Class KD 的正式结果**不能只报一个总体准确率，也不能只报存在等价结构的样本**。固定同时报告三层：

```text
A. 完整测试集
B. 非平凡严格等价结构子集
C. 单例子集（没有非平凡严格等价变体）
```

## 18.1 完整测试集

回答：

> 把 Class KD 放进正常 FC 训练流程后，整体 FC 能力有没有帮助？

即使只有部分数据真正触发 Class KD，也必须保留完整集结果。

## 18.2 等价结构子集

定义必须在 Student 训练前固定：

```text
至少存在一个：
surface 不同
但
canonical call 完全相同
的严格等价变体
```

这个子集是 Class KD 真正改变训练目标的位置，因此是**核心方法结果之一**。

## 18.3 单例子集

没有任何非平凡严格等价变体的样本。

这些样本上：

```text
Class KD == Candidate KD
```

因此它是天然机制对照。

理想结果：

```text
等价结构子集：Class KD 明显 > Candidate KD
单例子集：Class KD ≈ Candidate KD
完整测试集：Class KD 不退化，最好有整体提升
```

---

# 19. 如果完整集提升很小，如何判断

例如只有 30% 测试样本存在非平凡等价结构，则剩余约 70% 样本上 Class KD 与 Candidate KD 理论上没有区别，因此总体提升会被稀释。

所以出现：

```text
完整集：+0.3 ~ +1.0
等价结构子集：+3 ~ +5
单例子集：≈0
```

仍然可能是非常符合机制的正向结果。

但必须同时满足：

- 子集划分是在训练结果出现前固定；
- 不是根据涨点挑样本；
- 单例子集没有异常大幅变化；
- 完整集不能明显退化。

如果完整集明显下降，则不能声称 Class KD 是整体更好的 FC 训练方法，只能定位为等价场景专用方法。

---

# 20. 推荐增加“等价结构丰富程度”分析

在训练前按规则固定分桶，例如：

```text
Bucket 0：无非平凡等价结构
Bucket 1：只有一种/一个等价变体
Bucket 2：存在多个等价变体
Bucket 3：Teacher 对其他等价写法给予较高相对分数
```

报告：

```text
Class KD - Candidate KD
```

是否随等价结构变丰富而增大。

如果出现：

```text
无等价：≈0
弱等价：小幅提升
丰富等价：明显提升
```

这会成为很强的机制证据：**Class KD 恰好在它应该发挥作用的位置产生收益。**

---

# 21. 正式主结果表建议

```text
method,
overall_canonical_acc,
equiv_subset_canonical_acc,
singleton_subset_canonical_acc,
bfcl_acc,
invalid_rate
```

至少包含：

```text
SFT Continuation
Equivalence-Augmented SFT（正式阶段）
Candidate KD
Class KD
STAR-style strong KD baseline（正式阶段）
```

论文解释顺序：

```text
完整集 → 方法有没有整体价值
等价结构子集 → 方法在目标问题上是否有效
单例子集 → 提升是否符合机制，而不是 loss 随机变化
```

