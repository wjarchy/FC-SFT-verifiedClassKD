# FC Equivalence-Class KD Go/No-Go Pilot PRD v5 — FINAL

**项目**：小论文111  
**版本**：v5 FINAL  
**日期**：2026-08-19  
**状态**：当前唯一可执行 Pilot PRD  
**目的**：用最低实验成本获得可信的 **GO / WEAK-GO / NO-GO** 信号，而不是一次完成整篇论文的全部实验。

---

# 0. 本版本只回答一个 Research Question

Function Calling 的序列化输出与经过验证的完整调用语义之间可能存在 many-to-one 映射。

如果两个 surface-different 的调用：

```text
y_i != y_j
```

但经过确定性、Schema/Runtime-aware canonicalization 后：

```text
Canon(y_i, S) == Canon(y_j, S)
```

则它们属于同一个 functional equivalence class。

本 Pilot 只问：

> **Candidate-level KD 是否有必要继续拟合 Teacher 在同一个 functional equivalence class 内部的 surface preference？如果删除这部分 intra-class matching，只蒸馏不同 canonical call 之间的概率结构，Student 的 Function Calling 能力能否更好，或至少不退化？**

本版本**不再同时证明**：

- 小模型为什么会学习 schema/name shortcut；
- Student capacity 是否是该现象的唯一原因；
- canonicalizer / verifier 本身是不是一个创新；
- Class KD 是否一定提升 schema robustness；
- RL 是否不如 KD；
- 多 Student、多 Teacher、多 Seed、多 Benchmark 是否全部成立。

这些问题要么已有相关文献支撑，要么只有在本 Pilot 出现正向信号后才值得继续投入。

---

# 1. 当前方法链路与 Baseline 定位

当前论文逻辑固定为：

```text
SFT
 ↓
Token-level KD / top-k FKL
 ↓
Complete-call Candidate Space
 ↓
Functional Equivalence-Class Space
```

四个实验角色必须区分清楚：

### A. SFT Continuation

普通 CE/SFT。

作用：

> 控制从公共 Student S0 继续训练相同步数本身带来的收益。

---

### B. Token KD / top-k FKL

这是**已有 FC 文献证明有效的强 baseline**。

Pilot 使用 STAR 中的 top-k Forward KL 作为外部强基线。

截至 2026-08-19 已核对 STAR 原文：

```yaml
teacher_top_k: 100
kd_learning_rate_reference: 3e-6
kd_batch_size_reference: 128
```

STAR 的 top-k FKL 本身不使用 CKD tail penalty。

注意：

> 本 Pilot 只复现/适配 **top-k FKL baseline**，不加入 CKD，也不加入 Sim-RL。

Token KD 的**具体训练序列/prefix 构造**在 STAR 精读完成后最终锁定；在此之前可以先实现通用 top-k FKL loss/cache 接口，但不要提前把“Gold-prefix 版”写死成 STAR 的精确复现。

---

### C. Candidate KD

Candidate KD **不是社区公认的 FC 强基线**。

它在本项目中的角色只是一个机制桥梁：

> 把 token-level supervision 提升到 complete-call sequence space，使我们能够明确研究“完整调用之间的概率结构”。

它保留：

```text
inter-class matching
+
intra-class matching
```

---

### D. Equivalence-Class KD

这是本 Pilot 的唯一核心研究变量。

它只保留：

```text
inter-class matching
```

并删除：

```text
同一 canonical call 内部不同 surface realization 之间的 Teacher preference
```

---

# 2. 最关键的科学控制

真正的因果比较只发生在：

```text
Candidate KD
vs
Equivalence-Class KD
```

二者必须：

- 同一个 S0；
- 同一个 Train Split；
- 同一个 Candidate Pool；
- 同一个 Candidate 顺序；
- 同一个 sequence score；
- 同一个 candidate temperature；
- 同一个 optimizer；
- 同一个 LR；
- 同一个 batch / grad accumulation；
- 同一个训练步数；
- 同一个 seed；
- 同一个 evaluation pipeline。

唯一差别：

```text
Candidate KD = inter-class + intra-class
Class KD     = inter-class
```

Token KD 是**外部强 baseline / performance guardrail**，不是 Candidate→Class 因果分解的一部分。

因此不要要求 Token KD 与 Candidate/Class 在所有 loss 结构和超参数上完全一样；Token KD 优先遵守已发表方法的合理配置。

---

# 3. 数学定义

为避免 Markdown 数学渲染问题，本 PRD 的关键公式同时使用纯文本表达。

## 3.1 Complete-call Sequence Score

对输入 x 和完整候选调用 y：

```text
score_M(y | x)
=
(1 / |y|^alpha)
*
sum_t log p_M(y_t | x, y_<t)
```

Pilot 默认：

```yaml
candidate_length_alpha: 1.0
candidate_temperature: 2.0
```

candidate temperature 只属于 Candidate/Class 分布，不是 STAR Token-KD temperature。

---

## 3.2 Candidate Distribution

```text
pi_M(y_i | x)
=
softmax(score_M(y_i | x) / tau)
```

其中所有候选都来自同一个固定 Candidate Pool。

---

## 3.3 Candidate KD

```text
L_candidate_KD
=
KL(pi_T || pi_S)
```

Candidate 分支默认：

```text
L_Candidate
=
L_SFT
+
lambda_candidate * L_candidate_KD
```

Pilot 固定：

```yaml
lambda_candidate: 1.0
```

若 loss scale 明显失衡，只允许做一次 sanity calibration，不做网格搜索。

---

## 3.4 Canonical Equivalence Mapping

定义：

```text
g_S(y) = Canon(y, S)
```

其中 S 是当前 Tool Schema / 明确 Runtime Contract。

仅当：

```text
Canon(y_i, S) == Canon(y_j, S)
```

才认为：

```text
y_i ~ y_j
```

---

## 3.5 Class Distribution

对 canonical class e：

```text
q_M(e | x)
=
sum_{y_i: g_S(y_i)=e} pi_M(y_i | x)
```

---

## 3.6 Equivalence-Class KD

```text
L_class_KD
=
KL(q_T || q_S)
```

Class 分支：

```text
L_Class
=
L_SFT
+
lambda_class * L_class_KD
```

Pilot 固定：

```yaml
lambda_class: 1.0
```

Candidate 与 Class 的 CE/SFT 部分必须完全相同。

---

## 3.7 Candidate KL 分解

对 candidate class mapping g：

```text
KL(pi_T || pi_S)
=
KL(q_T || q_S)
+
sum_e q_T(e) *
KL(pi_T(. | e) || pi_S(. | e))
```

即：

```text
Candidate KD
=
Inter-class Matching
+
Intra-class Matching
```

而：

```text
Class KD
=
Inter-class Matching
```

本 Pilot 的唯一机制问题：

> **Intra-class Matching 是任务有价值知识，还是可以删除的 surface-level constraint？**

---

# 4. Class 的定义：彻底取消错误粗分类

旧方案曾将候选划分成：

```text
correct
wrong_function
wrong_arguments
invalid
```

v5 **禁止**用这些 error taxonomy 定义 KD Class。

例如：

```text
get_weather(city="Shanghai")
get_weather(city="New York")
```

虽然对于“北京天气”这个 Query 都可能属于 `wrong_arguments`，但它们是两个不同的完整调用语义，因此必须属于两个不同的 canonical classes。

同理：

```text
wrong_function_A
wrong_function_B
```

不能因为都属于 wrong function 就聚合。

错误标签只允许作为：

```text
analysis_tag
```

例如：

```text
candidate_type = wrong_function
candidate_type = wrong_argument
candidate_type = invalid
```

但它们**不得参与 class probability aggregation**。

---

# 5. Deterministic Canonicalizer Constitution

Canonicalizer 是本 Pilot 最关键的任务定义组件。

核心原则固定为：

> **Only collapse distinctions that are guaranteed to be semantically irrelevant by the formal parser, schema, or runtime/benchmark contract. Never infer equivalence from natural-language similarity.**

中文：

> **只合并由 Parser、Schema 或明确 Runtime/Benchmark 契约能够确定为无语义差异的表示变化；绝不根据自然语言相似性猜测功能等价。**

---

# 6. Canonicalizer Pipeline

统一实现：

```text
Raw Model Output
 ↓
Production-equivalent Tool-call Parser
 ↓
Exact Function Resolution
 ↓
Schema Validation
 ↓
Conservative Type-aware Canonicalization
 ↓
Stable Canonical Signature
```

接口建议：

```python
validate_call(raw_call, schema)
canonicalize_call(parsed_call, schema, explicit_runtime_rules)
evaluate_call(query, canonical_call, gold_call)
```

三者职责必须分离。

---

# 7. `validate_call`：只判断合法性

检查至少包括：

- Tool-call / JSON parse；
- function name 是否存在；
- required arguments；
- unknown arguments；
- argument type；
- enum；
- nested schema。

返回类似：

```text
VALID
INVALID_PARSE
INVALID_FUNCTION
INVALID_MISSING_REQUIRED
INVALID_UNKNOWN_ARGUMENT
INVALID_TYPE
INVALID_ENUM
```

Verifier/Validator 不判断：

```text
“这个调用语义上是不是适合当前 Query”
```

---

# 8. `canonicalize_call`：只定义调用身份

推荐内部表示：

```python
CanonicalCall(
    function_name="get_weather",
    arguments={
        "city": "Beijing",
        "unit": "C"
    }
)
```

然后进行稳定序列化，例如：

```text
{"arguments":{"city":"Beijing","unit":"C"},"function":"get_weather"}
```

可额外保存 hash，但必须同时保留 human-readable signature。

---

# 9. v5 允许的 Canonicalization

## 9.1 Function Name

默认：

```text
Exact Match Only
```

不同 function identifier 永远不同 class，除非 Runtime/Benchmark 明确声明 alias。

---

## 9.2 JSON/Object Key Order

允许递归忽略 object key 顺序。

例如：

```json
{"a":1,"b":2}
```

与：

```json
{"b":2,"a":1}
```

在同一 object schema 下 canonicalize 到相同 object。

---

## 9.3 Nested Object

递归 sort object keys。

---

## 9.4 JSON Whitespace / Escape Representation

只要 production parser 解析后得到完全相同 typed value，则 surface whitespace / JSON escape 写法差异自然消失。

---

## 9.5 Numeric Representation

必须：

```text
先 Schema Validate
再按已验证 numeric type 做稳定数值表示
```

例如 schema 明确为 `number` 时，可统一：

```text
1
1.0
1.000
```

到同一个 canonical numeric value。

不得通过 numeric normalization 把原本 Schema-invalid 的输入偷偷修成 valid。

---

# 10. v5 明确禁止的 Canonicalization

以下默认全部禁止：

### String Case

```text
"Beijing"
"beijing"
```

不同。

### Synonym / Entity Alias

```text
"Beijing"
"北京市"
```

不同。

```text
"NYC"
"New York"
```

不同。

### Function Alias 猜测

```text
get_weather
weather_lookup
```

不同。

### Parameter Alias 猜测

不同参数名不得根据自然语言 description 合并。

### Missing vs null

```text
{"city":"Beijing"}
```

与：

```text
{"city":"Beijing","date":null}
```

默认不同。

### Array Reordering

```text
["Beijing","Shanghai"]
```

与：

```text
["Shanghai","Beijing"]
```

默认不同。

本 Pilot 第一版不支持 order-insensitive array 语义。

### Date Semantic Normalization

不自动处理：

```text
2026-08-19
August 19, 2026
```

### Unit Conversion

不自动处理：

```text
1 km
1000 m
```

### LLM / Embedding Equivalence

禁止。

### Fuzzy String Matching

禁止。

---

# 11. Default 参数的特殊规则

**仅仅 Schema 中存在 `default` 字段，不足以证明 omitted 与 explicit-default 在 Runtime 中等价。**

只有以下情况允许补 default：

1. 实际 Runtime 明确保证参数缺失时自动使用该 default；
2. Benchmark / Dataset evaluator 明确规定 omitted default 与 explicit default 等价；
3. 代码层函数签名或正式 contract 能确定该行为。

否则：

```text
missing optional argument
```

和：

```text
explicit default argument
```

保持不同 canonical signatures。

这条规则优先级高于旧 PRD 中“schema 有 default 就补”的写法。

---

# 12. Invalid Candidate 的处理

不同 invalid candidates **绝不聚成一个 invalid class**。

例如：

```text
missing required
wrong type
malformed JSON
```

都保持 singleton identity。

建议：

```text
INVALID::<candidate_unique_id>
```

原因：

> 我们只知道它们都无效，不知道它们彼此功能等价。

---

# 13. Canonicalizer 不看 Query

Canonicalizer 定义：

```text
Canon(Call, Schema/Runtime Contract) -> Signature
```

而不是：

```text
Canon(Query, Call, Schema)
```

例如：

```text
get_weather(city="Shanghai")
```

不论 Query 问北京还是上海，其 canonical identity 始终相同。

是否对当前 Query 正确，由 evaluator 判断：

```text
Evaluate(Query, CanonicalCall, Gold) -> Correctness
```

不要把 correctness taxonomy 重新带回 Class 定义。

---

# 14. Canonicalizer Unit Tests

正式扫数据前至少覆盖：

## 应判为相同 Class

1. root object key reorder；
2. argument object key reorder；
3. nested object key reorder；
4. JSON whitespace 差异；
5. JSON escape 表示不同但 parser 后 typed string 完全相同；
6. numeric representation difference，前提是 Schema validation 后属于同一 numeric value；
7. omitted/explicit default，**仅当 Runtime/Benchmark contract 明确保证**。

## 必须判为不同 Class

8. 不同 function；
9. 同 function 不同 argument value；
10. missing vs null；
11. array order swap；
12. string case change；
13. synonym/entity alias；
14. malformed / invalid candidates 之间。

Gate：

```yaml
canonical_equivalence_precision_on_supported_rules: 100%
overall_parser_canonicalizer_accuracy: ">=99%"
```

任何无法确定的 case：

```text
unsupported_for_equivalence
```

宁可不合并。

---

# 15. 模型与资源

## 15.1 主模型

锁定官方模型 ID：

```yaml
student: Qwen/Qwen3-1.7B
teacher: Qwen/Qwen3-8B
enable_thinking: false
```

必须记录：

- exact model revision / commit；
- tokenizer revision；
- vocab size；
- BOS/EOS/PAD；
- chat template hash；
- tool-call template；
- `enable_thinking=false` 是否在所有训练/推理阶段一致。

Teacher 冻结，不训练。

---

## 15.2 辅助模型

```yaml
debug_student: Qwen/Qwen3-0.6B
```

仅用于：

- parser/template smoke；
- Tiny Overfit；
- loss toy tests；
- 11GB GPU 调试。

不进入第一轮主结论。

---

## 15.3 GPU 分工

### RTX 2080 Ti 11GB

- 数据清洗；
- canonicalizer；
- dataset adapter；
- Candidate pipeline；
- 0.6B Smoke/Tiny Overfit；
- loss unit tests；
- evaluation pipeline；
- 小样本 Teacher 量化预检查。

### RTX 4090 24GB

只有前置 Gate 通过后才租用：

- 8B Teacher multi-sampling；
- Teacher candidate scoring/cache；
- Token KD cache；
- 1.7B S0；
- 正式四分支 Pilot。

---

# 16. 数据源与 Split

第一轮只维护**一个主训练数据源**。

优先候选：

```text
ToolACE single-turn / single-call subset
```

备选：

```text
xLAM / APIGen 系列中 schema 清晰、gold call 稳定可解析的 single-call subset
```

选择标准：

- Tool Schema 完整；
- Gold call 可稳定 parse；
- 每条 Query 有多个 candidate tools；
- 参数有一定难度；
- single-turn；
- single-call；
- 与 BFCL 测试不直接重合。

推荐：

```yaml
train: 1500
validation: 200
internal_test: 400
opportunity_probe: 300
bfcl_subset: 200-400
seed: 42
```

所有 split 先固定 manifest。

---

# 17. Natural-Equivalence Held-out Set 不再强行规定 100–200

旧版写：

```text
natural_equiv_test = 100-200
```

但如果真实 Natural Equivalence Rate 只有 5%–10%，400 条 internal test 理论上只能自然获得约 20–40 条。

因此 v5 FINAL 改成：

> 从完全 held-out 的 `internal_test` 中进行同样的自然 multi-sampling，所有自然发现的 non-trivial gold-class equivalence cases 都作为 `natural_equiv_test`。

不强行凑数。

解释标准：

```text
>= 30 usable cases:
    可作为 Pilot 机制诊断

20–29:
    弱证据，只作方向参考

< 20:
    不允许对 held-out mechanism 做强结论
```

如果 Opportunity Gate 已 Strong Pass、但 internal_test 因样本数太小自然只得到 <30 条，可以额外增加**只用于 held-out evaluation 的样本**，不得进入训练/candidate construction。

---

# 18. Leakage Rules

BFCL：

- 不进入训练；
- 不进入 Candidate Pool construction；
- 不用于 lambda / candidate temperature 调参；
- 不用于 early stopping。

Opportunity Probe：

- 不进入正式 KD Train Candidate Pool；
- 只用于决定是否继续研究和确定 pipeline 是否有机会。

Natural-equivalence Test：

- 必须 held-out；
- 不进入任何训练 Candidate Pool；
- 不用于选择 checkpoint。

---

# 19. Phase A — Data Opportunity Gate

这是**第一生死检查**，优先级高于正式训练。

目标：

> 在不人为制造等价变体的情况下，Teacher 自然输出中是否真的存在足够多 surface-different、canonical-equivalent 的完整调用？

---

# 20. Teacher Natural Multi-Sampling

对：

```yaml
opportunity_probe: 300
```

每个 Query 使用完全相同的 Query + Schema，多次独立完整调用采样：

```yaml
teacher_samples_per_query: 8
sampling_temperature: 0.7
top_p: 0.95
max_new_tokens: fixed_by_tool_template
```

禁止为了提高 Equivalence Rate：

- 改 Schema；
- 参数重命名；
- function 重命名；
- schema reorder；
- 人工插入 equivalent candidate；
- LLM paraphrase candidate。

---

# 21. Sampling 只发现 Support，不估计 Probability

这是硬规则。

不能用：

```text
某 surface 在 8 次 sample 中出现 5 次
```

直接当：

```text
Teacher probability = 5/8
```

正确流程：

```text
8 Teacher generations
 ↓
Surface dedup
 ↓
Parse + Validate
 ↓
Canonicalize
 ↓
得到 unique candidate support
 ↓
Teacher 对每个 unique candidate 做 teacher-forced sequence scoring
 ↓
用统一 score + softmax 得到 candidate probability
 ↓
再聚合成 class probability
```

采样频率只用于：

```text
candidate discovery
```

概率必须来自：

```text
Teacher sequence score
```

---

# 22. Natural Equivalence 的定义

对一个 Query，存在 Natural Non-trivial Equivalence，当且仅当：

```text
存在 y_i != y_j
AND
两者都是 valid calls
AND
Canon(y_i,S) == Canon(y_j,S)
```

同时必须额外区分：

### Any-Class Natural Equivalence Rate

任意 canonical class 内有多个 surface realizations。

### Gold-Class Natural Equivalence Rate

Gold canonical class 内有多个 surface realizations。

当前 RQ 最关心：

```text
Gold-Class Natural Equivalence Rate
```

因为它直接对应：

> 多种不同 surface 表示是否能实现同一个正确 Function Call。

---

# 23. Opportunity Metrics

至少输出：

```text
Any-Class Natural Equivalence Rate
Gold-Class Natural Equivalence Rate
Mean Surface Multiplicity
Median Surface Multiplicity
Teacher Intra-class Entropy
Teacher Dominance Rate
Equivalence-Type Breakdown
Valid Generation Rate
Unique Surface Rate
```

`Equivalence-Type Breakdown` 至少区分：

```text
json/object ordering
numeric representation
explicit runtime default
wrapper/serialization only
other formally-supported rule
```

如果机会几乎全部来自极其琐碎的 whitespace/wrapper 差异，需要在结论中单独标注，不能包装成复杂 functional equivalence。

---

# 24. Opportunity Gate

以下阈值**只用于内部决策，不作为论文事实声明**。

### STRONG PASS

优先满足：

```text
Gold-Class Natural Equivalence Rate >= 10%
```

或至少：

```text
Any-Class Natural Equivalence Rate >= 10%
AND
Gold-Class Natural Equivalence Rate >= 5%
```

并且：

- 非平凡 class 不几乎全部 one-hot dominance；
- equivalence 不全部只来自 whitespace noise。

### WEAK PASS

```text
Gold-Class Natural Equivalence Rate in [5%,10%)
```

或：

```text
Any-Class >= 5%
```

但 gold-class evidence 较弱。

此时只进入 Gap Gate，不直接跑全部训练。

### PROVISIONAL NO-GO

```text
Gold-Class Natural Equivalence Rate < 5%
AND
Any-Class Natural Equivalence Rate < 5%
```

在最终判 Research No-Go 前，允许一个**唯一的低成本确认动作**：

```text
对 100 个 probe queries 将 samples/query 从 8 增至 16
```

若仍 <5%：

```text
Research NO-GO
```

不要靠人工 variant 把机会率重新制造出来。

---

# 25. Phase B — Tiny Overfit / Pipeline Gate

正式 1.7B 训练前：

用最多 32 条样本训练 debug model 或主模型小规模配置。

要求：

```text
Training-set Canonical/AST Accuracy >= 90%
```

它不是泛化指标，只证明：

- label mask；
- tool template；
- LoRA/QLoRA；
- generation；
- parser；
- canonical evaluator；

整条 pipeline 可学习。

不过：

```text
Pipeline Fail
```

不是 Research No-Go。

---

# 26. Phase C — Common Student S0

统一：

```text
Qwen3-1.7B
 ↓
Gold FC SFT
 ↓
Common Student S0
```

所有后续训练分支从同一个 S0 初始化。

S0 的目的：

1. 让 Student 具备基础 FC 能力；
2. 产生 Student hard errors；
3. 形成可蒸馏 Teacher–Student gap；
4. 保证 Candidate/Class 的起点一致。

建议在 internal sample 上观察：

```text
Canonical/AST Accuracy: 35%–80%
```

如果 >85%：

```text
任务可能过简单
```

如果 <20%：

```text
先排查 pipeline / task difficulty
```

---

# 27. Phase D — Student–Teacher Intra-class Gap Gate

只在 Opportunity Gate 至少 Weak Pass 后执行。

对有自然 equivalence 的 Query：

- 保留自然 Teacher candidate support；
- 加 Gold；
- 可加 S0 hard error；
- 可加 distinct wrong-function / wrong-argument candidates；
- **不加入人工 Strict Equivalent**。

Teacher 与 S0 对**完全相同 candidate set**重新计算 sequence score。

---

# 28. Gap Metrics

计算：

```text
Candidate KL
Inter-class KL
Intra-class KL
Intra-class Contribution Ratio
```

其中：

```text
r_intra
=
Intra-class KL contribution
/
Candidate KL
```

同时分别输出：

```text
all non-trivial classes
gold-equivalence classes only
```

统计：

- mean；
- median；
- p75；
- p90。

---

# 29. Gap Gate

### STRONG PASS

至少满足一项：

```text
median r_intra >= 10%
```

或：

```text
>=25% natural-equivalence samples have r_intra >=20%
```

并且 Opportunity Gate 不低于 Weak Pass。

### WEAK PASS

```text
median r_intra in [5%,10%)
```

### NO-GO

```text
median r_intra < 5%
AND
p90 r_intra < 10%
```

说明：

> Candidate KD 中准备删除的 intra-class term 本身几乎不存在，不值得继续。

---

# 30. Phase E — 正式 Train Candidate Pool

只有 Opportunity + Gap 两个 Gate 至少 Weak Pass 后构造。

每条 Train Query：

```yaml
max_candidates: 6
min_candidates: 3
```

优先级：

1. Gold；
2. 自然出现、与某 canonical class 等价的 alternative surface call，最多 2；
3. S0 hard error；
4. distinct wrong-function；
5. distinct wrong-argument；
6. 其他自然模型错误候选。

没有自然 equivalent：

```text
不补
```

此时该样本上：

```text
Candidate KD == Class KD
```

这不是问题。

它意味着：

> Class KD 只在真实存在 representation redundancy 的位置改变监督。

---

# 31. Candidate Pool Bookkeeping

每个 Candidate 保存：

```text
query_id
candidate_id
raw_surface
parsed_call
validity
canonical_signature
source_tag
analysis_tag
```

`source_tag` 可用：

```text
gold
teacher_sample
student_hard_error
wrong_function_rule
wrong_argument_rule
```

`analysis_tag` 不参与 class aggregation。

---

# 32. Phase F — Teacher Cache

## 32.1 Candidate Score Cache

保存：

```text
raw_logprob_sum
sequence_length
length_normalized_score
candidate_probability
canonical_signature
class_probability
source_tag
teacher_model_revision
tokenizer_hash
prompt_hash
```

Teacher 不得在 Student training step 中在线常驻。

---

## 32.2 Token KD Cache

截至本版已从 STAR 原文核实：

```yaml
top_k: 100
reference_kd_lr: 3e-6
reference_batch_size: 128
```

FKL 形式：

```text
L_FKL-topK
=
sum_{v in TeacherTopK}
P_T(v|prefix)
*
log(P_T(v|prefix) / P_S(v|prefix))
```

注意：

- 不加入 CKD tail penalty；
- 不使用旧版无依据的 `top_k=64`；
- 不默认加入 `temperature=2.0`；
- exact prefix/teacher-trajectory construction 在 STAR 精读后锁定；
- 在锁定前可以完成 loss/cache API 和 Toy Test，但不要宣称已经“复现 STAR”。

---

# 33. Phase G — 四条训练分支

```text
                 ┌─ A. SFT Continuation
Common S0 ───────┼─ B. Token KD / top-k FKL
                 ├─ C. Candidate KD
                 └─ D. Equivalence-Class KD
```

---

# 34. Branch A — SFT Continuation

目的：

> 控制“其他分支只是多训练了一段时间”。

使用普通 CE/SFT。

---

# 35. Branch B — Token KD / top-k FKL

目的：

> 已发表 FC KD 强 baseline。

优先按 STAR 的已公开 FKL 设置复现/适配。

不要为了“所有分支 LR 相同”故意破坏已发表 baseline。

本分支与 Class KD 的比较是：

```text
performance baseline comparison
```

不是严格的单变量 causal comparison。

---

# 36. Branch C — Candidate KD

```text
L_Candidate
=
L_SFT + lambda * CandidateKL
```

角色：

> complete-call sequence-space mechanism bridge。

---

# 37. Branch D — Equivalence-Class KD

```text
L_Class
=
L_SFT + lambda * ClassKL
```

角色：

> 核心研究方法。

---

# 38. 训练公平性

### 四分支共同固定

- S0；
- Train Split；
- 数据总量；
- total update steps；
- effective batch size 尽量一致；
- seed=42；
- validation cadence；
- generation config；
- evaluation pipeline。

### Candidate vs Class 必须额外完全一致

- optimizer；
- LR；
- LR schedule；
- Candidate Pool；
- Candidate order；
- candidate score；
- alpha；
- tau；
- lambda；
- answer mask。

### Token KD

允许采用其发表方法合理的 LR / loss 形式。

不得因为“统一超参数”把 strong baseline 故意跑弱。

---

# 39. 不做大规模超参数搜索

Pilot 固定：

```yaml
seed: 42

candidate:
  alpha: 1.0
  temperature: 2.0
  lambda_candidate: 1.0
  lambda_class: 1.0

token_fkl:
  top_k: 100
```

允许：

```text
一次 loss-scale / NaN / zero-gradient sanity calibration
```

禁止：

```text
每个方法单独大规模 grid search
```

---

# 40. 必须通过的 Toy / Unit Tests

## Token KD Toy Test

检查：

- top-k ids/log-probs 对齐；
- teacher top-k=100 接口；
- answer positions mask；
- prompt/tool-schema/padding 不参与 KD；
- cache reload 后 loss 可复现；
- 与手工小 vocab FKL 一致。

---

## Candidate Score Test

人工 2–3 个短序列手算：

```text
token log-prob sum
length normalization
softmax
candidate probability
```

要求：

```text
numerical error < 1e-6
probability sum == 1
```

---

## Canonical Aggregation Test

```text
A1: 0.30 -> canonical A
A2: 0.20 -> canonical A
B1: 0.50 -> canonical B
```

输出必须：

```text
A: 0.50
B: 0.50
```

---

## KL Decomposition Test

数值验证：

```text
CandidateKL
=
InterClassKL
+
WeightedIntraClassKL
```

误差：

```text
< 1e-6
```

---

## Permutation Test

重新随机排列 Candidate list：

```text
Candidate KD loss unchanged
Class KD loss unchanged
```

---

## Gradient Test

确保：

```text
loss.backward()
```

后：

- Student 参数有非零 gradient；
- Teacher 无 gradient；
- cached teacher score 不进入 autograd；
- Candidate/Class 对应参数确实更新。

---

# 41. Evaluation：只做三层证据

v5 不建设新大 Benchmark。

---

# 42. Internal Generation Test

所有模型独立 autoregressive generation。

禁止：

- Teacher rerank；
- Candidate rerank；
- canonicalizer repair；
- Verifier correction；
- LLM Judge 修答案。

输出：

```text
Parse Success
Function Name Accuracy
Argument Accuracy
Canonical / AST Accuracy
Invalid Rate
```

主指标：

```text
Canonical / AST Accuracy
```

---

# 43. Natural-Equivalence Held-out Mechanism Test

对 held-out natural equivalence cases，用统一 Candidate support 对**四个模型全部**计算：

```text
Candidate KL
Inter-class KL
Intra-class KL
Correct Canonical-Class Mass
Canonical Accuracy
```

注意：

> SFT Continuation 和 Token KD 虽然训练时没有 Candidate/Class loss，评测时仍然可以、也应该计算这些 KL 指标。

因此**不写 N/A**。

这能回答：

```text
SFT 学成什么分布？
Token KD 学成什么分布？
Candidate KD 是否更像 Teacher 的完整 sequence preference？
Class KD 是否更像 Teacher 的 inter-class structure、但不复制 intra-class preference？
```

---

# 44. v5 最理想的机制结果

例如：

```text
Class KD:
  Inter-class KL ↓
  Canonical Accuracy ↑ / =
  Intra-class KL 不下降，甚至略高
```

这意味着：

> Student 没有精确复制 Teacher 对同一 canonical call 内不同 surface realizations 的 preference，但它仍然学好了真正任务相关的 class-level distinction。

这直接支持：

```text
intra-class preference is not necessary for task-level imitation
```

但不要立刻进一步声称：

```text
“因为小模型容量被释放，所以性能提升”
```

capacity mechanism 留到 GO 后再验证。

---

# 45. BFCL Single-turn Compatible Subset

作用：

> external validity guardrail。

只看当前 Pilot 兼容范围。

输出：

```text
AST / function-call accuracy
parse success
invalid rate
```

统一比较：

```text
SFT Continuation
Token KD / FKL
Candidate KD
Equivalence-Class KD
```

BFCL 不参与训练和调参。

---

# 46. 可选 Surface-Equivalence Stress Diagnostic

只有主流程跑通后、成本很低时才做。

允许人为构造：

- JSON/object key reorder；
- parser-safe serialization variation；
- Runtime contract 明确的 explicit/implicit default。

用途：

```text
stress test / case study
```

禁止用它证明：

```text
“真实 FC 天然有很多 equivalence”
```

---

# 47. Pipeline Fail vs Research No-Go

必须严格区分。

## Pipeline Fail

例如：

- Tiny Overfit 不过；
- parser/canonicalizer unit test 失败；
- Token FKL Toy Test 失败；
- Candidate score/KL decomposition 不一致；
- Teacher generation 大量无法 parse；
- BFCL adapter 明显异常；
- loss NaN / zero gradient。

处理：

```text
STOP
→ 修 pipeline
→ 重跑 Gate
```

不能写成：

```text
“Class KD 无效”
```

---

# 48. STRONG GO

前置要求：

```text
Opportunity Gate = Strong Pass
Gap Gate = Strong Pass
```

训练后：

### A. Candidate vs Class 出现机制优势

至少一个：

```text
Class canonical accuracy > Candidate canonical accuracy
```

或：

```text
Class held-out Inter-class KL
相对 Candidate 降低 >=10%
AND
Canonical Accuracy 不下降 >0.5 point
```

### B. 能站在 Token KD 强 baseline 旁边

建议内部判断：

```text
Class KD >= Token KD - 1.0 point
```

在 Internal 或 BFCL 至少一个主要指标成立，另一处无明显退化。

### C. External Guardrail

```text
BFCL:
Class KD 不比最佳非-Class baseline 下降 >1.0 point
```

满足 A+B+C：

```text
STRONG GO
```

进入正式论文阶段。

---

# 49. WEAK GO

典型：

```text
Class KD > Candidate KD
on natural-equivalence mechanism metrics
```

且：

```text
Inter-class KL 明显降低
```

但：

```text
overall / BFCL
仍略低于 Token KD 约 1–2 points
```

结论：

> 机制存在，但还不足以证明 Class KD 是更强通用 FC KD。

下一步只允许：

- 精读 STAR 后精确补齐 Token baseline；
- 提升自然 candidate support 质量；
- 第二 Seed；
- 少量必要 attribution。

不要立刻扩十几个 ablation。

---

# 50. RESEARCH NO-GO

## 数据层

确认 sampling coverage 后仍：

```text
Gold-Class Natural Equivalence Rate < 5%
AND
Any-Class Natural Equivalence Rate < 5%
```

或：

```text
median r_intra < 5%
AND
p90 r_intra < 10%
```

说明：

> 这个问题在当前 FC 协议下 opportunity 太弱。

---

## 方法层

pipeline 已可靠，但：

```text
Class KD consistently < Candidate KD
AND
no Inter-class KL gain
AND
no natural-equivalence subset gain
```

---

## 外部有效性

```text
Class KD only improves self-built mechanism diagnostic
BUT
BFCL degrades >2 points
vs Token/Candidate baseline
```

则：

```text
NO-GO
```

停止继续往 Class KD 上堆模块。

---

# 51. GO 后才允许恢复的实验

只有 v5 GO 后：

- STAR CKD；
- 第二 Seed；
- 第二 Student size；
- ACEBench；
- capacity effect；
- entropy buckets；
- Candidate K sensitivity；
- gradient conflict；
- stronger natural candidate mining；
- RL baseline / ToolRL comparison；
- 统计显著性与置信区间。

顺序：

```text
先证明方法成立
→ 再解释为什么
→ 最后扩泛化
```

---

# 52. 最终执行顺序

```text
P0  Environment / exact model revision / tokenizer / template
 ↓
P1  Dataset adapter + deterministic canonicalizer
 ↓
P2  Canonicalizer unit/audit tests
 ↓
P3  Teacher natural multi-sampling on 300 opportunity queries
 ↓
P4  Re-score unique candidates with Teacher
 ↓
P5  Data Opportunity Gate
     ├─ FAIL → optional 16-sample confirmation → NO-GO
     └─ PASS
 ↓
P6  0.6B Tiny Overfit + all Loss Toy Tests
 ↓
P7  Train common 1.7B S0
 ↓
P8  Student–Teacher Intra-class Gap Gate
     ├─ FAIL → NO-GO / WEAK-GO
     └─ PASS
 ↓
P9  Build + freeze Train Candidate Pool
 ↓
P10 Cache Teacher Candidate Scores
 ↓
P11 Lock exact STAR top-k FKL sequence construction after paper read
 ↓
P12 Cache Token FKL information
 ↓
P13 Four branches:
     SFT-cont / Token FKL / Candidate KD / Class KD
 ↓
P14 Internal + Held-out Natural-Equiv + BFCL
 ↓
P15 GO / WEAK-GO / NO-GO
```

---

# 53. 你现在可以立即开始的部分

在 STAR 尚未完成精读时，**现在就可以做**：

```text
P0
P1
P2
P3
P4
P5
```

也就是：

1. 数据 Adapter；
2. deterministic canonicalizer；
3. canonicalizer unit tests；
4. 300-query Teacher natural multi-sampling；
5. unique candidate teacher re-scoring；
6. Data Opportunity report。

这些步骤已经足以给出第一次真正的：

```text
这个题有没有数据机会？
```

的答案。

不要等四个模型训练完才发现 Class 根本不存在。

---

# 54. Data Opportunity Report 固定输出

建议生成：

```text
reports/opportunity_report.json
reports/opportunity_report.csv
reports/opportunity_examples.jsonl
```

字段至少：

```text
query_id
num_teacher_samples
num_valid_samples
num_unique_surface
num_unique_canonical
has_any_nontrivial_equiv
has_gold_nontrivial_equiv
gold_class_surface_multiplicity
max_class_surface_multiplicity
teacher_intra_entropy
teacher_dominance
equivalence_rule_types
example_surfaces
canonical_signatures
```

汇总：

```text
Any-Class Natural Equivalence Rate
Gold-Class Natural Equivalence Rate
Valid Generation Rate
Mean/Median Multiplicity
Entropy distribution
Dominance distribution
Equivalence-Type Breakdown
```

---

# 55. Definition of Done

v5 Pilot 最终必须产出：

```text
[ ] dataset/split manifest
[ ] exact model revisions
[ ] tokenizer/template hashes
[ ] deterministic canonicalizer
[ ] canonicalizer audit report
[ ] natural teacher sampling cache
[ ] teacher re-scored candidate cache
[ ] Data Opportunity report
[ ] Any-Class Natural Equivalence Rate
[ ] Gold-Class Natural Equivalence Rate
[ ] equivalence-type breakdown
[ ] Teacher intra-class entropy/dominance
[ ] Tiny Overfit report
[ ] Common Student S0
[ ] Student–Teacher intra-class gap report
[ ] frozen Train Candidate Pool manifest
[ ] Token FKL cache
[ ] Candidate score cache
[ ] Token FKL Toy Test
[ ] Candidate Score Test
[ ] Canonical Aggregation Test
[ ] KL Decomposition Test
[ ] Permutation Test
[ ] Gradient Test
[ ] SFT Continuation checkpoint
[ ] Token FKL checkpoint
[ ] Candidate KD checkpoint
[ ] Equivalence-Class KD checkpoint
[ ] Internal result table
[ ] Held-out Natural-Equivalence result table
[ ] BFCL result table
[ ] GO / WEAK-GO / NO-GO decision
[ ] explicit Pipeline-Fail vs Research-No-Go classification
```

---

# 56. 最终结果主表

脚本固定输出：

```text
results_main.csv
```

字段：

```text
method
internal_canonical_acc
natural_equiv_canonical_acc
heldout_candidate_kl
heldout_inter_kl
heldout_intra_kl
heldout_correct_class_mass
bfcl_acc
invalid_rate
decision
```

行：

```text
SFT Continuation
Token KD / FKL
Candidate KD
Equivalence-Class KD
```

**四种方法全部计算 held-out Candidate/Inter/Intra KL。**

Intra-class KL 对 Class KD **不是越低越好**。

---

# 57. v5 最终判定原则

### 情况 A：问题不存在

```text
自然等价率很低
或
intra-class contribution 几乎为 0
```

→ **NO-GO**

---

### 情况 B：问题存在，但删除类内偏好有害

```text
Candidate KD 中确实有明显 intra-class supervision
但 Class KD 稳定更差
```

→ **NO-GO**

---

### 情况 C：问题存在，Class KD 有机制信号，但不如 Token KD

```text
Class > Candidate on mechanism
but
Class < Token KD on overall performance
```

→ **WEAK-GO**

---

### 情况 D：问题存在，Class KD 在 canonical semantics 上优于 Candidate KD，并能与 Token KD 持平或更好，BFCL 不退化

→ **STRONG GO**

此时才值得升级成正式论文实验。

---

# 58. 代码 Agent 开工前必须先回复

在写正式训练代码前，代码 Agent 必须先给出：

1. 对 v5 唯一 RQ 的理解；
2. 数据源与 split manifest；
3. exact Qwen model IDs / revisions；
4. tool/chat template；
5. deterministic canonical signature 的精确定义；
6. default / null / array / string 的处理；
7. Canonicalizer Unit Test 清单；
8. Teacher natural multi-sampling 配置；
9. 为什么 sampling frequency 不等于 Teacher probability；
10. Teacher re-scoring 实现；
11. Opportunity Metrics；
12. Opportunity STOP 条件；
13. S0 构造；
14. Candidate Pool 构造；
15. Candidate/Class KL 数学实现；
16. STAR-style top-k FKL API；
17. 所有 Toy Tests；
18. 2080 Ti / 4090 分工；
19. 明确确认第一轮不加入 CKD、RL、多 Student、多 Seed、ACEBench、LLM Verifier。

在这些内容确认前：

```text
不要进入正式四分支训练。
```

---

# 59. 当前第一步

**现在立刻做：**

```text
Step 1: 选定一个主数据源并固定 300 条 opportunity_probe
Step 2: 实现 deterministic canonicalizer
Step 3: 跑 14 类 canonicalizer unit tests
Step 4: Teacher 每条 Query 自然采样 8 次
Step 5: surface dedup + validate + canonicalize
Step 6: Teacher 重新 sequence-score unique candidates
Step 7: 输出 Data Opportunity Report
Step 8: 根据 Gate 决定是否继续 S0/KD
```

这一步得到结果之前：

> **不要租大规模 GPU 跑正式四分支，也不要继续扩展 Class KD 方法。**

---

# 60. Source / Baseline Notes

- STAR: *Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models*, ICLR 2026.
  - FKL baseline: top-k Forward KL.
  - Paper uses `k=100`.
  - KD training reference: learning rate `3e-6`, batch size `128`.
  - CKD and Sim-RL are **not** part of v5 Pilot.
- Qwen official model family:
  - `Qwen/Qwen3-0.6B`
  - `Qwen/Qwen3-1.7B`
  - `Qwen/Qwen3-8B`

---

**END — FC Equivalence-Class KD Go/No-Go Pilot PRD v5 FINAL**
