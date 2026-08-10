# Function Calling 等价类蒸馏 Pilot 实验 PRD

## v3 返修重点

本版相对 v2 的关键变化：

1. 用 **RQ1–RQ4** 重构整个 Pilot 的科学问题与证据链；
2. 明确区分：RQ1 等价调用鲁棒性、RQ2 容量与类内偏好、RQ3 教师软类别知识、RQ4 外部有效性；
3. 将原 `Valid-Set Contrastive` 正式解释并重命名为 **Class-Hard / Valid-Set Supervision**；
4. 新增 **Class Label Smoothing** 强 Baseline；
5. 新增 Candidate KL 的 **inter-class / intra-class 精确分解**及数值单测；
6. 新增 **Intra-class Entropy Buckets**、**Capacity × Entropy** 机制分析；
7. 候选池默认从 K=8 统一调整为 **K=6**，敏感性测试为 K=4/6/8；
8. 明确 0.6B 与 1.7B 两个学生规模承担不同实验职责；
9. 新增 BFCL / ACEBench Benchmark Adapter、Benchmark Leakage Check 与外部评测 Milestone；
10. Pilot 成功标准不再只看总分，而是逐 RQ 判断证据是否成立。

---

> 文档用途：交给代码 Agent / Vibecoding Agent，作为从零搭建实验环境、实现 Pilot、运行对比实验和输出可验收结果的唯一需求基线。  
> 项目目标：用最小工程与算力成本，验证“验证器定义的 Function Call 类级蒸馏”是否相较 Vanilla SFT、Token-level KD 与 Candidate-level KD 提供稳定增益。  
> 当前阶段：**Pilot，不追求完整顶会实验规模，不实现动态训练闭环、真实工具执行或多轮 Agent。**
> 版本：**v3.0——在 v2 资源方案基础上，重新按 RQ1–RQ4 组织科学问题、机制实验、外部 Benchmark 与归因 Baseline；明确 Class-Hard、Label Smoothing、Intra-class Entropy 和等价扰动评测。**

---

## 0. 强制执行原则

1. **严格控制变量**  
   除 Vanilla SFT 外，所有候选级方法尽量使用相同的数据、学生初始化、候选池、验证器、训练步数、优化器、推理设置和评测脚本。方法之间只允许改变“如何利用候选及教师分布”。

2. **先诊断，后训练**  
   必须先完成无训练诊断并通过 Go/No-Go 门槛，才允许启动完整 Pilot 训练。

3. **验证器高精度优先**  
   宁可将无法可靠判断的候选标为 `unknown` 并排除，也不能把错误候选归入正确/等价类。

4. **教师只在训练阶段使用**  
   测试时学生必须独立直接生成 Function Call，不允许教师重排、验证器修复或候选搜索参与最终预测。

5. **教师分数必须离线缓存**  
   禁止在每个训练 Step 重复运行教师模型。

6. **所有关键步骤必须可复现**  
   数据划分、候选生成、随机种子、模型版本、Prompt 模板、验证器版本和超参数必须写入配置与日志。

7. **不得擅自扩大范围**  
   Pilot 不实现 RL、DPO、GRPO、多轮轨迹、真实 API 环境、LLM Judge 等价判断、每 Epoch 动态候选刷新。

---

# 1. 项目背景与研究问题

## 1.1 基础任务

给定：

- 用户请求 \(x\)
- 候选工具及其 Schema 集合 \(\mathcal T\)
- 标准 Function Call \(y^*\)

学生模型生成一个单轮、闭域、单函数调用：

\[
(x,\mathcal T)\rightarrow y
\]

目标输出示例：

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "Chengdu",
    "date": "2026-07-31"
  }
}
```

## 1.2 研究假设

本 Pilot 不再把问题表述为“设计一个新的 KD Loss 是否涨分”，而是聚焦于一个核心矛盾：

> **现有小模型 Function Calling 的监督粒度，与 Function Calling 真正的结构正确性粒度存在错位。**

标准 SFT / Token KD 主要在 Token 或唯一 Gold 序列上监督；Candidate KD 虽然提升到完整调用级，但仍要求学生拟合教师对每个具体候选调用的类内偏好。实际 Function Calling 评测更关心：

- 是否选择了正确函数；
- 是否生成了正确参数；
- 调用结构是否有效；
- 多个严格等价调用是否都应视为正确。

因此提出三个机制假设和一个外部有效性问题：

- **H1 / Equivalence Hypothesis**：将严格等价调用聚合到同一 `correct` 类，可降低模型对表面序列形式的过拟合。
- **H2 / Capacity-Limited Hypothesis**：对容量有限的小模型，完整拟合教师的类内候选偏好可能形成不必要的优化负担；学生越小、教师类内分布越不确定，Class KD 相对 Candidate KD 的收益应越明显。
- **H3 / Soft-Structure Hypothesis**：教师类别软分布包含 Hard Label 和普通 Label Smoothing 不具备的错误结构信息，例如 Wrong Function 与 Wrong Argument 的相对混淆程度。
- **H4 / External Validity**：上述收益不能只存在于自建候选或诊断集上，必须在独立标准 Function Calling Benchmark 上保持稳定。

## 1.3 Pilot Research Questions（必须逐项回答）

| RQ | 科学问题 | 对应方法 | 最关键 Baseline | 主要验证位置 | 支持假设的预期现象 |
|---|---|---|---|---|---|
| **RQ1** | 严格等价的 Function Call 是否被现有监督不必要地区别对待？ | Verifier-defined equivalence aggregation + Class KD | SFT / Token KD / Candidate KD | Equivalence Diagnostic + 标准 AST 评测 | 等价/格式扰动后的 Robustness Drop 更小 |
| **RQ2** | 完整拟合类内候选偏好，是否会给容量有限学生带来不必要的优化负担？ | Class KD | Candidate KD | 0.6B vs 1.7B + Intra-class Entropy Buckets | 学生越小、高类内熵样本上 Class KD 相对收益越明显 |
| **RQ3** | 教师类别软分布是否比 Verifier Hard Label / 普通平滑包含更多结构知识？ | Teacher Soft Class Distribution | Class-Hard / Class Label Smoothing / Hard Negative | Similar-tool / Wrong-Argument subsets | Class KD 在困难混淆样本上更强 |
| **RQ4** | 方法是否只对自建候选和诊断数据有效？ | 外部标准评测 | 所有主要 Baseline | BFCL + ACEBench 的单轮兼容子集 | 标准 Benchmark 上仍保持稳定收益或至少不下降 |

### 1.3.1 RQ 之间的逻辑关系

四个 RQ 不是四个平行故事，而是一条证据链：

```text
RQ1：为什么要把等价调用合并？
  ↓
RQ2：为什么不能继续完整拟合同一类别内部的偏好？
  ↓
RQ3：那为什么不直接使用 Verifier Hard Label？
  ↓
RQ4：前述机制在独立 Benchmark 上是否成立？
```

最终希望支持的结论不是“新 Loss 涨分”，而是：

> **Verifier-defined structural class 是容量受限 Function Calling 学生更合适的蒸馏粒度之一。**

### 1.3.2 主要方法的粒度关系

```text
Candidate KD
  = 类间结构知识 + 类内候选偏好
          ↓ 去除 / 边缘化类内偏好
Class KD
  = 教师软类别结构
          ↓ 去除教师软知识
Class-Hard
  = Verifier 硬类别监督
```

因此：

- Candidate KD 是“过细”的主要对照；
- Class-Hard 是“过粗”的主要对照；
- Class KD 位于二者之间，保留教师结构软知识但不强制复制全部类内偏好。

# 2. Pilot 范围

## 2.1 必须包含

- 单轮 Function Calling
- 闭域：测试时提供候选工具 Schema
- 单个函数调用
- **两个容量档学生用于机制验证：Qwen3-0.6B 与 Qwen3-1.7B**
  - 0.6B：本地全流程 + RQ2 容量效应
  - 1.7B：主学生与正式结果
- 一个同家族教师模型：Qwen3-8B
- QLoRA 训练
- 静态候选池
- 确定性 AST / Schema 验证器
- 无训练诊断
- 4 个核心方法：Vanilla SFT / Token-level KD / Candidate-level KD / Class-level KD
- 4 个归因 Baseline：Random Negative / Hard Negative / **Class-Hard / Valid-Set** / **Class Label Smoothing**
- 标准测试集与机制诊断测试集
- **Intra-class Entropy 分桶分析**
- **Similar-tool / Wrong-Argument 子集分析**
- **BFCL + ACEBench 单轮兼容子集外部评测**
- 至少两个随机种子的确认实验

## 2.2 暂不包含

- 多轮 Function Calling
- 并行或串行多工具调用
- Tool Retrieval
- 真实 API 执行与外部环境状态
- RL / DPO / GRPO
- 每轮动态生成新候选
- LLM Judge 判断广义语义等价
- 不同 Tokenizer 的 Token KD
- 7B 学生全参训练
- 跨任务扩展到 Text-to-SQL
- 大规模超参数搜索

---

# 3. 模型选型、默认方案与验收

所有模型、精度和训练参数必须通过 YAML 配置，不得写死在业务代码中。

## 3.1 模型选型原则

模型选择不得只看通用能力排名，必须同时满足以下研究与工程条件：

1. **学生足够小且仍具备基础结构生成能力**  
   学生太强会出现天花板效应；学生太弱则可能连合法 Function Call 都无法稳定输出。

2. **教师显著强于学生**  
   教师必须在正确调用、正确类别和困难候选排序上明显优于学生，否则没有足够的蒸馏空间。

3. **主教师与主学生尽量使用相同 Tokenizer**  
   Candidate KD 与 Class KD 可以跨 Tokenizer，但 Token-level KD 需要逐 Token 对齐。主实验必须优先采用同家族模型。

4. **支持 Tool Calling 或容易通过稳定 Chat Template 表达工具 Schema**。

5. **能够在现有资源下复现**  
   当前可用资源为：
   - 实验室 RTX 2080 Ti 11GB；
   - 必要时自费租用 RTX 4090 24GB；
   - 不依赖商业闭源 API 作为核心教师。

6. **不得使用已过度专门化、导致学生初始 FC 性能接近饱和的模型作为唯一学生**。

---

## 3.2 候选模型评估表

以下评分仅代表其对本 Pilot 的适配度，不代表模型的综合能力排名。

| 模型 | 推荐角色 | 参数规模 | Pilot 适配度 | 主要优势 | 主要风险 |
|---|---|---:|---:|---|---|
| Qwen3-0.6B | 辅助学生 / 本地全流程学生 | 0.6B | 9.0/10 | 11GB 显存可完整开发；容量受限明显；容易观察 KD 收益 | 可能过弱，需要检查基础 FC 能力 |
| **Qwen3-1.7B** | **主学生** | 1.7B | **9.6/10** | 能力、提升空间、单卡成本平衡最好 | Candidate/Class KD 主实验更适合 24GB 卡 |
| Qwen3-4B | 后续扩展学生 | 4B | 7.8/10 | 适合验证容量规律 | 24GB 上 K 候选训练较紧 |
| **Qwen3-8B** | **主教师** | 8B | **9.7/10** | 与学生同系列；能力差距合适；可在 24GB 上高精度推理 | 11GB 上只能量化预诊断 |
| Qwen3-14B | 备用强教师 | 14B | 8.0/10 | 同系列、更强 | 24GB 上高精度不现实，成本明显增加 |
| Qwen3.5-0.8B | 备选学生 | 0.8B | 6.8/10 | 新、轻量、支持工具使用 | 推理栈与依赖更新快，首个 Pilot 工程风险高 |
| Qwen2.5-Coder-1.5B-Instruct | 备选学生 | 1.5B | 8.2/10 | 结构化输出较强、系列成熟 | 与当前 Qwen3/STAR 路线可比性略弱 |
| Qwen2.5-Coder-7B-Instruct | 备选教师 | 7B | 8.1/10 | 与 BalanceSFT 更直接可比 | 更偏代码模型 |
| Llama-3.2-1B-Instruct | 第二家族学生 | 1B | 8.0/10 | 小、支持 Tool Calling | 模型访问与模板适配更复杂 |
| Llama-3.2-3B-Instruct | 第二家族学生 | 3B | 7.7/10 | 适合后续跨家族验证 | 11GB 不适合完整候选级训练 |
| Llama-3.1-8B-Instruct | Llama 家族教师 | 8B | 7.8/10 | 与 Llama 小模型配对自然 | 不应在首个 Pilot 同时维护第二套模板 |
| Granite-3.3-2B-Instruct | 第二家族学生 | 2B | 8.0/10 | 许可友好、支持 Function Calling | 社区 FC 对比生态弱于 Qwen |
| Granite-3.3-8B-Instruct | 第二家族教师 | 8B | 8.2/10 | 与 2B 配对清晰 | 不是首轮最低风险方案 |
| FunctionGemma-270M-IT | 极小学生 | 0.27B | 6.5/10 | 资源成本极低、专门面向 FC | 已专门后训练，且规模过小 |
| Gemma-3-1B-IT | 备选学生 | 1B | 6.4/10 | 轻量 | FC 不是主要定位 |
| Phi-4-mini-instruct | 备选学生 | 3.8B | 6.7/10 | 结构与工具能力较强 | 学生偏大，教师配对成本高 |
| SmolLM2-1.7B-Instruct | 跨家族扩展学生 | 1.7B | 7.3/10 | 开放、提升空间可能较大 | 缺少自然的同 Tokenizer 大教师 |

### 选型验收标准

代码 Agent 必须输出：

- 模型 ID；
- Revision / Commit Hash；
- 参数规模；
- Tokenizer Vocab Size；
- Special Token IDs；
- Chat Template Hash；
- 是否支持原生工具模板；
- 许可与访问状态；
- 下载体积；
- 在目标 GPU 上的实测峰值显存。

---

## 3.3 最终推荐模型方案

### 主学生

```yaml
student:
  model_name_or_path: "Qwen/Qwen3-1.7B"
  role: "main_student"
  enable_thinking: false
  load_in_4bit: true
  quant_type: "nf4"
  double_quant: true
  compute_dtype: "auto_by_gpu"
  use_lora: true
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  gradient_checkpointing: true
  use_cache: false
```

用途：

- 正式 Pilot 的主要学生；
- 四个核心方法；
- 归因 Baseline；
- 两个随机种子确认。

### 辅助学生

```yaml
student_low_capacity:
  model_name_or_path: "Qwen/Qwen3-0.6B"
  role: "local_dev_and_capacity_student"
  enable_thinking: false
  load_in_4bit: true
```

用途：

- 2080 Ti 上完成完整流水线；
- Smoke Test；
- Tiny Overfit；
- 所有 Loss 的低成本调试；
- 验证“小模型越小，Class KD 收益是否更明显”。

### 主教师

```yaml
teacher:
  model_name_or_path: "Qwen/Qwen3-8B"
  role: "main_teacher"
  enable_thinking: false
  frozen: true
  formal_scoring_dtype: "bfloat16"
  local_prediagnostic_quantization: "4bit"
```

用途：

- 统一候选序列打分；
- 正式 Class/Candidate KD 教师分布；
- Token KD Top-k Logits 缓存；
- 教师—学生能力差距诊断。

### 备用教师

```yaml
teacher_backup:
  model_name_or_path: "Qwen/Qwen3-14B"
  enabled: false
```

仅当以下任一条件满足时启用：

- Qwen3-8B `TeacherClassTop1 < 80%`；
- 教师 AST Accuracy 未达到 80%；
- 教师与 1.7B 学生差距小于 5 个百分点；
- 教师对 Hard Negative 的排序质量明显不足。

不得在未完成 8B 教师预诊断前直接租更贵显卡运行 14B。

---

## 3.4 Tokenizer 一致性

启动时必须验证：

```python
assert student_vocab_size == teacher_vocab_size
assert student_special_token_ids == teacher_special_token_ids
assert student_tokenizer_hash == teacher_tokenizer_hash
```

还必须核对：

- BOS/EOS/PAD Token；
- Chat Template；
- Tool Call 特殊标记；
- Thinking/Non-thinking 模式输出格式。

若不一致：

- Candidate KD / Class KD 可以继续；
- Token KD 必须拒绝启动；
- 日志输出 `TOKEN_KD_BLOCKED_TOKENIZER_MISMATCH=true`。

不得仅凭模型属于同一系列而假设 Tokenizer 一定一致。

---

## 3.5 单轮单调用的难度要求

Pilot 保持：

> 单轮、最终只输出一个 Function Call，但每条 Query 提供多个候选工具。

不得采用“只提供唯一函数”的简单设置。

每条样本建议：

```yaml
task_difficulty:
  tools_per_query: 6
  correct_tools: 1
  semantically_similar_distractors: 2
  unrelated_distractors: 3
```

参数难度至少覆盖部分：

- 两个以上必填参数；
- Optional 参数；
- 默认参数；
- Enum；
- 日期或数字规范化；
- 相似参数名；
- 多实体干扰；
- 缺失信息；
- Seen API 与 Unseen API。

### 难度校准 Gate

在正式蒸馏前，必须评测 Initial Student、Vanilla Student 和 Teacher：

| 指标 | 推荐范围 |
|---|---:|
| Vanilla Student AST Accuracy | 40%–80% |
| Teacher AST Accuracy | ≥80% |
| Teacher − Vanilla Student | ≥8 个百分点，最低不得低于 5 |
| Teacher 正确、Student 错误样本比例 | ≥15% |
| Student 错误中语义/结构错误比例 | ≥50% |
| Student 错误中纯格式错误比例 | ≤30% |

若 Vanilla Student AST Accuracy > 85%，按以下顺序提高难度：

1. 增加相似工具；
2. 增加 Unseen API；
3. 增加 Schema Rename；
4. 增加参数复杂度；
5. 减少训练数据；
6. 将 0.6B 设为主学生。

若 Vanilla Student AST Accuracy < 20%，优先检查：

- Chat Template；
- Tool Schema 序列化；
- Label Mask；
- 数据质量；
- 截断；
- Parser 和指标。

不得使用 KD 掩盖基础训练错误。

---

## 3.6 LoRA 与 QLoRA 决策

### LoRA

- 基础模型使用 FP16/BF16；
- 基础权重冻结；
- 仅训练低秩 Adapter；
- 速度与数值行为更直接；
- 显存高于 QLoRA。

### QLoRA

- 基础模型以 4-bit NF4 加载；
- 基础权重冻结；
- LoRA Adapter 用 FP16/BF16 计算与更新；
- 大幅减少基础权重显存；
- 适合 11GB/24GB 单卡和 K 候选训练。

### Pilot 强制选择

```yaml
finetuning:
  method: "qlora"
  base_quantization: "nf4_4bit"
  trainable_parameters: "lora_only"
```

原因：

- Candidate/Class KD 的激活值远高于普通 SFT；
- 需要为 K 个候选、较长 Tool Schema 和训练 Workspace 留显存；
- 研究目标不是比较 LoRA 与 QLoRA。

后续完整论文可补 LoRA 一致性实验，但不属于 Pilot 必须项。

---

## 3.7 Token KD 的存储方式

禁止缓存完整词表 Logits。

采用：

> **Top-k Token-level Forward KL**

默认：

```yaml
token_kd:
  top_k: 64
  temperature: 2.0
  cache_teacher_logits: true
  cache_full_vocab_logits: false
```

每个 Gold 输出位置保存：

- Top-k Token ID；
- Top-k Log Probability；
- 剩余概率质量或归一化所需统计；
- Attention/Label Mask；
- 教师模型、Tokenizer 和 Prompt Hash。

验收：

- 教师不在学生训练时常驻显存；
- 缓存可重复加载；
- 缓存 Hash 与数据、教师、模板绑定；
- Padding、Prompt 与 Tool Schema Token 不参与 KD；
- 报告缓存大小与生成耗时。

---

# 4. 硬件、显存、精度、磁盘与双机协作方案

## 4.1 已知硬件条件

### 实验室资源

- GPU：RTX 2080 Ti
- 显存：11GB
- 架构：Turing
- 主要用途：免费开发、0.6B 全流程、1.7B Smoke Test、8B 量化预诊断

### 租用资源

- 首选：RTX 4090 24GB
- 主要用途：8B 正式教师缓存、1.7B 正式候选级训练、第二随机种子
- 不优先租用：A100 80GB、H100 等高价卡

---

## 4.2 双阶段 GPU 任务分配

| 任务 | 实验室 2080 Ti | 租用 4090 |
|---|---:|---:|
| 数据处理、验证器、指标 | 是 | 不需要 |
| 0.6B Vanilla SFT | 是 | 可选 |
| 0.6B Candidate/Class KD | 是 | 可选提速 |
| 1.7B Vanilla Smoke/Tiny Overfit | 是 | 是 |
| 1.7B 正式 Candidate/Class KD | 不推荐 | **必须优先使用** |
| 8B INT4 小样本预诊断 | 是 | 可选 |
| 8B BF16 正式候选打分 | 否 | **是** |
| 8B Top-k Token KD Cache | 小规模可调试 | **正式生成** |
| 两个 Seed 正式主结果 | 不推荐 | **是** |
| 4B 学生扩展 | 否 | 24GB 较紧，后续考虑 32/48GB |

### 强制原则

- 教师与学生不得为了方便而在同一训练阶段同时常驻显存；
- 教师正式分数必须离线缓存；
- 先在 2080 Ti 上排除代码和数据错误，再租卡；
- 租卡阶段不得重新从零开始调试基础 Parser、Loss 和数据结构。

---

## 4.3 2080 Ti 默认配置

2080 Ti 不支持高效原生 BF16，默认使用 FP16：

```yaml
hardware_profile: "rtx2080ti_11gb"

precision:
  fp16: true
  bf16: false
  grad_scaler: true

attention:
  implementation: "sdpa"
  flash_attention_2: false
  fallback: "eager"

training:
  max_length: 1024
  candidates_per_sample: 4
  candidate_chunk_size: 1
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  use_cache: false
```

执行顺序：

1. Qwen3-0.6B，K=4 跑通；
2. 尝试 K=6；
3. Qwen3-1.7B 仅做 Vanilla/Tiny Overfit/小样本 Loss 调试；
4. Qwen3-8B 仅以 4-bit、Batch 1 做 200–500 条预诊断。

---

## 4.4 RTX 4090 默认配置

```yaml
hardware_profile: "rtx4090_24gb"

precision:
  bf16: true
  fp16: false

attention:
  implementation: "flash_attention_2_or_sdpa"

training:
  max_length: 1536
  candidates_per_sample: 6
  candidate_chunk_size: 2
  original_sample_micro_batch: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  use_cache: false
```

K=8 仅在 K=6 主实验稳定后尝试。

教师正式打分优先级：

```text
BF16 > INT8 > INT4
```

若 BF16 OOM：

1. Batch 降至 1；
2. 缩短 Prompt；
3. 使用 INT8；
4. 最后才使用 INT4。

---

## 4.5 显存适配估算

以下为工程估算，最终必须以实测峰值为准。

| 工作负载 | 预计显存 | 推荐设备 |
|---|---:|---|
| Qwen3-0.6B Vanilla QLoRA | 4–7GB | 2080 Ti |
| Qwen3-0.6B Class KD，K=4–6 | 8–11GB | 2080 Ti |
| Qwen3-1.7B Vanilla QLoRA | 7–11GB | 2080 Ti/4090 |
| Qwen3-1.7B Class KD，K=6 | 14–19GB | 4090 |
| Qwen3-1.7B Class KD，K=8 | 17–23GB | 4090，接近边界 |
| Qwen3-4B Vanilla QLoRA | 14–20GB | 4090 |
| Qwen3-4B Class KD | 28GB 以上 | 32/48GB 卡 |
| Qwen3-8B 教师 BF16 推理 | 19–23GB | 4090 |
| Qwen3-8B 教师 INT8 | 11–15GB | 2080 Ti/4090 |
| Qwen3-8B 教师 INT4 | 7–10GB | 2080 Ti |
| Qwen3-14B 教师 BF16 | 32GB 以上 | 40/48GB 卡 |

`doctor.py` 和训练日志必须记录：

- Allocated VRAM；
- Reserved VRAM；
- Peak Allocated VRAM；
- 每 Step 时间；
- 是否发生 OOM；
- OOM 前配置。

---

## 4.6 OOM 自动降级策略

出现 `CUDA out of memory` 时，不得随机修改大量参数。严格按顺序降级：

1. `micro_batch_size = 1`；
2. K 从 8 降到 6，再降到 4；
3. `max_length` 从 1536 降到 1280/1024；
4. `candidate_chunk_size` 降到 1；
5. 开启或确认 `gradient_checkpointing=true`；
6. 缩短冗余 Tool Description；
7. LoRA Rank 从 16 降到 8；
8. 教师 BF16 改 INT8；
9. 更换更大显存 GPU；
10. 最后才缩小主学生。

每次降级必须写入 `DECISIONS.md`，并记录对实验可比性的影响。

---

## 4.7 Gradient Checkpointing 要求

`gradient_checkpointing` 在本文中指 **Activation Checkpointing**，不是保存模型文件。

作用：

- 前向时不长期保存部分层内激活值；
- 反向时重新计算这些前向结果；
- 以额外计算时间换取显存。

Pilot 训练默认开启。

验收：

- 开启前后同一 Toy Batch 的 Loss 差异在合理浮点误差内；
- 开启后峰值显存明显下降；
- 训练日志记录速度下降比例；
- 禁止把它和磁盘 Checkpoint 混为一谈。

---

## 4.8 FP16 / BF16 使用规范

### 2080 Ti

- 使用 FP16；
- 启用 GradScaler；
- 禁用 BF16；
- 禁止把不支持的 BF16 配置静默映射成其他精度。

### 4090

- 训练与正式教师打分优先 BF16；
- BF16 的优势是动态范围接近 FP32，训练通常比 FP16 稳定；
- FP16 尾数更细，但动态范围小，需要 Loss Scaling。

### 混合精度要求

不假设所有张量均为同一精度：

- 矩阵乘法与激活：FP16/BF16；
- 部分归一化、Softmax、Loss：允许 FP32；
- LoRA 优化器状态：通常 FP32 或 8-bit；
- 4-bit Base 权重：仅存储量化权重，计算时反量化；
- 教师正式候选概率不得未经审计直接使用低精度量化结果。

---

## 4.9 教师量化审计

2080 Ti 上的 Qwen3-8B INT4 仅用于预诊断，不直接视为正式教师。

正式实验若使用 INT8/INT4，必须在 100–500 条样本上与 BF16 对照：

- 候选分数 Spearman 相关系数；
- Candidate Top-1 Agreement；
- Class Top-1 Agreement；
- Teacher Correct-Class Top-1；
- 类别分布平均 KL。

推荐门槛：

```text
Spearman ≥ 0.95
Class Top-1 Agreement ≥ 95%
TeacherClassTop1 下降 ≤ 1 个百分点
```

不达标时必须使用 BF16 正式缓存。

---

## 4.10 `max_length` 与长度审计

`max_length` 指以下序列化内容的总 Token 数：

\[
	ext{System}+	ext{Tools}+	ext{User}+	ext{Assistant Candidate}
\]

对 Candidate/Class KD，每个候选都形成一条完整序列。

数据准备阶段必须统计：

- P50；
- P90；
- P95；
- P99；
- 最大长度；
- 截断样本比例；
- Tool Schema 被截断比例；
- Gold/候选输出被截断比例。

强制验收：

- Gold 输出被截断比例 = 0%；
- 正确工具 Schema 被截断比例 = 0%；
- 不得用固定 2048 掩盖大量冗余工具描述；
- 默认 2080 Ti 使用 1024，4090 使用 1536，再根据实际 P99 调整。

推理使用独立的：

```yaml
generation:
  max_new_tokens: 256
```

---

## 4.11 预诊断定义与资源用途

预诊断位于 Smoke Test 与正式 Pilot 之间：

```text
Smoke Test
→ 200–500 条预诊断
→ Go/No-Go
→ 正式 Pilot
```

预诊断不追求论文结果，只验证：

- 教师是否明显强于学生；
- 教师类别分布是否可信；
- 学生是否存在结构/语义错误；
- 候选池是否包含 Hard Negative；
- 验证器是否达到高精度；
- Class KD 是否存在基本可学习信号。

2080 Ti 上允许使用：

- Qwen3-0.6B/1.7B 学生；
- Qwen3-8B INT4 教师；
- 200–500 条诊断样本；
- Batch 1；
- 不做完整正式训练。

预诊断通过后，才租 4090 生成 BF16 正式教师缓存。

---

## 4.12 磁盘需求

100GB 不是要求用户自行拥有一块 100GB 实体硬盘，而是租卡实例的可写数据盘容量。

### 精简 Pilot

```text
最低可工作空间：100GB
推荐可工作空间：150–200GB
```

估算：

| 内容 | 空间 |
|---|---:|
| Python/PyTorch 环境 | 10–20GB |
| 0.6B/1.7B 模型 | 5–8GB |
| 8B BF16 模型缓存 | 16–20GB |
| 数据与候选池 | 2–10GB |
| 教师候选分数 | <数 GB |
| Top-k Token KD Cache | 1–5GB |
| LoRA Adapter 与日志 | 5–20GB |
| 临时/重复缓存缓冲 | 20–40GB |

完整论文扩展到第二家族、14B 教师、多 Seed 时，可能需要 200–400GB，但不属于首轮 Pilot 的前置要求。

---

## 4.13 双机数据协作方案

不得频繁同步完整模型权重。

### 两台机器分别直接下载

以下内容在实验室和租卡机器分别从官方模型源下载：

- Qwen3 模型；
- 原始公开数据集；
- Python 依赖。

### 需要版本化同步的实验资产

```text
artifacts/pilot_v1/
├── split_ids/
├── processed_data/
├── candidates/
├── verifier_results/
├── teacher_scores/
├── token_kd_cache/
├── best_adapters/
├── predictions/
├── manifests/
└── checksums.sha256
```

其中：

- 实验室 → 云端：Split、处理数据、候选池、验证器输出；
- 云端 → 实验室/持久存储：正式教师缓存、最佳 Adapter、最终结果；
- 所有文件必须有 SHA256；
- 同步必须使用 `rsync`、`rclone`、对象存储或平台文件存储，不使用手工零散复制。

### 不长期保留

- 每个 Step 的 Checkpoint；
- 失败 Run 的模型副本；
- 重复 Hugging Face Cache；
- 无分析价值的临时预测。

配置：

```yaml
checkpointing:
  save_strategy: "best_and_last"
  save_total_limit: 2
  save_full_base_model: false
  save_lora_adapter_only: true
```

---

## 4.14 是否购买 SSD 的决策

Agent 不得默认要求购买 SSD。

先执行：

```bash
nvidia-smi
df -h
lsblk
free -h
du -sh ~/.cache/huggingface 2>/dev/null
```

并确认：

- 实验室是否有 `/data`、NAS 或共享存储；
- 用户是否能物理接触服务器并挂载移动硬盘；
- 账户磁盘配额；
- 云平台文件存储/对象存储可用性。

仅当同时满足以下条件时建议购买 1TB SSD：

- 实验室无 NAS/共享大盘；
- 用户能够物理连接或挂载 SSD；
- 需要长期保留模型缓存和实验资产；
- 网络同步成本过高。

若只能远程 SSH 且无挂载权限，购买个人 SSD 不能直接解决服务器存储问题，应使用平台文件存储或对象存储。

---

## 4.15 资源预算与停止规则

利用实验室 2080 Ti 后，推荐将租卡工作控制在：

\[
50	ext{–}120\ 	ext{RTX 4090 GPU 小时}
\]

建议 Pilot 自费预算：

```text
GPU + 临时数据盘 + 失败重跑：约 400–900 元的预算上限
商业 API Token：核心实验 0 元
```

以上是预算控制目标，不是平台固定报价。

强制停止条件：

- 预诊断 No-Go；
- 验证器未达到 Precision 门槛；
- 教师与学生无明显差距；
- 0.6B/小样本 Class KD 无任何正向趋势；
- 正式租卡前核心 Loss 与候选聚合测试未全部通过。

---

## 4.16 环境交付物与验收

### 推荐技术栈

- Python 3.11
- PyTorch
- Transformers
- Datasets
- PEFT
- Accelerate
- bitsandbytes
- jsonschema
- Pydantic
- Hydra / OmegaConf
- pytest
- rich / tqdm
- pandas
- matplotlib
- TensorBoard 或 Weights & Biases

### 必须提供

- `pyproject.toml` 或锁定依赖文件；
- `README.md`；
- `.env.example`；
- `Makefile`；
- `scripts/doctor.py`；
- `scripts/smoke_test.sh`；
- `configs/hardware/rtx2080ti.yaml`；
- `configs/hardware/rtx4090.yaml`；
- `scripts/package_artifacts.sh`；
- `scripts/sync_artifacts.sh`。

### `doctor.py` 验收

必须输出：

- Python/CUDA/PyTorch 版本；
- GPU 型号和总显存；
- BF16/FP16 支持；
- bitsandbytes 可用性；
- Flash Attention/SDPA 可用性；
- 学生/教师 Tokenizer 一致性；
- 系统内存；
- 磁盘可用空间；
- HF Cache 路径；
- 最终 `ENVIRONMENT_OK=true`。

### Smoke Test 验收

10 分钟内完成：

- 读取 4 条样本；
- 构造 K=4 候选；
- 验证器分类；
- Qwen3-0.6B 一次前向/反向；
- Class KD Toy Loss；
- 保存并读取 Adapter；
- 输出显存峰值；
- 退出码 0。


# 5. 项目目录结构

必须至少采用如下结构：

```text
fc_equiv_kd/
├── README.md
├── pyproject.toml
├── Makefile
├── configs/
│   ├── data/
│   ├── models/
│   ├── methods/
│   └── pilot.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── candidates/
│   └── cache/
├── src/fc_equiv_kd/
│   ├── data/
│   │   ├── adapters.py
│   │   ├── preprocess.py
│   │   └── splits.py
│   ├── prompting/
│   │   ├── templates.py
│   │   └── serializers.py
│   ├── verifier/
│   │   ├── parser.py
│   │   ├── schema_checker.py
│   │   ├── normalizer.py
│   │   ├── classifier.py
│   │   └── signatures.py
│   ├── candidates/
│   │   ├── equivalent.py
│   │   ├── counterfactual.py
│   │   ├── model_generate.py
│   │   └── builder.py
│   ├── scoring/
│   │   ├── sequence_score.py
│   │   └── teacher_cache.py
│   ├── losses/
│   │   ├── vanilla_sft.py
│   │   ├── token_kd.py
│   │   ├── candidate_kd.py
│   │   ├── class_kd.py
│   │   ├── random_negative.py
│   │   ├── hard_negative.py
│   │   ├── class_hard.py
│   │   ├── class_label_smoothing.py
│   │   └── hierarchical_kd.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── collators.py
│   │   └── registry.py
│   ├── evaluation/
│   │   ├── generation.py
│   │   ├── metrics.py
│   │   ├── perturbations.py
│   │   ├── entropy_analysis.py
│   │   ├── error_subsets.py
│   │   ├── bfcl_adapter.py
│   │   ├── acebench_adapter.py
│   │   └── reports.py
│   └── cli/
├── tests/
│   ├── unit/
│   ├── property/
│   └── integration/
├── scripts/
│   ├── doctor.py
│   ├── smoke_test.sh
│   ├── run_diagnostic.sh
│   ├── run_rq1.sh
│   ├── run_rq2.sh
│   ├── run_rq3.sh
│   ├── run_rq4.sh
│   └── run_pilot.sh
├── outputs/
├── reports/
└── DECISIONS.md
```

---

# 6. 数据接口与数据处理

## 6.1 统一样本格式

所有原始数据必须转换为统一 JSONL：

```json
{
  "sample_id": "train_000001",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "查询成都明天的天气"}
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "查询天气",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "date": {"type": "string"},
          "unit": {"type": "string", "default": "celsius"}
        },
        "required": ["city", "date"]
      }
    }
  ],
  "gold_calls": [
    {
      "name": "get_weather",
      "arguments": {
        "city": "Chengdu",
        "date": "2026-07-31"
      }
    }
  ]
}
```

## 6.2 数据源与处理要求

### Pilot 数据源

优先从公开 Function Calling 数据中筛选单轮、单调用样本。代码必须通过 Adapter 支持至少：

- `xLAM Function Calling 60K`
- `ToolACE`
- 本地统一 JSONL 数据

首轮不要求同时使用全部数据源。默认策略：

1. 选择一个公开数据源构建主 Pilot；
2. 固定 Train / Validation / Internal Test；
3. BFCL / ACEBench 只做外部 Benchmark，不得反向用于训练、候选构造或超参数选择。

### Pilot 规模目标

默认：

- Train：约 3,000
- Validation：约 500
- Internal Test：约 1,000
- Mechanism Diagnostic：约 500

具体数量可根据合法样本筛选后调整，但所有变化必须写入 Manifest。

### 数据处理要求

- 原始文件只读，不允许覆盖。
- 每次处理生成 Manifest：数据源、数据版本、原始样本数、单轮/单调用筛选规则、过滤原因、最终样本数、Seed、Hash。
- 数据划分必须固定并保存 ID 列表。
- 禁止同一 `sample_id` 出现在多个 Split。
- 对工具名、Schema 高度重复情况输出统计报告。
- 必须执行 Benchmark Leakage Check，至少做：
  - Query 近重复检查；
  - Tool Schema / Function Name 高度重复检查；
  - 已知 Benchmark Sample ID 检查。

## 6.3 数据验收标准

- 100% 样本能被统一 Schema 校验。
- 100% Gold Call 可被验证器解析。
- 至少 98% Gold Call 被验证为 `correct`；其余必须列入 `data_quality_issues.jsonl`。
- Train/Validation/Test 无样本 ID 重叠。
- 随机重跑处理脚本，输出 Hash 一致。

---

# 7. Prompt 与序列化

## 7.1 训练输入

使用模型原生 Chat Template；工具 Schema 放入 System 或 Tool 区域。

## 7.2 标签 Mask

Vanilla SFT 仅对 Assistant 的 Function Call 输出计算 Token CE：

- System：Label = -100
- User：Label = -100
- Tool Schema：Label = -100
- Assistant Call：正常 Label

## 7.3 推理要求

测试时：

- Greedy Decode
- Temperature = 0
- 固定最大输出长度
- 不使用教师
- 不使用验证器修复
- 不做候选重排
- 不做约束解码，除非所有方法统一启用

## 7.4 验收标准

- 给定固定输入和 Seed，序列化结果字节级一致。
- Assistant Label Mask 单元测试通过。
- 推理输出可由统一 Parser 处理。

---

# 8. 验证器设计

## 8.1 验证器职责

验证器不是开放世界语义 Judge。它仅在给定：

- 当前 Query
- Tool Schema
- Gold Call / Gold Calls
- 确定性规范化规则

的前提下，对候选调用做高精度结构判断。

## 8.2 验证签名

每个候选输出：

```json
{
  "parseable": true,
  "function_exists": true,
  "function_correct": true,
  "required_args_complete": true,
  "extra_args_valid": true,
  "types_valid": true,
  "values_match": true,
  "canonical_form": "...",
  "class_label": "correct",
  "is_valid": true,
  "abstain": false,
  "reasons": []
}
```

## 8.3 第一版类别

```text
correct
wrong_function
wrong_arguments
invalid
unknown
```

`wrong_arguments` 内部保留细分原因：

- missing_required
- extra_argument
- wrong_key
- wrong_type
- wrong_value

## 8.4 安全等价规则

只允许：

1. JSON 字段顺序变化
2. 空格、引号等序列化差异
3. Schema 明确声明的默认参数展开
4. 明确的大小写、数字和日期规范化
5. Schema 明确声明为无序的集合排序

禁止：

- 将不同函数仅凭名称或描述相似判为等价
- 使用 LLM Judge 推断广义语义等价
- 因一次执行结果相同就判为等价
- 猜测未声明的默认值或参数别名

## 8.5 Abstain

以下情况必须输出 `unknown`：

- Schema 没有定义必要规则
- Gold 存在明显歧义
- 参数语义需要外部知识或真实环境
- 不同函数可能相似但无法确定等价
- 规范化可能改变语义

`unknown` 默认不参与 Class KD。

## 8.6 验证器验收标准

### 单元测试

必须覆盖：

- 合法 JSON / 非法 JSON
- 正确函数 / 不存在函数 / 存在但错误函数
- 缺失必填参数
- 非法额外参数
- 类型错误
- Enum 错误
- 数字、布尔、日期规范化
- 默认参数展开
- 字段重排
- 嵌套对象

要求：全部通过。

### Property-based Test

必须验证：

\[
C(C(y))=C(y)
\]

规范化幂等性。

等价关系满足：

- 自反
- 对称
- 传递

要求：全部通过。

### 人工抽检

Pilot 至少人工检查 100 个候选，覆盖所有类别。

验收目标：

- `correct/equivalent` 类 Precision ≥ 98%
- 总体分类准确率 ≥ 95%
- 无法可靠判断时必须进入 `unknown`
- 输出 Coverage

若不达标，不允许启动 Class KD 训练。

---

# 9. 候选池构造

## 9.1 每条样本默认 K=6

首个可信 Pilot 统一使用：

\[
K=6
\]

推荐组成：

| 来源 | 数量 | 说明 |
|---|---:|---|
| Gold | 1 | 原始正确调用 |
| 严格等价变体 | 1 | 安全规则生成；无法生成则记录 `no_equivalent_variant` |
| 学生 On-policy Hard Error | 1 | Vanilla SFT 初始学生的高分错误 |
| 错函数反事实 | 1 | 从相似候选工具中替换 |
| Wrong Argument | 1 | 缺参数 / 错 Key / 错类型之一 |
| Wrong Value / Invalid | 1 | 优先错参数值；无法构造时保留低比例 Invalid |

原则：

- RQ1 需要保证一部分样本存在至少两个严格合法表示；
- RQ2/RQ3 需要保证候选跨越多个结构类别，而不是 6 个近重复字符串；
- 对没有安全等价变体的样本，不允许强行制造“语义等价”。

敏感性分析默认比较：

\[
K\in\{4,6,8\}
\]

只有完整论文阶段再考虑 K=12。

## 9.2 On-policy 候选

使用 Vanilla SFT 初始学生 `S0`：

- Beam Search 或低温采样
- 去重
- 保留验证器确认的高概率错误
- 候选生成一次后静态缓存

## 9.3 候选记录格式

```json
{
  "sample_id": "train_000001",
  "candidate_id": "train_000001_c03",
  "source": "student_on_policy",
  "serialized_call": "...",
  "parsed_call": {},
  "verifier_signature": {},
  "class_label": "wrong_arguments",
  "is_valid": false,
  "canonical_form": null,
  "initial_student_score": -0.47,
  "teacher_score": -0.22
}
```

## 9.4 候选池验收标准

- 至少 95% 样本有 1 个 Gold 和 1 个错误候选。
- 至少 90% 样本有一个结构合法但语义错误候选。
- 每条样本候选去重后数量 ≥ 5。
- 各类别数量统计写入报告。
- K=4、6、8 三种截取方式可配置；正式主实验默认 K=6。
- 相同配置重复构建，候选池 Hash 一致。

---

# 10. 完整候选序列打分

## 10.1 分数定义

对模型 \(M\in\{T,S\}\)：

\[
s_M(y_i\mid x)
=
\frac{1}{L_i^\alpha}
\sum_{t=1}^{L_i}
\log p_M(y_{i,t}\mid x,y_{i,<t})
\]

默认：

\[
\alpha=1
\]

仅候选输出 Token 参与打分。

## 10.2 教师打分

- 教师冻结
- 无梯度
- 离线批量计算
- 保存 float32 分数
- 缓存 Key 包含：
  - 模型 Revision
  - Tokenizer Hash
  - Prompt Template Hash
  - 候选池 Hash
  - \(\alpha\)
  - 温度

## 10.3 学生打分

训练时学生候选分数必须保留梯度。

将同一样本的 K 个候选展开为 Batch，计算后按 `sample_id` 重新聚合。

## 10.4 数值验收

必须包含 Toy Test：

- 手工给定 3 个候选 Token Log Probability
- 程序分数与手算误差 < \(10^{-6}\)
- Prompt Token 不得计入分数
- 长度归一化开关正确
- Teacher Cache 重载后值完全一致

---

# 11. 实验方法定义

---

## 11.1 Vanilla SFT

### 输入

只使用 Gold。

### Loss

\[
\mathcal L_{\mathrm{SFT}}
=
-\frac{1}{|y^*|}
\sum_t \log p_S(y_t^*\mid x,\mathcal T,y_{<t}^*)
\]

### 验收

- 在 32 条样本 Tiny Set 上可过拟合。
- 训练 Loss 持续下降。
- Tiny Set AST Accuracy ≥ 90%。

---

## 11.2 Token-level KD

### 前提

学生与教师 Tokenizer 必须一致。

### Loss

\[
\mathcal L_{\mathrm{TokenKD}}
=
\frac{1}{|y^*|}
\sum_t
D_{\mathrm{KL}}
\left(
p_T^\tau(\cdot\mid x,y_{<t}^*)
\|
p_S^\tau(\cdot\mid x,y_{<t}^*)
\right)
\]

总目标：

\[
\mathcal L
=
\mathcal L_{\mathrm{SFT}}
+
\lambda_{\mathrm{TKD}}\tau^2\mathcal L_{\mathrm{TokenKD}}
\]

### 验收

- Teacher Logits 无梯度。
- Padding 与 Prompt Token 不参与 KL。
- Toy 分布 KL 与手算误差 < \(10^{-6}\)。
- \(\lambda=0\) 时结果退化为 Vanilla SFT。

---

## 11.3 Candidate-level KD

### 候选分布

\[
\pi_M(y_i\mid x)
=
\frac{\exp(s_M(y_i)/\tau_d)}
{\sum_j\exp(s_M(y_j)/\tau_d)}
\]

### Loss

\[
\mathcal L_{\mathrm{CandidateKD}}
=
D_{\mathrm{KL}}(\pi_T\|\pi_S)
\]

总目标：

\[
\mathcal L
=
\mathcal L_{\mathrm{SFT}}
+
\lambda_{\mathrm{CKD}}\tau_d^2\mathcal L_{\mathrm{CandidateKD}}
\]

### 验收

- 每条样本的候选概率和为 1。
- Teacher/Student 使用相同候选池顺序。
- 候选置换后 Loss 不变。
- 只有一个候选时 KD Loss 为 0。

---

## 11.4 Class-level KD（核心方法）

### 类别聚合

验证器定义：

\[
g_x(y_i)=e
\]

候选概率：

\[
\pi_M(y_i\mid x)=\frac{\exp(s_M(y_i)/\tau_d)}{\sum_j\exp(s_M(y_j)/\tau_d)}
\]

类别概率：

\[
q_M(e\mid x)=\sum_{y_i:g_x(y_i)=e}\pi_M(y_i\mid x)
\]

### Class KD Loss

\[
\mathcal L_{\mathrm{ClassKD}}=D_{\mathrm{KL}}(q_T\|q_S)
\]

总目标：

\[
\mathcal L=\mathcal L_{\mathrm{SFT}}+\lambda_{\mathrm{Class}}\tau_d^2\mathcal L_{\mathrm{ClassKD}}
\]

### Candidate KL 的精确分解（RQ2 理论依据）

定义类内条件分布：

\[
\pi_M(y\mid e,x)=\frac{\pi_M(y\mid x)}{q_M(e\mid x)}
\]

则：

\[
D_{\mathrm{KL}}(\pi_T\|\pi_S)=D_{\mathrm{KL}}(q_T\|q_S)+\sum_e q_T(e)D_{\mathrm{KL}}\left(\pi_T(\cdot\mid e,x)\|\pi_S(\cdot\mid e,x)\right)
\]

记：

\[
\mathcal L_{\mathrm{inter}}=D_{\mathrm{KL}}(q_T\|q_S)
\]

\[
\mathcal L_{\mathrm{intra}}=\sum_e q_T(e)D_{\mathrm{KL}}\left(\pi_T(\cdot\mid e,x)\|\pi_S(\cdot\mid e,x)\right)
\]

因此：

\[
\mathcal L_{\mathrm{CandidateKD}}=\mathcal L_{\mathrm{inter}}+\mathcal L_{\mathrm{intra}}
\]

本 Pilot 的核心可证伪假设是：

> 对容量受限学生，`intra-class` 项中的部分信息属于低优先级或不稳定的类内偏好；去除该项后，结构级泛化可能更好。

不得在实验前写成“类内信息一定无用”。

### Intra-class Entropy

对类别 \(e\)：

\[
H_T(Y\mid E=e,x)=-\sum_{y:g_x(y)=e}\pi_T(y\mid e,x)\log \pi_T(y\mid e,x)
\]

报告时至少对 `correct` 类计算；样本按 Low / Medium / High 三个分位桶分析。

RQ2 关键统计量：

\[
\Delta_{\mathrm{Class-Candidate}}=\mathrm{Metric}_{\mathrm{ClassKD}}-\mathrm{Metric}_{\mathrm{CandidateKD}}
\]

检查其是否随学生规模减小、教师类内熵升高而增大。

### 验收

- 类别概率和为 1。
- 类内候选顺序变化不影响 Class KD Loss。
- `unknown` 候选默认从聚合中排除并重新归一化。
- Candidate KL 数值必须满足分解恒等式，Toy Test 误差 < \(10^{-6}\)。
- Toy 示例：候选概率 `[0.3, 0.2, 0.1, 0.4]`，前三项同类，聚合类别概率必须为 `[0.6, 0.4]`。

---

## 11.5 Random Negative

### 负样本

从验证器确认错误的候选中按类别分层随机抽取固定数量。

### Loss

\[
\mathcal L_{\mathrm{RN}}
=
-\log
\frac{\exp(s_S(y^+)/\tau_c)}
{\exp(s_S(y^+)/\tau_c)+
\sum_{y^-\in N_{\mathrm{rand}}}\exp(s_S(y^-)/\tau_c)}
\]

总目标：

\[
\mathcal L
=
\mathcal L_{\mathrm{SFT}}
+
\lambda_{\mathrm{RN}}\mathcal L_{\mathrm{RN}}
\]

### 验收

- 固定 Seed 时负样本选择一致。
- 与 Hard Negative 使用相同候选数和相同 Loss。
- 不得把错误候选当作 CE 正标签。

---

## 11.6 Hard Negative

### 负样本

使用 Vanilla SFT 学生 `S0` 的离线分数：

\[
N_{\mathrm{hard}}
=
\operatorname{TopM}_{y\in N(x)}s_{S0}(y)
\]

优先保留结构合法但语义错误的候选。

### Loss

与 Random Negative 完全相同。

### 验收

- Hard Negative 平均初始学生分数高于 Random Negative。
- Random 与 Hard 唯一差异是负样本选择方式。
- 统计各错误类型占比。
- 第一版使用 Static Hard Negative。

---

## 11.7 Class-Hard / Valid-Set Supervision

该 Baseline 用于回答 RQ3：如果已有高精度 Verifier，是否只需要使用硬类别标签，而不需要教师软类别分布？

### 正确类别 / 正集合

\[
V(x)=\{y:\mathrm{verifier}(x,y)=\mathrm{correct}\}
\]

\[
q_S(\mathrm{correct}\mid x)=\sum_{y\in V(x)}\pi_S(y\mid x)
\]

### Loss

\[
\mathcal L_{\mathrm{ClassHard}}=-\log q_S(\mathrm{correct}\mid x)
\]

等价地：

\[
\mathcal L_{\mathrm{ClassHard}}=-\log\frac{\sum_{y\in V(x)}\exp(s_S(y)/\tau_c)}{\sum_{y\in \mathcal C(x)}\exp(s_S(y)/\tau_c)}
\]

因此原 PRD 的 `Valid-Set Contrastive` 在当前定义下，本质上就是 **Class-Hard**。

总目标：

\[
\mathcal L=\mathcal L_{\mathrm{SFT}}+\lambda_{\mathrm{CH}}\mathcal L_{\mathrm{ClassHard}}
\]

### 验收

- 多个合法候选均进入正确类。
- 不使用教师分数。
- 若所有候选均属于正确类，Loss 接近 0。
- 与 Class KD 使用完全相同的候选池和 Verifier。

---

## 11.8 Class Label Smoothing

用于排除：Class KD 的收益只是因为 Soft Label 本身，而不是教师特定的结构知识。

设类别数为 \(E\)，Gold 类为 \(e^*\)，平滑系数 \(\epsilon\)：

\[
\tilde q(e^*)=1-\epsilon
\]

其他类别：

\[
\tilde q(e)=\frac{\epsilon}{E-1}
\]

Loss：

\[
\mathcal L_{\mathrm{CLS}}=D_{\mathrm{KL}}(\tilde q\|q_S)
\]

默认首轮：`epsilon=0.1`；必要时仅在 Validation 比较 `{0.05, 0.1, 0.2}`。

### 验收

- 不读取 Teacher Cache。
- 除 Gold 类外，其余类别概率严格均匀。
- `epsilon=0` 时退化为 Class-Hard。
- 与 Class KD 使用相同类别集合。

---

## 11.9 Optional：Hierarchical KD（非首轮必跑）

仅当 RQ2 显示 Class KD 优于 Candidate KD，但部分低熵样本上 Candidate KD 更好时，才允许探索：

\[
\mathcal L_{\mathrm{HKD}}=\lambda_{\mathrm{inter}}\mathcal L_{\mathrm{inter}}+\lambda_{\mathrm{intra}}\mathcal L_{\mathrm{intra}}
\]

其中 \(0<\lambda_{\mathrm{intra}}<\lambda_{\mathrm{inter}}\)。该项属于完整论文扩展，不得阻塞 Pilot 主结论。

---

# 12. 训练公平性

所有方法必须统一：

- 学生 Base Checkpoint
- 数据 Split
- 训练样本数
- Epoch
- 学习率
- 有效 Batch Size
- LoRA 参数
- 最大长度
- 优化器
- 调度器
- 推理设置
- 评测脚本
- Seed

候选级方法统一：

- 候选池
- 候选序列化
- 评分函数
- 长度归一化
- 候选数量 K

KD 方法统一：

- 教师
- 教师 Revision
- 温度设置范围
- 教师缓存

每次实验输出 `run_manifest.json`。

---

# 13. 无训练诊断

必须先运行 `bash scripts/run_diagnostic.sh`。诊断不只决定“能不能训练”，还要判断 RQ1–RQ3 是否存在足够信号。

## 13.1 基础指标

- TeacherClassTop1
- StudentClassTop1
- \(D_{KL}(q_T\|q_S)\)
- Candidate Top-1 vs Class Top-1
- Teacher Class Entropy \(H(q_T)\)
- Multi-valid Coverage \(\rho_{multi}\)

## 13.2 RQ1 诊断：等价候选覆盖

必须统计：

- 有 2 个及以上严格合法调用的样本比例；
- 等价变体来源分布；
- 等价变体与 Gold 的 Token Edit Distance；
- Teacher / Student 对等价候选的分数差；
- Verifier 对等价类的 Precision / Coverage。

若等价变体几乎全部只是空格变化，RQ1 信号不足。

## 13.3 RQ2 诊断：类内熵与容量信号

对 `correct` 类和至少一个错误类计算 \(H_T(Y\mid E=e,x)\)。输出：

- P25 / P50 / P75；
- Low / Medium / High Entropy 桶；
- 每桶样本数；
- Teacher 类内 Top-1 Margin；
- 0.6B / 1.7B 初始学生对同一桶的候选排序差异。

高熵桶必须有足够样本，否则不能对 RQ2 下强结论。

## 13.4 RQ3 诊断：错误结构覆盖

统计至少：

- Wrong Function；
- Wrong Arguments；
- Invalid；
- missing_required / wrong_key / wrong_type / wrong_value；
- Similar-tool 样本比例；
- Teacher 在错误类别上的非均匀程度。

若 Teacher 对所有错误类概率几乎一致，则 Soft-Structure Hypothesis 信号不足。

## 13.5 Go / No-Go 门槛

进入正式训练前建议至少满足：

- TeacherClassTop1 ≥ 80%；
- 教师比 Initial / Vanilla Student 高至少 5 个百分点，优先 ≥8；
- 至少 90% 样本有一个有效 Hard Negative；
- 至少 10%–15% 样本有两个以上严格合法表示，**或** RQ1 专用诊断集达到该覆盖；
- Similar-tool / Wrong-Argument 困难样本数量足以单独报告；
- Intra-class Entropy 三个桶均有可分析样本；
- K=4、6、8 时教师类别 Top-1 结论不发生明显反转；
- Verifier `correct/equivalent` Precision ≥ 98%。

若不满足，生成 `NO_GO_REPORT.md`，必须明确是哪一个 RQ 缺乏可检验信号。

# 14. 评测设计

评测分为两层：

1. **机制诊断评测**：回答 RQ1–RQ3 为什么成立；
2. **外部标准 Benchmark**：回答 RQ4 是否具有外部有效性。

训练时的 Candidate Pool **绝不进入最终推理**。最终推理仍为 \((x,\mathcal T)\rightarrow \hat y\)，不得使用教师、候选答案或 Verifier 修复输出。

## 14.1 标准指标

必须输出：Parse Success Rate、Function Name Accuracy、Argument Key P/R/F1、Argument Value Accuracy、Argument Type Accuracy、AST / Canonical Call Accuracy、Invalid Call Rate、Valid-Set Hit Rate、Wrong Function Rate、Wrong Argument Rate、各错误类型数量与比例。

## 14.2 RQ1：Equivalence / Surface Perturbation Diagnostic

至少包括：

- JSON / 参数字段重排；
- Schema 明确默认参数的展开 / 省略；
- 安全数值 / 日期 / 大小写 / 无序集合规范化；
- 多合法调用子集。

核心指标：

\[
\mathrm{Drop}=\mathrm{Metric}_{original}-\mathrm{Metric}_{equiv\_perturbed}
\]

重点比较：

\[
\mathrm{Drop}_{ClassKD}<\mathrm{Drop}_{CandidateKD}
\]

并同时比较 SFT / Token KD。

## 14.3 RQ2：Capacity × Intra-class Entropy

学生至少：Qwen3-0.6B、Qwen3-1.7B。

核心统计：

\[
\Delta_{Class-Candidate}=\mathrm{Metric}_{ClassKD}-\mathrm{Metric}_{CandidateKD}
\]

检查是否出现 \(\Delta_{0.6B}>\Delta_{1.7B}\)。该关系是机制预测，不是强制成功条件。

按 Teacher Correct-class Intra-class Entropy 划为 Low / Medium / High，每桶单独报告 AST Accuracy、Valid-Set Hit、Class KD − Candidate KD、Candidate Top-1 / Class Top-1。

## 14.4 RQ3：困难混淆子集

### Similar-tool subset

候选工具包含语义相近函数，重点看 Wrong Function Rate 和 Function Name Accuracy。

### Wrong-Argument subset

覆盖 missing_required / wrong_key / wrong_type / wrong_value，重点看 Argument Accuracy 与各错误子类下降比例。

核心比较：Class-Hard / Class Label Smoothing / Hard Negative / Class KD。

如果 Class KD 优于 Class-Hard 和 Label Smoothing，才支持“Teacher Soft Structural Knowledge”而非普通软标签解释。

## 14.5 RQ4：外部标准 Benchmark

### BFCL

优先使用与当前任务一致的单轮、单调用 / AST 兼容子集。使用官方或复现一致的评测逻辑，保存原始预测，不将测试标签用于训练或候选构造。

### ACEBench

优先使用与单轮 Function Calling 设置兼容的 Normal / Special 子集。使用统一模型推理接口，保存逐样本输出，与 BFCL 分开报告。

### 外部评测成功判断

- Class KD 相比 Vanilla SFT 不出现明显退化；
- 相比 Candidate KD / Token KD 至少一个标准 Benchmark 或关键单轮子集有稳定正向趋势；
- 自建 Diagnostic 的收益不能伴随标准 Benchmark 明显下降。

## 14.6 评测验收

- 所有方法使用同一推理脚本；
- 推理过程中不使用教师、候选池和 Verifier 修复；
- 指标脚本在人工 Toy Cases 上 100% 正确；
- 输出逐样本预测与错误标签；
- RQ1–RQ4 每个问题都能从生成的 CSV/JSON 中独立复算。

# 15. Pilot 实验矩阵

Pilot 按“先核心趋势、再机制归因、最后外部验证”执行。

## 15.1 Stage A：核心可行性（1.7B 主学生）

首轮 Seed = 42：Vanilla SFT / Token KD / Candidate KD / Class KD。

目的：确认 Class KD 是否有正向趋势，确认 Candidate / Class KD 管线正确，初步回答 RQ1 / RQ4。

## 15.2 Stage B：RQ1 等价鲁棒性

对 Stage A 四种方法统一跑 Original Internal Test、Equivalence Diagnostic、BFCL AST 兼容子集，输出 Original Metric / Perturbed Metric / Robustness Drop。

## 15.3 Stage C：RQ2 容量与类内熵

重点跑 Candidate KD / Class KD，学生为 0.6B 与 1.7B。统一做 Low / Medium / High Entropy Buckets 与 `ClassKD - CandidateKD` 差值分析。

若资源不足，0.6B 在 2080 Ti 完成，1.7B 正式实验在 4090 完成。

## 15.4 Stage D：RQ3 Soft Structural Knowledge 归因

1.7B 上增加 Random Negative / Hard Negative / Class-Hard / Class Label Smoothing。

重点比较 Similar-tool 与 Wrong-Argument subset。

最关键因果链：`Class-Hard → Class Label Smoothing → Class KD`，以及 `Hard Negative → Class KD`。

## 15.5 Stage E：RQ4 外部 Benchmark

至少对 Vanilla SFT / Token KD / Candidate KD / Class KD / 最强非教师 Baseline 评测 BFCL + ACEBench。不得只把最优方法送外部 Benchmark。

## 15.6 Stage F：重复性与候选敏感性

补 Seed 43：Vanilla SFT / Candidate KD / Class KD / 最强归因 Baseline。

候选数敏感性：\(K\in\{4,6,8\}\)。首轮只对 Candidate KD / Class KD 做 K 敏感性。

# 16. Pilot 成功与停止标准

Pilot 成功标准必须分别对应 RQ1–RQ4，禁止只用一个总 AST Accuracy 判断。

## 16.1 RQ1：Equivalence 成功信号

- Class KD 在 Equivalence Diagnostic 上的 Robustness Drop 小于 Vanilla SFT；
- 相比 Candidate KD，Class KD 在等价/格式扰动场景有稳定优势，或相同准确率下 Drop 更小；
- 原始标准测试性能不因鲁棒性提升而明显下降。

若只有空格变化有效，不能宣称解决广义等价问题。

## 16.2 RQ2：Capacity / Intra-class 成功信号

强机制证据包括任一：

- `ClassKD - CandidateKD` 在 0.6B 大于 1.7B；
- High Intra-class Entropy 桶中 Class KD 相对 Candidate KD 的优势明显高于 Low Entropy 桶；
- Candidate KD 的类内拟合改善，但不转化为 AST / Valid-Set 指标提升。

若 Candidate KD 在所有规模和熵桶均稳定优于 Class KD，应否定或修改 Capacity-Limited Hypothesis。

## 16.3 RQ3：Soft Structural Knowledge 成功信号

- Class KD > Class-Hard；
- Class KD > Class Label Smoothing；
- Class KD > Hard Negative，或在 Similar-tool / Wrong-Argument 子集有明显优势；
- 错误类型分析与教师类别混淆方向一致。

只有 Class-Hard 提升而 Class KD 无额外收益时，不得宣称教师软类别知识有效。

## 16.4 RQ4：外部有效性成功信号

- BFCL / ACEBench 不出现明显退化；
- 至少一个外部 Benchmark 或关键单轮子集上，Class KD 相比 Vanilla SFT / Candidate KD 有稳定正向趋势；
- 第二 Seed 方向可解释。

## 16.5 Pilot 弱成功

RQ1 或 RQ3 至少一个机制问题成立，Class KD 相比 Vanilla SFT 有稳定正向趋势，外部 Benchmark 不明显下降，则值得进入更大实验，但论文主张需收窄。

## 16.6 Pilot 强成功

建议同时满足：

1. Class KD 比 Vanilla SFT 的标准 AST / Canonical Accuracy 有稳定增益；
2. Class KD 比 Candidate KD 至少在一个核心机制场景有明显优势；
3. RQ1 Robustness Drop 明显减小；
4. RQ2 至少出现“更小模型”或“高熵桶”其中一种符合预测的规律；
5. Class KD 优于 Class-Hard 与 Label Smoothing；
6. Class KD 优于或明显补充 Hard Negative；
7. BFCL / ACEBench 至少一个标准评测有稳定正向结果；
8. 两个 Seed 方向一致；
9. K=4、6、8 下核心结论不反转。

## 16.7 停止 / 转向条件

- Teacher 无法稳定识别正确类别；
- Verifier 等价判断达不到高 Precision；
- 严格等价变体极少且无法构成 RQ1 诊断；
- Candidate KD 在 0.6B / 1.7B 和全部熵桶持续优于 Class KD；
- Class KD 与 Class-Hard / Label Smoothing 没有可区分收益；
- 提升全部可由 Hard Negative 解释；
- K 稍变结果即反转；
- 两个 Seed 一个涨一个跌且无法解释；
- 只有自建 Diagnostic 提升，BFCL / ACEBench 明显下降。

失败时输出 `PILOT_FAILURE_ANALYSIS.md`，按失败 RQ 给出转向：RQ1 → Valid-Set Training；RQ2 → Hierarchical KD；RQ3 → Class-Hard / Structure-aware Hard-negative SFT；RQ4 → 检查数据分布与 Benchmark 对齐。

# 17. 输出与报告

每次运行必须输出：

```text
outputs/<run_id>/
├── config_resolved.yaml
├── run_manifest.json
├── train_log.jsonl
├── metrics.json
├── predictions.jsonl
├── error_analysis.json
├── checkpoint/
└── environment.txt
```

最终 Pilot 报告：

```text
reports/
├── diagnostic_report.md
├── verifier_audit.md
├── candidate_statistics.md
├── rq1_equivalence.md
├── rq2_capacity_entropy.md
├── rq3_soft_structure.md
├── rq4_external_benchmarks.md
├── pilot_results.md
├── ablation_results.md
└── go_no_go_decision.md
```

## 17.1 `pilot_results.md` 必须包含

- 数据与模型配置；
- 所有方法主指标；
- 两个 Seed 均值与差异；
- 候选来源与类别统计；
- Verifier Precision / Coverage；
- RQ1：Original / Perturbed / Robustness Drop；
- RQ2：0.6B vs 1.7B；Low / Medium / High Entropy；
- RQ3：Class-Hard / Label Smoothing / Hard Negative / Class KD；
- RQ4：BFCL + ACEBench；
- 错误类型变化与失败案例至少 20 条；
- 每个 RQ 单独给出 `Supported / Partially Supported / Not Supported`；
- 是否满足强成功标准；
- 后续建议。

# 18. CLI 与一键运行

必须实现：

```bash
python -m fc_equiv_kd.cli.prepare_data --config configs/pilot.yaml
python -m fc_equiv_kd.cli.build_candidates --config configs/pilot.yaml
python -m fc_equiv_kd.cli.audit_verifier --config configs/pilot.yaml
python -m fc_equiv_kd.cli.score_teacher --config configs/pilot.yaml
python -m fc_equiv_kd.cli.run_diagnostic --config configs/pilot.yaml
python -m fc_equiv_kd.cli.train --method vanilla_sft --config configs/pilot.yaml
python -m fc_equiv_kd.cli.train --method token_kd --config configs/pilot.yaml
python -m fc_equiv_kd.cli.train --method candidate_kd --config configs/pilot.yaml
python -m fc_equiv_kd.cli.train --method class_kd --config configs/pilot.yaml
python -m fc_equiv_kd.cli.train --method class_hard --config configs/pilot.yaml
python -m fc_equiv_kd.cli.train --method class_label_smoothing --config configs/pilot.yaml
python -m fc_equiv_kd.cli.evaluate --run_dir outputs/<run_id>
python -m fc_equiv_kd.cli.evaluate_diagnostics --run_dir outputs/<run_id>
python -m fc_equiv_kd.cli.evaluate_benchmark --benchmark bfcl --run_dir outputs/<run_id>
python -m fc_equiv_kd.cli.evaluate_benchmark --benchmark acebench --run_dir outputs/<run_id>
```

并提供：

```bash
bash scripts/run_pilot.sh
```

一键完成：

1. 环境检查
2. 数据处理
3. 候选构建
4. 验证器审计
5. 教师打分
6. 无训练诊断
7. Go/No-Go 判断
8. 核心四方法训练
9. RQ1 机制评测
10. 若核心趋势为正，再按配置运行 RQ2 / RQ3
11. BFCL / ACEBench 外部评测
12. 生成分 RQ 报告

若诊断 No-Go，脚本必须停止，不得继续消耗 GPU。

---

# 19. 测试要求

## 19.1 单元测试

覆盖：

- 数据 Adapter
- Prompt Mask
- Parser
- Schema 验证
- Canonicalization
- 候选去重
- 序列打分
- Softmax
- KL
- 类别聚合
- Random/Hard Negative 选择
- Class-Hard / Valid-Set Loss
- Class Label Smoothing Loss
- Candidate KL = Inter-class KL + Intra-class KL 分解恒等式
- Intra-class Entropy
- Robustness Drop
- 指标计算

## 19.2 集成测试

- 8 条样本跑通全部流水线
- 32 条样本 Tiny Overfit
- 教师缓存可加载
- 训练中断后可恢复
- 输出文件完整

## 19.3 测试验收

```bash
pytest -q
```

必须：

- 全部通过
- 无跳过核心测试
- 核心数学测试误差 < \(10^{-6}\)

---

# 20. Agent 工作协议

代码 Agent 必须遵守：

1. 先阅读本文档，再生成 `IMPLEMENTATION_PLAN.md`。
2. 每完成一个 Milestone，更新 `STATUS.md`。
3. 所有研究定义变更写入 `DECISIONS.md`。
4. 不得擅自新增 RL、DPO、多轮 Agent 等功能。
5. 发现需求冲突时暂停并记录，不得自作主张改变实验含义。
6. 每个模块先写测试，再实现。
7. 所有异常必须显式报错，禁止静默跳过。
8. 不得把教师输出直接作为测试预测。
9. 不得用测试标签参与候选生成或训练。
10. 测试集只用于最终评测；超参数选择使用 Validation。

---

# 21. Milestone 与验收

## M0：环境与骨架
交付项目结构、依赖锁定、Doctor、Smoke Test；验收 `ENVIRONMENT_OK=true` 且 Smoke Test 退出码 0。

## M1：数据与 Vanilla SFT
交付 xLAM / ToolACE / Local JSONL Adapter、固定 Split、Benchmark Leakage Check、Prompt 模板、Vanilla SFT、基础评测。验收 Gold 可解析率 ≥98%，Tiny Overfit AST ≥90%。

## M2：Verifier + RQ1 等价审计
交付 Parser、Schema Checker、Normalizer、类别分类器、Equivalence Generator、Audit Report。验收 Property Test 全通过，`correct/equivalent` Precision ≥98%，RQ1 等价覆盖可分析。

## M3：候选池与序列打分
交付 K=6 默认候选生成器、On-policy 静态候选、规则反事实、教师缓存、学生可微打分。验收 ≥90% 样本有 Hard Negative，Toy Score 与 Candidate KL 分解误差 < \(10^{-6}\)。

## M4：无训练诊断
交付 RQ1–RQ3 Diagnostic Report 与 Go/No-Go。验收 TeacherClassTop1 ≥80%，Entropy Buckets、Similar-tool、Wrong-Argument 覆盖足够，候选数敏感性可接受。

## M5：核心四方法 + RQ1
交付 Vanilla SFT / Token KD / Candidate KD / Class KD / Equivalence Diagnostic。验收同预算、Loss 测试通过、输出 RQ1 结果。

## M6：RQ2 容量与类内熵
交付 0.6B 与 1.7B 的 Candidate KD / Class KD，以及 Entropy Bucket Report。验收明确 RQ2 是否被支持。

## M7：RQ3 归因 Baseline
交付 Random Negative / Hard Negative / Class-Hard / Class Label Smoothing / Similar-tool / Wrong-Arg Report。验收可区分“硬类别”“普通软标签”“教师软结构”。

## M8：RQ4 外部 Benchmark
交付 BFCL Adapter、ACEBench Adapter、外部结果表。验收推理不使用 Candidate Pool / Teacher / Verifier 修复，所有主要方法统一评测。

## M9：重复性与最终报告
交付第二 Seed、K=4/6/8 敏感性、最终报告、Go/No-Go、运行说明。验收 RQ1–RQ4 分别给出结论，所有产物可复现。

# 22. 预计时间与算力

## 22.1 执行顺序

本 PRD 不把时间估算作为验收条件。实际执行严格按照：

```text
M0–M4：环境 / 数据 / Verifier / 候选 / Go-No-Go
   ↓
M5：核心四方法 + RQ1
   ↓
M6：RQ2 容量 × 类内熵
   ↓
M7：RQ3 归因 Baseline
   ↓
M8：RQ4 外部 Benchmark
   ↓
M9：第二 Seed + K 敏感性 + 最终报告
```

任何阶段若触发对应停止条件，不得为了“跑完计划”继续消耗资源。

## 22.2 算力

目标范围：

\[
50\text{–}120\ \text{RTX 4090 等效 GPU 小时}
\]

优化要求：

- 教师离线缓存
- 4-bit/8-bit 教师推理
- 学生 LoRA/QLoRA
- 静态候选
- 首轮单 Seed
- Tiny Set 调试后再跑全量

---

# 23. 最终 Definition of Done

Pilot 只有同时满足以下条件才算完成：

- [ ] 环境可一键复现
- [ ] 数据处理与 Split 固定
- [ ] Benchmark Leakage Check 完成
- [ ] Vanilla SFT 可正常训练与评测
- [ ] Verifier 通过高精度审计
- [ ] 候选池可复现，默认 K=6，K=4/6/8 可配置
- [ ] 教师分数完成离线缓存
- [ ] 无训练诊断完成并给出 Go / No-Go
- [ ] 四个核心方法完成同预算对比
- [ ] Class-Hard 与 Class Label Smoothing 完成
- [ ] Hard Negative 归因完成
- [ ] 0.6B / 1.7B Candidate KD vs Class KD 容量比较完成
- [ ] Intra-class Entropy 分桶完成
- [ ] Equivalence Robustness Drop 完成
- [ ] Similar-tool / Wrong-Argument 子集分析完成
- [ ] BFCL + ACEBench 外部评测完成
- [ ] 至少两个 Seed 的关键结果可比较
- [ ] 训练成本与失败案例完整记录
- [ ] 最终报告分别回答 RQ1–RQ4：
  - RQ1：等价调用聚合是否降低表面格式敏感性？
  - RQ2：Class KD 相对 Candidate KD 的收益是否与学生容量和类内熵有关？
  - RQ3：Class KD 是否优于 Class-Hard / Label Smoothing / Hard Negative？
  - RQ4：BFCL / ACEBench 是否保持稳定，方法是否只在自建诊断集有效？
- [ ] 明确最终可支持的论文主张与是否进入完整论文阶段

# 24. Agent 开始执行前必须回复的内容

代码 Agent 在写任何业务代码前，必须先提交：

1. 对本 PRD 的理解摘要；
2. 分阶段实现计划；
3. 拟使用的具体学生/教师模型；
4. 拟使用的数据源与 Adapter 方案；
5. GPU/显存适配方案；
6. 预计目录结构；
7. 预计风险点；
8. 需要用户确认的阻塞问题；
9. 首个可运行 Smoke Test 的时间点；
10. RQ1–RQ4 的实验映射与预期产物；
11. Benchmark Leakage 防护方案；
12. 不会在 Pilot 中实现的功能清单。

用户确认后再开始正式实现。
