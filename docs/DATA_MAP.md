# Fast Pilot 数据地图 — 每个阶段读什么、写什么

> 路径均相对于 `fc_class_kd/` 根目录。  
> **机器分工不固定**：任何有 GPU 的机器可跑训练/冒烟；无 GPU 可跑数据处理与 `pytest`。
>
> **当前 checkout 提示（2026-08-10）**：Git 中只有 `data/fixtures/smoke_4.jsonl`；下文列出的
> ToolACE processed/splits 条数来自旧机器记录，原 JSONL 尚未同步到本机，不能视为当前已复现。

---

## 图：数据在流水线里怎么流

```text
[HF 下载] xLAM / ToolACE
        ↓  scripts/build_data.py
data/processed/all_records.jsonl          ← 筛选+去重后的全集
        ↓  deterministic_split (seed=42)
data/splits/*.jsonl                       ← 固定划分（实验真相源）
        ↓
   ┌────┴────┬──────────────┬─────────────────┐
   ↓         ↓              ↓                 ↓
 Step2-4   Step3         Step5            Step8
 smoke/    预诊断        候选池(待建)      评测
 S0训练    diagnostic    candidates/      internal_test
           .jsonl        (待建)           equiv...
```

---

## 按 PRD Step 对照表

| Step | 阶段 | **读取（输入）** | **写出（输出）** | 当前状态 |
|------|------|------------------|------------------|----------|
| **1** | 环境 + 数据 + Verifier | HF 数据集（缓存不在本目录） | 见下方「目录说明」 | ⚠️ 代码已有；当前 checkout 无全量 splits；人工 Verifier Gate 未做 |
| **2** | 0.6B Smoke | `data/splits/train.jsonl` **或** `data/fixtures/smoke_4.jsonl`（无全量数据时自动 fallback） | 无数据文件；终端 `SMOKE_OK` | 待有 GPU 的机器跑 |
| **3** | Pre-Diagnostic | `data/splits/diagnostic.jsonl`（约 206 条） | `reports/diagnostic.md`（待建脚本） | 未跑 |
| **4** | 训练 S0（Vanilla SFT） | `data/splits/train.jsonl` | `outputs/s0/` LoRA（待建） | 未跑 |
| | 验证 | `data/splits/val.jsonl` | val metrics | |
| **5** | 冻结 K=6 候选池 | `data/splits/train.jsonl` + **S0 权重** | `data/candidates/train.jsonl`（**待建**） | 未做 |
| **6** | Teacher 分数缓存 | `data/candidates/*.jsonl` | `data/cache/teacher_scores/`（**待建**） | 未做 |
| **7** | 三条分支训练 | `train.jsonl` + `candidates` + `cache` | `outputs/sft_cont/` 等（**待建**） | 未做 |
| | 训练中验证 | `data/splits/val.jsonl` | | |
| **8a** | Internal Test | `data/splits/internal_test.jsonl` | `reports/fast_pilot_results.md` | 未跑 |
| **8b** | Equivalence | `data/splits/equivalence_diagnostic.jsonl` | Robustness Drop 指标 | 未实现完整 paired evaluation |
| | 变体 helper | `data/diagnostics/equivalence_variants.jsonl` | | 代码已有；当前文件不存在，且协议需先修订 |
| **8c** | BFCL | **外部** BFCL 数据（不进 train） | BFCL 分数 | 未接 |

---

## 目录说明（项目里每个文件夹干什么）

### `data/raw/` — 原始下载（可选本地备份）

| 文件 | 说明 |
|------|------|
| `.gitkeep` | 默认 **空**；`build_data.py` 直接从 HuggingFace 读，可不落盘 raw |

### `data/processed/` — 处理中间结果

| 文件 | 说明 | 阶段 |
|------|------|------|
| **`all_records.jsonl`** | 通过 Fast Pilot 筛选的全量记录（旧机器记录约 2688 条，ToolACE-only；当前 checkout 不含） | Step 1 产出；`build_data.py --skip-download` 时作为切分输入 |

### `data/splits/` — **最重要：固定实验划分**

| 文件 | 条数（旧机器记录） | 用途 |
|------|-------------|------|
| **`train.jsonl`** | 1654 | Step 2 冒烟样本；Step 4 S0 **训练**；Step 5 **候选池**（仅 train）；Step 7 KD **训练** |
| **`val.jsonl`** | 248 | Step 4/7 **训练中验证**、调 early stopping（不做最终论文结论） |
| **`internal_test.jsonl`** | 413 | Step 8 **自建主评测**（AST、错函数率等） |
| **`diagnostic.jsonl`** | 206 | Step 3 **预诊断**（Teacher vs Student，不用于训练） |
| **`equivalence_diagnostic.jsonl`** | 165 | Step 8 **等价鲁棒性**评测（原始条） |

**注意：** BFCL 测试集 **不得** 出现在以上任何文件中（PRD 要求）。
这些条数不是当前 checkout 的本地文件清单；恢复后应另存来源 revision、构建命令和文件 hash。

### `data/diagnostics/` — 诊断衍生数据（非主划分）

| 文件 | 说明 | 阶段 |
|------|------|------|
| **`equivalence_variants.jsonl`** | 由 `scripts/build_equivalence.py` 从 equiv split 展开的 gold-call 变体；当前 checkout 未生成，且不能单独构成鲁棒性评测 | Step 8b 辅助输入 |
| （待建）`pre_diagnostic_scores.jsonl` | 预诊断逐条打分 | Step 3 |

### `data/candidates/` — **待建** Step 5

| 文件（计划） | 说明 |
|--------------|------|
| `train.jsonl` | 每条 train 样本 **K=6 冻结候选** + Verifier 类别 + 来源标签 |
| 无 val/test 候选文件 | PRD：候选池按训练集构造；评测时学生 **直接生成**，不搜候选 |

### `data/cache/` — **待建** Step 6

| 文件（计划） | 说明 |
|--------------|------|
| `teacher_scores/train.parquet` 或 `.jsonl` | 每条样本 K 候选的 Teacher π、类分布 q |
| `student_scores/`（可选） | 预诊断或 KD 用 Student 分数 |

### `outputs/` — 模型与训练产物（不进 git）

| 目录（计划） | 阶段 |
|--------------|------|
| `outputs/s0/` | Step 4 S0 LoRA |
| `outputs/sft_continuation/` | Step 7 分支 A |
| `outputs/candidate_kd/` | Step 7 分支 B |
| `outputs/class_kd/` | Step 7 分支 C |

### `reports/` — 人类可读报告

| 文件 | 阶段 |
|------|------|
| `verifier_audit.json` | Step 1 Verifier 抽检 |
| `diagnostic.md` | Step 3（待建） |
| `fast_pilot_results.md` | Step 8 主表 |
| `go_no_go.md` | Step 9 |

---

## 单条 `train.jsonl` 里有什么（白话）

每一行是一条「考题」：

- `query`：用户说什么  
- `tools`：可选工具列表（带参数说明）  
- `gold_call`：标准答案（调哪个函数、什么参数）  

**候选、Teacher 分数** 不在 split 里，在 Step 5/6 单独文件。

---

## 机器与阶段（弹性分工）

| 需要 GPU | 典型命令 | 哪台机器都能跑 |
|----------|----------|----------------|
| 否 | `build_data.py`, `audit_verifier.py`, `build_equivalence.py`, `pytest` | 公司 PC ✅ |
| **是** | `smoke_test.py`, `train_sft.py`, `cache_teacher_scores.py` | **任意有 NVIDIA GPU + CUDA 的机器** |

当前机器没有可用 CUDA → **本机现在跑不了 0.6B 冒烟**；换有卡的机器即可，不必固定 2080 Ti。

协同见：`docs/WORKFLOW_COMPANY_PERSONAL.md`

---

## AST 是什么？

在本项目 / PRD 里，**AST 不是** 泛指「编译器里的抽象语法树源码」，而是：

> **把模型输出的 Function Call，用 Verifier 规范化之后，判断是否与标准答案在结构上等价。**

白话：

1. 模型吐一段 JSON（函数名 + 参数）  
2. Verifier 解析、按 schema 检查、做规范化（key 排序、默认参数等）  
3. 若类别是 **`correct`** → 这条算 **AST 对**  
4. 指标 **`AST Accuracy`** = 判对条数 / 总条数  

和「字符串一模一样」不同：key 顺序不同、等价默认参数写法不同，**仍可能 AST 对**。

相关指标（Step 8 Internal Test）：

| 指标 | 白话 |
|------|------|
| Parse Success | JSON 能解析吗 |
| Function Name Accuracy | 函数名对吗 |
| Argument Accuracy | 参数内容对吗（更严） |
| **AST / Canonical Call Accuracy** | Verifier 判 **整体调用正确**（主指标） |
| Wrong Function / Wrong Argument / Invalid Rate | 错在哪一类 |

Tiny Overfit 要求「训练集 AST ≥ 90%」= 几十条训练样本上，九成以上被 Verifier 判为正确调用。
