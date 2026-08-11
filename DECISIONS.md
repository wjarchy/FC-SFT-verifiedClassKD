# Research and Engineering Decisions

所有会影响实验含义、可比性或可复现性的决定记录在此。

## D001 — 候选 surface 与 AST 分离

- 日期：2026-08-11
- 状态：Accepted
- 决定：候选必须同时保存 `serialized_call` 与 `parsed_call`。Candidate sequence score 使用前者，
  Verifier/canonicalization 使用后者。
- 原因：字段顺序不同的 JSON 在 Python 字典层面相等，但模型看到的是不同 Token 序列；若只存 AST，
  Class KD 的“类内多表面形式”研究对象会在进入训练前消失。

## D002 — Default omission 采用保守白名单

- 日期：2026-08-11
- 状态：Accepted
- 决定：只有 optional 参数存在明确 Schema default、Gold 值与 default 等价、删除后仍通过
  `calls_equivalent` 时，才生成 implicit-default 变体。
- 原因：删除非默认值会制造假等价并污染 `correct` 类。

## D003 — `unknown` 使用 mask + renormalization

- 日期：2026-08-11
- 状态：Accepted
- 决定：`unknown` 不参与 Class KD；Teacher 和 Student 在保留候选上分别重新归一化。
  全部候选均被排除时显式报错。
- 原因：类别向量必须是和为 1 的合法分布，且 Candidate KL 分解需在同一保留空间成立。

## D004 — 关键数据不一致一律 fail-fast

- 日期：2026-08-11
- 状态：Accepted
- 决定：预测数不等于样本数、概率/标签长度不一致、非正温度、空序列分数、无监督输出 Token、
  Gold 截断均显式报错。
- 原因：静默截断或默认修复会虚高指标或改变不同方法的有效训练数据。

## D005 — Qwen3 Prompt 使用 non-thinking 且不静默 fallback

- 日期：2026-08-11
- 状态：Accepted
- 决定：调用原生 Chat Template 时显式 `enable_thinking=False`；模板不兼容时失败，不自动切换到
  另一种 Prompt。Prompt tokens 必须是完整训练序列的严格前缀。
- 原因：Prompt 格式和 label mask 是所有训练分支的共同控制变量。

## D006 — 当前 equivalence 文件不是生成鲁棒性结果

- 日期：2026-08-11
- 状态：Accepted
- 决定：现有 `build_equivalence.py` 产物只用于 reference/candidate/Verifier audit，并显式标记
  `suitable_for_generation_robustness_drop=false`。
- 原因：它没有改变 query/tools，模型输入不变，不能据此报告 generation RobustnessDrop。
- 后续：在 Step 2 前冻结 paired model-visible perturbation protocol。

## D007 — Qwen3 最低 Transformers 版本

- 日期：2026-08-11
- 状态：Accepted
- 决定：项目声明 `transformers>=4.51.0,<5`；完整精确版本等 GPU Smoke 后锁定。
- 原因：低于 4.51.0 不支持 `qwen3` model type。
