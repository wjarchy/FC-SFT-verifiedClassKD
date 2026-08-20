# 候选池说明

> 当前目录中的 S0 predictions、`train_unfrozen.jsonl`、`diagnostic_unfrozen.jsonl` 和严格 K=6
> 规则均属于 V4 历史资产。不得把它们冻结成 V5 正式候选池。

当前文件分三层：

- `s0_train_predictions.jsonl`、`s0_diagnostic_predictions.jsonl`：公共 S0 的自由生成与真实错误。
- `train_unfrozen.jsonl`、`diagnostic_unfrozen.jsonl`：每条严格 K=6，但尚未通过人工审核。
- `train.jsonl`、`diagnostic.jsonl`：只有人工审核达到门槛后才生成的冻结版。

Train 与 Diagnostic 不能交叉使用。清单状态不是 `frozen=true` 的文件不得用于 Teacher 完整缓存
或蒸馏训练。

V5 正式候选池只有在 Data Opportunity Gate 与 Gap Gate 至少 Weak Pass 后才构造，计划放在
`data/candidates/v5/`。每条允许 3–6 个候选，优先 Gold、自然 Teacher 等价 surface、S0 hard
error 和不同语义错误；没有自然 equivalent 时不人工补 Strict Equivalent。错误类型只作为
`analysis_tag`，class aggregation 只使用 `canonical_signature`。

详见：`docs/数据与文件流向.md`
