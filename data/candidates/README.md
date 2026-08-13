# K=6 候选池

当前文件分三层：

- `s0_train_predictions.jsonl`、`s0_diagnostic_predictions.jsonl`：公共 S0 的自由生成与真实错误。
- `train_unfrozen.jsonl`、`diagnostic_unfrozen.jsonl`：每条严格 K=6，但尚未通过人工审核。
- `train.jsonl`、`diagnostic.jsonl`：只有人工审核达到门槛后才生成的冻结版。

Train 与 Diagnostic 不能交叉使用。清单状态不是 `frozen=true` 的文件不得用于 Teacher 完整缓存
或蒸馏训练。

详见：`docs/数据与文件流向.md`
