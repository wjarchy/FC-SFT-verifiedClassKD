# V7 分析阈值冻结

## 时间边界

本文件和 configs/v7_analysis_thresholds.json 在任何恢复的 Candidate/Class KD outcome 产生前生成。阈值只来自训练侧 Equiv-All-1393 fixed-support cache，Internal 结果没有参与阈值选择。

## 固定阈值

| 分析变量 | q1 | q2 | Equiv-All 桶数 low / mid / high |
|---|---:|---:|---:|
| S0 correct functional class probability | 0.58942356 | 0.61558665 | 465 / 464 / 464 |
| Teacher normalized intra-class entropy | 0.96580232 | 0.98335794 | 465 / 464 / 464 |

quantile 方法：排序后 index=(n-1)q，相邻值线性插值。

边界：low ≤ q1；q1 < mid ≤ q2；high > q2。

## S0 cache

| Split | N | Correct-class confidence min / median / max | Correct class argmax |
|---|---:|---:|---:|
| Equiv-All | 1,393 | 0.5150 / 0.6027 / 0.7210 | 1,393/1,393 |
| Internal-Equiv | 248 | 0.5172 / 0.6018 / 0.6884 | 248/248 |

fixed-support argmax 全对不代表自由生成全对。View A 使用 S0 free-generation Correct/Wrong；View B 使用这里冻结的连续置信度桶，两者不得替换。

## 资产

| 资产 | SHA256 |
|---|---|
| data/cache/v7/s0/manifest.json | 321951fc3977e279463863971f71574eb450b2dd2d4eb8911319e14e33829c75 |
| S0 Equiv-All scores | 56dddd411fa5828e78cdba7f5a3e53cec1e380f303de60f621799d876a56e214 |
| S0 Internal-Equiv scores | 5b5f836565bff6667045ae2ada47559c8583ca407fb971b68f7edc04e47e45f0 |
| configs/v7_analysis_thresholds.json | 94a62913f3cbb6543fb89f9ae915feca60a325b60b9f3dc3dfa8fd79e8db607f |

大型 cache JSONL 由 Git 忽略；两个 manifest、冻结配置、报告和根 artifact 清单进入 Git。
