# V7 S0 Difficulty Analysis

## 结论摘要

- 置信度桶按训练侧 Equiv-All 的 S0 正确功能类概率预先冻结，不使用 Internal outcome 定阈值。
- 从高置信度到低置信度的 Class - Candidate 依次为：-1.23%, 0.00%, 0.00%。
- 难度上升时 delta 单调不减：true。
- S0 Wrong 与 S0 Correct 的相对 delta 差：-1.39%。

## 总体恢复基线

| Subset | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full Internal | 500 | 363/500 (72.60%) | 355/500 (71.00%) | -1.60% | [-4.00%, 0.80%] | 23 / 15 | 0.2559 |
| Internal-Equiv | 248 | 160/248 (64.52%) | 159/248 (64.11%) | -0.40% | [-3.63%, 2.82%] | 9 / 8 | 1.0000 |
| Internal-Singleton | 252 | 203/252 (80.56%) | 196/252 (77.78%) | -2.78% | [-6.35%, 0.79%] | 14 / 7 | 0.1892 |

## View A：S0 Correct vs Wrong

| S0 状态 | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 176 | 150/176 (85.23%) | 150/176 (85.23%) | 0.00% | [-3.98%, 3.98%] | 6 / 6 | 1.0000 |
| wrong | 72 | 10/72 (13.89%) | 9/72 (12.50%) | -1.39% | [-6.94%, 4.17%] | 3 / 2 | 1.0000 |

## View B：S0 fixed-support confidence

阈值：q1=0.58942356，q2=0.61558665；分桶规则为 low ≤ q1，q1 < mid ≤ q2，high > q2。

| 置信度 | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 75 | 45/75 (60.00%) | 45/75 (60.00%) | 0.00% | [-5.33%, 5.33%] | 2 / 2 | 1.0000 |
| mid | 92 | 57/92 (61.96%) | 57/92 (61.96%) | 0.00% | [-6.52%, 6.52%] | 5 / 5 | 1.0000 |
| high | 81 | 58/81 (71.60%) | 57/81 (70.37%) | -1.23% | [-6.17%, 2.47%] | 2 / 1 | 1.0000 |

## A/B/C

| 等价类型 | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 183 | 117/183 (63.93%) | 118/183 (64.48%) | 0.55% | [-3.28%, 4.37%] | 5 / 6 | 1.0000 |
| B | 19 | 16/19 (84.21%) | 15/19 (78.95%) | -5.26% | [-15.79%, 0.00%] | 1 / 0 | 1.0000 |
| C | 46 | 27/46 (58.70%) | 26/46 (56.52%) | -2.17% | [-10.87%, 6.52%] | 3 / 2 | 1.0000 |

## Teacher 类内 entropy（同步保存，主分析之后解释）

阈值：q1=0.96580232，q2=0.98335794；由训练侧 Equiv-All Teacher cache 预先冻结。

| Teacher entropy | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 97 | 74/97 (76.29%) | 71/97 (73.20%) | -3.09% | [-9.28%, 3.09%] | 6 / 3 | 0.5078 |
| mid | 67 | 37/67 (55.22%) | 37/67 (55.22%) | 0.00% | [-5.97%, 5.97%] | 2 / 2 | 1.0000 |
| high | 84 | 49/84 (58.33%) | 51/84 (60.71%) | 2.38% | [-2.38%, 7.14%] | 1 / 3 | 0.6250 |

## 统计口径

- 所有比较均为同一样本上的配对比较；Delta 为 Class accuracy - Candidate accuracy。
- 区间是 seed 42、10,000 次的 paired percentile bootstrap 95% CI；McNemar p 是 discordant pair 的双侧精确二项检验。
- 单 seed、多个小子组的区间较宽；子组结果用于机制判断，不把未校正的 p 值当作独立确证。
- 样本级表：[v7_s0_difficulty_samples.jsonl](v7_s0_difficulty_samples.jsonl)，SHA256 345d126b42411d948c8029a743ac576b421632ab8ac71e072b3ed4b7c18bd3b3。
