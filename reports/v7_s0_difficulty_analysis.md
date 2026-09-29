# V7 S0 Difficulty Analysis

## 结论摘要

- 置信度桶按训练侧 Equiv-All 的 S0 正确功能类概率预先冻结，不使用 Internal outcome 定阈值。
- 从高置信度到低置信度的 Class - Candidate 依次为：-2.47%, -1.09%, -2.67%。
- 难度上升时 delta 单调不减：false。
- S0 Wrong 与 S0 Correct 的相对 delta 差：0.88%。

## 总体恢复基线

| Subset | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full Internal | 500 | 360/500 (72.00%) | 306/500 (61.20%) | -10.80% | [-14.20%, -7.60%] | 63 / 9 | 0.0000 |
| Internal-Equiv | 248 | 156/248 (62.90%) | 151/248 (60.89%) | -2.02% | [-5.24%, 0.81%] | 10 / 5 | 0.3018 |
| Internal-Singleton | 252 | 204/252 (80.95%) | 155/252 (61.51%) | -19.44% | [-25.00%, -14.29%] | 53 / 4 | 0.0000 |

## View A：S0 Correct vs Wrong

| S0 状态 | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 176 | 147/176 (83.52%) | 143/176 (81.25%) | -2.27% | [-6.25%, 1.70%] | 9 / 5 | 0.4240 |
| wrong | 72 | 9/72 (12.50%) | 8/72 (11.11%) | -1.39% | [-4.17%, 0.00%] | 1 / 0 | 1.0000 |

## View B：S0 fixed-support confidence

阈值：q1=0.58942356，q2=0.61558665；分桶规则为 low ≤ q1，q1 < mid ≤ q2，high > q2。

| 置信度 | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 75 | 45/75 (60.00%) | 43/75 (57.33%) | -2.67% | [-8.00%, 2.67%] | 3 / 1 | 0.6250 |
| mid | 92 | 56/92 (60.87%) | 55/92 (59.78%) | -1.09% | [-5.43%, 3.26%] | 3 / 2 | 1.0000 |
| high | 81 | 55/81 (67.90%) | 53/81 (65.43%) | -2.47% | [-8.64%, 3.70%] | 4 / 2 | 0.6875 |

## A/B/C

| 等价类型 | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 183 | 115/183 (62.84%) | 112/183 (61.20%) | -1.64% | [-4.37%, 1.09%] | 5 / 2 | 0.4531 |
| B | 19 | 17/19 (89.47%) | 16/19 (84.21%) | -5.26% | [-15.79%, 0.00%] | 1 / 0 | 1.0000 |
| C | 46 | 24/46 (52.17%) | 23/46 (50.00%) | -2.17% | [-13.04%, 8.70%] | 4 / 3 | 1.0000 |

## Teacher 类内 entropy（同步保存，主分析之后解释）

阈值：q1=0.96580232，q2=0.98335794；由训练侧 Equiv-All Teacher cache 预先冻结。

| Teacher entropy | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 97 | 70/97 (72.16%) | 67/97 (69.07%) | -3.09% | [-10.31%, 4.12%] | 7 / 4 | 0.5488 |
| mid | 67 | 37/67 (55.22%) | 37/67 (55.22%) | 0.00% | [-4.48%, 4.48%] | 1 / 1 | 1.0000 |
| high | 84 | 49/84 (58.33%) | 47/84 (55.95%) | -2.38% | [-5.95%, 0.00%] | 2 / 0 | 0.5000 |

## 统计口径

- 所有比较均为同一样本上的配对比较；Delta 为 Class accuracy - Candidate accuracy。
- 区间是 seed 42、10,000 次的 paired percentile bootstrap 95% CI；McNemar p 是 discordant pair 的双侧精确二项检验。
- 单 seed、多个小子组的区间较宽；子组结果用于机制判断，不把未校正的 p 值当作独立确证。
- 样本级表：[v7_s0_difficulty_samples.jsonl](v7_s0_difficulty_samples.jsonl)，SHA256 d07490bff875e2745f3db1e9f3ca62af5fb750ebb104d32d1e293193251197f3。
