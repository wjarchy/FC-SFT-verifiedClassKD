# V7 S0 Difficulty Analysis

## 结论摘要

- 置信度桶按训练侧 Equiv-All 的 S0 正确功能类概率预先冻结，不使用 Internal outcome 定阈值。
- 从高置信度到低置信度的 Class - Candidate 依次为：0.00%, 4.35%, 6.67%。
- 难度上升时 delta 单调不减：true。
- S0 Wrong 与 S0 Correct 的相对 delta 差：-9.03%。

## 总体恢复基线

| Subset | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full Internal | 500 | 359/500 (71.80%) | 371/500 (74.20%) | 2.40% | [-0.20%, 5.20%] | 17 / 29 | 0.1038 |
| Internal-Equiv | 248 | 153/248 (61.69%) | 162/248 (65.32%) | 3.63% | [-0.81%, 8.06%] | 11 / 20 | 0.1496 |
| Internal-Singleton | 252 | 206/252 (81.75%) | 209/252 (82.94%) | 1.19% | [-1.59%, 4.37%] | 6 / 9 | 0.6072 |

## View A：S0 Correct vs Wrong

| S0 状态 | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 176 | 139/176 (78.98%) | 150/176 (85.23%) | 6.25% | [0.57%, 11.93%] | 7 / 18 | 0.0433 |
| wrong | 72 | 14/72 (19.44%) | 12/72 (16.67%) | -2.78% | [-9.72%, 4.17%] | 4 / 2 | 0.6875 |

## View B：S0 fixed-support confidence

阈值：q1=0.58942356，q2=0.61558665；分桶规则为 low ≤ q1，q1 < mid ≤ q2，high > q2。

| 置信度 | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 75 | 41/75 (54.67%) | 46/75 (61.33%) | 6.67% | [0.00%, 13.33%] | 1 / 6 | 0.1250 |
| mid | 92 | 53/92 (57.61%) | 57/92 (61.96%) | 4.35% | [-3.26%, 11.96%] | 5 / 9 | 0.4240 |
| high | 81 | 59/81 (72.84%) | 59/81 (72.84%) | 0.00% | [-7.41%, 7.41%] | 5 / 5 | 1.0000 |

## A/B/C

| 等价类型 | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 183 | 111/183 (60.66%) | 119/183 (65.03%) | 4.37% | [-0.55%, 9.29%] | 7 / 15 | 0.1338 |
| B | 19 | 15/19 (78.95%) | 16/19 (84.21%) | 5.26% | [0.00%, 15.79%] | 0 / 1 | 1.0000 |
| C | 46 | 27/46 (58.70%) | 27/46 (58.70%) | 0.00% | [-10.87%, 13.04%] | 4 / 4 | 1.0000 |

## Teacher 类内 entropy（同步保存，主分析之后解释）

阈值：q1=0.96580232，q2=0.98335794；由训练侧 Equiv-All Teacher cache 预先冻结。

| Teacher entropy | N | Candidate | Class | Delta | Bootstrap 95% CI | Candidate-only / Class-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 97 | 72/97 (74.23%) | 76/97 (78.35%) | 4.12% | [-3.09%, 11.34%] | 4 / 8 | 0.3877 |
| mid | 67 | 34/67 (50.75%) | 37/67 (55.22%) | 4.48% | [-4.48%, 14.93%] | 4 / 7 | 0.5488 |
| high | 84 | 47/84 (55.95%) | 49/84 (58.33%) | 2.38% | [-3.57%, 9.52%] | 3 / 5 | 0.7266 |

## 统计口径

- 所有比较均为同一样本上的配对比较；Delta 为 Class accuracy - Candidate accuracy。
- 区间是 seed 42、10,000 次的 paired percentile bootstrap 95% CI；McNemar p 是 discordant pair 的双侧精确二项检验。
- 单 seed、多个小子组的区间较宽；子组结果用于机制判断，不把未校正的 p 值当作独立确证。
- 样本级表：[v7_s0_difficulty_samples.jsonl](v7_s0_difficulty_samples.jsonl)，SHA256 493bd342b7be7f9ee6c26cc3d4a07c141407d91b07a579e2d43138aff9909015。
