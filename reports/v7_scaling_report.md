# V7 Data Scaling / Diversity Rescue

## 固定比较

| Setting | Unique samples | Sample exposures | Optimizer steps |
|---|---:|---:|---:|
| Balanced-336 × 1 | 336 | 336 | 21 |
| Balanced-336 × 4 | 336 | 1,344 | 84 |
| Equiv-All-1393 × 1 | 1,393 | 1,393 | 88 |

后两项 exposure 接近；前者重复同一小集合，后者增加 unique query 并保留自然 A/B/C 分布。

## 主结果

| Setting | Subset | N | Candidate | Class | Delta | Paired bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Balanced-336 × 1 | Full Internal | 500 | 360/500 (72.00%) | 306/500 (61.20%) | -10.80% | [-14.20%, -7.60%] |
| Balanced-336 × 1 | Internal-Equiv | 248 | 156/248 (62.90%) | 151/248 (60.89%) | -2.02% | [-5.24%, 0.81%] |
| Balanced-336 × 1 | Internal-Singleton | 252 | 204/252 (80.95%) | 155/252 (61.51%) | -19.44% | [-25.00%, -14.29%] |
| Balanced-336 × 4 | Full Internal | 500 | 363/500 (72.60%) | 355/500 (71.00%) | -1.60% | [-4.00%, 0.80%] |
| Balanced-336 × 4 | Internal-Equiv | 248 | 160/248 (64.52%) | 159/248 (64.11%) | -0.40% | [-3.63%, 2.82%] |
| Balanced-336 × 4 | Internal-Singleton | 252 | 203/252 (80.56%) | 196/252 (77.78%) | -2.78% | [-6.35%, 0.79%] |
| Equiv-All-1393 × 1 | Full Internal | 500 | 359/500 (71.80%) | 371/500 (74.20%) | 2.40% | [-0.20%, 5.20%] |
| Equiv-All-1393 × 1 | Internal-Equiv | 248 | 153/248 (61.69%) | 162/248 (65.32%) | 3.63% | [-0.81%, 8.06%] |
| Equiv-All-1393 × 1 | Internal-Singleton | 252 | 206/252 (81.75%) | 209/252 (82.94%) | 1.19% | [-1.59%, 4.37%] |

## S0 Correct / Wrong

| Setting | S0 状态 | N | Candidate | Class | Delta | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Balanced-336 × 1 | correct | 176 | 83.52% | 81.25% | -2.27% | [-6.25%, 1.70%] |
| Balanced-336 × 1 | wrong | 72 | 12.50% | 11.11% | -1.39% | [-4.17%, 0.00%] |
| Balanced-336 × 4 | correct | 176 | 85.23% | 85.23% | 0.00% | [-3.98%, 3.98%] |
| Balanced-336 × 4 | wrong | 72 | 13.89% | 12.50% | -1.39% | [-6.94%, 4.17%] |
| Equiv-All-1393 × 1 | correct | 176 | 78.98% | 85.23% | 6.25% | [0.57%, 11.93%] |
| Equiv-All-1393 × 1 | wrong | 72 | 19.44% | 16.67% | -2.78% | [-9.72%, 4.17%] |

## S0 confidence tertile

| Setting | 置信度 | N | Candidate | Class | Delta | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Balanced-336 × 1 | low | 75 | 60.00% | 57.33% | -2.67% | [-8.00%, 2.67%] |
| Balanced-336 × 1 | mid | 92 | 60.87% | 59.78% | -1.09% | [-5.43%, 3.26%] |
| Balanced-336 × 1 | high | 81 | 67.90% | 65.43% | -2.47% | [-8.64%, 3.70%] |
| Balanced-336 × 4 | low | 75 | 60.00% | 60.00% | 0.00% | [-5.33%, 5.33%] |
| Balanced-336 × 4 | mid | 92 | 61.96% | 61.96% | 0.00% | [-6.52%, 6.52%] |
| Balanced-336 × 4 | high | 81 | 71.60% | 70.37% | -1.23% | [-6.17%, 2.47%] |
| Equiv-All-1393 × 1 | low | 75 | 54.67% | 61.33% | 6.67% | [0.00%, 13.33%] |
| Equiv-All-1393 × 1 | mid | 92 | 57.61% | 61.96% | 4.35% | [-3.26%, 11.96%] |
| Equiv-All-1393 × 1 | high | 81 | 72.84% | 72.84% | 0.00% | [-7.41%, 7.41%] |

## A/B/C

| Setting | 等价类型 | N | Candidate | Class | Delta | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Balanced-336 × 1 | A | 183 | 62.84% | 61.20% | -1.64% | [-4.37%, 1.09%] |
| Balanced-336 × 1 | B | 19 | 89.47% | 84.21% | -5.26% | [-15.79%, 0.00%] |
| Balanced-336 × 1 | C | 46 | 52.17% | 50.00% | -2.17% | [-13.04%, 8.70%] |
| Balanced-336 × 4 | A | 183 | 63.93% | 64.48% | 0.55% | [-3.28%, 4.37%] |
| Balanced-336 × 4 | B | 19 | 84.21% | 78.95% | -5.26% | [-15.79%, 0.00%] |
| Balanced-336 × 4 | C | 46 | 58.70% | 56.52% | -2.17% | [-10.87%, 6.52%] |
| Equiv-All-1393 × 1 | A | 183 | 60.66% | 65.03% | 4.37% | [-0.55%, 9.29%] |
| Equiv-All-1393 × 1 | B | 19 | 78.95% | 84.21% | 5.26% | [0.00%, 15.79%] |
| Equiv-All-1393 × 1 | C | 46 | 58.70% | 58.70% | 0.00% | [-10.87%, 13.04%] |

## 科研判断

- Balanced-336 从 1 轮增加到 4 轮后，Internal-Equiv 的 Class-Candidate delta 从 -2.02pt 改善到 -0.40pt，但仍未转为 Class 优势；近似相同 exposure 下，Equiv-All-1393×1 的 delta 为 +3.63pt。
- Equiv-All 的目标 subset 提升为 9/248，paired bootstrap 95% CI [-0.81pt, 8.06pt]，McNemar exact p=0.1496；Full Internal 同时为 +2.40pt，95% CI [-0.20pt, 5.20pt]，没有观察到明显整体退化。
- 证据方向支持“更多 unique query 与自然 A/B/C 分布”的组合条件比重复小集合更重要，不支持“只要增加 exposure 就足够”。由于 Equiv-All 同时改变 unique diversity 和类别分布，本实验不能进一步把两者因果拆开。
- 当前仅单 seed，两个主区间均跨 0；因此这是有方向一致性的 rescue 信号，不是已确证的 universal superiority。
- S0 confidence 在 Equiv-All 下呈 high 0.00pt、mid +4.35pt、low +6.67pt，但 S0 free-generation Wrong 组为 -2.78pt；difficulty 机制只得到部分支持，不满足 Weak-S0 的明确触发条件。

## 直接对比

- Balanced-336 × 4 相对 Balanced-336 × 1 的 Internal-Equiv delta 变化：1.61%。
- Equiv-All-1393 × 1 相对 Balanced-336 × 4 的 Internal-Equiv delta 变化：4.03%。
- 所有区间均来自同一样本配对的 seed 42、10,000 次 percentile bootstrap。单 seed 下不把区间重叠解释成等价性证明。
