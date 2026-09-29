# V7 Teacher Entropy Analysis

Teacher 正确功能类内部的 normalized entropy 三分位阈值由训练侧 Equiv-All Teacher cache 预先冻结；Internal outcome 没有参与阈值选择。

| Setting | Teacher entropy | N | Candidate | Class | Delta | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Balanced-336 × 1 | low | 97 | 72.16% | 69.07% | -3.09% | [-10.31%, 4.12%] |
| Balanced-336 × 1 | mid | 67 | 55.22% | 55.22% | 0.00% | [-4.48%, 4.48%] |
| Balanced-336 × 1 | high | 84 | 58.33% | 55.95% | -2.38% | [-5.95%, 0.00%] |
| Balanced-336 × 4 | low | 97 | 76.29% | 73.20% | -3.09% | [-9.28%, 3.09%] |
| Balanced-336 × 4 | mid | 67 | 55.22% | 55.22% | 0.00% | [-5.97%, 5.97%] |
| Balanced-336 × 4 | high | 84 | 58.33% | 60.71% | 2.38% | [-2.38%, 7.14%] |
| Equiv-All-1393 × 1 | low | 97 | 74.23% | 78.35% | 4.12% | [-3.09%, 11.34%] |
| Equiv-All-1393 × 1 | mid | 67 | 50.75% | 55.22% | 4.48% | [-4.48%, 14.93%] |
| Equiv-All-1393 × 1 | high | 84 | 55.95% | 58.33% | 2.38% | [-3.57%, 9.52%] |

## 科研判断

- Equiv-All-1393×1 的 low/mid/high entropy delta 分别为 +4.12pt、+4.48pt、+2.38pt，没有随 entropy 升高而增强。
- Balanced-336×4 虽从 low -3.09pt 变到 high +2.38pt，但该 setting 总体没有 Class 优势；三种训练设置之间方向不一致。
- 因而 Teacher 类内 entropy 不能解释本轮 Equiv-All rescue，最多保留为未确证的探索性异质性。

## 解释边界

- 这里检验 Class - Candidate 是否随 Teacher 类内偏好不稳定性上升而改善。
- 分桶是机制分析，不更换 Full Internal 或 Internal-Equiv 主指标。
- 多个子组共享样本和模型，未做多重比较校正；结论以方向一致性、效应量和区间共同判断。
