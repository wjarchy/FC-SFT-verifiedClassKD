# V7 Formal 数据恢复报告

## 结论

V7 必需的 Formal 数据已恢复并冻结，可继续训练 Formal S0。恢复版本满足 PRD 的规模和结构约束；唯一已知偏差是 Train 的 A 类及 `Equiv-All` 比历史约数各多 2 条。由于旧机器未提交精确行 ID 与最终构建脚本，本报告不把恢复版本表述为逐样本复刻。

## 固定输入与命令

- xLAM revision：`26d14ebfe18b1f7b524bd39b404b50af5dc97866`
- ToolACE cleaned revision：`74befe8a24c760ccad2b5d58bcf35d89b3bbd2be`
- ToolACE schema revision：`6bda777c88d21e5a204703c1ee45597a8fa4f734`
- BFCL 泄漏检查 revision：`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- 随机种子：42
- 生成命令：`HF_HOME=/root/autodl-tmp/hf_cache PYTHONPATH=src /root/miniconda3/envs/wj/bin/python scripts/v7_build_formal_data.py --seed 42`
- 机器可读定义与逐文件哈希：[`data/processed/v7/manifest.json`](../data/processed/v7/manifest.json)

## 恢复口径

沿用仓库已有的 ToolACE+xLAM 双源、严格 schema 检查、BFCL 排除和规范化问题去重。等价规则优先级固定为 C→B→A：

- A：JSON 对象键顺序改变，包括嵌套对象；
- B：数值 schema 下语义相同的 JSON number 表示改变；
- C：仅删除有明确契约依据的默认参数，或按 schema 处理显式无序集合。

xLAM 默认值只在“非空、Python 类型完全相同、值与已通过 schema 的 Gold 完全相同”时接受。恢复过程不读取任何 Student/Teacher 输出。

## 数据统计

| 划分 | 行数 | ToolACE | xLAM | A | B | C | singleton |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 3,000 | 1,343 | 1,657 | 1,096 | 112 | 185 | 1,607 |
| Validation | 300 | 131 | 169 | 110 | 10 | 17 | 163 |
| Internal | 500 | 245 | 255 | 183 | 19 | 46 | 252 |
| Internal-Equiv | 248 | 96 | 152 | 183 | 19 | 46 | 0 |
| Equiv-All | 1,393 | 434 | 959 | 1,096 | 112 | 185 | 0 |
| Balanced-336 | 336 | 55 | 281 | 112 | 112 | 112 | 0 |
| Reserve | 14,229 | 333 | 13,896 | 5,501 | 928 | 1,523 | 6,277 |

Internal 从与 Train/Validation 相同的校准池分层抽取 248 条等价样本和 252 条 singleton，避免完整剩余池被 xLAM 占比主导。Train、Validation、Internal 之间不存在 ID 重叠。

## 与历史控制数的偏差

| 指标 | V7 PRD 历史约数 | 恢复值 | 偏差 |
|---|---:|---:|---:|
| A | 1,094 | 1,096 | +2 |
| B | 112 | 112 | 0 |
| C | 185 | 185 | 0 |
| Equiv-All | 1,391 | 1,393 | +2 |

4,650 条校准池是在新模型运行前，仅依据历史聚合控制数选择的。该恢复定义现已冻结；后续不会为了提高 Class KD 指标而更改数据。

## 完整性验证

- 八个 JSONL 的 SHA256、字节数、行数、唯一 ID、来源和类别计数均与 manifest 一致。
- `equiv_all`、`internal_equiv`、`balanced_336` 共 1,977 条逐条通过 K=4 构造：Gold 与 Equivalent 同一功能类，Wrong Argument 与 Wrong Function 为另外两个不同功能类，四个 surface 均唯一。
- 70 项单元测试通过，包括 KL 精确分解 `KL_candidate = KL_inter + KL_intra` 和两种 KD loss 的有限梯度检查。

## 风险边界

恢复版本可以支持 V7 内部的公平受控比较，但不能用来声称精确复现旧机器某一条未保存 prediction。论文报告必须明确它是依据固定 revision、历史规则与聚合控制数重建的新冻结 cohort。
