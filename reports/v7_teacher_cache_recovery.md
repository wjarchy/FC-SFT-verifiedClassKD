# V7 Teacher Cache 恢复报告

## 结论

固定 Qwen3-8B 已对冻结 K=4 candidate pool 完成 sequence scoring。Equiv-All 1,393 行和 Internal-Equiv 248 行都保存 candidate score/probability、class probability、正确类内部概率及 entropy；所有概率与哈希校验通过。

## 固定定义

| 项目 | 值 |
|---|---|
| Teacher | Qwen3-8B |
| Revision | b968826d9c46dd6066d109eabc6255188de91218 |
| Precision | BF16 |
| Candidate order | Gold / Equivalent / Wrong Argument / Wrong Function |
| Class index | [0, 0, 1, 2] |
| Temperature | 2 |
| Length normalization alpha | 1 |
| Max length | 2,560 |
| Candidate micro-batch | 2 |

## 工程恢复

首次 candidate batch 4 在第 173 个长样本的完整词表 log-softmax 处需要额外 3.39 GiB，触发 OOM；异常 traceback 又保留了相关 tensor，使进程内 fallback 不可靠。修复后 scorer 使用显式 candidate micro-batch 并把该值写入 run signature。

正式 cache 从零统一以 batch 2 重算；未复用首次 batch-4 的 172 行 partial。90 项回归测试通过，正式运行峰值显存 21,595.4 MiB。

## 完整性与 sanity

| Split | N | Correct class argmax | Correct mass min / median / max | Normalized intra entropy median |
|---|---:|---:|---:|---:|
| Equiv-All | 1,393 | 1,393/1,393 | 0.5241 / 0.6982 / 0.8618 | 0.9758 |
| Internal-Equiv | 248 | 248/248 | 0.5527 / 0.7066 / 0.8383 | 0.9762 |

- 每行 candidate probability 长度 4、class probability 长度 3，值有限且各自和为 1。
- normalized intra-class entropy 全部位于 [0,1]。
- Teacher class argmax 只作为 cache sanity；V7 没有重新启动 Teacher Gate。
- entropy 整体偏高，但机制分析使用在 Equiv-All 预先冻结的相对三分位，不按 outcome 改阈值。

## 资产

| 资产 | 路径 | SHA256 |
|---|---|---|
| Manifest | data/cache/v7/teacher/manifest.json | 7c16f7777c649a3710dfeacf97af77a8c56bf8479b73eea799d716b0a4b93c43 |
| Equiv-All scores | data/cache/v7/teacher/teacher_equiv_all_scores.jsonl | 124c3e7abd70fbea67bd3772550fd696dd6a55cb20829ba48270906e6c48d06c |
| Internal-Equiv scores | data/cache/v7/teacher/teacher_internal_equiv_scores.jsonl | ee5e5bd4f9c23b1daf90b0427c8ad68375c600facb821bd81b08efc81a05b2b9 |

大型 score JSONL 由 Git 忽略；manifest、报告、生成参数和哈希进入 Git。
