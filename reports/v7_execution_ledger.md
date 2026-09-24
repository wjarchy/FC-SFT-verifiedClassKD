# V7 执行账本

> 当前执行基线：[ClassKD_V7_Lean_Execution_PRD.md](../ClassKD_V7_Lean_Execution_PRD.md)。
> 本账本只记录 V7 恢复与续跑，不把 V4/V5/V6 历史摘要冒充本次结果。

## 2026-09-24：灾后恢复启动

- 起始分支：`main`；起始提交：`07d72d62cb6a880388412382308567398e230256`。
- V7 工作分支：`recovery/v7-rescue`。
- 工作树起始状态：只有用户恢复的 `ClassKD_V7_Lean_Execution_PRD.md` 未跟踪；数据、模型、checkpoint、Teacher cache 和预测均不存在。
- GPU：NVIDIA GeForce RTX 3090 24GB；CUDA 与 BF16 可用。
- 可复用环境：Conda `wj`，Python 3.11.16、PyTorch 2.5.1+cu124、Transformers 4.57.6、PEFT 0.20.0、bitsandbytes 0.50.0。
- 旧代码回归：64 项测试全部通过。
- 固定数据 revision 已可访问：xLAM `26d14ebf...`、ToolACE cleaned `74befe8a...`、ToolACE schema `6bda777c...`。
- 已确认 Hugging Face 登录有效；下载缓存使用 `/root/autodl-tmp/hf_cache`。

### Formal 数据恢复调查

- xLAM 原始 60,000 条；旧严格筛选通过 19,642 条；精确问题去重后 18,251 条。
- 按现有 BFCL 排除和更严格规范化去重后为 17,924 条。
- ToolACE 严格恢复后为 2,103 条。
- 单用 xLAM 的结构比例无法解释历史 `A/B/C ≈ 1094/112/185`；旧构建器的 ToolACE+xLAM 双源抽样能得到接近该比例的自然分布。
- 剩余小差异指向旧机器未提交的 V6 等价构造实现，重点是递归数值表示、嵌套对象换序和安全默认值契约。恢复时只使用 PRD 允许的严格规则，不按模型结果挑样本。

### 下一步

1. 补齐 V7 专用 Formal 构建、K=4、功能类聚合和 KL 分解测试。
2. 完成 GPU 端到端 smoke 并冻结恢复口径。
3. 生成 Formal 数据及其 manifest，再恢复 Formal S0。
