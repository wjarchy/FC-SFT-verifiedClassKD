# Fast Pilot Implementation Plan

执行基线：`FC_ClassKD_Fast_Pilot_PRD_v4.md`
建立日期：2026-08-11

## 执行原则

- v4 是首轮唯一范围；v3 只用于核对定义与风险。
- 每个研究关键模块先补回归测试，再改实现。
- 数据、模型、Tokenizer、Prompt、候选池和运行配置必须可追溯。
- 任何 Gate 未通过都停止进入下一阶段，不用 KD 掩盖数据、Prompt 或 Verifier 错误。

## 阶段计划

### P0：静态正确性加固

- [x] 分离候选 surface string 与 parsed AST。
- [x] 修复 key-reorder 候选被字典相等/去重吞掉的问题。
- [x] 只允许省略与 Schema 明确 default 相等的 optional 参数。
- [x] 排除 `unknown` 后重新归一化并保持 KL 分解恒等式。
- [x] 嵌套参数候选按完整 surface string 去重。
- [x] 预测数量不一致时拒绝评测。
- [x] Qwen3 Prompt 强制 non-thinking，检查标签边界并拒绝 Gold 截断。
- [x] 明确现有 equivalence helper 不能单独计算 generation RobustnessDrop。

### P1：环境与数据（PRD Step 1）

- [x] 建立 Python 3.11 Conda 环境 `wj`。
- [ ] 修复 NVIDIA 驱动/用户态库版本失配。
- [ ] 冻结并记录完整 Python/CUDA 依赖。
- [ ] 确认首轮数据源后重建 processed data 与固定 splits。
- [ ] 输出数据版本、筛选参数、split IDs 和 SHA256 manifest。
- [ ] 完成 BFCL leakage check。

### P2：Verifier 与 Equivalence Gate

- [ ] 生成覆盖四主类的盲标样本与标注说明。
- [ ] 完成人工 Verifier audit：correct precision ≥98%，overall accuracy ≥95%。
- [ ] 冻结真正改变模型可见输入的 paired equivalence protocol。
- [ ] 为候选表面偏好与生成鲁棒性分别定义指标。

### P3：GPU Step 2

- [ ] 在 Qwen3-0.6B 真实 tokenizer/model 上通过 Prompt、non-thinking 和截断审计。
- [ ] 运行 4 条 fixture Smoke Test。
- [ ] 实现并通过 32 条以内 Tiny Overfit：训练集 AST ≥90%。
- [ ] 验证端到端 Class KD → LoRA 梯度。

### P4：Pre-Diagnostic 与正式 Fast Pilot

- [ ] 200–300 条 Teacher/Student/Candidate Pre-Diagnostic。
- [ ] 通过 Teacher gap、candidate quality、equivalence coverage Gate。
- [ ] 训练公共 S0，冻结 K=6 Candidate Pool，缓存 8B Teacher scores。
- [ ] 同预算训练 SFT-cont / Candidate KD / Class KD。
- [ ] 统一完成 Internal / Equivalence / BFCL 与 Go/Weak-Go/No-Go。

## 首轮明确不做

第二 seed、Token KD、Class-Hard、Label Smoothing、ACEBench、K 敏感性、梯度冲突、
Hierarchical KD、multi-turn、RL/DPO/GRPO 均不阻塞 Fast Pilot。
