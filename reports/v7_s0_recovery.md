# V7 Formal S0 恢复报告

## 结论

Formal S0 已按 V6/V7 Formal Data 定义重训并固定。选中 checkpoint-step-000564；Internal AST 为 80.40%，Internal-Equiv AST 为 70.97%。训练、选模和预测均已落盘并用 SHA256 绑定，后续 Candidate/Class 分支共用这一份 S0。

## 固定定义

| 项目 | 值 |
|---|---|
| Base model | Qwen3-1.7B |
| Revision | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| Train / Validation | 3,000 / 300 |
| Seed | 42 |
| QLoRA | 4-bit NF4，r=16，alpha=32，dropout=0.05 |
| LR / effective batch | 2e-4 / 16 |
| Epoch / optimizer steps | 3 / 564 |
| Max length | 2,560 |
| 选模规则 | 最高 Validation AST；并列时较低 Validation loss，再并列时较早 epoch |

## 训练与选模

| Epoch | Step | Train loss | Validation loss | Validation AST |
|---:|---:|---:|---:|---:|
| 1 | 188 | 0.063386 | 0.040184 | 235/300 = 78.33% |
| 2 | 376 | 0.030121 | 0.029397 | 244/300 = 81.33% |
| 3 | 564 | 0.017655 | 0.031723 | 248/300 = 82.67% |

epoch 3 的 Validation loss 高于 epoch 2，但预注册 primary selection metric 是 AST accuracy，因此选择 epoch 3，不在看到结果后切换指标。训练峰值显存为 6,868.4 MiB。

## Internal 预测

| Split | N | AST correct | AST accuracy | Wrong argument | Wrong function | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| Full Internal | 500 | 402 | 80.40% | 88 | 10 | 0 |
| Internal-Equiv | 248 | 176 | 70.97% | 68 | 4 | 0 |

Internal-Equiv 预测由 Full Internal 结果按冻结 subset ID 顺序派生，并逐条验证记录内容一致；没有对同一批样本执行第二次生成。

## 资产与哈希

| 资产 | 路径 | SHA256 |
|---|---|---|
| 训练 run manifest | outputs/v7_formal_s0_seed42/run_manifest.json | 3571f0eeb9ae8332797ba1b4ad2e2a6d86b5d3515c8a7ccada184fd862c3d446 |
| training summary | outputs/v7_formal_s0_seed42/training_summary.json | a59e7681fc9baa504468176003aec71ec789239726987dd5422255a97b8b11a3 |
| selected S0 | outputs/v7_formal_s0_seed42/selected_s0.json | 12e75369d081605aff775f95aae3535efe50eaed931b9938cd9604e08c6446d9 |
| selected adapter | outputs/v7_formal_s0_seed42/checkpoint-step-000564/adapter/adapter_model.safetensors | 6bb07558d42735f32ae1676fc3eaae64c786523f8bb55bb4078a90b9ef06d2d5 |
| Internal predictions | data/candidates/v7/s0_predictions/s0_internal_predictions.jsonl | e3d8c454432c2d083987e3437fa23caa99064dada13829356820c9f8d3c7975b |
| Internal-Equiv predictions | data/candidates/v7/s0_predictions/s0_internal_equiv_predictions.jsonl | 12e5dbacadf12339a32436d4a7db9edd7cf5888dcf1698082416b00acae37f7c |

模型、checkpoint 和预测 JSONL 由 Git 忽略；根 artifacts_manifest.json 保存路径、哈希和生成命令。

## 科研边界

- 该 S0 是所有 V7 KD 分支共享的初始化，不能按后续效果另换 checkpoint。
- S0 Internal-Equiv 比 Full Internal 难约 9.43 个百分点，说明 difficulty 分层有实际动态范围；这不预言 Class KD 的方向。
- Weak-S0 是否执行只由后续预注册 S0 Difficulty 趋势决定。
