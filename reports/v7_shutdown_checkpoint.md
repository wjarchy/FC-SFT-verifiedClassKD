# V7 关机保存点

## 当前安全边界

用户要求关机，执行已停在一个完整实验边界：

- Formal 数据、Formal S0、K=4 pool、Teacher cache、S0 fixed-support cache 均已完成并已推送 manifest/report。
- S0 confidence 与 Teacher entropy 三分位已在任何恢复 KD outcome 前冻结。
- Balanced-336×1 Candidate KD 已完成整个 1 epoch / 21 optimizer steps。
- Balanced-336×1 Candidate 尚未评测；对应 Class KD 尚未启动。
- 未启动 scaling、gradient 或 Weak-S0。

因此不存在半个 optimizer step、未关闭的 output file、只看一侧模型 outcome 后改方案或 Candidate/Class 使用不同数据的问题。

## Candidate checkpoint

| 项目 | 值 |
|---|---|
| 路径 | outputs/v7_kd/balanced336_e1_candidate |
| Final checkpoint | checkpoint-step-000021 |
| Exposures / steps | 336 / 21 |
| Mean KD loss | 0.03708983 |
| Peak GPU | 18,363.7 MiB |
| Adapter SHA256 | e458945adc6a16c80d11f306f8ab761f1cf812734c26e3708c9704bb4eeea0a2 |
| Run manifest SHA256 | e10657d4387135fb9eb34698561c0d54818ca59e19ac7ab9378d98ffba912a67 |
| Training summary SHA256 | 42928115342c2531867725e768c9d453762380538c06358e4a8f000bed53289f |
| Training log SHA256 | 20c33f309a60c617385230a846632da42a1eaf66c6a4b9cd53899200ce303255 |
| Optimizer SHA256 | d14458c2c48dfbf71e634f4ec2f38fd5570aba7c92247d9de7ded61fed6189cf |
| RNG state SHA256 | a4fb68dee6f8613f425fd070fe8e48579779e95d42787c13bc64ed2dfc81c699 |
| Trainer state SHA256 | 252eca41016ce96f8350488e8668b1849f2f8da7da2f691b73d303306bdf145b |

## 下次恢复顺序

1. 确认 branch 为 recovery/v7-rescue，并拉取最新远端提交。
2. 检查根 artifacts_manifest.json、configs/v7_experiment_matrix.json 和本报告。
3. 不重跑 Candidate；先启动 Balanced-336×1 Class：

       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src /root/miniconda3/envs/wj/bin/python scripts/v7_train_kd.py --config v7_rtx3090 --objective class --data-path data/processed/v7/balanced_336.jsonl --pool-path data/candidates/v7/balanced_336_k4.jsonl --teacher-cache data/cache/v7/teacher/teacher_equiv_all_scores.jsonl --s0-dir outputs/v7_formal_s0_seed42 --output outputs/v7_kd/balanced336_e1_class --epochs 1 --batch-size 16 --learning-rate 0.0002 --temperature 2 --loss-weight 1 --alpha 1 --max-length 2560 --seed 42 --resume auto

4. Class 完成后，用冻结的 batched evaluator 分别生成两个模型的 Full Internal predictions。
5. 运行 scripts/v7_analyze_s0_difficulty.py，提交并推送 Priority 1 结果；随后继续 scaling。

Candidate 目录包含 step-0 与 step-21 的 adapter、optimizer、RNG 和 trainer state，可用 resume auto 自检已完成状态；无需重新训练。
