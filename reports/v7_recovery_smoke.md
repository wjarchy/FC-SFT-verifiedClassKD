# V7 恢复 Smoke 报告

## 结论

Phase 0 通过，可以启动 Formal S0。代码、数值恒等式、真实 1.7B 4-bit QLoRA 前反向、Class KD 端到端梯度、adapter 保存和最长样本显存均正常。正式训练尚未在本报告前启动。

## 执行基线

- 分支：`recovery/v7-rescue`
- smoke 前代码提交：`4abec37efb36ba21c17ec6658d9e547b0e58fd9c`
- GPU：NVIDIA GeForce RTX 3090 24,576 MiB；driver 580.76.05；BF16 可用
- Python 3.11.16
- PyTorch 2.5.1+cu124；CUDA 12.4
- Transformers 4.57.6；PEFT 0.20.0；bitsandbytes 0.50.0
- Datasets 5.0.1；Accelerate 1.14.0

## 固定模型

Student 为 `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，存放于 `/root/autodl-tmp/models/Qwen3-1.7B`。主要文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `model-00001-of-00002.safetensors` | `169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5` |
| `model-00002-of-00002.safetensors` | `912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9` |
| `model.safetensors.index.json` | `0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d` |
| `config.json` | `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197` |
| `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |

离线加载确认：Qwen3、hidden size 2,048、28 layers、config vocab 151,936；tokenizer 可用。Teacher 已固定为 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`，本阶段未加载 Teacher。

## 代码与数值 Gate

- 73 项 pytest 全部通过，覆盖 V7 等价规则、K=4 功能类、Candidate/Class loss 反向传播和显式数据路径绑定。
- 本轮新增/修改的 Python 文件通过 Ruff 与 Python compile；全库既有文件仍有历史 lint 债务，本轮不做无关重排。
- 固定示例的 KL 分解：`KL_candidate=0.149475211619`、`KL_inter=0.022730254805`、`KL_intra=0.126744956814`；恒等式残差为 `2.776e-17`。

## 真实 GPU smoke

命令使用 `scripts/smoke_test.py --config rtx3090 --model /root/autodl-tmp/models/Qwen3-1.7B --data-source fixture --samples 4`，并强制离线加载。

- 四条 fixture 的候选构造与 evaluator 可运行；
- SFT loss：1.4807，有限；
- SFT backward 后 LoRA 有非零梯度，冻结 base 无梯度；
- Class KD loss：0.002017，端到端经过 candidate sequence scores 和 class aggregation 后 LoRA 梯度非零；
- 临时 adapter 保存成功；
- 峰值 GPU memory：4,256.8 MiB；
- `SMOKE_OK=true`。

## Formal 长度与显存 Gate

旧 2,048 上限会使 `xlam_4469` 的 Gold 被截断，runner 已按设计抛错，没有静默训练。全量 native chat template 审计如下：

| 划分 | P50 | P95 | P99 | Max | 超过 2,048 |
|---|---:|---:|---:|---:|---:|
| Train | 578 | 1,036 | 1,257 | 2,392 | 2 |
| Validation | 596 | 1,080 | 1,442 | 1,893 | 0 |
| Internal | 604 | 1,043 | 1,396 | 1,965 | 0 |

`configs/v7_rtx3090.yaml` 将 V7 的 `max_length` 固定为 2,560。Equiv-All、Balanced-336、Internal-Equiv 的全部 K=4 编码最大分别为 2,059、2,059、1,967；没有 candidate 超过 2,560。最长 Train 样本 `xlam_4469` 的真实 2,392-token 前反向通过：loss 0.248720，峰值 8,047.3 MiB。

Formal 预编码最终通过：Train=3,000、Validation=300，无截断；三轮、batch 16（micro=1、gradient accumulation=16），每轮 188 optimizer steps，总计 564。该上限在任何正式结果产生前冻结，并由所有后续分支共用。
