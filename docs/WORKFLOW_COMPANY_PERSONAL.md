# 公司 PC ↔ 个人 PC ↔ 实验室 协同工作方案

> 项目：`fc_class_kd`（Function Calling Class KD — Fast Pilot）  
> GitHub：`https://github.com/wjarchy/FC-SFT-verifiedClassKD`  
> 更新：2026-08-10

---

## 1. 三台环境分工

| 环境 | 网络 | 适合做什么 | 不适合做什么 |
|------|------|------------|--------------|
| **公司 PC** | 严格内网，GitHub push/pull 易失败 | 读文档、改代码（本地）、跑 `pytest`、Verifier、无 GPU 任务 | `git push`、大模型训练、HF 大文件下载（视策略而定） |
| **个人 PC** | 正常公网 | `git push/pull`、HF 登录/xLAM、`build_data.py`、协调同步 | 长期 GPU 实验（若无卡） |
| **实验室 2080 Ti** | 通常较好 | Smoke test、训练、评测、`git clone`（若网络允许） | 依赖公司 GitHub 推送 |

---

## 2. 我们今天在公司 PC 上试过什么（及结果）

### 2.1 HTTPS + Personal Access Token

| 步骤 | 结果 |
|------|------|
| `git push` via `https://github.com/wjarchy/...` | 失败 |
| 错误 | `curl 55 Send failure` / `HTTP 302` / 连接中途断开 |
| 登录框不弹出 | Windows 凭据里已有过期项 `git:https://github.com`，静默失败 |

**结论**：公司网络 + HTTPS 推送不可靠；凭据问题叠加网络问题。

### 2.2 SSH（端口 443）

| 步骤 | 结果 |
|------|------|
| 生成密钥 | `~/.ssh/id_ed25519_github` + `~/.ssh/config`（`ssh.github.com:443`） |
| 公钥加到 GitHub | 认证阶段可连上 |
| `git push` | 对象打包完成后传输中断：`Permission denied` / `hung up` |

**结论**：443 上 SSH 能握手，但 **push 数据流被公司策略拦截**；公司 PC 不宜作为 push 端。

### 2.3 Git Bundle（离线包）

| 步骤 | 结果 |
|------|------|
| `git bundle create fc_class_kd.bundle main` | 成功（约 26KB） |
| 内容 | **仅 git 已跟踪文件**（源码与配置），不含 `data/splits` 等大文件 |

**结论**：作为 **绕过公司 push 限制** 把 commit 带到个人 PC 的手段，**有效**。

---

## 3. 最终选择的工作流（推荐）

```text
                    ┌─────────────────────────────────────┐
                    │           GitHub（代码真相源）         │
                    │  wjarchy/FC-SFT-verifiedClassKD     │
                    └──────────────▲───────────────┬──────┘
                                   │ push/pull     │ clone/pull
              个人 PC（主开发+推送） │               │ 实验室 2080 Ti
                    ┌──────────────┴───┐       ┌────┴────────────┐
                    │ 写代码、push、     │       │ 训练、评测、     │
                    │ build_data、HF   │       │ smoke test      │
                    └──────────┬───────┘       └─────────────────┘
                               │
                    bundle / 网盘（公司 push 失败时）
                               │
                    ┌──────────┴───────┐
                    │   公司 PC        │
                    │ 改代码、pytest   │
                    │ pull/clone 或    │
                    │ 收 bundle        │
                    └──────────────────┘

        大文件（数据、模型、outputs）──► 数据云盘 / U盘（不进 git）
```

| 内容类型 | 同步方式 | 说明 |
|----------|----------|------|
| 源码、配置、测试 | **GitHub** | 个人 PC push，其他机器 `clone` / `pull` |
| 公司 PC 无法 push 时 | **Git bundle** | 个人 PC 生成或接收 bundle，再 push |
| `data/processed/`、`data/splits/` | **云盘 / U盘** 或 **个人 PC 重跑** | 默认 **不进 git**（见 `.gitignore`） |
| 模型权重、训练 `outputs/` | **云盘 / 实验室本地** | 绝不进 git |

---

## 4. 重跑 `build_data.py` 能否与当前划分一致？

### 4.1 可以一致的条件（确定性）

划分逻辑使用 **固定 seed**（默认 `42`）+ 对 `all_records` 下标 shuffle，**在以下条件全部相同时会得到相同 splits**：

| 条件 | 当前公司 PC 状态 |
|------|------------------|
| `--seed` | `42` |
| `--sources` | 当时仅成功 **`toolace`**（无 xLAM） |
| 数据源版本 | `tryumanshow/ToolACE-Qwen-cleaned`（HF 上若更新则可能变） |
| 筛选代码版本 | 与当前 `adapters.py` / `filters.py` 相同 |
| `all_records.jsonl` 内容 | 2688 条（ToolACE-only） |
| split 目标数量 | 不足时 **按比例自动缩小**（与当时行为一致） |

**在个人 PC 上复现「当前」划分的最稳命令**（ToolACE-only）：

```bash
cd fc_class_kd
python scripts/build_data.py --sources toolace --seed 42 \
  --train-size 2000 --val-size 300 --internal-test-size 500 \
  --diagnostic-size 250 --equivalence-size 200
```

若 HF 数据集与代码未变，应得到与当前相近的 manifest：

| split | 当前约数 |
|-------|---------|
| train | 1654 |
| val | 248 |
| internal_test | 413 |
| diagnostic | 206 |
| equivalence_diagnostic | 165 |

### 4.2 会不一致的情况

- 个人 PC 加上 **xLAM**（`--sources xlam toolace`）→ 池子变大，划分 **必然变化**（这是预期升级，不是 bug）
- HF 上游数据集更新
- 修改 adapter / filter / 去重逻辑
- 使用不同的 `--train-size` 等且总量足够、不再触发「按比例缩小」

### 4.3 100% 一致的做法（推荐实验室用）

从公司 PC 或云盘拷贝这两类路径（不依赖重跑）：

```text
data/processed/all_records.jsonl
data/splits/*.jsonl
```

或在个人 PC 已有 `all_records.jsonl` 后只重新切分（不重新下载）：

```bash
python scripts/build_data.py --skip-download --seed 42 \
  --train-size 2000 --val-size 300 --internal-test-size 500 \
  --diagnostic-size 250 --equivalence-size 200
```

---

## 5. 个人 PC 操作清单（从手机里的 bundle 开始）

### 5.1 把代码推到 GitHub（首次）

```bash
# 1. 从 bundle 恢复仓库
git clone /path/to/fc_class_kd.bundle -b main fc_class_kd
cd fc_class_kd

# 2. 配置远程。若 bundle clone 已把 bundle 路径设为 origin，用 set-url
git remote set-url origin git@github.com:wjarchy/FC-SFT-verifiedClassKD.git
# 若完全没有 origin，则改用：
# git remote add origin git@github.com:wjarchy/FC-SFT-verifiedClassKD.git
# 或 HTTPS + PAT：
# git remote set-url origin https://github.com/wjarchy/FC-SFT-verifiedClassKD.git

# 3. 推送
git push -u origin main

# 4. 确认
git log -1
git branch -vv    # 应显示 [origin/main]
```

浏览器打开仓库，应能看到 `Initial Fast Pilot scaffold...` 等文件。

### 5.2 环境安装

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/doctor.py
pytest -q
```

### 5.3 数据（二选一）

**A. 与当前公司 PC 完全一致（推荐）**

从公司 PC / 云盘 / 手机拷贝：

- `data/processed/all_records.jsonl`
- `data/splits/*.jsonl`

**B. 在个人 PC 重新生成**

```bash
hf auth login   # 若需要 xLAM
python scripts/build_data.py --sources toolace --seed 42
# xLAM 开通后：
# python scripts/build_data.py --sources xlam toolace --seed 42
```

### 5.4 日常开发后推送

```bash
git add <files>
git commit -m "描述"
git push
```

---

## 6. 公司 PC 如何同步个人 PC / GitHub 上的新代码

### 6.1 首选：`git pull`（若公司网络允许 **拉取**）

```powershell
cd "D:\Users\Desktop\专业实践\xiaolunwen\Class KD\pilot\fc_class_kd"
git pull origin main
```

说明：公司可能 **拦 push 但不拦 pull**；需实测。若 `pull` 也失败，用下面 bundle 方式。

### 6.2 备选：个人 PC 生成 bundle → 传到公司

**在个人 PC：**

```bash
cd fc_class_kd
git bundle create fc_class_kd-update.bundle main
# 若有多个 commit 且公司仓库落后：
# git bundle create fc_class_kd-update.bundle main ^<公司已有的commit>
```

**在公司 PC：**

```bash
git clone /path/to/fc_class_kd-update.bundle -b main fc_class_kd_new
# 或在已有仓库：
# git pull /path/to/fc_class_kd-update.bundle main
```

### 6.3 仅 `clone` 够不够？

| 场景 | 是否够用 |
|------|----------|
| 公司 **第一次** 拿 GitHub 上最新代码 | `git clone git@github.com:wjarchy/FC-SFT-verifiedClassKD.git`（pull 可用时） |
| 个人 PC 已 push 新 commit，公司已有旧仓库 | 需要 `git pull`，不能只 clone 另一个目录（除非重新 clone 全新目录） |
| 公司完全不能访问 GitHub | 只能用 **bundle** 或 **拷贝整个项目文件夹（含 `.git`）** |

**大文件**（数据、模型、训练输出）：**不要指望 git**；用云盘同步到 `data/`、`outputs/`。

---

## 7. 实验室 2080 Ti 首次上手

```bash
git clone git@github.com:wjarchy/FC-SFT-verifiedClassKD.git
# 或 HTTPS
cd FC-SFT-verifiedClassKD

pip install -e ".[dev]"
python scripts/doctor.py
pytest -q
```

数据：从云盘拷 `data/processed/` + `data/splits/`，或在能访问 HF 的机器上 `build_data.py`。

有 GPU 后按 PRD §13：smoke test → pre-diagnostic → S0 训练 → …

---

## 8. 公司 PC 上已配置的 SSH（备忘）

路径：`%USERPROFILE%\.ssh\`

| 文件 | 作用 |
|------|------|
| `config` | `Host github.com` → `ssh.github.com:443`，密钥 `id_ed25519_github` |
| `id_ed25519_github` | 私钥（勿泄露） |
| `id_ed25519_github.pub` | 公钥（已加 GitHub） |
| `known_hosts` | 含公司网络下的 host 指纹 |

Git 远程已设为：`git@github.com:wjarchy/FC-SFT-verifiedClassKD.git`

---

## 9. 安全提醒（内网 / OPPO 环境）

- **勿**将 `HF_TOKEN`、GitHub PAT、SSH 私钥提交到 git 或上传公网
- **勿**将 `.env`、模型权重、含敏感信息的原始数据放进 GitHub
- 大文件用 **公司合规的内网云盘**；第三方网盘标注【非官方】并遵守公司规定

---

## 10. 快速对照表

| 我想… | 怎么做 |
|--------|--------|
| 个人 PC 首次上线 GitHub | §5.1 bundle → push |
| 个人 PC 日常提交 | `git commit` + `git push` |
| 公司与当前划分完全一致 | 拷贝 `all_records.jsonl` + `splits/*.jsonl` |
| 重跑但尽量一致 | §4.1 相同 sources + seed + 版本 |
| 公司拿最新代码 | `git pull` 或收 bundle |
| 实验室拿代码 | `git clone` |
| 同步大数据 | 云盘 / U盘，路径见 §3 |

---

## 11. 当前进度快照（2026-08-10）

- [x] 工程骨架、Verifier、数学 helper、单元测试（本机复核 18 passed）
- [x] 旧机器曾处理 ToolACE（约 2688 条，仅 toolace）；当前 checkout 未带 processed/splits
- [x] 旧报告记录 200 条 Gold self-check 100%；这只是自动 sanity check
- [ ] 人工标注的 Verifier precision / overall accuracy Gate
- [ ] xLAM 入池（待 HF 数据集权限稳定后 `--sources xlam toolace`）
- [x] GitHub `main` 已建立，可通过 SSH push/pull
- [ ] 2080 Ti smoke test 及后续 GPU 流程

准确的研究进度与证据边界以根目录 `STATUS.md` 为准。
