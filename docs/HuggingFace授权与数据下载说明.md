# Hugging Face 授权与数据下载说明

> 授权、固定 revision 和原始下载记录继续有效。下文 ToolACE+xLAM 合并与五划分是 V4 历史工作；
> V5 要先选一个主数据源，并单独固定 Train/Validation/Internal/Opportunity Probe manifest。

## 当前状态

授权和下载已经在 2026-08-12 完成。当前登录账号为 `wjcole`；xLAM 固定版本为 `26d14ebfe18b1f7b524bd39b404b50af5dc97866`，原始文件 SHA256 为 `4ef5c6f0dc552f2231f93f5853a9ef431e9e806d7aa514d0f6b615606ce576c6`。

文件已经放入数据盘上的 Hugging Face 缓存，未占用系统盘。xLAM 已完成结构审计、BFCL 泄漏检查和正式抽样，最终进入正式数据 1,147 条。正式合并数据共 3,250 条，验证结果见 `reports/合并数据质量验证.json`。

## 首次授权步骤（已完成，仅供重建机器时参考）

### 第一步：网页同意数据使用条款

1. 登录自己的 Hugging Face 账号。
2. 打开 `https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k`。
3. 填写页面要求的 First Name、Last Name、Country、Affiliation。
4. 点击 `Agree and access repository`。
5. 页面能看到数据文件后，说明账号已获得访问权限。

### 第二步：在 AutoDL 终端登录

不要把令牌发到聊天中。打开 Hugging Face 的 `Settings → Access Tokens`，创建一个只读令牌，然后在 AutoDL 项目目录运行：

```bash
cd /root/FC_ClassKD
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
source /etc/network_turbo >/dev/null
export HF_HOME=/root/autodl-tmp/hf_cache
.venv/bin/hf auth login
```

在终端提示中粘贴只读令牌。不要使用 `--add-to-git-credential`，也不要把令牌写入项目 `.env`、命令脚本或聊天记录。

令牌不得写入项目文件、命令脚本或聊天记录。

## Codex 后续执行的命令原则

所有下载命令都会在单条 Shell 中先清除 `127.0.0.1:17890`，再使用 AutoDL 学术加速，并把缓存放在 `/root/autodl-tmp/hf_cache`。不会修改全局代理。

V4 下载后已经按以下顺序完成：

1. xLAM 行数、单调用数量和参数类型检查。
2. Gold 在自身 schema 下的合法性检查。
3. 与 ToolACE 的完全相同问题去重。
4. 与 BFCL 的近重复检查。
5. 一次性重建 Train、Validation、Internal Test、Diagnostic 和 Equiv Diagnostic，并更新全部文件哈希。

V5 不直接沿用第 5 项的混合划分。后续只复用授权、下载缓存、source revision、结构审计和 BFCL
去重证据；选定主数据源后另建 V5 manifest。

旧的 ToolACE-only 数据已备份到数据盘：`/root/autodl-tmp/fc_class_kd_backup_toolace_20260812/`。该绝对路径只用于本机恢复，不写入正式数据清单。
