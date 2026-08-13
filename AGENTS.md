# FC_ClassKD agent rules

## Network and proxy rule (this AutoDL machine)

- The user-configured `127.0.0.1:17890` proxy is reserved for OpenAI/Codex traffic only.
- Before any shell-based download or network command unrelated to OpenAI (including
  Hugging Face, GitHub, Git, pip, curl, wget, and dataset/model downloads), explicitly
  unset both uppercase and lowercase proxy variables:

  ```bash
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  ```

- For Hugging Face or GitHub downloads, prefer AutoDL's temporary academic accelerator:

  ```bash
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  source /etc/network_turbo >/dev/null
  ```

- Keep Hugging Face model/dataset caches on the data disk, for example
  `HF_HOME=/root/autodl-tmp/hf_cache`, rather than the system disk.
- Apply proxy changes only inside the individual shell command. Do not overwrite the
  user's global OpenAI/Codex proxy configuration.

## 实验文档铁律

- 实验进度、决策、执行流程和结果报告使用简明中文，避免堆叠没有解释的英文缩写和术语。
- 文档优先使用中文文件名；`README.md`、`AGENTS.md` 等工具约定文件名可以保留。
- 模型名、代码字段、命令、文件格式和 PRD 已定义的指标可以保留英文，首次出现时说明中文含义。
- 新增或修改实验文档时，必须同步更新目录链接，保证用户可以直接监控整个实验过程。
