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

- 实验进度、决策、执行流程和结果报告必须使用通俗、直接、简明的中文，不能默认读者熟悉科研或工程术语。
- 能用中文说清楚的内容，不使用英文名词、英文缩写或专业黑话。正文、章节标题、表格列名和图表标签都按此执行。
- 必须保留的模型名、代码字段、命令、文件格式、数据集代号和 PRD 已定义的正式判定标签可以使用原文，但首次出现时必须立即解释中文含义。
- 专业指标第一次出现时，必须先说明“它衡量什么、数值意味着什么”。例如不能只写 `delta`、`CI`、`p-value`、`entropy` 或 `gradient` 而不给中文解释。
- 报告先写结论和影响，再写原因、限制和复现细节；优先使用短句，避免中英文混排和连续堆叠术语。
- 每次交付用户可读文档前，必须做一次语言检查，替换不必要的英文和缩写；技术准确性不能成为表达晦涩的理由。
- 文档优先使用中文文件名；`README.md`、`AGENTS.md` 等工具约定文件名可以保留。
- 新增或修改实验文档时，必须同步更新目录链接，保证用户可以直接监控整个实验过程。
