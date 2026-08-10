# 冒烟专用：4 条合成样本（已进 git，无需拷贝 data/splits）

`scripts/smoke_test.py` 在找不到 `data/splits/train.jsonl` 时会自动用本文件。

强制使用：`python scripts/smoke_test.py --data-source fixture`
