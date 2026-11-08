# 手工筛选参考图训练配对

网页并排显示MC原图、FLUX参考图和Qwen参考图。每个参考版本构成独立图片对，可以只选其中一对。

```bash
PYTHONPATH=src python scripts/review_reference_pairs.py
```

默认地址 `http://127.0.0.1:8765`，读取 `outputs/spatial_reference_20261104/pairs.json`。原生MC像素按最近邻放大，参考图保持完整比例，可点击放大。

审核与训练使用是两项独立选择：

- **可用**：确认配对质量，不自动加入训练。
- **用于本次训练**：明确选择该图片对，同时确认配对可用；可以随时单独取消选用。
- **排除 / 待定**：不用于训练；排除会取消已有的训练选用。

支持文本、原文件名、文件名分组、原始划分和训练选用状态筛选。逐项模式可分别审查两个参考版本，浏览模式支持批量操作。所有修改自动保存在本机SQLite，支持撤销，刷新或重启服务后仍保留。

快捷键：`1/3` 分别切换两张参考图的训练选用，`2/4` 分别排除；`A` 全选训练、`X` 全部排除、`S` 稍后、方向键切换物品、`U` 撤销。输入备注时不触发快捷键。

## 导出与训练

点击右上角“导出训练清单”。训练部分仅包含明确选用、且审核可用的配对；默认沿用原验证集并去掉手工排除项，也可改成仅使用确认可用的验证配对。验证样本不会进入训练。

导出前核验原图哈希，并在本机生成：

- `selection.json`：训练器使用的源行号、配对ID、数据指纹与划分。
- `selected_pairs.jsonl`：便于核对的完整配对记录。
- `selection.zip`：上述清单与摘要，不包含图片。

网页显示服务器本地清单位置，也可下载文件。用清单准备**新实验目录**：

```bash
PYTHONPATH=src python scripts/prepare_reviewed_reference.py \
  --selection outputs/pair_review/exports/导出编号/selection.json \
  --out outputs/reference_reviewed
```

此步骤复用原视觉缓存，保持原始feature索引及训练/验证划分，不改动旧实验，不启动训练。训练步数默认保留来源配置，可通过 `--steps` 调整；开始正式训练前应根据筛选规模重新确定预算。

准备完成后，`scripts/train_visual_reference.py` 会读取新目录中的 `selection.json`，采样器只从所选配对抽样。训练检查点记录选择清单哈希，不能把另一份选择清单无声用于原断点：

```bash
PYTHONPATH=src python scripts/train_visual_reference.py train \
  --out outputs/reference_reviewed --device cuda:0
```

## 保存与访问

默认筛选记录在 `outputs/pair_review/review.sqlite3`，导出文件在同目录的 `exports/`。请保留该目录；它不会随Git提交。可用 `--state-dir` 指定另一轮独立筛选记录，用 `--pairs` 指定其他符合原始索引协议的配对清单。

服务默认只监听本机。远程机器可通过SSH端口转发，在自己的浏览器打开上述地址：

```bash
ssh -N -L 8765:127.0.0.1:8765 iflab@服务器地址
```

网页不加载生成模型，不使用GPU，不上传原始图片或筛选记录。当前版本已验证审核与训练开关独立、撤销和恢复、图片完整性、导出，以及真实训练采样对筛选结果的遵守；正式筛选与浏览器测试使用不同数据库。
