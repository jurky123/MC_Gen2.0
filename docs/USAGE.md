# 使用说明

以下命令用于安装、生成、训练和评估。示例中的本机路径需按自己的环境替换；模型权重与数据不随仓库提供。

## 当前选用版本

当前选用扩容文生图实验的 **4128 步**。本机推理文件为 `checkpoints/mcgen-step4128.pt`，原始训练快照仍保留；这是人工比较生成效果后的选择，不是自动误差最小的 4644 步。

```bash
pip install -e '.[embedding]'
python scripts/download_qwen_vl_embedding.py --out /home/iflab/models/Qwen3-VL-Embedding-8B
mcgen sample --checkpoint checkpoints/mcgen-step4128.pt \
  --prompt 'diamond sword' --seed 20261112 --steps 32 --text-scale 2 \
  --out outputs/diamond_sword.png --device cuda:0
```

以上是本机路径示例。编码器的下载目录需与检查点记录一致；编码器文件经过校验，不能把旧 Qwen3-8B 替换到此检查点中。生成器权重目前仅在本地保留，不随 Git 发布。模型配置、文件哈希和数据规模见 [4128 步记录](models/step4128.json)。

下方通用命令同时保留历史实验的 text/reference/joint 能力；当前推荐的 4128 步模型只用于文生图。

## 安装与验证

```bash
pip install -e '.[test]'
mcgen smoke --out outputs/my_smoke
python -m pytest -q
```

本机解释器 `/home/iflab/miniconda3/envs/mcgen/bin/python`。
不安装时使用 `PYTHONPATH=src python -m mcgen ...`。
历史模型的 Qwen 已下载至 `/home/iflab/models/Qwen3-8B`，固定官方 commit
`b968826d9c46dd6066d109eabc6255188de91218`，权重哈希已核验。

## 数据准备

本机可直接使用 `data/catalogs/native16_v3/catalog`。它由完整旧数据重新划分，隔离跨集合重复，并将目标无损压缩成原生 16px；旧数据未改动。

```bash
# 重新从旧项目迁移时，使用新目录
mcgen migrate --project . --out data/catalogs/new_source --size 16 --include-text
mcgen prepare --catalog data/catalogs/new_source --out data/catalogs/new_prepared
mcgen audit --catalog data/catalogs/new_prepared/catalog
```

`prepare` 输出固定项目划分、隔离清单、原生 mmap、alpha 检测和审计报告。
可用 `--project-plan plan.json` 指定已审查的项目归属。新素材以 PNG JSONL 导入：

```bash
mcgen import-images --manifest samples.jsonl --out data/new_assets --size 16
```

每行包含 `id,lineage,prompt,asset_type,tileable,target`，可选 `reference,license,style`。
图片路径相对于 manifest。asset_type 为 item/tile/other；tileable 为 true/false/null。
多来源先 `mcgen merge --catalog ... --out ...`，再统一 prepare，不能拼接独立划分。

## 预算与训练

`configs/v3/` 包含 item/tile 各自的 text/reference/joint 六组短试验配置；`other` 不进入这些实验。
默认三种任务使用同一批配对目标，保证对照的数据范围一致；全量文生图扩展配置位于 `configs/v3/scale/`。
历史 token 特征配置保留 Qwen 三层特征、独立文字/参考图注意力。当前扩容模型使用一个冻结的 Qwen Embedding 向量和纯文本主干，见 [当前模型记录](EMBEDDING_TEXT_20261103.md)。

```bash
mcgen budget --config configs/v3/item_joint.yaml \
  --catalog data/catalogs/native16_v3/catalog --out outputs/item_budget.json

mcgen train --config configs/v3/item_joint.yaml \
  --catalog data/catalogs/native16_v3/catalog --out outputs/item_joint \
  --device cuda:0 --text-device cuda:1
```

先查看预算中的任务曝光；默认 2,000 步是试验，不代表百万数据训练充分。
断点续训在同一命令上加 `--resume outputs/item_joint/latest.pt`，配置、数据指纹和文本塔身份必须一致。
训练保存 latest、分任务 best flow、预算和指标。flow 最优不代表生成质量最优。
`configs/v3/ablations/` 提供注意力共享、Qwen 层和宽度对照配置。

## 生成与服务

```bash
mcgen sample --checkpoint outputs/item_joint/latest.pt \
  --prompt 'red crystal sword' --out outputs/sword.png --device cuda:0 --text-device cuda:1
mcgen sample --checkpoint outputs/item_joint/latest.pt \
  --reference reference.png --prompt 'red crystal sword' \
  --out outputs/sword_ref.png --device cuda:0 --text-device cuda:1
mcgen serve --checkpoint outputs/item_joint/latest.pt --device cuda:0 --text-device cuda:1
```

省略 prompt 可做纯参考图生成。text 专用模型拒绝参考图；reference 专用模型必须有参考图。
`--export-scale 2` 为最近邻显示放大，`--palette-colors 16` 为显式后处理。原始评测不应用这些修饰。
参考图保留透明度和长宽比；checkpoint 的 alpha 输出契约应用于 CLI/API 导出。
API 提供 `/health`、`POST /generate`，请求支持 prompt、reference_png（base64 PNG）、seed、steps、text_scale、ref_scale、solver、export_scale、palette_colors。默认监听 127.0.0.1。

## 配对评测与语义盲评

```bash
mcgen evaluate --checkpoint outputs/item_joint/latest.pt \
  --catalog data/catalogs/native16_v3/catalog --split test --n 64 \
  --out outputs/item_test --device cuda:0 --text-device cuda:1

mcgen benchmark --checkpoint outputs/item_joint/latest.pt \
  --manifest benchmarks/semantic_v1.jsonl --out outputs/item_semantics \
  --device cuda:0 --text-device cuda:1
mcgen review --directory outputs/item_semantics
```

evaluate 使用同目标、同种子的条件对照和 resize/palette 基线，并按素材类型、生成器汇总。
评估独立外部 catalog 时须加 `--training-catalog` 指向训练时 catalog，自动核查已知重叠。

benchmark 固定任务标准、多个种子，导出盲评 CSV。参考图案例须提供 reference、reference_sha256 和 provenance。
隐藏 `key.json` 完成人工评分后运行 review；未填的评分始终显示待评，不自动冒充质量通过。
