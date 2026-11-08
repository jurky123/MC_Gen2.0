# 使用说明

以下命令用于安装、生成、训练和评估。示例中的本机路径需按自己的环境替换；当前选用权重通过 Hugging Face 提供，完整数据不随 Git 仓库提供。

## 当前选用版本

文生图基础模型保留扩容实验的 **4128 步**。[Hugging Face 下载目录](https://huggingface.co/datasets/Risposta/MC_Gen/tree/85470a51e497cea27ceb4d639e822440f660ac1a/checkpoints/step4128)，推理文件为 `checkpoints/mcgen-step4128.pt`，原始训练快照仍保留；这是人工比较生成效果后的选择，不是自动误差最小的 4644 步。

参考图实验按本次选择保留 **[8,000步五任务版本](#five-task-reference-8000)**，下载和命令见下方。

```bash
pip install -e '.[embedding]'
curl -L --fail --create-dirs -o checkpoints/mcgen-step4128.pt \
  https://huggingface.co/datasets/Risposta/MC_Gen/resolve/85470a51e497cea27ceb4d639e822440f660ac1a/checkpoints/step4128/mcgen-step4128.pt
python scripts/download_qwen_vl_embedding.py --out ../Qwen3-VL-Embedding-8B
MCGEN_EMBEDDING_MODEL=../Qwen3-VL-Embedding-8B mcgen sample \
  --checkpoint checkpoints/mcgen-step4128.pt \
  --prompt 'diamond sword' --seed 20261112 --steps 32 --text-scale 2 \
  --out outputs/diamond_sword.png --device cuda:0
```

以上命令从项目根目录执行。通过 `MCGEN_EMBEDDING_MODEL=/path/to/Qwen3-VL-Embedding-8B` 可指定其他下载目录；只改变文件位置，仍会校验模型文件和检查点中的编码器身份。不能把旧 Qwen3-8B 替换到此检查点中。生成器权重是推理文件，不含优化器或冻结的 Qwen 模型；若需继续原训练，应使用本地保留的完整训练快照。模型配置、文件哈希和数据规模见 [4128 步记录](models/step4128.json)。

校验下载文件：

```bash
curl -L --fail -o checkpoints/SHA256SUMS.txt \
  https://huggingface.co/datasets/Risposta/MC_Gen/resolve/85470a51e497cea27ceb4d639e822440f660ac1a/checkpoints/step4128/SHA256SUMS.txt
(cd checkpoints && sha256sum -c SHA256SUMS.txt)
```

<a id="five-task-reference-8000"></a>

## 五任务参考图：8000步

按本次选择保留 **8,000步文字最佳检查点**，支持文生图、纯参考图和图文联合。这个检查点未冻结时间MLP；12,000步冻结续训仅作为[对照](../examples/time_freeze_reference/README.md)。较早版本完整保留，复杂轮廓与装饰仍不稳定。

```bash
curl -L --fail --create-dirs -o checkpoints/mcgen-reference-five-task-text-step8000.pt \
  https://huggingface.co/datasets/Risposta/MC_Gen/resolve/c14c8a4f1b869d5b57bcff9ebe2c18cd745a97a2/checkpoints/reference-five-task-text-step8000/mcgen-reference-five-task-text-step8000.pt

MCGEN_EMBEDDING_MODEL=../Qwen3-VL-Embedding-8B mcgen sample \
  --checkpoint checkpoints/mcgen-reference-five-task-text-step8000.pt \
  --reference examples/time_freeze_reference/inputs/val_00_reference.png \
  --prompt "Tiger's-eye battleaxe" --conditioning-mode visual_tokens \
  --steps 24 --text-scale 1 --ref-scale 1 --seed 2026110907 \
  --out outputs/five_task_axe.png --export-scale 8 --device cuda:0
```

Qwen3-VL-Embedding-8B需单独下载，方法见上文。省略prompt为纯参考图；文生图省略reference和conditioning-mode。下载文件仅包含推理权重，本地完整训练快照及优化器继续保留。

[权重与SHA256](https://huggingface.co/datasets/Risposta/MC_Gen/tree/c14c8a4f1b869d5b57bcff9ebe2c18cd745a97a2/checkpoints/reference-five-task-text-step8000) · [少量真实样本](../examples/time_freeze_reference/README.md) · [实验记录](FIVE_TASK_REFERENCE_20261108.md)

## 参考图生成（实验）

参考图实验采用 **2151 步适配器 + 冻结的4128步底座**，支持纯参考图和图文联合，也保留原文生图输出。色偏、浅色过亮和复杂造型不稳仍存在。[真实抽样对照](../examples/reference_generation/README.md)。

```bash
curl -L --fail --create-dirs -o checkpoints/mcgen-reference-step2151.pt \
  https://huggingface.co/datasets/Risposta/MC_Gen/resolve/3f99aab1d0f4658197b533590bba4703cb712da7/checkpoints/reference-step2151/mcgen-reference-step2151.pt

MCGEN_EMBEDDING_MODEL=../Qwen3-VL-Embedding-8B mcgen sample \
  --checkpoint checkpoints/mcgen-reference-step2151.pt \
  --reference examples/reference_generation/inputs/test_03_reference.png \
  --prompt 'orange diamond axe' --conditioning-mode multimodal_embedding \
  --steps 32 --ref-scale 2 --seed 202611042 \
  --out outputs/reference_axe.png --export-scale 8 --device cuda:0
```

省略 `--prompt` 即为纯参考图生成。没有参考图时，省略 `--conditioning-mode`，按原文生图方式使用。此模式中 `--ref-scale` 引导整个图像/图文 embedding，不提供独立的文字引导调节。

[权重与 SHA256](https://huggingface.co/datasets/Risposta/MC_Gen/tree/3f99aab1d0f4658197b533590bba4703cb712da7/checkpoints/reference-step2151) · [完整896张对照](https://huggingface.co/datasets/Risposta/MC_Gen/tree/3f99aab1d0f4658197b533590bba4703cb712da7/evaluations/reference-step2151) · [训练与验证记录](REFERENCE_ADAPTER_20261104.md)。

## 空间参考图实验

新增独立视觉 token 通路，保留原文本条件和文生图能力。本轮图文联合更接近配对目标，纯参考图弱于上面的旧版；装饰多样性仍需验证。[评估与比较边界](SPATIAL_REFERENCE_EVALUATION_20261104.md)。

```bash
curl -L --fail --create-dirs -o checkpoints/mcgen-spatial-reference-step1787.pt \
  https://huggingface.co/datasets/Risposta/MC_Gen/resolve/e1b7f0499c6fd1b9b35e64ca7cdcc8bed65b4d99/checkpoints/spatial-reference-step1787/mcgen-spatial-reference-step1787.pt

MCGEN_EMBEDDING_MODEL=../Qwen3-VL-Embedding-8B mcgen sample \
  --checkpoint checkpoints/mcgen-spatial-reference-step1787.pt \
  --reference reference.png \
  --prompt 'redstone helmet' --conditioning-mode visual_tokens \
  --steps 32 --text-scale 2 --ref-scale 2 --seed 202611042 \
  --out outputs/spatial_helmet.png --export-scale 8 --device cuda:0
```

文字和参考图引导分别通过 `--text-scale`、`--ref-scale` 调节。省略 prompt 可试纯参考图；没有参考图时按原文生图方式使用。显存不足时可加 `--text-device cpu`，速度会受 CPU 性能影响。

[权重与校验](https://huggingface.co/datasets/Risposta/MC_Gen/tree/e1b7f0499c6fd1b9b35e64ca7cdcc8bed65b4d99/checkpoints/spatial-reference-step1787) · [训练记录](SPATIAL_REFERENCE_20261104.md)。`reference.png` 使用自己的参考图；本轮不公开原始图片及完整评估包。

下方通用命令同时保留历史实验的 text/reference/joint 能力；文生图推荐原4128步模型，参考图实验按上述各自的输入协议使用。

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
