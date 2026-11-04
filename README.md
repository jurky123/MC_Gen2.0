<div align="center">

# MC Gen



Text-to-Image · Reference-to-Image (experimental) · 16×16 RGBA

[参考图效果](#参考图生成实验) · [生成方式](#生成方式) · [数据与规模](#数据与规模) · [开始使用](#开始使用) · [数据处理抽样](examples/data_review/README.md)

</div>

![MCGen：从训练集与测试集真实文本标签生成的物品示例](docs/images/showcase/showcase.png)

[PDF 原图](docs/images/showcase/showcase.pdf) · [样本与提示词](docs/images/showcase/manifest.json)


使用原始文件名提取文本标注生成。现提供参考图生成实验版，支持纯参考图和图文联合输入。

相比MC_Gen上一版，使用了Flow Matching进行训练，并把编码器换成了Qwen3-VL-Embedding-8B。并且用了更多数据。
尝试使用VLM精标注，但是很难得到不错的标注，所以还是恢复了文件名提取文本标注。提示词尽量使用形容词+名词的形式。
原因主要猜测还是对于MC素材这种低像素风格目前的VLM能力不强，所以没法从低分辨率信息得到精确描述。
训练数据已上传huggingface：https://huggingface.co/datasets/Risposta/MC_Gen
参考图实验使用之前由flux和qwen_image生成的高清图片-MC素材配对，冻结原文生图模型，只训练新增适配器。
展示图使用GPT 6.1sol制作（GPT审美还是太差了）

如果觉得存在侵犯知识版权的问题可以联系我删除checkpoint

## 参考图生成（实验）

![参考图实验：训练内与训练未见样本的真实对照](docs/images/reference_generation/preview.png)

每组依次为：**参考图、MC 原图、纯文本、纯参考图、图文联合**。固定抽样，保留失败结果。

已完成训练内、训练未见各64个新物品的双种子对照，共896张输出。常见物品能成形，浅色过亮、色偏和复杂造型不稳仍明显。原有文生图输出保持一致。

[参考图权重与用法](docs/USAGE.md#参考图生成实验) · [样本与来源](examples/reference_generation/README.md) · [完整对照下载](https://huggingface.co/datasets/Risposta/MC_Gen/tree/main/evaluations/reference-step2151)

## 生成方式

```mermaid
flowchart LR
    A[文本提示] --> B[冻结的 Qwen Embedding]
    B --> C[像素空间 Flow Matching 生成器]
    N[随机噪声] --> C
    C --> D[16×16 透明物品图]
```

从零训练，直接在 Minecraft 物品纹理上学习文字与像素的对应关系。

## 数据与规模

| 项目 | 文生图基础模型 |
| --- | --- |
| 输出 | 16×16，RGBA 透明背景 |
| 生成器 | **81,409,028 参数（约 81.4M）**，512 宽、12 层 Transformer |
| 文本编码器 | **Qwen3-VL-Embedding-8B**，冻结；不计入生成器参数量 |
| 训练集 | **311,510 张** |
| 验证集 | **40,808 张** |

纹理主要来自 [OVAWARE/16xModdedMinecraft](https://huggingface.co/datasets/OVAWARE/16xModdedMinecraft) 及其图文衍生数据，覆盖多种 Minecraft 模组。项目整理的数据导出位于 [Risposta/MC_Gen](https://huggingface.co/datasets/Risposta/MC_Gen)，可通过[下载脚本](scripts/download_dataset_hf.py)获取。

源表包含 1,034,057 条纹理记录；上表是经过去重、原生像素校验和清理后的实际使用量。文本监督由文件名和可恢复的资源路径提取，保留材质、颜色和抽象名称。按来源项目划分数据，隔离跨集合重复。


## 开始使用

[下载文生图权重](https://huggingface.co/datasets/Risposta/MC_Gen/resolve/85470a51e497cea27ceb4d639e822440f660ac1a/checkpoints/step4128/mcgen-step4128.pt)（约 326 MB） · [SHA256 校验](https://huggingface.co/datasets/Risposta/MC_Gen/resolve/85470a51e497cea27ceb4d639e822440f660ac1a/checkpoints/step4128/SHA256SUMS.txt)

Qwen3-VL-Embedding-8B 需单独下载，脚本会固定版本并校验文件。

```bash
pip install -e '.[embedding]'

curl -L --fail --create-dirs -o checkpoints/mcgen-step4128.pt \
  https://huggingface.co/datasets/Risposta/MC_Gen/resolve/85470a51e497cea27ceb4d639e822440f660ac1a/checkpoints/step4128/mcgen-step4128.pt
python scripts/download_qwen_vl_embedding.py --out ../Qwen3-VL-Embedding-8B
MCGEN_EMBEDDING_MODEL=../Qwen3-VL-Embedding-8B mcgen sample \
  --checkpoint checkpoints/mcgen-step4128.pt \
  --prompt 'diamond sword' --seed 20261112 \
  --steps 32 --text-scale 2 --out outputs/diamond_sword.png --device cuda:0
```

生成器权重通过 Hugging Face 分发；Git 仓库提供代码、展示原图与少量可核验的数据样本，完整数据和特征缓存不随 Git 提交。

[安装、模型准备与更多命令](docs/USAGE.md) · [文本与数据处理](docs/RESOURCE_TEXT_20261103.md) · [项目进度](PROGRESS.md)

## License

Research and study only. Do not use the trained models commercially.
