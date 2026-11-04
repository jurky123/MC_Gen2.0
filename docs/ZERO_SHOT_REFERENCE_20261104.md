# 参考图：零额外训练试验

第一版直接复用用户选定的 **4128 步文生图生成器**，冻结全部权重。高清参考图经 Qwen3-VL-Embedding-8B 编码为一个 4096 维向量，再送入已有的条件投影与生成流程。纯图像输入不附带文件名、标签或视觉描述；图文输入由 Qwen 联合编码，未使用向量加权平均。

[Qwen 官方说明](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)支持图文共享语义空间，但这不保证图像向量符合生成器训练时的文本条件分布。本轮用于检验直接迁移能否成立，不代表模型已学会参考图条件。

## 使用

```bash
PYTHONPATH=src python -m mcgen.cli sample \
  --checkpoint checkpoints/mcgen-step4128.pt \
  --reference /path/to/reference.png \
  --conditioning-mode multimodal_embedding \
  --ref-scale 2 --steps 32 --device cuda:1 \
  --out outputs/reference_sample.png --export-scale 16
```

添加 `--prompt "blue gem"` 可试验图文联合输入。这个模式中，`--ref-scale` 控制整个联合 embedding 的引导强度；`--text-scale` 不提供独立的文字控制。原有文生图和空间参考图模式保持默认行为，必须显式选择实验模式。

Python 对应 `Runtime.generate([""], [image], conditioning_mode="multimodal_embedding", ref_scale=2)`。API 请求同样支持 `conditioning_mode` 字段。

## 对照与复现

```bash
PYTHONPATH=src python scripts/zero_shot_reference.py \
  --out outputs/zero_shot_reference_20261104 --device cuda:1
```

脚本需要本地现有配对数据和下载过的官方实现，具体路径可见参数与代码。输出目录必须是新目录；已完成准备时可用 `--phase run`，生成完成后用 `--phase render` 制作面板。

- 数据：现有清理后目标与旧高清配对取交集，保留训练/验证划分，不用测试集。共有 **11,594 对训练数据（5,716 个目标）、1,280 对验证数据（639 个目标）**，并非每个目标都恰好两对。
- 抽样：训练、验证各六个目标，每个目标含 FLUX、Qwen 两种原始高清参考图，共 24 张。目标及参考图在生成前固定并记录来源和哈希；类别只用于分散抽样，不参与纯图像条件。
- 输出：文本、纯图像（CFG 1/2/3）、第二个随机种子、图文联合、空条件对照。除第二种子列外，同一行固定种子，采用 32 步 Heun；保存原始 16×16 RGBA，不修轮廓或调色。
- 图像：保持比例，EXIF 转正、透明通道合成到白底，按官方图像预处理输入，像素预算上限 512²；不读取旧的 128 像素参考缓存。超长输入报错，不截断或回退为空文本。
- 协议：最后有效 token 的 L2 归一化向量。分别对纯图像、图文联合及原有文本输入核对官方实现；文本身份和 checkpoint 校验保持不变，新增图像协议单独记录身份。

发现 BF16 图像 embedding 对 batch 形状敏感，故现已固定逐张编码，缓存与线上一致；对照图也按此协议重新生成。纯图像直接迁移仍不稳定，图文联合更接近可用物品，部分颜色能受参考影响。后续采用[冻结原生成器的条件适配训练](REFERENCE_ADAPTER_20261104.md)。

输出包括 `cases.json`、`data_audit.json`、`protocol_check.json`、`results.json`、原始输入、生成 PNG 和四张对照面板。面板中的 MC 原图是对应目标示例；零训练模型并未学习本轮高清配对，不能把与目标逐像素相同作为要求。训练集面板也不等于参考图条件的训练集。
