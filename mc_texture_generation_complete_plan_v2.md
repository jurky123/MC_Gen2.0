# Minecraft 低分辨率材质生成模型：完整数据、模型、训练与部署方案

> 版本：v2.3
> 日期：2026-09-11
> 目标：构建 Minecraft / voxel-game 低分辨率文生纹理模型。数据改为像素风素材与真实 MC 纹理；主模型、8GB 显存目标和 Rectified Flow Transformer 范式保持不变。

> **v2.3 路线覆盖说明：** 本节及“0. 最终方案概要”是当前有效方案。后文 MatSynth、ambientCG、Material Maker、pseudo-pixel、SigLIP2 和旧 Stage A/A.5/C/D 内容仅作历史参考，不再作为数据获取、文本条件编码或训练依据。

---

# 0. 最终方案概要

主路线：

```text
Stage 1：全量数据预训练
Kenney + itch.io free pixel-art assets + NathMen12 MC TextToImage
        ↓
Stage 2：Minecraft 弱标注训练
NathMen12/16xModdedMinecraft-TextToImage + existing MC weak labels
        ↓
Stage 3：Minecraft 精标注精调
James-A/Minecraft-16x-Dataset + 后续人工复核精标注 MC 子集
        ↓
MC-FlowDiT-Base
Pixel-space Rectified Flow Transformer
        ↓
16–24 step Heun / Euler inference
        ↓
可选 4–8 step distillation
```

当前数据约定：删除并停用 MatSynth、ambientCG 等真实世界通用材质；Stage 1 使用全部像素素材和全部 MC 数据；Stage 2 以 `NathMen12/16xModdedMinecraft-TextToImage` 的 1,034,057 条图文数据为主，并保留现有 MC 弱标签；Stage 3 使用 James-A/Minecraft-16x-Dataset，并允许后续加入人工复核的精标注 MC 子集。项目限定为学习研究用途，但仍保存来源 URL、作者、页面标签和许可字段以保证可追溯。

## 0.1 当前数据落盘状态

| 数据 | 当前状态 | 训练用途 |
|---|---|---|
| NathMen12 MC TextToImage | 1,034,057 条已完整下载并转换为 32x32 mmap；train/val/test 为 941,104 / 43,184 / 49,769 | Stage 1 + Stage 2 主 MC 数据 |
| Kenney Pixel Assets | 14 包、2,106 个源图片已处理为 4,312 个去重样本 | Stage 1 |
| itch.io Free Pixel Art | 原始下载达到 30.022 GiB 硬上限；连通域切分进行中，本次记录点为 557,124 条 manifest | Stage 1 |
| James-A/Minecraft-16x-Dataset | 已完整下载 1,519 条；train/validation/test 为 1,366 / 70 / 83，具有颜色、图案、光照、对称性、平铺方向、用途和整体描述等细粒度字段 | Stage 3 首个高质量精调集 |

旧 OVAWARE 本地子集和 Modrinth 提取集继续保留作兼容、校验或补充，但不替代 NathMen12 主集。固定网格方式生成的旧 itch 切片为废弃中间产物，不可混入训练。

## 0.2 spritesheet 与独立物件处理规范

像素素材网页中的单张文件经常包含多个透明隔开的角色、道具或 tile，因此原图不能直接当作一个训练样本。当前处理规则如下：

1. 读取 alpha 通道，以非透明像素的二维连通区域定位独立物件并计算包围盒。
2. 仅当文件名明确标注 8x8、16x16、32x32 等 tile 规格时，允许优先按该显式规格切格；不再猜测固定网格。
3. 对包围盒内容保持长宽比，使用 nearest-neighbor 降采样或上采样，居中放入 32x32 透明画布。该方式避免双线性插值污染像素边缘，也避免无条件拉伸造成形变。
4. 跳过空白、近纯色和无法可靠拆分的全不透明大图；按输出内容 SHA 精确去重。
5. 在 JSONL manifest 中保存源文件、所属素材/压缩包、包围盒、切分方法和输出路径，使筛选与重建可追溯。

处理完成后需进行分层抽样复核，重点检查极端长宽比对象、动画条带、粒子图、字体图、UI atlas 和意外保留的整张场景。通过复核后，才将 Kenney、itch 与全部 MC 样本合并为 Stage 1 mmap。Stage 1 合并构建目前尚未完成。

## 0.3 文本条件编码器定案

文本条件编码器固定为 `Qwen/Qwen3-VL-Embedding-8B`。选择 8B 是因为其语义质量更好，能更充分地覆盖短纹理描述和细粒度属性，且本项目可本地离线编码，成本可接受。其原生 embedding 为 4096 维，并支持 instruction-aware 表示；本项目首版保留原生 4096 维，不额外训练降维器。

必须严格遵守以下用途边界：

- Qwen3-VL-Embedding-8B **只用于文本条件注入**。
- 输入始终是 prompt 文本；不向它输入训练图片或图文混合内容。
- 不用它做图片去重、相似检索、数据筛选、caption 生成、caption 评分或质量检查。
- 数据去重继续使用确定性的内容 SHA/像素规则，数据处理与文本编码保持解耦。

统一条件流程：

```text
canonical prompt text
        ↓
frozen Qwen3-VL-Embedding-8B（text-only）
        ↓
L2-normalized pooled embedding [4096]
        ↓
offline FP16 mmap [N, 1, 4096]
        ↓
MC-FlowDiT text projection 4096 → hidden_size
```

编码 instruction 固定为：

```text
Represent this Minecraft or pixel-art texture description for conditional image generation.
```

Stage 1、Stage 2、Stage 3 与推理服务必须固定模型 revision、tokenizer、instruction、文本清洗规则和 L2 normalization。MC-FlowDiT 训练期间不加载 Qwen 权重；classifier-free guidance 的空条件使用同一编码器对空/负条件模板离线编码，或使用训练得到的 null condition，但二者只能选定一种并保持训练与推理一致。

James-A 数据集的 `overall_texture_description` 作为详细文本视图；另由已有结构化字段按固定模板组成短文本视图。训练时可在两种文本视图间随机采样，但两者都仅以文本形式送入 Qwen 编码器。不得把对应图片送入 Qwen 编码器。

推荐主模型：

```yaml
model: MC-FlowDiT-Base
resolution: 32
patch_size: 2
image_tokens: 256

hidden_size: 512
heads: 8
double_stream_blocks: 3
single_stream_blocks: 6
mlp_ratio: 3.0

text_dim: 4096
max_text_tokens: 1
position_encoding: 2D RoPE
qk_norm: RMSNorm
activation: SwiGLU
conditioning: AdaLN-Zero
objective: Rectified Flow / Flow Matching

params: ~45-50M
```

单卡训练目标：

```text
8GB VRAM
BF16 / FP16
PyTorch SDPA
fused AdamW
torch.compile
CPU EMA
offline text embedding
mmap dataset

默认不开：
activation checkpoint
CPU offload
DeepSpeed
8-bit optimizer
```

---

# 1. 任务定义

本项目生成对象不是一般自然图片，而是：

- 16×16 / 32×32 / 64×64 block texture；
- Minecraft / voxel-game 风格；
- 强 pixel-art 特征；
- 有限 palette；
- 大量局部重复；
- 很多纹理要求 seamless tiling；
- prompt 语义相对结构化；
- 数据规模较小；
- 训练与推理成本应远低于通用文生图模型。

典型输入：

```text
dark mossy stone bricks
blue crystal ore embedded in dark deepslate
weathered copper tiles
oak planks with small cracks
```

典型输出：

```text
16×16 / 32×32 RGB texture
```

---

# 2. 为什么使用 Pixel-space，而不是 VAE latent

对于 32×32 RGB：

\[
32\times32\times3=3072
\]

只有 3072 个 RGB 标量。

使用传统：

```text
RGB → VAE → latent → diffusion → VAE decode
```

没有明显必要，而且 VAE 会引入：

- 单像素边界平滑；
- palette 漂移；
- 小结构丢失；
- checker / interpolation artifact；
- tile seam；
- 像素艺术风格损失。

因此本项目选择：

\[
\boxed{\text{Pixel-space generative modeling}}
\]

即直接对 RGB pixel patch 做 Flow Matching。

---

# 3. 为什么使用 Flow Matching + Transformer

当前高性能图像生成模型的主流架构已经从：

```text
U-Net + DDPM
```

逐渐转向：

```text
Transformer + Flow / Rectified Flow
```

代表路线包括：

- Stable Diffusion 3：Rectified Flow + MMDiT；
- FLUX.1：Flow Matching Transformer；
- FLUX.2：Double-stream → Single-stream multimodal Transformer；
- 新一代少步生成模型大量采用 Flow + distillation。

本项目采用：

\[
\boxed{
\text{Pixel-space}
+
\text{Rectified Flow}
+
\text{MMDiT / FLUX-style Transformer}
}
\]

但按 32×32 小图进行极度缩小。

---

# 4. MC-FlowDiT 架构

## 4.1 Image Tokenization

默认：

```text
resolution = 32×32
patch_size = 2×2
```

得到：

\[
16\times16=256
\]

image tokens。

每个 patch：

\[
2\times2\times3=12
\]

维 RGB 输入，通过线性层投影到：

\[
d=512
\]

。

---

# 4.2 双流阶段：MMDiT-style

前 3 层使用 image/text 两个独立 stream：

```text
Image stream ─┐
              ├── Joint Attention
Text stream ──┘
```

特点：

- image/text 独立 QKV / MLP；
- attention 时联合交互；
- 比普通 cross-attention 对 prompt alignment 更充分。

因为本任务文本很短，不需要很多双流层。

推荐：

```text
Double-stream blocks = 3
```

---

# 4.3 单流阶段：FLUX-style

之后把：

```text
[text tokens | image tokens]
```

合并成一个 sequence。

使用：

```text
Joint Attention
+
SwiGLU MLP
+
shared timestep / condition modulation
```

推荐：

```text
Single-stream blocks = 6
```

这样能够：

- 降低参数量；
- 增强 text-image interaction；
- 保持当前主流 architecture 风格。

---

# 4.4 Normalization

推荐：

```text
RMSNorm
QK-Norm
```

对 attention：

\[
q'=\frac{q}{RMS(q)}
\]

\[
k'=\frac{k}{RMS(k)}
\]

提高训练稳定性。

---

# 4.5 位置编码

推荐：

\[
\boxed{\text{2D RoPE}}
\]

而不是固定 learned positional embedding。

坐标：

\[
(h,w)
\]

直接进入二维 rotary encoding。

优点：

- 与 Transformer 当前主流设计一致；
- 方便后续 32 → 64 resolution transfer；
- 不需要重新插值 learned embedding。

---

# 4.6 条件调制

推荐：

\[
\boxed{\text{AdaLN-Zero}}
\]

timestep embedding 和 condition 生成：

```text
shift
scale
gate
```

然后：

\[
h'=
h+
g\cdot
F(\mathrm{AdaLN}(h))
\]

block 初始接近 identity，有利于稳定训练。

---

# 4.7 MLP

使用：

\[
\boxed{\text{SwiGLU}}
\]

推荐：

```text
mlp_ratio = 3.0
```

---

# 4.8 参数规模

主配置：

```yaml
hidden_size: 512
heads: 8
double_stream_blocks: 3
single_stream_blocks: 6
mlp_ratio: 3.0
```

预计：

\[
\boxed{45\text{–}50M}
\]

参数。

推荐同时训练三档：

| Model | 参数量 | 作用 |
|---|---:|---|
| Tiny | 20–25M | 快速 debug / ablation |
| Base | 45–50M | 主模型 |
| Large | 80–120M | scaling study |

不推荐第一版直接训练 450M。

---

# 5. 16×16 Native 版本

如果最终主要目标是经典 Minecraft 16×16：

```text
resolution = 16
patch_size = 1
```

此时仍然：

\[
16\times16=256
\]

tokens。

但每个 token 对应一个真实像素。

优点：

- pixel fidelity 更强；
- palette 更准确；
- seam 更容易控制；
- 不存在 2×2 unpatchify 内部耦合。

推荐做重要 ablation：

```text
32×32 / patch=2
vs
16×16 / patch=1
```

---

# 6. Flow Matching 训练目标

数据：

\[
x_0\sim p_{data}
\]

噪声：

\[
z\sim \mathcal N(0,I)
\]

时间：

\[
t\sim U(0,1)
\]

线性路径：

\[
x_t=(1-t)x_0+t z
\]

即：

\[
x_t=x_0+t(z-x_0)
\]

目标 velocity：

\[
v^\*=z-x_0
\]

模型：

\[
v_\theta(x_t,t,c)
\]

损失：

\[
L_{flow}
=
E\left[
\|v_\theta(x_t,t,c)-v^\*\|_2^2
\right]
\]

第一版直接使用：

```text
MSE
+
uniform timestep sampling
```

稳定以后再测试：

```text
logit-normal timestep sampling
```

。

---

# 7. Minecraft 特化：Tileability

## 7.1 Toroidal Shift Augmentation

对于 tileable texture：

```python
x = torch.roll(
    x,
    shifts=(dy, dx),
    dims=(-2, -1)
)
```

这相当于在二维 torus 上随机平移。

比普通 crop augmentation 更符合 seamless texture 的分布。

---

# 7.2 Tile Loss

通过 velocity 预测恢复：

\[
\hat{x}_0=x_t-t\hat v
\]

定义边缘 loss：

\[
L_{tile}
=
\|Left-Right\|_1
+
\|Top-Bottom\|_1
\]

推荐比较：

```text
border_width = 2~3 pixels
```

而不只是单像素。

总 loss：

\[
L
=
L_{flow}
+
\lambda_{tile} L_{tile}
\]

起始：

```text
lambda_tile = 0.03
```

建议只在：

```text
t < 0.7
```

计算 tile loss。

---

# 8. 数据总体设计

最终数据 curriculum：

```text
通用真实材质
       ↓
Pixel-art bridge
       ↓
Minecraft-style texture
       ↓
Structured weak text
       ↓
Detailed natural language
```

不要一开始就用少量 MC caption 从零训练。

---

# 9. Stage A：通用材质预训练

## 9.1 核心数据：MatSynth

推荐把 MatSynth 作为主数据集。

主要特点：

- 4K PBR materials；
- tileable；
- 多材料类别；
- 有 metadata；
- 可按 license 筛选；
- 包含 basecolor / normal / roughness 等。

本任务只使用：

```text
basecolor
+
metadata
```

第一版最保守：

\[
\boxed{\text{只使用 CC0}}
\]

主要类别：

```text
stone
wood
metal
concrete
ground
terracotta
marble
plaster
fabric
plastic
ceramic
```

这些和 MC block texture 非常匹配。

---

# 9.2 Patch 构建

不要直接：

```text
4K → 32×32
```

而应：

```text
4K basecolor
  ↓
随机 crop：
128 / 256 / 512 / 1024
  ↓
area resize
  ↓
32×32
```

每个独立 material 采：

```text
32–128 patches
```

最终建议：

\[
\boxed{200K\text{–}400K}
\]

generic texture patches。

---

# 9.3 Material Maker

作为 procedural texture 补充。

第一版只抓：

```text
CC0
```

利用 procedural seed 生成真实不同实例。

目标：

\[
\boxed{50K\text{–}150K}
\]

additional textures。

---

# 10. Stage A.5：Pixel-art Bridge

自然材质和 MC texture domain gap 仍然很大。

所以单独构建：

\[
\boxed{\text{pixel bridge}}
\]

。

---

# 10.1 Pseudo-pixelization

从通用 texture 自动生成：

```text
32×32
16×16 → nearest upscale
palette quantization
posterization
pixelation
```

palette size：

```text
8 / 16 / 32 / 64
```

推荐 mixture：

```text
40% 直接低分辨率
30% palette quantized
20% pixelized + quantized
10% 原始 texture
```

---

# 10.2 Kenney

使用许可证明确的 CC0 pixel-art asset。

适合：

- tiles；
- platformer blocks；
- patterns；
- pixel assets。

建议：

\[
10K\text{–}30K
\]

有效 tiles / patterns。

---

# 10.3 OpenGameArt

可作为补充，但必须：

```text
逐 item 验证实际 license
```

第一版完全可以暂时不依赖 OGA。

---

# 10.4 Stage A.5 规模

最终：

\[
\boxed{50K\text{–}150K}
\]

pixel-style samples。

推荐：

```text
70% pseudo-pixel
20% Kenney
10% verified OGA
```

---

# 11. Stage B：Minecraft Domain Dataset

核心平台：

\[
\boxed{\text{Modrinth}}
\]

同时抓：

```text
Resource Packs
+
Open-source Mods
```

原因：

Resource pack：

- vanilla-style 纹理多；
- 同类重绘多。

Open-source mod：

- 新材料更多；
- ore / machine / stone / wood / fantasy block diversity 更强。

---

# 12. MC Texture 路径

主要提取：

```text
assets/<namespace>/textures/block/**/*.png
```

支持：

```text
16×16
32×32
64×64
```

第一版跳过：

```text
*.png.mcmeta
```

对应的 animated texture。

---

# 13. License 策略

## 自动接受

第一版优先：

```text
CC0-1.0
CC-BY-4.0
```

如果希望未来发布与商业使用最干净：

```text
只用 CC0
```

。

---

## 人工审核

下面这些不能只看 repo license：

```text
MIT
Apache-2.0
GPL
```

必须额外确认：

```text
textures/assets 是否同 license
```

因为：

> 代码 license 不一定覆盖图像资产。

---

## 自动排除

```text
ARR
custom restrictive
no-AI
AI-training prohibited
mixed license unclear
asset provenance unclear
```

---

# 14. 明确排除的数据

## Faithful

Faithful 当前 license 明确禁止使用作品训练 AI / neural networks。

因此：

\[
\boxed{\text{排除}}
\]

---

## Mojang Vanilla Textures

Mojang 原版 asset 有明确版权与 EULA 限制。

为了保证公开数据集和最终模型的许可干净：

\[
\boxed{\text{第一版不将 Vanilla asset 纳入训练数据}}
\]

可以保留用于：

```text
private evaluation
reference
style statistics
```

但不作为公开训练数据主来源。

---

# 15. MC 数据规模

严格 license filtering 后，不要强求固定数量。

目标：

```text
Minimum: 10K
Recommended: 20K–50K
Strong: 50K–100K
```

对约 45M 模型：

\[
20K\text{–}50K
\]

高质量 MC texture 已经足以做有效 domain adaptation。

---

# 16. 数据去重

必须做：

## Exact Dedup

```text
SHA256 / SHA512
```

## Near Dedup

```text
pHash
DCT hash
low-resolution RGB L2
```

资源包之间经常有：

- 完全复制；
- 小幅改色；
- 轻微锐化；
- 小幅 palette 修改。

---

# 17. Dataset Split

不能：

```text
random image split
```

而应该：

```text
project_id split
```

或：

```text
material family split
```

保证一个资源包不会同时出现在 train/test。

---

# 18. Stage C：弱标签自动构建

MC 资源结构天然适合自动标注。

解析：

```text
textures/block/
models/block/
blockstates/
lang/en_us.json
namespace
```

例如：

```text
mossy_limestone.png
```

得到：

```json
{
  "material": "limestone",
  "state": "mossy",
  "form": "block"
}
```

---

# 18.1 model JSON

例如：

```json
{
  "parent": "minecraft:block/cube_all",
  "textures": {
    "all": "example:block/mossy_limestone"
  }
}
```

可以建立：

```text
texture → block identifier
```

映射。

---

# 18.2 lang JSON

例如：

```json
{
  "block.example.mossy_limestone": "Mossy Limestone"
}
```

得到：

```text
human-readable block name
```

---

# 18.3 结构化语义

例如：

```text
weathered_cut_copper
```

自动转：

```json
{
  "material": "copper",
  "form": "cut",
  "state": "weathered"
}
```

最终生成 weak prompt：

```text
weathered cut copper, pixel-art block texture
```

---

# 19. Stage D：VLM 高质量 Caption

这一步不再固定某一个模型，而是先 benchmark。

推荐候选：

```text
Qwen3-VL-Flash
Qwen3-VL-30B-A3B-Instruct
Qwen3-VL-235B-A22B-Instruct
```

---

# 20. 为什么先 Benchmark

16×16 / 32×32 Minecraft texture 是非常特殊的 domain。

通用 benchmark：

```text
MMMU
MathVista
ChartQA
```

不能代表：

```text
材质识别
颜色判断
pattern
像素结构
tileability
```

因此先建立：

\[
\boxed{\text{MC-Texture-CaptionBench}}
\]

---

# 21. MC-Texture-CaptionBench

人工选：

\[
500\text{–}1000
\]

张。

覆盖：

```text
wood
stone
brick
ore
metal
soil
sand
plant
glass
machine
decorative
fantasy
emissive
```

人工标：

```text
material
form
state
dominant color
pattern
surface
details
caption
```

比较：

```text
Material Accuracy
Form Accuracy
State F1
Color F1
Attribute F1
Hallucination Rate
JSON Valid Rate
Human Caption Preference
Cost / image
Latency
Throughput
```

最终：

\[
\boxed{\text{按质量选择模型，不按参数量选择}}
\]

---

# 22. 当前 VLM 推荐优先级

## 全量粗标

```text
Qwen3-VL-Flash
```

优势：

- API 极便宜；
- 高吞吐；
- 适合 structured JSON。

---

## 高质量标注

当前优先 benchmark：

\[
\boxed{\text{Qwen3-VL-30B-A3B-Instruct}}
\]

特点：

```text
30B total
~3B active language params/token
MoE
```

在短 caption workload 下性价比很好。

---

## 最高质量候选

如果 benchmark 显示明显更强：

\[
\boxed{\text{Qwen3-VL-235B-A22B-Instruct}}
\]

由于 API 成本仍然很低，可以直接用于全部 Stage D 数据。

---

# 23. VLM 输入格式

不要直接给：

```text
16×16 PNG
```

。

应该构造两个视图。

---

## View 1

```text
16/32px texture
       ↓
nearest-neighbor
       ↓
512×512
```

禁止：

```text
bilinear
bicubic
```

避免破坏 pixel boundary。

---

## View 2

```text
original texture
       ↓
4×4 repeat
       ↓
nearest upscale
       ↓
512×512
```

让 VLM 看：

- pattern；
- seam；
- orientation；
- repeated structure。

---

# 24. 同时提供 Metadata

VLM 输入还包含：

```text
filename
block_id
display_name
namespace
known weak tags
```

原则：

\[
\boxed{\text{不要让 VLM 猜已经知道的事实}}
\]

VLM 主要负责补充：

```text
visual properties
```

。

---

# 25. VLM 输出 JSON

推荐：

```json
{
  "material": "stone",
  "form": "bricks",
  "state": ["mossy", "weathered"],
  "dominant_colors": ["dark gray", "muted green"],
  "pattern": "rectangular masonry",
  "surface": "rough",
  "directionality": "horizontal",
  "details": [
    "small cracks",
    "irregular moss patches"
  ],
  "emissive": false,
  "tileability": "likely",
  "short_caption": "dark gray mossy stone bricks",
  "detailed_caption": "A dark gray pixel-art stone brick texture with irregular muted-green moss patches and small cracks.",
  "uncertainty": 0.08
}
```

---

# 26. VLM Prompt 原则

System prompt：

```text
You are annotating low-resolution voxel-game block textures.

Use supplied metadata as ground truth when available.
Describe only visually observable properties.
Do not invent a Minecraft item or block name.
If the material cannot be determined, output "unknown".
Do not infer gameplay function, lore, rarity or provenance.
Preserve uncertainty.
Return valid JSON only.
```

标注任务：

```text
consistency > creativity
```

推荐：

```text
temperature = 0
max_tokens = 256
```

---

# 27. API 标注成本

按照约：

```text
700 input tokens / sample
180 output tokens / sample
```

估计。

目前 API 成本低到：

\[
\boxed{\text{不应该为了省几十元明显牺牲标注质量}}
\]

粗略人民币级别：

```text
50K Qwen3-VL-8B / 30B API：
几十元

50K 235B：
约百元级
```

实际价格需以调用当天官方价格为准。

因此如果：

```text
235B 在 MC-Texture-CaptionBench 明显更准
```

完全可以全量使用 235B API。

---

# 28. 本地 VLM GPU 方案

如果数据不能上传云端或希望完全可复现，可以本地部署。

## A100 40GB

推荐：

```text
Qwen3-VL-8B BF16
```

优先追求 batch throughput。

---

## A100 80GB

推荐：

```text
Qwen3-VL-30B-A3B BF16
```

并把：

```text
max_model_len
```

限制为：

```text
4096 / 8192
```

，不浪费显存给超长 context。

---

## H100 / H200 / Ada 48G+

推荐：

```text
Qwen3-VL-30B-A3B FP8
```

。

---

# 29. 最终 VLM 策略

最终建议：

```text
Step 1
先做 500–1000 张 MC-Texture-CaptionBench

Step 2
比较 Flash / 30B-A3B / 235B

Step 3
如果差异很小：
→ 30B-A3B

如果 235B 明显更好：
→ 235B API

如果需要完全离线：
→ 30B-A3B local
```

人工最终复核：

\[
\boxed{1K\text{–}3K}
\]

golden samples。

---

# 30. Text Encoder

最终生成模型不直接使用 VLM。

VLM 只负责：

```text
dataset annotation
```

。

生成模型的 text condition 固定使用：

\[
\boxed{\text{Qwen3-VL-Embedding-8B（text-only）}}
\]

。

训练前：

```text
canonical prompt text
  ↓
frozen Qwen3-VL-Embedding-8B（禁止输入图片）
  ↓
L2-normalized pooled embedding [4096]
        ↓
offline FP16 mmap [N,1,4096]
```

训练时：

\[
\boxed{\text{完全不加载 text encoder}}
\]

。

---

# 31. 为什么预计算文本 Embedding

节省：

- VRAM；
- forward time；
- tokenizer / text model 开销。

训练 GPU 上只有：

```text
MC-FlowDiT
```

。

---

# 32. 完整训练 Curriculum

## Stage A — Generic Material

数据：

```text
MatSynth CC0
+
Material Maker CC0
```

规模：

\[
200K\text{–}400K
\]

+ procedural 50K–150K。

训练目标：

```text
unconditional texture prior
```

建议：

```yaml
steps: 100k-150k
lr: 3e-4
effective_batch: 128-256
weight_decay: 0.03
```

---

# 33. Stage A.5 — Pixel Bridge

数据：

```text
pseudo-pixel materials
+
Kenney CC0
```

规模：

\[
50K\text{–}150K
\]

建议：

```yaml
steps: 20k-50k
lr: 1e-4
```

---

# 34. Stage B — MC Domain Adaptation

数据：

```text
90% audited MC
10% pixel-prior replay
```

避免完全遗忘 generic texture prior。

建议：

```yaml
steps: 30k-80k
lr: 1e-4
tile_loss_weight: 0.03
```

---

# 35. Stage C — Weak Text Alignment

数据：

```text
MC image
+
structured metadata caption
```

加入：

```text
condition dropout = 10–15%
```

用于 CFG。

建议：

```yaml
steps: 20k-50k
lr: 5e-5 ~ 1e-4
```

---

# 36. Stage D — High-quality VLM Fine-tune

数据：

```text
5K–20K high-quality captions
```

如果 API 很便宜且质量足够好：

```text
可以全量 MC 数据都生成 detailed captions
```

。

建议：

```yaml
steps: 5k-20k
lr: 2e-5 ~ 5e-5
```

避免过拟合。

---

# 37. 8GB 单卡训练 Infra

主模型 45–50M 参数时目标是：

\[
\boxed{\text{高吞吐训练，而不是勉强塞进去}}
\]

---

# 38. Precision

优先：

```text
BF16
```

如果 GPU 不支持 BF16：

```text
FP16 + GradScaler
```

禁止：

```text
纯 FP32 activation
```

。

---

# 39. Attention

使用：

```python
torch.nn.functional.scaled_dot_product_attention
```

。

让 PyTorch 自动选择：

```text
FlashAttention
memory-efficient SDPA
math fallback
```

。

不要显式 materialize：

\[
QK^T
\]

attention matrix。

---

# 40. Optimizer

45M 主模型：

```python
torch.optim.AdamW(
    params,
    fused=True
)
```

推荐：

```yaml
betas: [0.9, 0.95]
weight_decay: 0.03
```

。

不要默认用 AdamW8bit。

因为 45M optimizer state 本身不大，fused AdamW 通常更快。

---

# 41. torch.compile

推荐 benchmark：

```python
model = torch.compile(model)
```

再比较：

```text
default
max-autotune
max-autotune-no-cudagraphs
```

。

8GB 显存下不要盲目开启额外 CUDA Graph workspace。

---

# 42. Activation Checkpoint

45M Base：

\[
\boxed{\text{默认关闭}}
\]

。

只有：

- 64×64 profile；
- 更大模型；
- batch 无法满足 throughput；

再开启。

checkpoint 是：

```text
用计算换显存
```

会降低训练速度。

---

# 43. EMA

EMA 权重放：

```text
CPU
```

。

推荐：

```yaml
ema_decay: 0.9999
update_every: 8
```

。

不要长期占 GPU VRAM。

---

# 44. Batch

先自动 probe：

```text
32
64
128
256
```

选择最大显存不超过：

```text
7.2GB
```

的 microbatch。

Base 第一目标：

```text
micro_batch = 128
```

。

尽量：

\[
\boxed{\text{microbatch 大，gradient accumulation 小}}
\]

。

---

# 45. DataLoader

推荐：

```python
pin_memory=True
persistent_workers=True
prefetch_factor=2~4
```

GPU copy：

```python
x.cuda(non_blocking=True)
```

optimizer：

```python
optimizer.zero_grad(set_to_none=True)
```

。

---

# 46. 数据存储

不要训练时读几十万小 PNG。

推荐：

```text
images.uint8.mmap
metadata.parquet
splits.json
```

固定 32×32：

```text
uint8[N,32,32,3]
```

。

1M 张也只有约：

\[
3.07GB
\]

。

---

# 47. 所有昂贵预处理都离线

训练 loop 中禁止：

```text
4K crop
PNG decode
VLM
text encoder
palette clustering
complex resize
```

。

训练时只做：

```text
mmap
→ simple augmentation
→ GPU
```

。

---

# 48. 64×64 Profile

32×32 / p2：

\[
256
\]

tokens。

64×64 / p2：

\[
1024
\]

tokens。

attention 理论计算约：

\[
16\times
\]

。

因此第一版：

\[
\boxed{32×32 主训练}
\]

。

如果以后需要 64：

```text
32 pretrained
→ 64 finetune
```

。

---

# 49. 推理范式

从：

\[
x_1=z
\]

沿 flow ODE 积分到：

\[
x_0
\]

：

\[
\frac{dx}{dt}=v_\theta(x,t,c)
\]

。

---

# 50. Solver

第一轮 benchmark：

```text
Euler
Heun
Midpoint
```

默认：

\[
\boxed{\text{Heun}}
\]

。

step sweep：

```text
8
12
16
24
32
```

第一版推荐：

\[
\boxed{16\text{–}24}
\]

steps。

---

# 51. Classifier-Free Guidance

Stage C/D：

```text
condition_dropout = 0.10~0.15
```

推理：

\[
v=
v_u+
s(v_c-v_u)
\]

CFG sweep：

```text
1.0
1.5
2.0
2.5
3.0
```

预计最佳 CFG 不会很高。

---

# 52. 少步蒸馏

Base 稳定以后：

```text
Teacher:
16–24 steps

Student:
4–8 steps
```

。

可以做：

```text
step distillation
guidance distillation
```

。

不要第一版就做。

---

# 53. 最终推理 Pipeline

```text
User Prompt
    ↓
Text normalization
    ↓
Qwen3-VL-Embedding-8B text-only encoder
    ↓
4096-d pooled condition token
    ↓
Gaussian noise [3,H,W]
    ↓
MC-FlowDiT
    ↓
Flow ODE solver
    ↓
RGB texture
    ↓
clamp / uint8
    ↓
optional palette postprocess
    ↓
PNG + tiled preview
```

---

# 54. 推理 API

示例：

```json
POST /generate

{
  "prompt": "dark mossy stone bricks with small cyan crystals",
  "seed": 1234,
  "steps": 20,
  "cfg": 2.0,
  "size": 32
}
```

返回：

```text
texture.png
tiled_preview.png
seed
model_version
sampling_config
```

。

---

# 55. 部署

第一版：

```text
PyTorch
torch.compile
safetensors
FastAPI
```

。

不要一开始 TensorRT。

路线：

```text
PyTorch correctness
→ compile benchmark
→ production benchmark
→ TensorRT only if needed
```

。

---

# 56. Evaluation

不要只看 FID。

必须加入 MC-specific metrics。

---

# 56.1 Seam Score

\[
S_{seam}
=
\frac12
(
\|L-R\|_1+
\|T-B\|_1
)
\]

越低越好。

同时输出：

```text
4×4 tiled preview
```

。

---

# 56.2 Prompt Alignment

使用固定 prompt 集进行人工盲评，并由下节的轻量属性分类器统计材质、形态、状态和颜色命中率。Qwen3-VL-Embedding 不接收生成图片，也不承担图文相似度评分。

---

# 56.3 Attribute Accuracy

训练轻量 classifier：

```text
material
form
state
color
```

衡量 prompt 是否真正控制视觉属性。

---

# 56.4 Diversity

同一 prompt：

```text
32 seeds
```

测：

```text
pHash distance
RGB distance
LPIPS
palette diversity
```

。

---

# 56.5 Memorization

对每个生成结果找最近训练样本：

```text
pHash
RGB L2
LPIPS
```

。

小数据场景必须检查 memorization。

---

# 57. 标准 Validation Visualization

每个 prompt 固定输出：

```text
single texture
4×4 tiled preview
palette
prompt
seed
CFG
steps
```

。

---

# 58. 必做 Ablation

## Data

```text
scratch
vs
generic pretrain
vs
+ pixel bridge
vs
+ MC adaptation
```

---

## Architecture

```text
20M
vs
45M
vs
100M
```

```text
16/p1
vs
32/p2
```

```text
Double+Single
vs
All Single
```

---

## Training

```text
tile loss on/off
toroidal augmentation on/off
```

---

## Text

```text
weak metadata
vs
VLM detailed caption
```

---

## Inference

```text
Euler / Heun
8 / 12 / 16 / 24 / 32 steps
CFG 1–3
```

---

# 59. 推荐项目目录

```text
mc-texture-gen/
├── configs/
│   ├── data/
│   ├── model/
│   └── train/
│
├── src/
│   ├── data/
│   │   ├── matsynth_builder.py
│   │   ├── materialmaker_builder.py
│   │   ├── modrinth_crawler.py
│   │   ├── license_audit.py
│   │   ├── mc_extract.py
│   │   ├── dedupe.py
│   │   ├── weak_labels.py
│   │   ├── vlm_caption.py
│   │   └── build_mmap.py
│   │
│   ├── model/
│   │   ├── rope2d.py
│   │   ├── norm.py
│   │   ├── modulation.py
│   │   ├── double_stream.py
│   │   ├── single_stream.py
│   │   └── mc_flow_dit.py
│   │
│   ├── train/
│   │   ├── flow.py
│   │   ├── losses.py
│   │   ├── ema.py
│   │   └── train.py
│   │
│   ├── infer/
│   │   ├── solver.py
│   │   ├── sample.py
│   │   └── api.py
│   │
│   └── eval/
│       ├── seam.py
│       ├── attributes.py
│       ├── diversity.py
│       └── memorization.py
│
├── data/
├── checkpoints/
├── outputs/
└── scripts/
```

---

# 60. 推荐 model.yaml

```yaml
model:
  name: MCFlowDiTBase

  image_size: 32
  in_channels: 3
  patch_size: 2

  hidden_size: 512
  num_heads: 8
  mlp_ratio: 3.0

  double_stream_blocks: 3
  single_stream_blocks: 6

  text_dim: 4096
  max_text_tokens: 1

  qk_norm: rmsnorm
  rope: 2d
  activation: swiglu
  bias: false

  modulation:
    type: adaln_zero
    shared_by_stage: true

  cond_dropout: 0.12
```

---

# 61. 推荐 train.yaml

```yaml
train:
  precision: bf16
  compile: true

  optimizer:
    name: adamw
    fused: true
    lr: 3.0e-4
    betas: [0.9, 0.95]
    weight_decay: 0.03

  scheduler:
    type: cosine
    warmup_steps: 1000

  grad_clip: 1.0

  activation_checkpointing: false

  batch:
    auto_probe: true
    preferred_micro_batch: 128
    max_vram_gb: 7.2

  ema:
    enabled: true
    device: cpu
    decay: 0.9999
    update_every: 8

  flow:
    timestep_sampling: uniform
    loss: mse

  tile_loss:
    enabled: true
    weight: 0.03
    max_t: 0.7
    border_width: 2
```

---

# 62. Environment

建议：

```text
Python 3.11
PyTorch 2.x current stable
CUDA matching driver
```

依赖：

```text
torch
torchvision
transformers
datasets
huggingface_hub
safetensors
numpy
pillow
opencv-python-headless
einops
pyarrow
xxhash
imagehash
fastapi
uvicorn
```

可选：

```text
bitsandbytes
accelerate
webdataset
vllm
```

---

# 63. 最终执行顺序

## Phase 0 — Smoke Test

```text
5K procedural textures
20M Tiny model
```

验证：

```text
dataset
loss
SDPA
compile
EMA
sampling
tile preview
```

。

---

## Phase 1 — Build Generic Dataset

```text
MatSynth CC0
Material Maker CC0
```

生成：

```text
250K–400K generic textures
```

。

---

## Phase 2 — Pixel Bridge

生成：

```text
50K–150K pseudo-pixel
+
Kenney assets
```

。

---

## Phase 3 — MC Dataset

```text
Modrinth crawler
→ project license
→ asset audit
→ block texture extraction
→ dedupe
→ project split
```

目标：

```text
20K–50K clean MC textures
```

。

---

## Phase 4 — Weak Labels

解析：

```text
filename
models
blockstates
lang
```

生成 structured weak labels。

---

## Phase 5 — VLM Benchmark

构建：

```text
500–1000 MC-Texture-CaptionBench
```

比较：

```text
Qwen3-VL-Flash
Qwen3-VL-30B-A3B
Qwen3-VL-235B-A22B
```

。

---

## Phase 6 — Detailed Captions

按 benchmark 结果选择 VLM。

对：

```text
5K–20K
```

或全量 MC 数据生成 detailed captions。

人工检查：

```text
1K–3K
```

golden set。

---

## Phase 7 — Train Base

```text
45M MC-FlowDiT
32×32
patch=2
```

训练：

```text
Stage A
→ A.5
→ B
→ C
→ D
```

。

---

## Phase 8 — Scaling / Native16

训练：

```text
20M
45M
100M
```

和：

```text
16×16 / patch=1
```

。

---

## Phase 9 — Few-step Model

Base 训练稳定以后：

```text
16–24 step teacher
→
4–8 step student
```

。

---

# 64. 第一版推荐最小规模

如果希望尽快做出结果：

```text
Generic:
250K MatSynth CC0 patches

Pixel Bridge:
80K pseudo-pixel samples

MC:
20K–30K audited textures

VLM:
10K detailed captions

Model:
45M

Resolution:
32×32 / p2

Training:
BF16 + SDPA + fused AdamW + compile

Inference:
Heun 20 steps
CFG ~2
```

这已经足够做出一个完整第一版。

---

# 65. 后续研究价值最大的方向

不建议下一步只是：

```text
45M → 450M
```

更值得做：

## Material-family generation

一次生成：

```text
stone
stone bricks
chiseled stone
cracked stone
mossy stone
ore-in-stone
```

共享 material/style latent。

---

## Multi-face block generation

一次生成：

```text
top
side
bottom
```

并保持材质一致。

---

## Palette-conditioned generation

输入：

```text
prompt
+
palette
```

生成统一风格资源包。

---

## Toroidal attention / positional encoding

从网络结构上把空间视作：

\[
\mathbb Z_H\times\mathbb Z_W
\]

周期空间。

---

## Resource-pack generation

最终从：

```text
single texture generation
```

升级成：

```text
coherent texture pack generation
```

，这是比单图生成更有产品与研究价值的方向。

---

# 66. 核心参考路线

架构：

- Stable Diffusion 3 — Rectified Flow + MMDiT
- FLUX.1 / FLUX.2 — Flow Matching Transformer
- Pixel-space Diffusion / PixelDiT

数据：

- MatSynth
- Material Maker
- Kenney
- Modrinth

文本 / VLM：

- Qwen3-VL-Embedding-8B（仅 text-only 条件注入）

训练 Infra：

- PyTorch SDPA
- fused AdamW
- torch.compile
- CPU EMA
- mmap / sequential dataset storage

---

# 67. 最终结论

本项目最合适的第一版不是一个大模型，而是：

\[
\boxed{
\textbf{约45M参数的 Pixel-space Rectified Flow Transformer}
}
\]

结合：

\[
\boxed{
\textbf{
通用材质预训练
→ Pixel Bridge
→ MC 域适配
→ Metadata 弱标注
→ VLM 高质量文本对齐
}
}
\]

。

数据量目标：

\[
\boxed{
250K\text{–}500K\ generic
+
20K\text{–}50K\ MC
}
\]

已经足够现实。

训练目标：

\[
\boxed{
8GB\ VRAM
+
高吞吐
}
\]

而不是靠 checkpoint / offload 勉强运行。

VLM 标注由于 API 成本只在几十到百元级，应：

\[
\boxed{
\text{先 benchmark，再按准确率选模型}
}
\]

而不是为了节省少量 API 成本牺牲 caption 质量。

最终开发重点应该放在：

1. 数据许可与去重；
2. MC domain quality；
3. pixel fidelity；
4. tileability；
5. prompt adherence；
6. 训练吞吐；
7. 少步推理。

这是一条可以直接进入实现的 v2.0 主方案。
