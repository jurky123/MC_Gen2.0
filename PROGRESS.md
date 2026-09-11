# MC-Gen2.0 项目进度

> 更新日期：2026-09-11；当前方案：v2.3（数据处理与条件编码定型）

## 新训练课程

- Stage 1：全部 Kenney、itch.io 免费像素风素材，加全部 MC 纹理进行预训练。
- Stage 2：全部 MC 纹理加文件名、模型、语言文件和命名空间生成的弱标注。
- Stage 3：精标注 MC 子集精调；首个组成部分确定为 James-A/Minecraft-16x-Dataset，后续还可加入人工复核的精标注 MC 数据。

旧的真实材质预训练和 pseudo-pixel bridge 路线已废止。

## 当前数据状态

- legacy MC：31,713 张 16x16 图片，已构建 `data/build/mc_b16/`。
- Modrinth：100 个资源包，提取 12,883 张 block 纹理，精确去重后约 10,852 张。
- NathMen12/16xModdedMinecraft-TextToImage：两个 Parquet 分片已完整落盘，共 1,034,057 条；已转换为 `data/build/mc_text2image32/`，包含 32x32 RGB mmap、metadata Parquet 和 train/val/test 划分（941,104 / 43,184 / 49,769）。这是 Stage 1/2 的主 MC 数据集。
- Kenney Pixel Assets：14 包已经下载并解包，原始目录约 53.1 MB、共发现 2,106 个 PNG/JPG；连通域处理已经完成，在 `data/processed/kenney_components32/` 得到 4,312 个去重 32x32 样本，处理清单无报错。
- itch.io Free Pixel Art Assets：原始下载达到 30.022 GiB 后停止；透明连通域处理已完成，独立清单得到 1,862,160 个样本；与其他 Stage 1 来源做全局内容去重后纳入 1,859,029 条。
- evilsocket/alucard-sprites：已完整下载 6 个 Parquet、312,550 条；独立处理得到 282,120 个唯一 32x32 样本，精确重复 30,430 条、错误 0，并保留原始 `text` 作为 `weak_prompt`；全局去重后纳入 Stage 1 共 282,044 条。
- Kaggle ebrahimelgazar/pixel-art：已下载并解压；实际含 89,400 个 JPEG，但只有 1,722 个唯一原图，最终处理为 1,722 个唯一 32x32 样本。
- nyuuzyou/OpenGameArt-OGA-BY-3.0：与图像训练相关的 2D Art 两个 ZIP（约 2.51 GB）及其元数据/署名文件已完整下载并通过 ZIP 校验；处理得到 4,574 个唯一 32x32 样本，全局去重后纳入 Stage 1 共 4,555 条。音乐、音效和 3D 内容不进入本项目。
- James-A/Minecraft-16x-Dataset：已完整下载 1,519 条（train 1,366 / validation 70 / test 83）；处理得到 1,498 个唯一 32x32 精调样本，精确重复 21 条、错误 0，并保留精细描述作为文本条件。
- Stage 1 统一训练集已升级为 RGBA：`data/build/stage1_32_rgba/`，共 3,185,719 条 32x32x4，train/val/test = 3,004,255 / 91,566 / 89,898。透明像素素材保留 alpha 通道，MC 纹理 alpha 全不透明；来源与旧 RGB 版一致。
- Stage 1 统一训练集：已完成 `data/build/stage1_32/` 构建，共 3,185,719 条，其中 MC 1,034,057 条、Kenney 4,312 条、itch 1,859,029 条、Kaggle 1,722 条、Alucard 282,044 条、OpenGameArt 4,555 条。划分为 train 3,004,255 / val 91,566 / test 89,898；mmap、Parquet 和划分文件的行数与字节数均已校验一致。

## 像素素材处理约定

- spritesheet 不再默认整图缩放或盲目固定网格切片。
- 对含透明通道的图片，以 alpha 非透明像素的二维连通区域计算每个物件的包围盒；只有文件名明确给出 8x8、16x16、32x32 等规格时才优先使用显式网格。
- 裁剪后的物件保持长宽比，使用 nearest-neighbor 进行降采样或上采样，并居中放入 32x32 透明画布；保留原始像素边缘，不引入平滑插值。
- 跳过无法可靠拆分的全不透明大图以及空白、近纯色等低信息样本；输出按内容 SHA 去重。
- 每个样本在 JSONL 清单中保留来源文件、压缩包/素材目录、裁剪框、切分方法等信息，以便复查与重新构建。
- 旧的固定网格试验目录 `data/processed/itch_pixel32/` 属于废弃中间产物，不进入训练。

## 已移除数据

- `data/raw/matsynth/` 已删除。
- `data/raw/ambientcg/` 已删除。
- MatSynth、ambientCG、Material Maker 和 pseudo-pixel 不再进入训练。

项目仅用于学习研究。仍记录像素素材的来源、作者和许可声明，保证可追溯。

## 框架状态

- MC-FlowDiT 模型、训练、推理、评估和 API 框架已完成。
- 冒烟测试已通过；MC 弱标签和离线文本嵌入流水线已可用。
- 旧 OVAWARE 本地子集保留用于校验和兼容，后续训练以 NathMen12 TextToImage 版本为主。

## 训练配置

- 生成模型输入改为 4 通道 RGBA（`configs/model/*.yaml` 的 `in_channels: 4`）；数据加载器 `MmapImageTextDataset` 支持 3/4 通道，读取 3 通道 mmap 时自动补 alpha=不透明，读取 4 通道时原样保留。
- Stage 1 启用梯度累计：micro_batch 16 × `gradient_accumulation: 8` = 有效 batch 128；`steps: 234710`（约 10 个 epoch）。
- 训练循环每 `save_every` 步滚动保存 `checkpoints/stage_1/latest.pt`，并支持 `--resume`；不再只在 epoch 末保存。
- 推理端（`src/infer/sample.py`、`api.py`）按 `model.cfg.in_channels` 采样并输出 RGBA。
- 冒烟配置 `configs/train/smoke_rgba.yaml` 用于快速验证 4 通道 + 梯度累计 + 中间 checkpoint。

## 文本条件编码器

- 正式选择 `Qwen/Qwen3-VL-Embedding-2B`，不选 8B 版本，以兼顾语义质量、离线编码速度和本地资源占用。
- 该模型在本项目中只作为 prompt 的文本编码器：只输入文本，只输出生成模型需要的文本条件。
- 明确不使用它处理图片，也不用于去重、检索、筛选、标注生成或数据质检。
- 编码器全程冻结；训练前离线预计算 L2-normalized 2048 维 pooled embedding，以 FP16 保存为每样本一个条件 token。训练 MC-FlowDiT 时不加载 Qwen 权重。
- Stage 1、Stage 2、Stage 3 和推理端使用同一模型、同一 instruction 与同一规范化方式，避免条件空间漂移。
- 当前代码仍是 768 维 SigLIP2/占位编码接口，迁移到 Qwen3-VL-Embedding-2B 尚待实现和冒烟验证。

## 下一步

- 对 Stage 1 统一训练集做分来源随机抽样质检，重点检查极端长宽比、动画帧和残留整图。
- 完成像素素材的来源清单与许可元数据校验。
- 使用全部 MC 加弱标注构建 Stage 2。
- 为 James-A 精标注字段生成稳定的短描述/详细描述两种文本视图，并构建 Stage 3 数据集。
- 将文本条件流水线从 768 维 SigLIP2/占位实现迁移为冻结的 Qwen3-VL-Embedding-2B（2048 维单 token、离线 FP16）。
