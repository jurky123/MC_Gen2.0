# MC-Gen2.0 项目进度

> 更新日期：2026-09-11；当前方案：v2.2（数据处理阶段）

## 新训练课程

- Stage 1：全部 Kenney、itch.io 免费像素风素材，加全部 MC 纹理进行预训练。
- Stage 2：全部 MC 纹理加文件名、模型、语言文件和命名空间生成的弱标注。
- Stage 3：精标注 MC 子集精调；本阶段数据暂不获取。

旧的真实材质预训练和 pseudo-pixel bridge 路线已废止。

## 当前数据状态

- legacy MC：31,713 张 16x16 图片，已构建 `data/build/mc_b16/`。
- Modrinth：100 个资源包，提取 12,883 张 block 纹理，精确去重后约 10,852 张。
- NathMen12/16xModdedMinecraft-TextToImage：两个 Parquet 分片已完整落盘，共 1,034,057 条；已转换为 `data/build/mc_text2image32/`，包含 32x32 RGB mmap、metadata Parquet 和 train/val/test 划分（941,104 / 43,184 / 49,769）。这是 Stage 1/2 的主 MC 数据集。
- Kenney Pixel Assets：14 包已经下载并解包，原始目录约 53.1 MB、共发现 2,106 个 PNG/JPG；连通域处理已经完成，在 `data/processed/kenney_components32/` 得到 4,312 个去重 32x32 样本，处理清单无报错。
- itch.io Free Pixel Art Assets：下载已达到 30 GiB 硬上限并停止，实际落盘 30.022 GiB、2,752 个素材目录、27,957 个文件。当前使用透明区域连通域方案处理；截至本次落盘，`data/processed/itch_components32/manifest.jsonl` 已记录 557,124 个样本，处理进程仍在运行，因此该数字不是最终值。
- 精标注 MC：暂不获取。

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

## 下一步

- 等待 itch 连通域处理完成，抽样检查极端长宽比、动画帧和残留整图。
- 完成像素素材的最终统计、过滤和来源清单校验。
- 修复并完成 Stage 1 合并构建器，将 Kenney、itch 与全部 MC 数据构建为统一训练集。
- 使用全部 MC 加弱标注构建 Stage 2。
- Stage 3 保持占位，不获取精标注数据。
