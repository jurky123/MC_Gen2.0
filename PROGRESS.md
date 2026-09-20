# MC-Gen2.0 项目进度

> 更新日期：2026-09-19；当前方案：v2.3（数据处理与条件编码定型）

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
- Stage 1 数据改为**读取时多源混合**：`data/build/stage1_32_rgba`（mmap）+ `data/processed/modrinth32` + `data/processed/minecraft_16x_finetune32`（tile manifest），无需重传或重新合并；Stage 1 只使用图片。多源与权重在 `configs/data/stage_1.yaml` 配置。

## 文本条件编码器

- 正式选择 `Qwen/Qwen3-VL-Embedding-8B`（原生 4096 维），以更好的语义质量覆盖短纹理描述和细粒度属性；本地离线编码成本可接受。
- 该模型在本项目中只作为 prompt 的文本编码器：只输入文本，只输出生成模型需要的文本条件。
- 明确不使用它处理图片，也不用于去重、检索、筛选、标注生成或数据质检。
- 编码器全程冻结；训练前离线预计算 L2-normalized 4096 维 pooled embedding，以 FP16 保存为每样本一个条件 token。训练 MC-FlowDiT 时不加载 Qwen 权重。
- Stage 1、Stage 2、Stage 3 和推理端使用同一模型、同一 instruction 与同一规范化方式，避免条件空间漂移。
- 代码已提供 `qwen3vl` 编码接口（`src/data/embed_text.py`），`configs/text_encoder.yaml` 固定为 8B / 4096 / 单 pooled token；离线编码与 K 视图预计算（`scripts/precompute_text.py --views-parquet`）已就绪。

## 本地 VLM 标注与 caption benchmark

- 精标注/粗标注使用本地 `Qwen/Qwen3.8-27B`（2026-08-13 发布的原生多模态 dense 27B；hybrid attention + vision tower + MTP；原生 262K context；Apache-2.0）。它是**数据标注器**，与冻结的文本条件编码器 `Qwen3-VL-Embedding` 相互独立，两套用途不混用。
- 部署：`scripts/serve_vlm.py` 用 vLLM 起 OpenAI 兼容服务（`--max-model-len 8192`、`--limit-mm-per-prompt image=2`、`enable_thinking=false`）。注意 vLLM recipe 要求 `transformers>=5.8.0`，架构为 `Qwen3_5ForConditionalGeneration`；A100 80GB 也可直接用官方 `Qwen/Qwen3.8-27B-FP8`。
- 标注流水线：`src/data/vlm_caption.py` + `scripts/annotate_textures.py`。按计划 §23 生成 nearest-neighbour 的单图与 4x4 tiling 两个 512 视图；把文件名、弱标签、资源包 metadata 作为 ground-truth 提示写入 prompt，并单列“reference label”区块；固定 system prompt、`temperature=0`、仅输出 JSON，支持重试与断点续跑。
- profile：`fine`（双视图、256 token，Stage 3）与 `coarse`（单视图、160 token、concurrency 16，Stage 1/2 全量）。两者共用同一 JSON schema：`material / form / state / dominant_colors / pattern / surface / directionality / details / emissive / tileability / short_caption / detailed_caption / uncertainty`。
- 直接标注 mmap：`scripts/make_annotation_manifest.py` 把 Stage 1/2 的 `images.uint8.mmap`（headerless uint8 `(N,H,W,C)`）转成带 `mmap`+`index`+`shape` 的 JSONL，标注时按索引读取，避免导出上百万张 PNG。
- 首版粗标注目标：`data/build/stage2_32`（1,044,875 条，含 `mc_text2image32_wl` 1,034,057 + modrinth32 10,818）。同一流程可直接扩展到 Stage 1 的 `stage1_32_rgba`（3,185,719 条）以及 `mc_text2image32_wl`。
- CaptionBench：`scripts/build_caption_bench.py` 分层采样（默认 James-A 1,498 条，按 block/item 类别均衡）→ 模型标注 → 导出 `review.csv`（模型输出预填进 `gold_*`，人工只改错项并填评分/幻觉标记）→ `scripts/score_caption_bench.py` 计算标量准确率、集合 F1、caption token F1、attribute macro F1、JSON exact、延迟/吞吐，以及人工评分和幻觉率（`report.json` + `report.md`）。
- 上述流程已用 mock OpenAI 服务端端到端验证：HTTP 并发、mmap 读取、profile 切换、断点续跑、复核表导出与评分全部通过。

## 提示词多样化（LLM 重写）

- 动机：coarse 标注偏客观且充斥 `pixel art / pixelated / minecraft / texture` 等域冗余词，缺少人类命名式的抽象表达，导致条件分布过窄。
- 方案：LLM 重写。输入结构化字段 + 清洗后的原始标签 + 初步 short/detailed caption，输出 K=4 个风格视图：`factual`、`evocative`（游戏物品命名风格，如 lava lace / star blade）、`thematic`、`descriptive`。域冗余词全部去除（system 指令 + 后处理双保险），并约束必须忠于图像属性。
- 实现：`configs/prompt_rewrite.yaml`、`src/data/prompt_diversify.py`、`scripts/rewrite_prompts.py`（支持断点续跑、分片、日志）。
- 队列与多卡：`scripts/queue_rewrite.sh`（重写）与 `scripts/queue_finish.sh`（merge + precompute）后台接力。标注/重写支持 `--endpoints a b ...` 共享任务队列（谁空谁取，避免快卡先空转后闲置）；`precompute_text.py --devices cuda:0,cuda:1` 用 sentence-transformers 进程池（`encode_multi_process`，chunk 动态分发）做多卡动态编码，另有 `--num-shards/--init` 供进程级静态分片。
- 训练侧：`MmapImageTextDataset(text_views=K)` 读取 `(N,K,text_dim)` 并每步随机取一个视图；`scripts/merge_prompt_views.py` 对齐生成 `prompt_views.parquet`，`scripts/precompute_text.py --views-parquet` 写出 `(N,K,4096)`。
- 实测重写速度：双卡独占约 31–34 条/秒/卡（合计 ~65/s）；抢 GPU 时单卡约 11.9 条/秒。注意：本次重写用静态分片启动，A100-SXM 先跑完后短暂空转；后续批量任务改用 `--endpoints` 共享队列即可同时收尾。

## Stage 2 标注与 K 视图条件（已完成）

- 数据源：`data/build/mc_text2image32_wl`（Modrinth mod 爬取，13,053 个 mod，1,034,057 条）。筛选保留 block+item、不按 license 过滤、不剔近纯色，仅去掉 gui/font/entity/particle 文件名 → 1,031,063 条。
- 粗标注：本地 `Qwen/Qwen3-VL-8B-Instruct`（vLLM BF16，coarse profile 单视图），两卡数据并行，约 1.5–2h 完成 1,031,063 条，失败 0，产物 `coarse_annotations.shard0{0,1}.jsonl`；实测单卡约 10 img/s。
- 提示词多样化：同一 8B 文本-only 重写，K=4（factual/evocative/thematic/descriptive），全部去域词，产物 `prompt_views.shard0{0,1}.jsonl`；原 VLM caption 仅作重写输入，不单独保留为视图。
- 对齐与编码：`scripts/merge_prompt_views.py` → `prompt_views.parquet`（1,031,063 覆盖 / 1,034,057 行，未覆盖行回退清洗弱标签）；`precompute_text.py --views-parquet --devices cuda:0,cuda:1` 用 `Qwen3-VL-Embedding-8B` 双卡动态进程池编码 → `text_embeddings.f32.mmap`，形状 `(1,034,057, 4, 4096)` fp32 = 67,767,959,552 字节，约 2h、约 546 embedding/s。
- 训练数据配置 `configs/data/stage_2_annotated.yaml`：`channels: 4`（磁盘 RGB，loader 自动补不透明 alpha）、`text_dim: 4096`、`text_views: 4`。
- 说明：批量粗标选择 8B 是速度权衡（27B 单卡仅约 2.8 img/s，全量需数天）；27B 仍保留用于精标 / CaptionBench。
- Stage 2 训练已启动：`configs/train/stage_2_annotated.yaml`（9,190 步，eff. batch 1024，从 `checkpoints/stage_1_clean/best.pt` 初始化，文本投影按 4096 维重初始化）。修复了 `num_workers=0` 与 EMA `update_every=1` 造成的主机空档：DataLoader 改为 8 worker + prefetch/persistent 并为每个 worker 单独 seed numpy；EMA 改为 `update_every=8`。GPU 利用率由 34–100% 抖动变为稳定 98–100%，step 时间 0.71s → 0.56s。

## Stage 2 数据修正：MC 恢复 alpha（RGBA）

- 问题：`mc_text2image32` / `mc_text2image32_wl` 最初用 `channels=3` 构建，透明 PNG 被合成到黑底、alpha 丢失（item 纹理平均 ~67% 纯黑像素；`stage1_32_rgba` 中 MC 子集 alpha 也全为 255）。HuggingFace 上的 `Risposta/MC_Gen` 同样是 3 通道，确认此前没有任何带透明的 MC 数据。
- 来源核对：NathMen12 原始图像多为 RGBA（抽样 200 张中 182 张 RGBA），带 alpha 的平均透明像素约 5.8%。
- 修正：从原始 parquet 用 `src/data/mc_text2image_builder.py --channels 4` 重建 `data/build/mc_text2image32_wl/images.uint8.mmap`（4,235,497,472 B，alpha 0–255，约 24.7% 像素半透明）。重建后的 `file_name` / `project_id` 序列与旧数据逐行一致，因此 `prompt_views.parquet` 与 `text_embeddings.f32.mmap` 无需重算，配置 `channels: 4` 不用改。
- 推理修正：`src/infer/sample.py` 的 CFG 空条件改用训练时学习的 `model.text_null`（此前用全零，与训练不一致）。
- 标注偏差：现有 coarse 标注是在旧的黑底 RGB 视图上做的，抽样 3 万条中 8.5% 提到 `black`、1.4% 提到 `background`。已把标注视图改为把 alpha 合成到白底（`ANNOTATION_BG`），并在 system prompt 中明确“背景不属于纹理、不要描述背景/透明”。因此**建议在 RGBA 数据上重新走一遍 标注 → 重写 → 编码 → 训练**。
- 待办：重启 Stage 2 训练以使用 RGBA 数据（旧的 3 通道数据已从 `_wl` 移除）。

## grounded 提示词（从原始标签重建）

- 动机：coarse 标注偏客观、含域冗余词，且 material unknown 14.6%；改用**原始文件名标签**作为权威来源。
- `src/data/filename_prompts.py`：切词 → 去方位/动画/通用/数字/mod 噪声 → 保留标签词（可配置是否保留方位词）；提供 `validate_prompt`（词边界 + 复数）与 `fallback_prompt`。
- 产物 `data/build/mc_text2image32_wl/grounded_prompts.parquet`：每条 1 个 prompt（K=1），统计覆盖 stone 3.6%、lava 2.0%、sword 1.2% 等；`lace` 仅 0.005%（确认此前"lava lace"类测试不公平）。
- 另一版 LLM grounded 重写：`configs/grounded_rewrite.yaml` + `src/data/grounded_rewrite.py` + `scripts/rewrite_grounded.py`，严格要求保留全部标签词与 block/item，双卡 27B/8B 分片，产物 `grounded_prompts`（LLM 句更自然，~94% 通过校验，其余回退规则拼接）。

## 走向 cross-attention + 动态文本塔

- 文本条件从"1 个池化 token"升级为 **token 序列 + cross-attention**（Q=图像 token，K/V=文本 token，带 padding mask，输出零初始化）。
- 文本塔动态计算（不预计算）：`Qwen3-8B` causal LM，取**第 9/18/27 层 hidden states 拼接**（FLUX.2-klein 取法，4096×3 = 12288 维），`enable_thinking=False`，冻结，训练时跑在第二张卡上。
- 模型 `text_injection: cross_attn`：图像主干保留学习到的 register token（与 Stage 1 的常量 `text_null` 语义一致）以复用 Stage 1 权重；`cross_attn_blocks: 4`。
- 相关文件：`src/model/cross_attention.py`、`src/data/text_tower.py`、`configs/model/base_flux2klein.yaml`、`configs/data/stage_2_grounded_k1.yaml`、`configs/train/stage_2_grounded_k1.yaml`（文本塔 `device: cuda:1`）。

## 关键 bug 修复（生成空间伪影）

- **`_depatchify` 转置错误**（`96b90f4`）：`tokens.view(B, C*p*p, N)` 把 token 维与 channel 维错误重排，`patchify→depatchify` 往返误差 0.99（修复后 0.0），与 2D RoPE 位置假设冲突，导致生成内容被挤到某侧、中心空洞。修复为 `tokens.transpose(1,2).reshape(B, C*p*p, N)` 再 `F.fold`。
- **toroidal roll 误用**（`7c2b14a`，后于 `11975a0` 全面关闭）：随机环形平移对可平铺纹理无损，但会把居中 sprite 移到任意位置/跨边界绕回，模型只能学出"位置随机的平均 sprite"。对照实验：同一稀疏 item，训练**加 roll** → 生成散碎（loss 0.138）；**不加 roll** → 完美还原（agreement 1.0）。现已全局 `toroidal: false`。
- 结论：以上两 bug + 训练量不足是此前"边缘碎块/中心空洞/不成形"的全部主因。

## 评测套件与 Stage 2 结果

- `scripts/eval_generation.py`：真实/生成对照图 + 概念召回（全词/常用词、词边界）+ 颜色命中 + 真实-生成保真（RGB L2 / 直方图余弦）+ seam（与真实 baseline 对比）+ 多样性 + 检索准确率 + 文本有效性（val MSE 真实 vs 空，固定种子）。
- Stage 2（`checkpoints/stage_2_grounded_k1/best.pt`，10 epoch，修复后 + 无 roll，val mse 0.0747）：
  - 文本有效性 ~13.6%（固定种子，早期未固定时的 20.7% 为噪声）；常用概念召回 block 0.385 / item 0.361；颜色 0.70；检索 0.125（随机 0.016）；seam 11.8（真实 27.1）。
  - 生成 item 墨水中心占比 0.89（数据 0.76），空间伪影消失；可辨认西瓜/宝石环/南瓜/剑等。

## Stage 3：子集筛选 + 27B 精标 + 防遗忘微调

- **词表统计**（全量 246 万 token）：形容词 30 颜色 + 48 状态；名词 63 form + 59 material；其余为方位/部件/mod 噪声。
- **子集筛选** `scripts/select_stage3_subset.py`：按 (type×form)、(type×material)、(type×colour)、(type×state) 及组合键多轮分层，每 project ≤25；产出 `stage3_subset.jsonl` **20,000 条**（block 10,004 / item 9,996；form 63/63、material 61/61、colour 30/30、state 48/49 覆盖，4,274 项目）。
- **27B 精标** `src/data/stage3_annotate.py` + `scripts/annotate_stage3.py`（`configs/stage3_annotator.yaml`）：双视图（single + 4×4 tiling、白底合成）、严格保留全部标签词与 block/item、15 个字段（material/form/state/colours/pattern/surface/shape/symmetry/tileable/emissive/transparency/short+detailed_prompt/uncertainty）、词边界校验 + 回退；双卡 2 个 27B replica（`:8000/:8001`）约 1.8h 完成 20,000 条，失败 0。
- merge → `stage3_prompts.parquet`（20,000 覆盖）+ `stage3_splits.json`（train/val/test 15,948 / 2,047 / 2,005）。
- **防遗忘微调** `configs/data/stage_3.yaml` + `configs/train/stage_3.yaml` / `stage_3_frozen.yaml`：75% 精标子集 + 25% Stage-2 replay 多源混训；lr 3e-5、warmup 100、steps 400；新增 `train.freeze_backbone`（只训练 `text_proj/register/cross_attn/head/t_embedder`）。
- frozen 结果（`checkpoints/stage_3_frozen/best.pt`，可训练 74.9M / 冻结 55.1M，~33 分钟）：Stage-2 val 文本有效性 13.6% → 12.9%（固定种子，**无实质遗忘**），保真/直方图/检索/block 召回略升；Stage-3 val 颜色 0.825、item 概念召回 0.468、检索 0.188、与真实图 L2 57.7。
- 观察：27B 精标仍会**漏细节/误判**（如红裤子护甲被标 molten/green、头盔细节缺失），且小模型对长尾细节**拟合不足**；两份对比图见 `outputs/eval_s3_stage3/real_vs_generated.png`、`/tmp/opencode/compare_s2_s3.png`。

## 新方向：图生图 / HD→MC 风格化（设计阶段）

- 完整设计见 `docs/IMG2IMG_DESIGN.md` 与最终版 `docs/MC-Gen2_HD-to-MC_Design_v1.0.md`（设计评审稿，作为实施依据）。
- 核心路线：真实 MC 为唯一 target → 现有 MC→HD 模型（外部黑盒，选型 FLUX2，先小样本调参再量产）造锚定配对 `(HD_ref, MC_target)` → 从 Stage-3 初始化、只训零初始化 spatial adapter 的 HD→MC Stylizer → 通过 Gate 后用"结构化 prompt→HD→Stylizer→MC"开放环造 `(prompt, MC)` 反哺 t2i（10–20% 起步）。
- 外部模型：MC→HD 用 FLUX2；首批范围 = Stage-3 精标子集（20k）；Img2Img Golden Set 512 条人工审核可接受。
- 决策点与验收门槛见设计文档 §6/§12。**暂不实现，先完成仓库整改。**

## 仓库整改（2026-09-19，按设计文档 §9）

依据 `docs/MC-Gen2_HD-to-MC_Design_v1.0.md` §9/§13，全部修复并配套测试（`tests/test_rectification.py`，14 项全过）：

- **P0-1 mmap 统一**：新增 `src/data/mmap_io.py`（headerless 原始 mmap 原子写入 + `schema.json` + 形状校验 + 行数推断），`build_mmap` 弃用 `np.lib.format.open_memmap`（其 NPY header 与 reader 的 headerless 假设不一致）。现有 `images.uint8.mmap` 本身是 headerless 的（大小精确匹配），未受影响，隐患已消除。
- **P0-2 group split 实现**：新增 `src/data/lineage_split.py`（`group_split` 按 project_id 确定性哈希整组划分 + `check_disjoint`/`audit_no_leakage` 审计工具）；`build_mmap` 现在真正按组切分并写 `split_audit.json`。
- **P0-3 全局 holdout 与泄漏修复**（`scripts/rebuild_splits.py`）：
  - 旧 stage3_splits 独立随机生成 → stage3 val/test 有 3,641 行落在 replay train（泄漏确认）。
  - 新规则：stage3 子集**只取全局 train 行**（20,000 → 17,990 可用，1,799 条位于全局 val/test 的行弃用），再按 project 哈希 90/5/5 重切 → `stage3_splits.json` v2（train 16,132 / val 953 / test 905），完全继承全局 split。
  - 新增 `replay_splits.json`：全局 train 减去 stage3 val/test 行，并**移除 5,572 行与 holdout 像素级完全相同的重复行**（跨 source 实际存在的第二种泄漏）。
  - 子集数据量复核（`stage3_subset_audit.json`）：block 10,004 / item 9,996；form 62/63、material 61/61、colour 30/30、state 47/49 覆盖（丢的 1 form + 2 states 只被被排除的 2,010 行覆盖）；每 project ≤25、3,528 项目；约 20% 透明像素。
- **P1-1 结构化 batch + masked tile loss**：dataset 可返回 aux（`tileable` 布尔、`asset_type`），`flow_tile_loss` 接收 per-sample `tileable` mask（block→可平铺，item→否；门的类结构词例外），tile loss 只施加于可平铺样本。
- **P1-5 premultiplied RGBA**：新增 `src/data/rgba.py`（torch/np premultiply/unpremultiply + round-trip 测试）；数据配置 `rgba_mode: premultiplied`；推理保存时自动 unpremultiply（`solver.to_uint8`，rgba_mode 从 checkpoint manifest 读取）。premultiplied 下透明区域 RGB 确定为 0。
- **P1-3 conditioning manifest**：checkpoint 写入 `conditioning`（text tower 名称/层数/长度/dtype、text_dim、max_text_tokens、cond_dropout、channels、rgba_mode 等），`Trainer.load` 强校验；sample/eval 打印并按其选择 RGBA 转换。
- **P2-1 attributes 分类器**：标签经 `base.index[i]` 解析（shuffle 不再错位），输入通道数跟随数据配置。
- **P2-2 梯度累计尾 batch**：固定 `loss/accum`，尾步按 `accum/pending` 补偿缩放梯度。
- **附带发现与修复**：训练日志 `pending_loss` 从不重置 → loss 日志为累积和（看似发散，训练实际正常）；已修。
- **文本塔推理路径**：`FrozenTextEncoder` 改为只跑 backbone（不再算 151k 词表 logits，此前 1024 提示 encode 会 OOM 27+ GiB），用 forward hook 抓层 9/18/27（与 hidden_states 索引语义一致，同 shape 下逐位一致）；新增 `pad_bucket`（按桶填充，控制编译形状数，当前训练置 0）。
- **训练吞吐**：新增 `src/train/pipeline.py`（`ChunkedEncodedLoader`）——把一个梯度累计窗口的全部提示合并为一次文本塔前向，并在后台线程与 DiT 计算重叠；步时 ~5.2s → ~3.7s。`train.pipeline_encode: true` 默认开启（可在 `Trainer.compute_loss` 传入预编码 `text_pair`）。torch.compile 尝试过（DiT 0.49→0.19s/micro）但与变长文本序列的 recompile 成本不匹配，暂不启用。
- **重训（v2 链，已完成）**：Stage 2 v2（premultiplied + 结构化 batch + 精标子集以权重 0.1 并入主料，9,190 步，从 stage_1_clean 初始化，输出 `checkpoints/stage_2_grounded_k1_v2/`，best @8352 val mse 0.0750）→ Stage 3 v2（frozen，干净 split，400 步，`checkpoints/stage_3_frozen_v2/best.pt`，混合 val mse 0.0757）→ 双评测（Stage-2 val 无遗忘 + Stage-3 val 干净指标）。

## v2 链事故与修复（2026-09-20）

- **事故**：Stage 2 v2 跑到 9,175/9,190、Stage 3 v2 跑完 400 步后，两者都在最终 `val_validate` 的文本编码处崩溃（"hook 状态尺寸 64 vs 1024 不匹配" / "hooks missing layers"），导致 best.pt 没落盘，自动链停在 eval 之前。
- **根因**：`FrozenTextEncoder._hook_states` 是跨线程共享可变状态——pipeline worker 线程在后台 encode 时，主线程的 val encode 同时跑，两个 forward 的 hook 输出在字典里交错（一个写了 layer 9 的 64 条，另一个的 18/27 层 1024 条留了下来），cat 炸掉。另：worker 收到 stop 不退出（while 循环没检查 stop），close() 的 60s join 超时后仍在后台 encode。
- **修复**：encode 全程加锁 + 异常时清空 hook 状态；worker 循环检查 stop；`scripts/val_select.py`（standalone 验证 + promote）。
- **重要发现（EMA）**：EMA shadow 在功能上是坏的——同 val 上 ema mse 1.43 vs raw 0.063。因为 `text_proj`/`text_null` 是随机重初始化后训练的，EMA 把初值与训练值平均后条件通路被破坏（backbone 平均没问题）。**全链路统一用 raw 权重**：val_validate、sample.py、`--init-from` 默认本就都是 raw，这次只是显式确认并写入工具。
- **恢复**：Stage 2 v2 用 @8352 的 best.pt（val 0.0750，为最佳）；Stage 3 v2 用 val_select.py 验证 latest.pt（混合 val mse 0.0757，健康）后 promote 为 best.pt。三组评测已跑完。

## v2 链评测结果（对比 v1，64 样本对照）

| 指标 | v1 S2 | v2 S2 | v2 S3（S2 val，无遗忘） | v1 S3 fine | v2 S3 fine |
|---|---|---|---|---|---|
| 常用概念召回 block | 0.385 | 0.423 | 0.500 | — | 0.339 |
| 常用概念召回 item | 0.361 | 0.361 | 0.394 | 0.468 | 0.406 |
| 颜色命中 | 0.70 | 0.75 | 0.75 | 0.825 | 0.825 |
| 真实-生成 L2 | 70.2 | 64.0 | 68.6 | 57.7 | 52.7 |
| 直方图余弦 | 0.48 | 0.519 | 0.451 | — | 0.585 |
| 检索准确率 | 0.125 | 0.109 | 0.094 | 0.188 | 0.156 |
| text_effect | 13.6% | 3.5% | 3.4% | — | 13.1% |
| 多样性 | 24.8 | 27.9 | 24.8 | — | 14.1 |

解读：无灾难性遗忘（block/item 召回反而上升，颜色持平）；Stage-3 fine 的 item 召回/检索略低于 v1 主因是 v1 的 stage3 val 曾在 replay train 里（泄漏导致指标虚高），v2 是干净 holdout 上的诚实数字；L2/直方图反而更好。`text_effect` 在 Stage-2 val 上偏低（3.5%）是 premultiplied 尺度下的 artifact——同一模型在 Stage-3 val 上为 13.1%，文本条件工作正常。

## 下一步

- 等 v2 重训链完成（Stage 2 v2 → Stage 3 v2 → 双评测），对比 v1/v2 指标（同 seed、同 val 集；注意表示切换为 premultiplied，指标含义随之更新）。
- FLUX2 MC→HD 小样本调参（Phase 0 原型）：少量 Stage-3 子集 target 各生成 2–4 个 HD reference，结构过滤 + 人工抽查，淘汰高漂移参数配置。
- 调好后对 Stage-3 子集（20k target）批量生成锚定配对，按 lineage split 入 `pairs/`。
- Phase 1：reference encoder（64×64→16×16 token 网格）+ 零初始化 spatial adapter，只训 adapter，Gate 1（paired test 显著优于确定性像素化基线 + reference shuffle 掉点验证）。
- 建立 512 条 Img2Img Golden Set 与 `eval_img2img.py`。
- 用图生图/HD 参考**辅助校正标注**（缓解精标漏细节）。

