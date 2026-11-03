# MC-Gen2.0 项目进度

> **当前状态（2026-11-03）**：全量文生图 best 第 2,000 步已获使用者初步认可，作为优先试用候选；第 6,000 步不采用。训练按时停止，评估已完成，未更换服务默认模型。见[全量实验整理](docs/TEXT_FULLSCALE_20261103.md)。下方各阶段结论按时间保留。

> 2026-11-02 长训完成：保守分支到 22,000 步，最佳验证点为 8,000 步（loss 0.14679，终点 0.15180）。227 固定案例与 157 拟合探针三方对照完成：末点新训练图拟合略好，整体泛化未改善，暂不替换原恢复候选。见[长训结果](docs/TEXT_EXTENSION_20261102.md)。

> 2026-11-02 续训调整：高学习率重启分支在 17,303 步因持续退化主动停止；500 步 padding 对照两组均退化，不能归因于单一因素。主训练回到原 6,000 步模型及 Adam，使用原末尾 5e-6 学习率和完整 padding，继续至 22,000 步。当前目录 `outputs/text_expansion22k_conservative_20261102`。

> 2026-11-02 长程续训：按用户要求从 6,000 步接续到 22,000 步，保留 Adam 状态和全局抽样序列；固定数据及异常描述回退规则。已启用实测约 1.23 倍局部吞吐的尾部 padding 优化，训练进行中。见[续训记录](docs/TEXT_EXTENSION_20261102.md)。

> 2026-11-02 扩容进行中：按用户要求保留文件名/详细描述双监督，已构建 2,055 训练/216 验证图，全部候选图与固定 64 张文本样本完成复查；12 条异常详细描述回退到文件名监督。双文本名称选择实测 50.06%，6,000 步训练与 227 案例评估完成：原详细提示 MAE 40.235→34.526，新短名称验证 MAE 31.234→34.722，未获整体泛化提升；不替换原恢复版。补充 471 张拟合诊断显示新训练样本仍未学充分、旧样本拟合退化；每图曝光仅为之前的 28%，需要固定数据的长程对照后再归因。见[扩容方案与运行记录](docs/TEXT_EXPANSION_20261102.md)。

> 2026-11-02 文生图恢复更新：逐图筛查后构建 378 训练/43 验证的完整物品核心集，完成 4,000 步训练和固定前后对照。宽范围短名称前景 IoU 0.579→0.656，无前景输出 2/64→0/64；长描述有退化。另验证固定 50% 参数合并，未取得整体更优结果。保留完整恢复权重作为短提示试用候选，详见[恢复报告](docs/TEXT_RECOVERY_CORE_20261102.md)。

> 2026-11-02 最新：完成 1,152 张扩标、三组各 600 步的描述监督对照，以及 16 张干净物品的长/短提示隔离实验。混合监督将固定 64 条名称提示的前景 IoU 从全长描述组 0.522 恢复至 0.580（旧组 0.568）；泛化仍不稳定，未替换默认模型。详见[学习能力诊断](docs/TEXT_LEARNABILITY_20261102.md)。下方条目保留此前阶段记录。

> 2026-11-02 标注更新：Qwen3-VL-8B-Instruct 已下载并通过四个权重分片的官方 SHA256 校验；256 张视觉标注及名称合并处理结束，32 条自动候选、224 条带隔离标记，全部仍待复核。抽查发现误识别与格式限制问题，未投入训练。产物：`outputs/semantic_caption_batch256_20261102/comparison.html`；细节见 [描述审计](docs/TEXT_CAPTION_AUDIT.md)。

> 2026-11-02 当前状态：项目已重启为 v3。当前先做纯文生图，暂停高清参考图实验；正在修复文本监督。已建立 256 张描述审查集、查看 24 张并保留修订/隔离记录，尚未完成全量重标或开始新训练。详见 [描述审计](docs/TEXT_CAPTION_AUDIT.md) 与 [当前 README](README.md)。下方 2026-10-09 的暂停状态和训练方案为历史记录。

> 更新日期：2026-10-09；**项目状态：已暂停**（代码/权重/数据按原样保留，不再迭代）。
> 最终方案：v2.4（MC→HD 双生成器 + Stylizer v4 训练 + 全套评测已跑完）。

## 项目状态（2026-10-09 暂停）

- 训练与评测全部停止；GPU 空闲。最终保留的关键产物：
  - `checkpoints/stage_3_frozen_v2/`（t2i base）、`checkpoints/stylizer_v4/`（HD→MC）、
    `checkpoints/stylizer_v3/`（对照）、`checkpoints/stage_2_grounded_k1/`、`checkpoints/stage_1/`
  - `pairs/stylizer_v4/`（34,170 对）、`pairs/ood_benchmark/`（48 条 OOD 基准）
  - 评测：`outputs/eval_stylizer_v4_n64/`（公平 n=64 Gate-1）、`outputs/eval_stylizer_v3_on_v4_n64/`
- 数据集已同步到 Hugging Face **`Risposta/MC_Gen`**（`scripts/upload_dataset_hf.py`，
  新增 `mc_text2image32_wl`（21 个文件）与 `stage2_32`（6 个文件），卡片 `hf/README.md` 已更新；
  注意 `text_embeddings.f32.mmap`（~80 GB）因太大未上传，可用 `scripts/precompute_text.py` 重算；
  上传曾遇到 Xet `xorb not found` 错误（升级 `hf_xet` 到 1.7.0 + 清缓存后解决），
  另修复了 `ignore_patterns` 中 `**/x` 匹配不到根目录文件的坑）。
- 磁盘清理（2026-10-09，约 **104 GB**）：
  - `/tmp/opencode` 临时大目录（phaseD_*、dump、synth、e2e、pkg_online 等）：62 GB → ~0.1 GB
  - 废弃 checkpoint（debug/smoke/buggy/被 v2 或 v3/v4 取代的 stage_2 变体与 stylizer_phase1*）：
    51 GB → **14 GB**，保留 `stage_1`、`stage_1_clean`、`stage_2_grounded_k1`、
    `stage_3_frozen_v2`、`stylizer_v3`、`stylizer_v4`
  - 过时 pairs（tierd_*、anch_*、stylizer v1/v2、mchd_v1）：19 GB → **14 GB**，
    保留 `stylizer_v3/v4`、`ood_benchmark` 与全部 `mchd_*` 源锚点目录
  - `data/build/smoke*` 测试集已删；`data`（108 GB 数据集本体）未动
- 若重启项目：从 `PROGRESS.md` 末尾「已完成结论」与 `docs/MC2HD_RECIPE.md` 恢复上下文；单测 `python -m pytest tests/ -q`。

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
| 常用概念召回 block | 0.385 | 0.423 | 0.538 | — | 0.366 |
| 常用概念召回 item | 0.361 | 0.361 | 0.261 | 0.468 | 0.355 |
| 颜色命中 | 0.70 | 0.75 | 0.75 | 0.825 | 0.825 |
| 真实-生成 L2 | 70.2 | 64.0 | 67.6 | 57.7 | 51.3 |
| 直方图余弦 | 0.48 | 0.519 | 0.447 | — | 0.583 |
| 检索（白底 rgb） | 0.125 | 0.156 | 0.094 | 0.188 | 0.203 |
| text_effect | 13.6% | 20.1% | 20.4% | — | 24.9% |
| 多样性 | 24.8 | 35.8 | 34.6 | — | 18.4 |

解读（2026-09-20 设计评审后修正版评测）：
- 旧 v2 的 text_effect（3.5%）与 diversity 数字**无效**：text_effect 给模型喂了 straight RGBA（训练分布外），diversity 漏传 premultiplied 标记。修正后 text_effect 恢复健康（20–25%），旧 PROGRESS 里的"premultiplied artifact"解释是错的。
- retrieval 改为白底 rgb + alpha-mask 双指标；alpha-mask 三组同为 0.234（15/64）疑似大量全不透明 block 的 mask 完全相同导致的平局饱和，以白底 rgb 为主指标。
- Stage-3 v2 的 Stage-2 val item 召回 0.361→0.261 有下降（block 反升 0.423→0.538），n=64 + VLM 裁判噪声下属小幅波动，无灾难性遗忘；fine 集 L2/直方图/颜色均优于 v1，item 召回低于 v1 主因是 v1 stage3 val 有泄漏（虚高）。
- Stage-3 v2 已用 project-clean replay + 修复后代码重跑，best val mse 0.07586，in-loop 选点正常。

## MC→HD 路线定型的探索（2026-09-20）

目标是给 HD→MC Stylizer 造 `(HD, prompt, 真实 MC)` 锚定对，逐一实测了多种 MC→HD 方案：

- **FLUX.2-klein edit（图条件）**：保布局，但输出保留输入硬台阶；改 prompt/加噪去噪都不解决。
  - latent img2img（加噪 t0=0.2~0.8 再积分）**完全无用**：蒸馏模型只从纯噪声起步，中途起噪只会把图"磨糊"（四档输出几乎一致）。
  - 步数 8/16/24 输出几乎一致（8 步够用）。
- **Qwen-Image-Edit（20B，有 negative prompt）**：写实度强，但会重写布局/加场景；512 已低于其原生 ~1M 像素，256 勉强、**128 直接崩坏**（生成人脸/瓶子等无关内容）。
- **输入预处理是关键**：先 `NEAREST` 放大再低通滤波，可把输出的"网格台阶"从 61.6 降到 ~1.0：
  | 预处理 | 台阶比 | 备注 |
  |---|---|---|
  | nearest | 61.6 | 硬台阶 |
  | **bilateral+gauss12** | **1.5** | 推荐（边缘干净、细节保留好）|
  | gauss16 | 1.15 | 备选 |
  | gauss28 | 0.85 | 最平滑但结构损失大 |
  | bilateral / meanshift 单独 | 234 / 183 | 保边算子方向相反，反而留台阶 |
- **prompt 是第二关键**：
  - 写实措辞（"photorealistic PBR, studio render"）→ 照片级素材（此前"扁平卡通"是措辞导致）；
  - **去掉 `block` 一词**，否则模型把输入理解成立方体（苔藓地毯被画成 3D 方块）；
  - item 需加"居中白底"。
- **真实 MC 的原生分辨率**：抽样 3072 张 100% 为 **16×16**（我们存储为 32，即 2×2 色块）；而 Tier-D 早期用 `HD→32` 直接降采样得到的是 32×32 有效网格（比真实 MC 细 2 倍），SDEdit 还会打散网格（格子一致率 1.00→0.50）——旧 Tier-D 路径因此**废弃**。
- 结论：**MC→HD 只保留一条路**（见 `docs/MC2HD_RECIPE.md`），Tier-D 的 prompt→HD→MC 改由 Stylizer 承担。

## MC→HD 量产（17,085 对锚定）

- `scripts/build_mchd_pairs.py`：分层抽样（type×form×material 轮询）→ `NEAREST 384` → `bilateral+gauss12` → FLUX edit（8 步）→ HD 384；
  prompt = `"<Stage-3 精标描述，剔除 block>。photorealistic high-end game asset, ..."`；
  断点续跑 + 原子写入 + `config.json` 指纹 + `manifest.jsonl` provenance（含许可证）。
- 双卡分片跑完 **17,085/17,085**（Stage-3 精标全子集，block 8,475 / item 8,610），0.99s/张。
- **踩坑修复**：`stratified_rows` 里 `set` 迭代顺序受 Python hash 随机化影响 → 两个分片拿到不同的基准列表，导致 3,994 行重复生成、3,994 行漏掉（覆盖率 77%）。改为排序 token 后确定性，补齐缺失行。

## HD→MC Stylizer v1 → v3

架构：`MCFlowDiT` + `ReferenceEncoder`（stride 自适应，输出 16×16 token 与目标网格对齐）+ 零初始化 spatial adapter（位置 0/1）+ 独立 K/V 的 reference cross-attn×2；冻结主干只训参考通路（trainable 81.07M / frozen 55.08M）。

- **零初始化陷阱（重要）**：gate 与 proj 同时零初始化 → 两者梯度恒为 0，参考通路永远不学（表现为"打乱参考指标完全不变"）。修法：gate 零初始化 + proj 随机初始化（输出仍严格为 0）。
- **v1/v2（ref 64，target=HD 的确定性降采样/SDEdit）**：Gate-1 数值好看，但**免费 resize 就能赢**——target 是 ref 的确定性函数，模型学不到东西。已废弃该数据构造。
- **v3（当前基线）**：target = **真实 MC**（锚定对），ref 64→**128**，条件丢弃（ref 0.25 / text 0.15）+ 混入 30% **text-only** 真实数据（Stage-3 精标 15% + Stage-2 replay 15%，用零参考实现，统一 aux 键以便 batch 混合）。
  - 训练 4,000 步（冻结主干），覆盖 17,085 对（train 16,293 / val 792，按 project 分组）；
  - Gate-1：base t2i L2 60.3 / IoU 0.665 → **Stylizer L2 32.0 / IoU 0.902**；打乱参考退化到 55.8 / 0.704（参考确实被使用）；
  - 开环（自拟 prompt → FLUX 文生图 → Stylizer）：v2 的失败项全部修好——药水瓶（含软木塞）、青铜齿轮（带齿与中心孔）、骑士头盔（面甲缝）、黑曜石灯笼（暗框青焰）、金冠、匕首均可辨认；block/纹理类一贯稳定。
  - 产物：`checkpoints/stylizer_v3/best.pt`、`pairs/stylizer_v3/`、`outputs/eval_stylizer_v3/`、评审图 `scripts/review_stylizer.py`。

## MC→HD 第二生成器：Qwen-Image-2.1（2026-09-22）

目的：同一份 MC 出两张不同风格的 HD，作为 Stylizer 的**参考多样性**来源（多生成器混训是设计，不是权宜）。

- 配方（逐项 A/B 定型，见 `docs/MC2HD_RECIPE.md` v2）：
  `NEAREST 384 → gauss28 → Qwen-Image-2.1 edit 30 步，true_cfg=1.0（无 negative）`；
  prompt 只用原始 REALISM（P1），**不加任何"平滑"从句**（每加一句都变糊）。
- 与 FLUX 路径的两个关键差异：
  1. **Qwen 是步进蒸馏**：日志写 "Guidance scale 4.0 is ignored" 但 `true_cfg>1` 仍跑第二遍 forward（6.6s vs 3.4s）；
     `true_cfg<=1` 时 negative prompt 被整体忽略。CFG 关闭后体素伪影自己消失了 → 不需要 neg prompt。
  2. **FLUX 必须要 neg prompt**（meanabs 17–68），否则退回体素化；两条路径的 CFG 策略不同。
- `torch.compile`：必须用 `mode="max-autotune-no-cudagraphs"`——`reduce-overhead` 会撞 KV cache 的
  CUDA graph（"accessing tensor output of CUDAGraphs that has been overwritten"）。一次性编译成本 757s；
  收益 cfg4 6.60→5.71 s、cfg1 3.43→2.90 s/img。
- **8-bit 量化已排除**：A100 无 FP8 tensor core；`torchao int8_weight_only` 实测**更慢**
  （bf16 3.30s vs int8 4.41s，显存 35.7→28.6GB，画质 meanabs 2.07）；`Rin247` 的 FP8/INT8 权重
  是真量化（HTTP Range 读到 I8/F8_E4M3 + weight_scale）但布局非标准、diffusers 不认。torchao 已卸载。
- 产物：`pairs/mchd_qwen21_{a,b}`（满幅 6,019）+ `pairs/mchd_qwen21_cut_{a,b}`（抠图 11,066）。
- **许可证**：Qwen-Image-2.1 = Research License（非商用）→ 只进研究/评测数据；生产链路仍只用 FLUX（Apache-2.0）。

## 抠图（64.8%）的三个连锁 bug（2026-09-22）

17,085 对里 11,066（64.8%）是透明抠图（床侧板、矿粒、物品精灵），踩了三个坑：

1. **裁切 bug**：抠图合成白底后，两个模型都"填满画面"，输出变成被裁切的放大 3D 物件。
   → FLUX 侧加 item 前缀 `"single item centered on a plain white background, "` 修复
   （`pairs/mchd_cutout_flux_{a,b}`，11,066 张）；代价是图案型抠图（`olive_leaves` 等）被过度简化。
2. **紫色背景假象（非幻觉）**：RGBA 在 alpha=0 处的 RGB 是任意值（实测 (141,57,180)），
   用 `convert("RGB")` 裸看会显示成紫色。**显示/合成必须 alpha-over-white**。
   仓库多处旧脚本仍是裸 `convert("RGB")`，看图时要注意。
3. **硬 alpha 方块感（最后一个 bug）**：alpha-aware ref（抠图传 RGBA）原先只模糊 RGB、
   **保留原 NEAREST alpha 硬边** → 模型把锯齿轮廓当像素画复刻，输出仍有方块感。
   → 改为对**整个 RGBA 做 gauss28**（`gauss()` 对 alpha 用 σ/2）后方块感消失
   （`preprocess_ref` 已改，抠图行用 `--cutouts-only` 重生成 → `pairs/mchd_qwen21_cut_{a,b}`）。
   冒烟验证：绿镐/苔藓/床/绿宝石四行软边输出全部干净、布局正确、无放大。

## HD→MC Stylizer v4（2026-09-22，参考通路定型）

在 v3 之上改了三处，全部有单测（`tests/test_ref_conditioning.py`，15 例）：

1. **learned null-reference token**（`ref_null`）：参考通路**永远在跑**——要么吃真参考 token，
   要么吃这个可学的 null token。训练时 dropout / text-only 行、CFG 时"无参考"三者**共用同一条代码路径**。
   修复了 v3 的不一致：以前用**零图**当 null，但 `encoder(0)` 因 GroupNorm 仿射 ≠ 0（v3 实测两边恰好
   都是 0 是因为 conv/GN 全无 bias，训练后 `ref_null` 学到非零就分叉了）。
   → 数据侧显式发 `ref_present` 标志（`MmapPairDataset`=True / `zero_reference` 行=False），
   `compute_loss` 与 `val_validate` 都透传给模型。
2. **`ref_cross_positions` 可配置**：每个 ref cross-attn 块可以插在 `single_stream_blocks` 的任意位置
   （-1 = 尾部，即 v3 布局）。修了补丁里的**双跑 bug**：原先尾部循环用 `cross_at.get(k, -1)`
   按**块号**查**位置表**，导致中段块在尾部又跑一次。现在每个块严格跑一次（有测试覆盖 default/mid/重复位置/越界/无参考）。
3. **多生成器混训数据**（`scripts/build_stylizer_pairs_v4.py`）：
   同一 MC target 配**两个** HD ref（FLUX + Qwen），共享同一 prompt、同一真实 MC target；
   按 `project_id` 分组切分保证同 row 的两个变体永远同侧（有防泄漏断言）。
   满幅 6,019×2 + 抠图 11,066×2 = **34,170 对**，ref 128，target 真实 MC。
   两个生成器的 HD 统一**白底合成**后进 ref（Qwen 是 RGBA、FLUX 是 RGB，不统一会让模型把
   "透明背景"当成 Qwen 的风格特征）。

配置：`configs/{model,data,train}/stylizer_v4.yaml`；`freeze_backbone` 的可训练前缀已加 `ref_null`；
评测脚本（`eval_stylizer` / `probe` / `review` / `eval_ood`）都带 legacy 加载回退（v3 权重无 `ref_null`）。

**队列**：`scripts/run_stylizer_queue_v4.sh` 已跑完（`QUEUE-V4-DONE`，2026-09-22 18:42）。

### Stylizer v4 结果（2026-09-22）

- 训练：4,000 步，冻结主干（trainable 81.07M / frozen 55.08M），init `stage_3_frozen_v2`；
  best val flow_mse **0.09226**（step 4000）；产物 `checkpoints/stylizer_v4/{best,latest}.pt`。
- 数据：`pairs/stylizer_v4/` **34,170** 对（flux 17,085 + qwen 17,085），train 32,586 / val 1,584，
  project 分组无泄漏；QC 图 `/tmp/opencode/qc_v4/qc_v4.png`；单测 **42 passed**。
- **Gate-1**（`outputs/eval_stylizer_v4/`，val 前 16，first-n 有偏）：

  | ckpt × pairs | base t2i | Stylizer (dual CFG) | shuffled ref |
  |---|---|---|---|
  | v3 × v3 pairs（旧 FLUX-only val） | 60.3 / 0.665 | **32.0 / 0.902** | 55.8 / 0.704 |
  | v3 × **v4 pairs**（first-16） | 60.3 / 0.665 | 39.4 / 0.846 | 58.3 / 0.709 |
  | v4 × **v4 pairs**（first-16） | 60.0 / 0.664 | **38.4 / 0.884** | 53.0 / 0.805 |

- **Gate-1 公平随机采样 n=64**（2026-10-09 补测；seed 0，按 `generator` 分层 flux/qwen 各 32，
  同 seed 同样本；`eval_stylizer.py --order random`）：

  | ckpt × pairs | base t2i | Stylizer (dual CFG) | shuffled ref |
  |---|---|---|---|
  | v4 × v4 val（random64） | 49.0 / 0.589 | **23.5 / 0.872** | 61.1 / 0.699 |
  | v3 × v4 val（random64） | 49.0 / 0.589 | **23.1 / 0.882** | 47.2 / 0.727 |

  同一大样本下 **v3 与 v4 基本打平**（v3 略好 0.4 L2 / 0.01 IoU；shuffled 退化也更小）。
  first-16 上 v4 的优势来自有偏采样，不构成结论；`edge_rel` 均值被近零 target 边缘 outlier 拉爆，
  以 L2 / IoU 为准。产物：`outputs/eval_stylizer_v4_n64/`、`outputs/eval_stylizer_v3_on_v4_n64/`。
  采样函数 `src/data/pair_dataset.py::sample_balanced_indices` + 单测 `tests/test_sample_balanced.py`。
- **OOD**（`/tmp/opencode/ood_eval/report.json`）：g1_flux_edit:none L2 31.1 / IoU 0.938 /
  layout_corr 0.705；g3_qwen_edit:none L2 20.3 / IoU 1.000 / corr 0.807（Qwen 参考更贴 MC 分布）；
  scale90 最伤（L2 38.4 / corr 0.420）；unpaired t2i corr 0.668。
  MC 统计：生成图 colors ~450 / blocks2 ~0.18 vs 真实 MC colors 15 / blocks2 1.0（仍偏"照片感"）。
- **robustness**（`/tmp/opencode/probe/probe.json`，sr=1.0）：none 37.4/0.868 →
  crop5 38.2、shift4 40.5、scale95 41.9、blur 38.5、jpeg 37.8、color 37.0；
  平移/缩放有损但不崩，像素对齐捷径不主导。
- **review**：`/tmp/opencode/styl_v4_review/review.png`（n=16 first）与
  `/tmp/opencode/styl_v4_review_n32/review.png`（n=32 分层随机，按 generator 标注 HD 来源）。
- 已知 bug（当场修）：
  1. `eval_stylizer.py` 曾在加载 ckpt **之前**用 `--ref-size` 默认 0 建 `MmapPairDataset`
     → mmap 尺寸校验失败；已改为先读 `model_cfg.ref_size`（`probe` 同步修）。
  2. shuffled-ref 对照曾写成 `ds[global_index]`（把全局行号当 split 内位置），随机采样下
     `IndexError`；已改为 `ds[position]`（2026-10-09 修）。

## 已完成结论（暂停时快照）

- Stage 1/2/3 训练链路与评测链路完整可用；Stage-3 frozen（`stage_3_frozen_v2`）为最佳 t2i base。
- HD→MC：双生成器（FLUX.2-klein-4B + Qwen-Image-2.1）锚点 + Stylizer v4 数据管线跑通；
  参考条件化（adapter + cross-attn 零门控、`ref_null`/`ref_present`）单测覆盖。
- Stylizer 在公平 n=64 上与 v3 打平（23.5 vs 23.1 L2）；shuffled-ref 退化明显，参考确实起作用；
  生成纹理仍偏「照片感」（颜色数/块统计未收敛到真实 MC 分布）——若重启，优先补 Tier-D 合成数据消融。
- 数据与代码发布：HF `Risposta/MC_Gen`（数据）、GitHub `jurky123/MC_Gen2.0`（代码）。

## 暂停时不做 / 后续可选项（仅备忘）

- Tier-D 数据引擎：用 Stylizer v4 批量产 `(prompt, MC)` 合成对，按 Gate-4 做 t2i 混合消融（0/10/20%）。
- 与确定性 MC 化（`HD→16→32→量化16色→饱和×2.2`）做对照。
- 补齐 `eval_img2img.py` 与 512 条 Img2Img Golden Set。
- Stylizer Phase 2（解冻末端 blocks / LoRA、reference-strength 曲线、`ref_cross_positions` 消融）。


### Full-scale item cleaning (2026-11-02, v2)

User requested cleaning before deciding full-scale data composition; no new training started. Built `outputs/fullscale_clean_v2_20261102/catalog` from canonical native16_v3 item train/val targets. Retained 324,921 train / 41,869 val targets; quarantined 38,473 train / 3,047 val. Original pixels/catalog/split assignments and test set remain unchanged. v1 is superseded, retained for audit.

Cleaning is reversible rule-based triage with 402 distinct stratified train images plus 48 supplemental train review positions, not exhaustive visual certification. Quarantine includes assembly/render fragments, conflicting asset types, opaque assets awaiting inspection, placeholders and empty filename supervision. GUI suffix alone, generic blade names outside assembly context, abstract names, numeric variants and unusual complete designs are retained. Explicit visual overrides restore complete inventory objects, including square opaque items. Unreviewed quarantine can contain valid items; retained pool can still contain unrecognized fragments.

Filename-only train targets: 322,884; filename + retained eligible weak description: 2,037. With uniform target sampling and 50/50 caption choice within dual-caption images, descriptive prompts are only 0.3135% of text-conditioned draws. No all-corpus annotation, source balancing or shape constraints added. Category counts remain filename-token heuristics, with 77.49% outside the original 15 keywords, not a claim that this group is invalid or fully categorized.

Artifacts: `scripts/clean_fullscale_items.py`, v2 `report.json`, `decisions.jsonl`, `review/decisions.json`, `verification.json`. Initial review sheets and supplemental contact sheet remain under `outputs/fullscale_clean_v1_20261102/review/`. Full target pixel hashes checked during inventory creation; final catalog verification checks source/manifest hashes, all alias lineage split isolation, caption counts and nonempty normalized text. Next decision: inspect composition and determine full-scale training budget/conditioning cache, not reuse 22k-step small-data schedule unchanged.

### Full-scale 10k-caption queue launched (2026-11-03)

User authorized expanding automatic descriptions to 10,000 training images and then full training, requested detached queuing without continued monitoring. Queue root: `outputs/fullscale_text10k_20261103`; PID 3658536, CUDA 0. Parent and annotation child confirmed alive after launch. Follow `queue_status.json`, `queue.log`, annotation `status.json`, and later `training/metrics.jsonl`. No ongoing assistant monitor was scheduled.

Queue order: retain existing 2,037 eligible training descriptions; annotate a deterministic uniform train-only reserve until enough additional eligible descriptions exist; freeze exactly 10,000 dual-caption train targets while preserving all 324,921 cleaned train images and 41,869 cleaned val images; precompute exact frozen Qwen3-8B BF16 128-position features to disk; initialize model from accepted conservative step-8000 best_flow with fresh Adam; train 200,000 updates, microbatch 32 x accumulation 2, LR 1e-5 cosine to 1e-6, warmup 1,000, fixed flow validation 1,024 examples/checkpoint every 2,000 steps; evaluate final and best checkpoints against initialization on prior 227 plus 128 fixed cleaned-validation probes. Bad descriptions stay filename-only. No locked test or HD reference route, no production promotion.

Effective text exposure is approximately 35.45 draws/image. At 10k descriptions and target-uniform 50/50 eligible captions, descriptive draws are approximately 1.54% of text samples. Exact disk cache avoids repeatedly encoding the full corpus; full sequence padding retained. Cache stage estimates ETA and checks disk reserve, roundtrips and encoder identity. Queue stops on stage failure and writes resume command; annotation, cache cursor and checkpoint resume supported. Catalog and code/protocol hashes recorded. Planning estimate 1-3 days total under shared GPU load, not measured completion guarantee.

Validation before launch: 20 tests passed (disk text, training extension, caption supervision and target-balanced sampling); real Qwen cached tensors bitwise matched, isolated two-step GPU training/validation/checkpoint smoke passed (14.72 GiB peak reserved). Scripts: `scripts/queue_fullscale_text.py`; annotator now accepts --root. Trainer accepts an optional encoder, default behavior preserved. CPU/disk cache implementation: `src/mcgen/disk_text.py`.

### Speed estimate requested (2026-11-03 shortly after midnight)

Queue remains in annotation. Observed stable annotation throughput about 4,808 images/hour (mean 11.979 s/16 images); first 352/352 generated descriptions passed current automatic eligibility rules, not human certification. First queued batch target is 8,633 new annotations (24,000 is reserve size, not required workload). Isolated 40-update GPU probe with real model, microbatch32 x accumulation2 and exact disk feature reader measured mean 0.45317 s/update after 6 warmup updates, projecting 25.18h for 200k updates before full-corpus random I/O, validation/checkpoint and evaluation overhead. Probe used a tiny warm sample/cache while annotation concurrently ran, so it is not full-scale end-to-end throughput. Text precompute not yet measured. Current total remaining planning estimate 30-40h, subject to shared GPU/storage load. Evidence: `outputs/fullscale_text10k_20261103/speed_estimate.json` and `speed_probe_20261103/report.json`. Queue configuration/weights untouched; no continuing monitor requested or scheduled.

### User-scheduled stop: 2026-11-03 09:00 Asia/Shanghai

User explicitly clarified November 3, not November 4. Detached timer PID 3983728 is armed for `2026-11-03T09:00:00+08:00`, targeting only recorded queue session PGID 3658536 with PID start-time verification. Schedule/status: `outputs/fullscale_text10k_20261103/stop_schedule.json`, `stop_timer_status.json`. At deadline it writes STOP_REQUESTED.json and sends SIGTERM to the queue session, preventing subsequent stages. Future training child now handles SIGTERM by completing the current update and atomically saving latest.pt with optimizer state and stopped.json. Text-feature stage preserves durable cursor. If a process fails to exit within 120 seconds, only remaining members of this queue session are forcibly stopped. No automatic restart.

Training stop/resume exactness and isolated absolute-time timer tested; 6 relevant tests passed. Code changes are separately recorded in stop_support_code_identity.json because initial queue code_identity.json predates user-requested stop support. Timer state confirmed armed; no assistant monitoring required. A future resume should inspect the stop result and remove/archive the stop-request marker only with user authorization to resume.

### Stopped full-scale run evaluated (2026-11-03 afternoon)

User requested evaluation only. Training remained stopped at step 6,000; best_flow was step 2,000. Created immutable checkpoint/metrics snapshots under `outputs/fullscale_eval6000_20261103` and evaluated prior accepted conservative best8000, new best2000 and latest6000 on all 355 fixed prompt/seed cases (227 previous probes + 128 cleaned validation filenames), using 32-step Heun / CFG2. Baseline generation was pixel-identical across the two GPUs. Added paired project bootstrap reports for both new checkpoints. No locked test, no training restart or production promotion.

Fair same-1024 validation flow baseline measured without optimizer updates: prior .22663059, best2000 .21904988, step4000 .22266709, latest6000 .25206644. Prior / best2000 / latest6000 generation: core43 MAE29.8217 /29.7503 /33.6914; fullscale128 MAE45.0061 /44.3939 /43.8442 but alpha IoU .6386 /.5856 /.5530. Fullscale MAE project-bootstrap intervals cross zero for both candidates; alpha-IoU degradation intervals exclude zero. Novel36 requested-color pixel fraction .9427 /.9372 /.8758. No completely blank outputs among355 in any arm.

Visually inspected all36 novel/basic cases and first24 fullscale probes in three-way contact sheets, plus target images for8 overlapping fullscale cases. Latest6000 exhibits clearly broken/faded silhouettes and detached pixels. Best2000 mostly preserves old basic-category ability but does not demonstrate robust broad generalization improvement. Keep prior accepted model as default; retain best2000 for diagnosis, do not promote latest6000. Only3% of the planned200k updates ran (~1.064 expected text draws/target), which does not by itself explain the regression.

Artifacts: comparison.html (all355), comparison.json, visual_review.json, parent_flow.json, evaluation*/paired_analysis.json. Helper: scripts/compare_fullscale_stopped.py. Slow initial Qwen cold reads were observed; evaluation completed successfully.

### Repository preparation and current candidate (2026-11-03)

User reviewed full-scale best2000 and judged it basically usable and noticeably better than the preceding model. It is now the preferred experimental text-to-image candidate; this supersedes the earlier recommendation to keep the prior candidate first. Quantitative limitations and latest6000 regression remain documented. No training restart, service-default change or weight publication occurred.

Prepared v3 core/legacy compatibility changes, data/caption/queue/evaluation tools, configs, tests and historical experiment documentation for Git. README now reflects current scope and preferred candidate. Added `docs/TEXT_FULLSCALE_20261103.md`, portable reference config, aggregate evaluation JSON and three generated-only comparison sheets. Training data, raw annotations, caches, checkpoints and full runtime outputs remain ignored. Remote README additions from cb86865 were preserved in the historical README before merging remote history.

Pre-push verification: all138 tests passed in47.28s with two dependency deprecation warnings; offline wheel build passed. Sandboxed TestClient event-loop startup stalled, while the same isolated test and complete suite passed outside the sandbox. Staged files checked for model/data artifacts, size, whitespace and common credential patterns; new documentation links validated.

### Color/outline integrity diagnosis (2026-11-03)

User reported noisy colors and broken silhouettes and asked whether there is a bug. Diagnostic artifacts: `outputs/pixel_integrity_diagnosis_20261103/`. No training resumed, checkpoint promoted, default sampling changed, or new public showcase substituted.

Checked2,048 random cleaned train targets: content hashes valid and encode/decode roundtrip max uint8 error0.128 of those exactly matched their original source pixels. Recomputed48 frozen-text rows from beginning/middle/end in their original16-row batches; features and masks were bitwise identical. Analytic constant-velocity oracle reconstructs known RGBA exactly with Euler/Heun at1/7/32 steps;29 related tests passed. Attention fast/slow and train/eval forward comparisons showed no difference on the tested inputs. FP32/BF16 do differ, especially low-noise predictions, but generation precision switches did not consistently fix defects; valid noisy-input probes recorded separately.

Sampled18 fixed basic prompt/seed cases across sampling controls on accepted best2000, plus before/latest controls. Increasing32 to128 steps removes a few isolated pixels but leaves color variation and latest6000 geometry regression. CFG4 sharpens/saturates some outputs and changes colors/details, so it was not adopted as a fix. Raw outputs occasionally leave the valid premultiplied RGBA domain; this reflects model prediction error, not evidence of faulty decode on valid images.

Training sample average exact foreground colors17.52 versus100.17 in existing best basic36 probes (different cohorts, includes tiny RGB differences; coarse color bins8.53 vs12.53). Random64 visual inspection confirms cleanup omissions including guard/grip/sword-head assembly assets; clean complete inventory items still dominate this sample. These findings support output-distribution/optimization and residual data-quality concerns, but do not uniquely establish why continued fullscale training regressed. No confirmed causal implementation bug in checked paths. Next useful experiment is matched clean-subset/optimization controls, not attributing all issues to either data or step count without an ablation.

### Small pixel-quality continuation pilot (2026-11-03)

User authorized trying the proposed controlled experiment. Completed a 2x2 pilot from accepted fullscale best2000 with its Adam state and global step preserved: visually screened complete items versus category-matched random existing-corpus items, each 120 images (8 each of15 classes); LR1e-5 versus2e-6; exactly300 extra updates per arm. Filename-only supervision in all arms; same45 project-disjoint development-validation images, deterministic per-step randomness and original200k schedule. No fullscale restart, default promotion, postprocessing change or push. Artifacts and reproducible driver: `outputs/pixel_quality_pilot_20261103/`, `scripts/pilot_pixel_quality.py`.

Reviewed180 clean candidates,120 controls and45 validation candidates before training. Excluded3 clean candidates, selected first8 remaining per class. Controls include sword blades, shield-labelled rods, fiery-ingot dots and a near-empty book layer. Four problematic canonical targets exactly match their original32px dataset rows, so their defects were not introduced by the16px conversion; this does not audit upstream asset extraction. All300 updates and optimizer step2300 verified for all arms; shared baseline flow.16768648, final clean normal/low .14686653/.13766545, control normal/low .14100168/.13916894. Parent SHA unchanged.

Evaluated all143 fixed cases for all5 checkpoints (715 outputs), with32-step Heun/CFG2/FP32 and fixed prompt/seeds: novel36, clean-val45, clean-fit15, control-fit15, broad-val32. Reviewed generated novel36, first12 clean-val and all32 broad-val visually. Final endpoints used, not best checkpoint selection. Interactive local comparison includes all outputs and targets. Clean arms retain whole novel shield/gem forms; unfiltered normal continuation visibly breaks multiple shields and one red gem. Six novel shields: mean foreground pixels before154, clean-normal171.67, clean-low167.5, control-normal56.83, control-low130.5; components1/1/1/3.33/1. Both controls generate two nearly empty broad outputs (<8 foreground pixels), versus zero before/clean. These additional diagnostics were computed after visual review, not preregistered endpoints.

Clean-low novel color fraction .9385 versus before.9375; fractional visible alpha .0335 versus.1358, but exact foreground RGB count104.53 versus100.39 (tiny channel differences count; not distinct perceived hues), coarse bins11.28 versus12.67. Thus sharper edges on basic prompts do not resolve palette noise. Broad32 alpha IoU before.5905, clean-normal.6445, clean-low.6353; composite MAE49.896/51.219/50.331 worsens. Controls achieve better clean-val pixel metrics despite worse novel shape behavior. This supports prioritizing data quality and conservative continuation, not declaring a universal winner or isolated cause. Different images/projects, one training seed, small repeated subsets and only32 broad probes limit causal/generalization claims. Preferred best2000 remains unchanged; clean-low retained only as a diagnostic candidate.

### Training-label sample and original filename provenance audit (2026-11-03)

User requested sampled image/filename-label/automatic-label comparisons and verification that filenames are authentic original resource names. Exported72 unchanged training examples:48 uniform samples from10000 dual-caption targets,16 from314921 filename-only targets, plus8 separately marked nonrandom diagnostic probes. Fixed seed20261103; original captions preserved verbatim, missing automatic supervision explicitly marked; no new annotations or training changes. HTML search/filter gallery,9 contact sheets, CSV/JSON, source-evidence sheet and portable review ZIP are under `outputs/training_label_audit_20261103/`.

Checked all324921 training target filenames, mods and short-label extraction against converted metadata: zero mismatches. Checked all1034057 downloaded Parquet records against converted metadata for filename/project/mod/version URL/type: zero mismatches. All72 sample raw images exactly match native training targets. Entire downloaded corpus has null image.path and basename-only file_name (no directory separator). The exact upstream stage that discarded paths was not established.7749 training targets carry multiple distinct filename aliases but supervise a representative filename.

Downloaded11 original Mod archives from recorded version URLs (read-only ZIP/PNG inspection, no code execution). All12 selected probes match original basename and every alpha/visible RGB pixel;9 match complete RGBA,3 differ only in invisible RGB channels. Original source paths reveal semantic loss: skull.png is elixir/cap/skull.png; shield_* textures are under tool/cleaver/; two fiery_ingot.png targets come from sword/pommel/counterweight and sword/pommel/grip_loop. A book's _e texture is explicitly an emissive layer: original assets/minecraft/optifine/emissive.properties says suffix.emissive=_e, while its item model uses the base book texture. Filename authenticity does not imply whole-item semantics or verified in-game display name. Archive checks are sampled, not certification of every source.

Confirmed active automatic-caption error: diagnostic sample070 is a few-pixel grip-loop component labelled "A fiery ingot, shaped like a solid bar, with warm orange and yellow tones." This exact caption is present in current training; the audit did not generate it. Prioritize recovering resource-path/component/layer context and checking visual grounding before changing supervision. Saved archive hashes, source URLs, full matching paths and embedded configuration evidence. No model/catalog mutation, training restart or remote push. Added read-only audit/render helpers; export text/link checks and syntax/whitespace checks passed.

### Grounded label correction and agreed10k reannotation (2026-11-03)

User requested correcting both filename-derived and visual supervision. Implemented versioned policy in mcgen/grounded_labels.py: original-package basename AND normalized RGBA must match for path recovery; source paths/config classify components and emissive layers; sparse targets are quarantined. Frozen Qwen3-VL observes white/gray image previews without filenames or old captions, then a separate name pass receives source metadata plus the immutable observation. Abstract/unfamiliar names remain weak semantic information; unsupported words, semantic-token deletion, identifier-only codes, visual/name category conflicts and annotation-process commentary cannot silently enter both labels. Unknown file identity can retain independently supported visual supervision; component/layer conflicts exclude both from complete-item supervision. No old long-caption fallback.

First prototype was excessively conservative (68/72 quarantined); retained separately under outputs/grounded_label_revision_unverified_pathmatch_20261103, never promoted. Tightened path matching to require matching basename, balanced normal MC silhouette recognition, preserved unfamiliar abstract semantics and revalidated all72 independent observations under the final stricter decision policy. Original observation protocol and migration signatures retained. Final pilot:18 quarantined,54 independent visual-label candidates,41 filename-label candidates. This is automatic weak supervision, not a claim of human-certified correctness.20 policy and legacy compatibility tests pass; script syntax/whitespace checks pass.

Full train inventory:324921 targets;8579 original paths recovered from the11 available archives,6860 targets quarantined on provenance/content evidence,318061 filename candidates explicitly pending independent verification. Unrecovered paths are not marked verified and no whole-item name is guessed from a component material name. Inventory and72-case comparison: outputs/grounded_label_revision_20261103. Existing dataset/checkpoints remain historical evidence.

Asked whether to reannotate all324921 images or first redo existing10000 descriptions plus full filename audit. User explicitly accepted the latter. Staged10023 unique targets (10000 existing automatic-caption targets +72 audit probes with overlap), reused72 identical-input independent observations with recorded source signatures, and launched the remaining9951 across two GPU workers. Queue PID1231939; worker PIDs1231970/1231971. Both workers successfully finished their first8 cases. Scope and statuses under outputs/grounded_label_revision_10k_20261103; review.html and corrected_labels.csv/jsonl export the pilot now and refresh after completion. Batch is running, not completed. Queue runs annotation and export only; does not train, promote checkpoints or push. Filename candidates outside this scope remain explicitly unverified rather than claimed corrected.

### Grounded10k revision completion checked (2026-11-03)

User requested status. Both workers and export completed successfully, ~69min annotation time. All10023 staged targets processed exactly once, pending0;8762 automatic visual-label candidates, including6153 with retained file labels and2609 visual-only;1261 quarantined. Within the original10000 descriptions:8747 eligible and1253 quarantined; remaining23 targets are additional diagnostics. Full structural completion audit passed: input/output target sets identical, unique hashes/signatures valid, all quarantine records contain no training-caption variants, eligible records have independent visual observations and no known provenance/sparse-content flags, and no identifier underscores remain in eligible captions. All11 checked prior-error probes are quarantined, including skull cap, cleaver/shield parts, both fiery-ingot pommel components and emissive book layer. Saved completion_audit.json with export hash. This verifies batch completion and policy compliance, not human correctness of every automatic label or full-corpus filename verification. Training remains unstarted.

### Fresh revised-label samples for user inspection (2026-11-03)

Exported32 fresh targets excluding the prior72:16 dual-caption candidates,8 visual-only candidates,8 quarantined targets, uniformly sampled within each group with seed2026110307. Group sampling is illustrative, not corpus-prevalence weighted. Original16px pixels/hash checked; current labels rendered verbatim; no caption generation or quality-based selection.8 PNG contact sheets, searchable HTML-style card gallery, full JSON/CSV and a portable ZIP under outputs/current_label_samples_20261103. Renderer scripts/render_current_label_samples.py. Visually inspected pages1/5/7. Sample01 body_tinkers_thinking_burnt_ashes suggests a remaining component omission; sample27 shadow_helmet is a complete helmet excluded because its generated caption describes two color variations. These examples show remaining data/annotation errors and over-quarantine despite completed batch processing; samples were preserved unchanged for user inspection.

### Switch to deterministic file-information supervision (2026-11-03)

User abandoned visual labels and requested better directly extracted text for training. User accepted continuation from current preferred fullscale best2000 for10000 updates. Added mcgen/resource_text.py: deterministic CamelCase/separator normalization, explicit technical namespace and bow/frame-index handling, preservation of abstract names/material words/meaningful numbers/unknown words; path-backed component/layer rejection plus conservative filename rules for modular-tool parts, emissive/normal/specular suffixes, model/export identifiers, opaque codes and very sparse content. No VLM-generated labels or VLM-based keep/reject decisions are consulted. In particular shadow_helmet is restored, while body_tinkers_thinking_burnt_ashes, cleaver shield parts and fiery-ingot pommel layers are quarantined. Full original target pixels and split groups retained; no test inspection.

New run outputs/resource_text_training_20261103:311510 train,40808 validation targets, one short file-derived label each. Quarantine13411 train/1061 val. Normalized text changes15774 train/738 val. Data audit passed352318 unique targets, zero visual captions/reference images. Exported32 review examples (24 random +8 technical-marker-cleanup samples), with exact old/new labels in review/index.html.22 extraction regression tests and3 frozen-feature cache tests pass (25 total).

Added exact compatible-cache routing to DiskTextEncoder infrastructure; mixed-source ordering, repeats, null prompts, absent-key failure and identity mismatch covered by tests. Reused251782 required prompt features; computed9400 new Qwen3-8B features (~29.57GB) instead of copying the old .89TB cache. Delta features recomputed in first/middle/last original batches and match BF16 bits and masks exactly. Training queue PID186004; stages features->20-update smoke->10000-update continuation->68 fixed prompt/seed evaluation cases for before/best/latest. Initialization hash809fb3ecbafbbe42909d9c08fac8a90c495248596f04d69f52a99aab445de281 unchanged; fresh Adam with500-update warmup,LR2e-6,batch32x2,evaluate every1000,1024 validation targets. First smoke optimizer updates are running with finite loss and zero text-cache misses. No publication or model promotion.

Resource-text smoke completed successfully:20 optimizer updates plus1024 deterministic validation cases, flow MSE.21613976918, finite training losses and zero cache misses. Saved latest/best checkpoints, then queue advanced to the authorized10000-update continuation by exact same-config/catalog resume. Features completed with9400 delta rows and exact recomputation audit; no VLM labels used. Queue stage is train; accepted best2000 remains intact. Startup and first validation involve cold disk I/O, so no training-duration estimate based solely on feature speed or20-step smoke is asserted.

### In-flight resource-text generation comparison (2026-11-03 18:04)

User explicitly requested images now. Snapshotted original accepted best2000, current resource-text best1000 and latest saved2000 into outputs/resource_text_comparison_20261103_180359/checkpoints before evaluation; hashes/identities/actual steps recorded. Generated all68 frozen prompt/seed cases for each arm (204 unchanged native16px RGBA outputs),32-step Heun/CFG2/FP32 on GPU1 while training continues on GPU0.10 nearest-neighbor gray-background comparison sheets, full HTML gallery and image ZIP exported. No postprocessing or promotion.

Reviewed all36 basic cases and first8 broad prompts. Tools/helmets/apples/shields mostly preserve shapes; several second-seed gem outputs show more hollow/eroded interiors after1000/2000 updates. Color control remains similar (.93747/.93949/.94312 requested-color fraction); this metric does not capture hollow silhouettes. No broad semantic recovery demonstrated by the reviewed8 cases. Lower validation flow loss has not established a generation improvement. All204 files checked RGBA16x16. Training remains authorized/running; no settings or checkpoint selection changed. Helper scripts/compare_resource_text_interim.py.

### User switched generator to random initialization (2026-11-03)

User requested training from0. Terminated the prior continuation queue PID186004 and sent its training child PID310977 SIGTERM so the installed handler saved latest.pt/stopped.json at2340. The old wrapper logs its expected endpoint assertion because of this deliberate early stop; queue_status.json explicitly records user_requested_random_initialization_in_new_run. Old best/current weights remain preserved; no old queue evaluation proceeds.

New run outputs/resource_text_scratch_20261103 uses identical311510 training/40808 validation images, deterministic file-text prompts, architecture and frozen Qwen3-8B. Generator created with Trainer without initialize/resume; all optimizer state starts empty, global step0. Saved initial_random.pt and initialization_audit.json (model digest ad10cab2d54d29b6f243ee850c47a15a6db8f5dbc191c456a564bbb89c251892). Independently loaded that checkpoint and verified step0, initialization=None, empty optimizer and nonzero random backbone variance. Pretrained Qwen features only are reused through read-only cache/data links.

First stage10000 updates, LR2e-4 (existing scratch-training scale config default),1000 warmup, batch32x2,1024 validation cases every1000 updates, seed2026110309. First run20-update smoke then resumes only its own fresh checkpoint to10000; evaluation compares original preferred model versus new scratch best/latest. Queue PID2022940, first optimizer update recorded with finite loss1.79363 and zero cache misses; start from0 verified. Previous continuation first-step loss was~.2, so these fresh losses are not compared as improvement/regression. Driver scripts/run_resource_text_scratch.py and queue_resource_text_scratch.py passed syntax/whitespace checks. No default promotion or push.

### Scratch progress validation and generation comparison (2026-11-03 19:10)

User requested progress and a validation run. Scratch reached~2200 steps at request; fixed1024 validation flow had improved1.81085655 at20 to.21963863 at1000 and.18725163 at2000 (current best). Snapshotted accepted old best2000, abandoned-continuation best1000 and scratch best2000, then independently revalidated all3 on the same current catalog1024 examples/noise/timesteps: .21712088/.20046671/.18725163. Scratch score exactly matches training log. Ran all68 frozen prompt/seed probes for all3 (204 RGBA16px outputs) on GPU1 while scratch training continued. Old baseline outputs are pixel-identical to all68 previous comparison outputs. Artifacts outputs/resource_text_comparison_20261103_190936; helper compare_resource_text_interim.py now accepts run directory, continuation reference and optional common flow validation.

Reviewed36 basic and first8 broad cases. Scratch2000 has recognizable tool/helmet/round silhouettes, and several gems look more filled than the continued model. However gems/apples/shields are less distinct, some seed-conditioned outputs look similar across categories, and blue/purple control is inconsistent. Requested-color fraction .93747 old/.93949 continued/.83838 scratch is only a coarse proxy, penalizes legitimate brown handles, and does not by itself establish quality. Broad prompt semantics remain weak in the inspected8. Do not equate improved denoising loss with overall generation superiority; keep training to the authorized first10000-step endpoint. No promotion, schedule changes or data modifications.

### Repository update and portable processing examples (2026-11-03)

User explicitly requested uploading/pushing current work and including a small dataset subset for processing review. Prepared current file-text rules, exact merged text caches, scratch/continuation and generation-validation scripts, plus historical diagnostic/visual-label experiments marked historical in docs. README now distinguishes its earlier selected showcase from the current scratch experiment and links data review. Added concise docs/RESOURCE_TEXT_20261103.md.

Portable examples/data_review contains14 exact native16px images from Create Modular Tools (ZLT), with original MIT license copied from LICENSE_create_modular_tools inside the independently checked source archive. Included source version URL, original paths, archive SHA, per-image file/content hashes, old extracted text and exact current rule outputs. Selected7 kept bucket textures,4 rejected parts and3 deliberately disclosed component omissions (pommel/tool_wrap); not a representative sample or accuracy benchmark.12 clearly marked synthetic filename-only cases cover abstract names, numbers, CamelCase, frames and component/layer identifiers. Human-readable README/preview and scripts/check_data_review.py allow reproduction without full local datasets/caches. The sample check reproduces the current rule behavior and explicitly reports known semantic omissions rather than claiming correctness. Total example directory~208KB. No full datasets, archives, labels, model checkpoints or feature caches staged.

Pre-push checks:179 tests passed in33.19s,2 dependency deprecation warnings; portable sample hash/rule checks passed; new scripts syntax checked; whitespace checks passed. Remote fetched and main was unchanged from fbe8574. Active scratch training is unaffected by repository preparation.

### Correct multimodal embedding, larger generator and measured batch (2026-11-03)

User authorized switching to the proper Qwen embedding model, increasing generator capacity, budgeting by exposure and selecting a VRAM-efficient measured batch. Stopped small scratch queue PID2022940 and gracefully saved its trainer child at7768 (outputs/resource_text_scratch_20261103/training/stopped.json); preserved as reference. No old generator weights initialize the new experiment.

Downloaded Qwen/Qwen3-VL-Embedding-8B revision2c4565515e0f265c6511776e7193b22c0968ddc7. HF large-file transfers stalled, recovered missing shards from the official ModelScope mirror by ranged requests; all final files checked against pinned HF LFS SHA256 or Git blob hashes. Correct model uses pooled final attended token and L2 normalization, not ordinary Qwen intermediate-layer concatenation. Actual4-text comparison against official source digest0e447b6a80e05527f3fd437ab4f1617a6e7d0df102b8164348bbce787a41b267 gives max abs difference0 and cosine>=.99999988. Encoder identity7dc02f153dae1b1d277f4b39812e6e6f9fe56d53fb37e371717da06733114402. Empty conditioning is masked; no image or old visual label is supplied. Added runtime/backend support with old-cache compatibility.

New outputs/qwen_vl_embedding_large_20261103:512width12layers, pure text-only conditioning,81,409,028 parameters versus20,086,916 before. Eliminated dummy reference CNN/attention from this new architecture; historical separate/shared models unchanged. Same311510 train/40808 validation image/file-text pairs, no VLM labels, random initialization. Pooled feature cache shape261182x1x4096 BF16 (~2.14GB) replaces huge token feature caches. Precomputation currently running on GPU1. Probe data comes from actual native-grid images and actual official embeddings, published before full cache completion.

Measured isolated forward/backward/gradient clipping/AdamW steps on A100-SXM4-80GB GPU0,2 warmups+3 timed updates and validation forward. B64/128/256/384/512/640/672 allocated peaks7.47/13.82/26.43/39.05/51.66/64.29/67.45GiB. B70470.60GiB and860.5images/s versus B67267.45GiB and862.8images/s;B768OOM under90% allocator cap. Selected672 with68.68GiB reserved peak (~86% of80GiB), preserving margin. Throughput is model computation on repeated real-data probe examples, not end-to-end training ETA.

Final exposure budget7211 updates x672 x.9 /311510 =14.00023 expected text exposures/target; total15.55582 draws/target. Fresh AdamW LR2e-4; warmup232 updates (~half total-data exposure); fixed1024 validation every516 updates (~one text exposure). Trainer additionally records real/shuffled/null-text MSE at t=.1/.5/.9 on128 fixed validation cases and18 fixed prompt/seed samples per interval. Budget uses expected random-target exposure, not a per-image deterministic epoch guarantee. Batch trials and formula saved in batch_and_budget.json. Queue PID3450902 waits for the cache, then runs20-step smoke, the full budget, and final generation checks. At this entry it is finishing_embedding_cache, NOT yet formally training.

Full regression after compatibility fix:182 passed,2 dependency warnings in34.50s. Official pooled-embedding and text-only architecture tests added. Batch and configuration refinements do not mutate current data or old checkpoints. Changes not pushed in this turn.

### Select step4128, finish training, and refresh repository presentation (2026-11-03)

User compared the checkpoint images, explicitly selected4128 as preferred, requested stopping training and pushing the repository, then asked for a clearer and more attractive README. At inspection the queue had already completed naturally at7211 (queue_status complete); no further training was launched. Preserved the original latest7211 and automatic minimum-Flow checkpoint4644 without rewriting their meanings. Selected4128 independently from the previously frozen snapshot; source hash checked. Exported checkpoints/mcgen-step4128.pt for inference (326227394 bytes), retaining the exact configuration, encoder identity and every generator tensor while removing optimizer state. Export SHA256790d2264f5e3559894dba726b99a7109185dca2f40d009065bccdea4c29369e2. Selection records are in training/selected.json and docs/models/step4128.json. Weights remain excluded from Git, consistent with repository publication policy.

Compared4128/4644/5676/6708 on24 common, conceptual and previously viewed held-out prompts with2 fixed seeds (192 fresh outputs). Original historical18-prompt samples align at every saved evaluation interval. Significant late regression occurs between5160 and5676; mean training loss rises rapidly around5250-5350. Fixed validation Flow MSE4128=.1501915,4644=.1486997,5676=.4374699,7211=.6839257. Cause remains unresolved; these qualitative examples do not establish unbiased generalization quality. The already-viewed test cases are not treated as unseen. Compact records and images are published in docs/EMBEDDING_TEXT_20261103.md and docs/models/.

Refreshed README around actual4128 results, a short generation diagram and an explicit data/model table:311510 train,40808 validation,81409028 generator parameters; frozen Qwen3-VL-Embedding-8B is counted separately. Dataset source attribution links the upstream OVAWARE corpus and project-prepared Risposta export. Rebuilt showcase as a dark inventory gallery with16 named tools/equipment/gems/concept examples. Every sprite is byte-identical to a4128 source output, one fixed seed throughout; only background composition and nearest-neighbor display enlargement. Renderer validates original PNG hashes. Layout references were read from local copies of the public Causal-Forcing and Sparse-VideoGen READMEs after online tools failed; no external artwork or prose was copied.

Publication checks:181 tests passed,1 deselected in35.69s. The excluded API/runtime TestClient check stalls during Starlette/AnyIO portal startup in this sandbox; an empty FastAPI TestClient with no MCGen imports reproduces the same hang. Interrupted the two blocked check processes; no API implementation change made. All changed scripts parsed, documentation file links checked,16 showcase source files and hashes checked, portable data review reproduced14 images/12 filename cases (known omissions retained), and whitespace checks passed. Git publication is prepared below; do not infer a successful remote push from this entry alone.
