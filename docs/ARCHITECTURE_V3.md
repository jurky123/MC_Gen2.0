# MCGen v3：任务、数据与实验契约

## 边界

产品提供文生图、参考图生成两个功能；联合条件是参考图加可选文字。任务接口相同不代表必须共用一个检查点。物品和方块纹理分别建立 text/reference/joint 三组实验；`other` 不进入正式配置。保留 Qwen3-8B，冻结文本塔，原生 16px pixel-space flow，不使用 VAE。

这是可训练、可比较的工程基线，不是已经证明效果更好的算法。MC→HD→MC 训练对提供逆向恢复监督；它们不能证明真实照片适配或文字编辑能力。后两者仍需要有来源记录的真实参考图、编辑对和人工评审。

## 数据契约

`migrate/import-images/merge` 接受来源数据并保留其版本。正式实验先运行 `prepare`：

1. 按项目名称和 seed 固定分配 train/val/test，默认 80/10/10；可传入人工审查过的 project plan。增加记录不会重新抽签已有项目。
2. 找出跨集合出现的相同目标内容，将该内容的所有副本隔离，不移动项目、不随机拆散重复项。隔离记录和原因完整保存在 `quarantine.jsonl`。
3. 每个唯一目标验证内容 SHA、精确原生网格并只写一次 native mmap；检测 opaque/binary/continuous alpha。原图引用保留为 `source_target`，源 catalog 不变。
4. 新 catalog 验证项目、目标内容均不跨集合；独立版本指纹阻止错误续训。

项目同源关系仍取决于来源 lineage 的质量。此策略保证已知项目与精确重复不泄漏，不声称能识别重命名项目、近似图和未知素材继承关系。公共素材隔离比例是要检查的数据质量信号，不是越少越好。

本机已生成 `data/catalogs/native16_v3/catalog`：946,318 条记录；train 751,916、val 99,441、test 94,961；隔离 104,824 条／22,231 个唯一目标。最大项目 13,964 条。原生目标 857,244,672 bytes。审计、固定项目计划和隔离清单位于其父目录。

存储只压缩目标，参考图沿用只读 mmap。JSONL 逐行解析以避免额外的大字符串副本，但目前样本元数据与采样索引仍常驻内存，启动仍校验源 SHA；大型多节点训练需进一步引入索引数据库、可迁移相对路径和预取。这些优化不应削弱数据身份校验。

## 任务与预算

新配置显式设置 `train.task_mix`，包含 text/reference/joint/unconditional，概率总和必须为 1。先抽任务，再均匀抽该任务的唯一目标，最后抽该目标的 caption/reference 变体。不存在可用目标时拒绝启动，不将 reference 悄悄变为 text。

- text 专用：90% text + 10% unconditional。
- reference 专用：90% reference + 10% unconditional；文字始终屏蔽，推理必须有参考图。
- joint：45% text + 20% reference + 25% joint + 10% unconditional。

这些是待对照的试验设置，不宣称最优。显式模式不再叠加 `text_dropout/ref_dropout/paired_fraction`；旧配置未提供 task_mix 时保留原有兼容路径，并在预算报告标记 legacy_dropout。

默认六组试验使用 `target_scope: paired`，将 text/reference/joint 的目标范围限定为同一批配对目标，避免用不同数据规模比较任务架构。`configs/v3/scale/` 的文生图扩展配置显式使用 all。text 从所选范围内全部唯一目标采样，reference/joint 从有参考图的唯一目标采样。预算报告给出每个任务的期望抽样次数、每目标曝光；同一目标参与多个任务时应相加。正式 YAML 默认 2,000 步短试验，不再把固定步数当成全量训练覆盖。先检查 `mcgen budget`，再按实验目标增加步数或收窄数据范围。

`asset_type` 和 `style` 是检查点输出契约；style 非空时只使用同标签数据，不凭空赋予旧数据风格标签。opaque/binary 训练筛选要求准备过的 alpha 元数据；item 默认允许半透明，tile 默认 opaque。空文本始终为空，不会被元数据前缀变成有条件输入。

## 模型和图像

默认文字与参考图各有交叉注意力，再在像素 token 上融合，避免在一次 softmax 中竞争。原共享注意力保留为消融选项。参考图保持 RGBA 和长宽比，透明补边；已有 alpha 不再白底合成丢弃。对没有 alpha 的照片不伪造前景分割。

模型内部为 premultiplied RGBA [-1,1]，磁盘为 straight RGBA uint8。训练目标严格还原原生网格。API/CLI 可按检查点契约导出 opaque/binary alpha；Runtime 和评测保留原始输出，不用导出修饰掩盖模型缺陷。调色板量化也只属于显式导出操作。模型连续输出不能自动保证良好的像素簇、轮廓或自然平铺；这些必须独立评审。

正式模型不分配 byte 文本嵌入。2026-11-01 诊断确认 Qwen 中间层的共同首 token 存在极端幅值；新配置显式设置 `model.text_normalization: layernorm`，在每个 token 的线性投影前归一化。缺省值仍为 none，以保持旧检查点计算行为；不能仅切换旧检查点配置就宣称质量修复，须重新训练或做受控适配。诊断见 `docs/DIAGNOSIS_20261101.md`。Qwen 特征使用固定最大 token 长度，进程内 CPU LRU 按字节限额缓存，批内去重；只缓存冻结塔输出，绝不缓存可训练投影。键由实例内固定的文本塔身份、配置和文本决定，文本长度进入身份指纹；不把全量百万 caption 无限制写入磁盘。命中率、容量随训练记录。

检查点格式为 `mcgen.native-flow.v3`。旧版 v2 和 legacy 权重不能直接加载：预处理和模型语义已变。重构前源文件快照：`outputs/design_rebuild_20261101/core_before.tar.gz`。旧源码、数据和权重未删除。

## 评估和选点

- 去噪验证分别记录 text/reference/joint；分别保存 `best_<task>_flow.pt`，另有 `latest.pt` 和历史兼容的 `best_flow.pt`。flow 损失不是上线资格。
- 配对重建默认沿用检查点 target_scope，保证三个基线比较同一目标范围；显式 `evaluate --target-scope all` 可另做全量泛化评测，报告记录范围。
- 配对重建：黑白合成 MAE、alpha IoU、边缘、颜色等；按 asset 和 reference generator 分层，按项目 bootstrap。box/palette 基线不具有前景分割能力，不能把其 alpha 差异当成生成模型全面优势。
- 外部配对 catalog：`evaluate --training-catalog ...` 核对检查点来源，并拒绝与 train/val 的样本 ID、项目、目标哈希、参考源行重叠。复制后的参考图和未知来源关系需要额外人工审计，自动检查不会替代它。
- 文本语义：`benchmarks/semantic_v1.jsonl` 固定 12 个物品/方块案例及可观察标准，多个 seed；不以与某一张目标的像素距离代替语义。方块预览显示 3×3 平铺。
- 真实参考图案例可添加 reference、reference_sha256、provenance、reference_criteria；不把人工编写的文本案例或合成图冒充真实 OOD 数据。
- `benchmark` 导出随机顺序的盲评图片、`review.csv` 和单独 `key.json`；评审时隐藏 key。`review` 检查评分 ID、重复项、范围，显示待评数量并按任务汇总。不会自动填分或宣布发布通过。

## 对照顺序

同一数据划分、固定评测案例和曝光预算下，先对照 text-only、reference-only、joint。再逐项比较 `configs/v3/ablations/` 中 shared/separate、Qwen 最后一层/9+18+27 层、宽度 256/384。宽度变化需同时记录算力预算，不能仅比较训练步数。选择架构后才扩大训练。

验证产物集中于 `outputs/design_rebuild_20261101/`。真实 Qwen 接通和短训练证明实现可执行；模型质量、风格一致性、编辑服从和真实参考图泛化仍须完成对照训练及人工评分。
