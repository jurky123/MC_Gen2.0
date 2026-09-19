# MC-Gen 2.0 图生图 / HD→MC 风格化设计与仓库整改方案

> 版本：v1.0  
> 日期：2026-09-19  
> 状态：设计评审稿，暂不实现  
> 对应仓库：`jurky123/MC_Gen2.0`，审查基线 `main@acb6b39`  
> 建议用途：替换现有 `docs/IMG2IMG_DESIGN.md`，并作为后续实施和验收依据。

---

## 0. 执行摘要

本项目应把 HD→MC 建设为一个独立、可验证的条件生成器，再把它用于扩展文生图训练数据。MC→HD 不属于 MC-Gen 2.0 的研发范围：直接调用现有模型离线生成 HD reference，并记录模型、参数、许可和血缘信息。

推荐主线：

1. 使用真实 MC 纹理作为唯一可信 target；
2. 用现有 MC→HD 模型为同一 target 生成多个 HD reference，形成锚定配对；
3. 从现有 Stage 3 MC-FlowDiT 初始化 HD→MC Stylizer，新增零初始化参考图适配器；
4. 先证明学习式 Stylizer 显著优于 nearest、调色板量化和 dithering；
5. 再执行“结构化 prompt→现有高清文生图模型→Stylizer→MC”的开放环数据生成；
6. 过滤后得到新的 `(prompt, MC)` 数据，以低比例加入真实数据和 replay，微调现有 t2i；
7. 不把“现有 t2i→HD→MC→再训练现有 t2i”作为主循环，避免错误放大和分布收窄。

在图生图开发前，应先修复仓库中的 mmap 格式、group split、Stage 3 潜在泄漏、全样本 tile loss、评测选点和条件编码配置漂移等问题。

---

## 1. 当前系统与问题定义

### 1.1 当前系统

当前主线为 32×32 RGBA 像素空间 Rectified Flow Transformer：

- target：32×32 RGBA MC block/item 纹理；
- patch size：2，对应 16×16、共 256 个图像 token；
- 文本条件：冻结文本塔产生 token sequence，经 cross-attention 注入；
- 训练阶段：像素风预训练 → 全量 MC 文本对齐 → 2 万条精标子集 + replay 微调；
- 已具备基础颜色、常见概念和 block/item 生成能力；
- 主要短板是罕见概念、局部结构、轮廓细节和复合属性。

### 1.2 根本矛盾

目前瓶颈不只是标注不够长，而是文本无法无损表达高频视觉结构；同时，在 32×32 输出空间中也不可能保留 HD 的全部细节。因此 HD→MC 的目标不是像素级复原，而是：

> 从高清参考中选择最能保持身份、轮廓和材质辨识度的信息，并将其投影到真实 MC 纹理分布。

### 1.3 目标

- 输入 `(HD reference, optional text)`，输出 32×32 RGBA MC 风格纹理；
- 同时覆盖 block tile 和 centered item，但明确区分其损失与评测；
- 支持 reference strength 和文本编辑；
- 建立可追踪、无 split 泄漏的配对数据；
- 用 Stylizer 制造带原始 prompt 的新 MC 数据，并验证其对 t2i 的净增益；
- 保持现有 t2i checkpoint 独立，不因图生图实验被覆盖。

### 1.4 非目标

- 不训练或复现 MC→HD 模型；
- 首版不追求任意自然照片到 MC 的通用迁移；
- 不把 HD→MC 定义为超分辨率逆过程；
- 首版不训练 64×64 MC 主模型；
- 首版不统一 t2i 与 i2i 为同一个生产 checkpoint；
- 不以确定性像素化数据作为主要监督来源。

---

## 2. 系统边界与总体架构

### 2.1 三个系统角色

| 模块 | 是否由本项目训练 | 职责 |
|---|---:|---|
| 现有 MC→HD 模型 | 否 | 离线生成结构对应的 HD reference |
| HD→MC Stylizer | 是 | 将 HD reference 映射到真实 MC 纹理分布 |
| 现有 MC-FlowDiT t2i | 后期微调 | 消费新增 `(prompt, MC)` 数据，提升长尾语义与细节 |

MC→HD 仅作为外部数据生产工具。仓库中只需要数据适配器、任务清单、输出校验和 provenance，不需要其网络、训练代码或权重管理逻辑。

### 2.2 两阶段数据闭环

#### 阶段 A：训练 Stylizer

```text
真实 MC target ──► 现有 MC→HD 模型 ──► HD reference
      └────────────────────────────────► 监督 target
```

#### 阶段 B：反哺 t2i

```text
结构化 prompt ──► 现有高清生成模型 ──► HD image
       └──────────────────────────────────────┐
                                              ▼
                                      HD→MC Stylizer
                                              ▼
                                    (prompt, synthetic MC)
                                              ▼
                                     过滤 + t2i 微调
```

阶段 B 是开放环：新语义来自结构化 prompt 和外部高清模型，而不是来自现有 MC t2i 自身。

---

## 3. 数据方案

### 3.1 数据分层

#### Tier A：真实 MC 锚定配对——主训练数据

对真实 MC 纹理运行现有 MC→HD 模型：

```text
HD_ref = ExistingMC2HD(MC_target, parameters, seed)
pair   = (HD_ref, MC_target, prompt, metadata)
```

建议每个 target 生成 2–4 个变体：

- 低重绘强度：严格结构保持；
- 中等重绘强度：补充合理材质细节；
- 不同随机种子或外观；
- 可选的不同光照/锐度设置。

MC target 始终来自真实训练域。HD reference 可以带合理幻觉，但必须通过结构过滤。

#### Tier B：程序化 reference 扰动——预热和鲁棒性

从真实 MC 纹理构造放大、模糊、压缩、颜色漂移、锐化、局部缺损、非整数 resize 等 reference。其作用是让 reference adapter 学会基本对齐和输入鲁棒性，不承担新语义学习。

#### Tier C：确定性 HD→MC——baseline 和弱正则

包括 nearest/bicubic 下采样、median-cut、k-means、调色板约束和 ordered dithering。它们必须被保留为 baseline，但不作为大规模训练底料。正式训练中的建议占比为 0%–10%，由消融决定。

#### Tier D：Prompt→HD→MC——t2i 增强数据

当 Stylizer 通过验收后，使用丰富结构化 prompt 生成 HD，再转为 MC。必须保留原始 prompt，禁止用 VLM 重新猜测它作为唯一文本标签。

结构化字段至少包括：

- `asset_type`: block/item；
- `material`；
- `form`；
- `dominant_colors`；
- `silhouette`；
- `surface/pattern`；
- `details`；
- `symmetry/directionality`；
- `emissive/transparency`；
- `tileability`；
- `orientation`。

### 3.2 数据记录格式

建议图像继续使用 headerless uint8 mmap，但同时增加明确 schema：

```text
pairs/
  hd.uint8.mmap          # (N, 64, 64, 4)
  mc.uint8.mmap          # (N, 32, 32, 4)
  metadata.parquet
  splits.json
  schema.json
```

关键 metadata 字段：

| 字段 | 含义 |
|---|---|
| `sample_id` | 当前 pair 唯一标识 |
| `root_asset_id` | 原始 MC target 标识 |
| `lineage_id` | 所有派生变体共享的血缘标识 |
| `project_id` | 原资源包/项目 |
| `asset_type` | block/item |
| `tileable` | 是否应计算 seam/tile loss |
| `source_kind` | anchor/procedural/pixelize/prompt_hd |
| `generator_model` | 外部模型精确名称和版本 |
| `generator_params` | seed、strength、control 等参数 |
| `license_source` | 来源、许可与审计状态 |
| `prompt_id/prompt` | 原始结构化 prompt |
| `quality_flags` | 自动和人工过滤结果 |

### 3.3 Split 与去重

必须先去重/聚类，再按 `lineage_id` 划分。来自同一个 MC target 的所有 HD 变体必须进入同一 split。

最低规则：

1. exact SHA 去重；
2. alpha-aware pHash 或 feature 近重复聚类；
3. `root_asset_id/project_id` 分组；
4. group split；
5. 验证 train/val/test 的 lineage 交集为空。

Prompt→HD 数据还应按 concept family、prompt template 和生成器来源分组，防止只更换形容词的模板跨 split。

### 3.4 RGBA 表示

建议把训练表示切换为 premultiplied RGBA：

```text
rgb_train = rgb * alpha
target = concat(rgb_train, alpha)
```

原因是透明像素背后的 straight RGB 没有稳定语义。对于 reference：

- 保留原始 alpha/mask；
- 训练时随机合成到白/黑/灰/随机背景；
- mask 仍作为独立输入，防止模型把背景当纹理；
- item 与透明 block 分开统计 alpha 指标。

### 3.5 数据过滤

锚定配对过滤不应只依赖 VLM。建议组合：

- 下采样后的颜色偏移；
- 多尺度 edge/silhouette 相似度；
- alpha mask IoU；
- 主体中心和面积变化；
- VLM 判断是否为同一物体/材质；
- 困难类别人工抽检。

对每一种外部模型/参数组合单独统计通过率，淘汰高结构漂移配置。

---

## 4. 模型设计

### 4.1 推荐形态

首版采用独立 Stylizer checkpoint，但从最佳 Stage 3 MC-FlowDiT 初始化；不从零训练，也不立即与生产 t2i 合并。

推荐接口：

```python
velocity = model(
    x_t,
    timestep,
    text_tokens,
    text_mask,
    reference_rgba,
    reference_mask,
    reference_strength,
    asset_type,
)
```

### 4.2 Reference Encoder

当前 target 为 32×32、patch size 2，对应 16×16 target token 网格。首版使用：

```text
64×64×4 reference
→ convolutional stem
→ stride 4
→ 16×16×D
→ 256 spatial reference tokens
```

这比先把 reference 降到 32×32 再通道拼接更能保留高清结构，也不需要改变现有 `patch_embed` 输入维度。

### 4.3 注入方式

首版推荐零初始化 spatial adapter：

```python
image_tokens = image_tokens + gate * zero_proj(reference_tokens)
```

`zero_proj` 与 gate 初始为零，使刚初始化的 Stylizer 与原 t2i 行为接近。可以在 2–4 个主干位置加入 gated residual。

后续消融再考虑：

- reference cross-attention 全局 token；
- 多尺度 reference feature；
- ControlNet 式旁路；
- 与文本 K/V 合并。

不建议首版直接把文本和图像 token 拼成同一个 K/V 序列，因为两者结构、长度和职责不同，小数据下容易相互干扰。

### 4.4 训练参数解冻策略

阶段 1：只训练 reference encoder、adapter、gate。  
阶段 2：解冻最后 2–3 个 single-stream block 和输出 head，或添加 LoRA。  
阶段 3：只有当验证表明容量不足时才扩大解冻范围。

文本塔始终冻结；原生产 t2i checkpoint 不覆盖。

### 4.5 条件 dropout 与双 CFG

建议训练分布：

| 条件组合 | 比例 |
|---|---:|
| text + reference | 70%–80% |
| null text + reference | 10%–15% |
| text + null reference | 5%–10% |
| null text + null reference | 5% |

推理时采用三次 forward 的顺序式双 CFG：

\[
v=v_{\varnothing,\varnothing}
+s_t(v_{t,\varnothing}-v_{\varnothing,\varnothing})
+s_r(v_{t,r}-v_{t,\varnothing})
\]

其中 `s_t` 控制文本，`s_r` 控制参考保真度。严格分离条件交互需要四次 forward，首版不建议。

---

## 5. 训练目标

### 5.1 主目标

继续使用 conditional Flow Matching：

```text
x0 = MC target
condition = reference + optional text + asset type
```

主损失维持 velocity MSE，避免首版同时改变生成目标和条件架构。

### 5.2 辅助损失

建议总损失：

\[
L=L_{flow}+\lambda_\alpha L_{alpha-boundary}
+\lambda_s L_{structure}+\lambda_{tile}L_{seam}
\]

- `L_flow`：所有样本；
- `L_alpha-boundary`：item/透明纹理，约束 mask 和边界；
- `L_structure`：在重建的 `x0_hat` 上计算低权重多尺度 edge loss；
- `L_seam`：仅对 metadata 中 `tileable=true` 的样本；
- palette/unique-colour 约束暂不进入首版主损失，先作为评测指标。

不要给所有 item 施加 seam loss，也不要直接对 HD reference 和 MC output 做像素 L2。

---

## 6. 分阶段实施与验收门槛

### Phase 0：仓库整改与基线

- 修复 mmap 格式；
- 实现真正的 group split 和泄漏检查；
- 修复 per-sample tile loss；
- 固定 RGBA 表示；
- 建立 deterministic pixelization baseline；
- 建立 Img2Img Golden Set 和 paired evaluator。

**Gate 0**：数据可重复构建；所有 split lineage 交集为空；raw mmap round-trip 测试通过。

### Phase 1：小规模 Adapter 原型

- 约 10k 真实 MC target；
- 每个 target 2–4 个 HD reference；
- 64×64 reference → 32×32 MC；
- 只训练 adapter；
- 暂不引入复杂双 CFG。

**Gate 1**：在固定 paired test 和人工盲评上显著优于最佳 Path B；reference shuffle 后性能显著下降；alpha 和 item silhouette 不退化。

### Phase 2：有限解冻与可控性

- 解冻末端 blocks/head 或 LoRA；
- 加入 reference dropout；
- 加入文本编辑；
- 建立 reference-strength curve。

**Gate 2**：文本编辑有效但不会无故破坏 reference 身份；独立来源 HD 测试集不过度退化。

### Phase 3：规模化锚定配对

- 扩展到约 50k–200k pair；
- 增加来源和参数多样性；
- block/item、tileable、transparent、emissive 分桶监控。

**Gate 3**：各困难桶均有稳定增益，不只是总体平均指标上升。

### Phase 4：反哺 t2i

- 结构化 prompt→HD→Stylizer；
- 质量、许可、重复和教师偏差过滤；
- 合成数据从总 batch 的 10%–20% 开始；
- 保持真实 MC 至少 70%，其余为 replay/精标，具体由消融确定。

**Gate 4**：相同 base、步数和真实数据预算下，加入合成数据能提升长尾概念/复合属性，并且 Stage 2/3 回归指标不显著下降。

### Phase 5：可选扩展

- 16×16 native；
- 64×64 MC；
- top/side/bottom 多面一致生成；
- 显式调色板条件；
- 多 reference；
- 最终评估是否合并 t2i/i2i。

---

## 7. 评测方案

### 7.1 Paired i2i 指标

- premultiplied RGBA L1/L2；
- alpha IoU；
- alpha boundary F-score；
- 多尺度 edge F1；
- Lab 色差和 palette distance；
- silhouette consistency；
- tileable 样本 seam；
- item 中心占比、透明背景占比；
- VLM/人工的 identity、material、MC style 评分。

LPIPS 在 32×32 像素纹理上只作辅助，不作为主选点指标。

### 7.2 条件利用率

- reference shuffle；
- text shuffle；
- null reference；
- reference strength 曲线；
- reference 小幅平移/缩放/颜色扰动稳定性；
- 同 reference 多 seed 的结构保持和外观多样性。

### 7.3 Golden Set

建议固定 512–1000 个样本，覆盖：

- tile block；
- centered item；
- tool/armor；
- gem/ingot；
- transparent/emissive；
- directional/symmetric；
- long-tail material；
- HD 模型容易产生结构漂移的困难案例。

可复用 CaptionBench 的人工审核基础设施，但必须建立独立的 Img2Img schema，不能直接把 caption 指标当图生图指标。

### 7.4 t2i 增益消融

必须从同一 base checkpoint 运行：

- A：真实数据 + replay；
- B：真实数据 + replay + 合成数据；

保持总步数、有效 batch、真实样本预算、采样 prompt 和随机种子一致。重点评估长尾概念、属性组合、轮廓、调色板、多样性、记忆化和教师风格偏置。

---

## 8. 外部模型和许可策略

MC→HD 模型不需要研发，只进行黑盒验收和数据治理：

- 固定模型名称、revision、运行环境；
- 固定并记录 seed、strength、control、prompt；
- 校验输出是否允许用于后续模型训练；
- 保存许可/服务条款审计结果；
- 按配置统计结构漂移和过滤通过率；
- 对失败配置停止继续扩量。

用于 Prompt→HD 的模型可以与 MC→HD 模型不同。前者侧重 prompt adherence 和长尾概念，后者侧重结构保持。不要因为一个模型适合 MC→HD，就默认它也适合开放语义的高清素材生成。

---

## 9. 当前仓库问题与整改建议

以下结论基于 `main@acb6b39`。

### P0-1：mmap 写入和读取格式不一致

`src/data/build_mmap.py` 使用 `np.lib.format.open_memmap`，会写入 NPY header；`MmapImageTextDataset` 检测到 header 后却仍使用裸 `np.memmap` 从第 0 字节读取，没有应用 offset。由该函数构建的文件可能出现偏移和前部数据损坏。

**建议：**统一采用 headerless raw mmap，并增加 `schema.json`；或检测 NPY 后使用 `np.load(..., mmap_mode="r")`。所有通过文件大小推断 N 的脚本必须共享同一 mmap reader。增加写入→读取→逐像素一致性测试。

### P0-2：`split_by` 参数未实现

`build_mmap(..., split_by="project_id")` 接收参数，但当前实现仍对行索引随机划分，与文档的 project split 原则不一致。

**建议：**实现通用 `group_split(records, group_keys, seed)`；缺失 project_id 时回退到 lineage/root asset，而不是静默逐行随机切分。输出 split audit 报告。

### P0-3：Stage 3 可能发生跨 source 泄漏

精标 source 和 replay source 指向同一个底层百万纹理，却分别使用 `stage3_splits.json` 和全量 `splits.json`。若两套 split 独立生成，精标 val/test 样本可能进入 replay train。

**建议：**先建立唯一 global holdout；Stage 3 subset 只能从 global train 中选择；所有派生 split 必须继承 global split。增加跨 source ID/lineage intersection 检查，训练启动时发现泄漏直接失败。

### P1-1：Tile loss 被统一施加

`flow_tile_loss` 不接收 per-sample tileability mask。启用后 item、工具、护甲等非平铺纹理也被要求边缘相近。

**建议：**dataset 返回结构化 batch，包括 `tileable` 和 `asset_type`；只在 tileable 样本上计算 seam/tile loss。评测 seam 同样只统计 tileable block，并采用 alpha-aware 版本。

### P1-2：文档中的“双 val 选点”尚不存在

当前 `Trainer` 只根据单个 flow validation MSE 保存 best checkpoint；没有 i2i val、t2i regression val 或 forgetting-aware selection。

**建议：**增加 evaluator registry 和 checkpoint selector。Stylizer 至少记录 paired flow loss、alpha/edge/seam、condition utilization 和人工/VLM proxy；若修改共享主干，再跑 t2i regression。选点规则应配置化并保存完整指标，而不是只保存一个 MSE。

### P1-3：文本条件路线存在配置和文档漂移

仓库同时描述了离线 `Qwen3-VL-Embedding-8B` pooled embedding，以及在线 Qwen3-8B 多层 token sequence cross-attention；配置中可见 4096/12288 维和不同 token 长度。

**建议：**为 checkpoint 写入 `conditioning_manifest`，包括 encoder、revision、layers、instruction、normalization、tokenizer、max length、null condition 策略。训练、评测和推理加载时强校验。README、PROGRESS 和配置只保留当前正式路线，历史路线归档。

### P1-4：现有生成评测不足以支撑图生图

随机生成没有唯一 real target，`real-vs-generated RGB L2` 不能作为严格保真指标；当前 seam 忽略 tileability/alpha；VLM concept recall 强依赖 caption 表达。

**建议：**保留现有评测作为 t2i 趋势指标，新增独立 `eval_img2img.py` 和 paired benchmark。所有指标按 asset_type/tileability/transparency/source 分桶报告。

### P1-5：RGBA 语义未完全统一

当前直接对 RGBA 归一化训练，但透明像素背后的 RGB 可能任意，可能浪费模型容量并污染距离指标。

**建议：**统一 premultiplied RGBA；明确定义训练、推理、保存和评测的转换；标注/VLM 视图使用背景合成但始终保留原 alpha；加入 round-trip 和透明区域测试。

### P2-1：属性分类器标签可能与 shuffle 错位

`src/eval/attributes.py` 按 dataset 顺序生成标签，但 DataLoader 使用 shuffle，训练循环按累计位置取标签，图像与标签可能错位；分类器同时固定为 3 通道。

**建议：**dataset 直接返回 label，或使用显式 index；模型输入通道与 RGBA 规则统一；在修复前不要把该分类器用于正式验收。

### P2-2：梯度累计尾 batch 的缩放不精确

不足 `gradient_accumulation` 的 epoch 尾部仍按固定 accum 除 loss 后执行 optimizer step，会使该步梯度偏小。

**建议：**丢弃尾部、跨 epoch 累积，或按实际 pending 数重新缩放。训练日志记录真实 effective batch。

### P2-3：测试覆盖不足

当前 smoke test 不能覆盖新增条件系统和已有数据风险。

**建议新增：**

- raw/NPY mmap round-trip；
- group split 和跨 source 无泄漏；
- RGBA premultiply/unpremultiply；
- per-sample tile mask；
- text/reference dropout 四种组合；
- 三分支 CFG 数值测试；
- checkpoint conditioning manifest 兼容性；
- weighted MixDataset 采样比例；
- reference/target lineage 对齐。

---

## 10. 建议代码组织

```text
src/data/
  pair_dataset.py          # HDPairDataset + structured sample
  pair_builder.py          # manifest/mmap 构建
  lineage_split.py         # 去重、group split、leakage audit
  rgba.py                  # premultiplied RGBA 公共实现
  external_generation.py   # 外部模型任务清单与结果接入，不含模型训练

src/model/
  reference_encoder.py
  reference_adapter.py
  mc_flow_dit.py            # 接收可选 reference features

src/train/
  objectives.py             # flow/alpha/structure/tile masked loss
  evaluators.py             # evaluator registry

src/eval/
  img2img.py
  alpha.py
  structure.py
  condition_usage.py

scripts/
  build_hd_pairs.py
  audit_pair_splits.py
  train_img2img.py
  sample_img2img.py
  eval_img2img.py
  build_synthetic_t2i.py

configs/
  model/stylizer_adapter.yaml
  data/hd_pairs.yaml
  train/stylizer_phase1.yaml
  train/stylizer_phase2.yaml
```

不建议继续把所有任务塞进现有 `train.py` 的条件分支中。公共 flow、EMA、checkpoint 可以复用，但 dataset batch、loss、evaluator 和任务入口应明确分离，避免 t2i 训练循环不断增加特殊判断。

---

## 11. 关键消融矩阵

至少完成以下消融后才能决定最终架构：

| 实验 | 变量 | 目的 |
|---|---|---|
| A0 | nearest/median-cut/k-means/dither | 非学习基线 |
| A1 | channel concat vs spatial adapter | 验证参考注入方式 |
| A2 | adapter only vs last blocks/LoRA | 判断是否需要主干适配 |
| A3 | reference 32 vs 64 | 判断 HD 信息是否真正有效 |
| A4 | 单 reference vs 多变体 | 检验教师偏差 |
| A5 | straight vs premultiplied RGBA | 验证透明区域建模 |
| A6 | 无/有 structure loss | 判断辅助损失净收益 |
| A7 | 无/有 text condition | 判断文本编辑能力与冲突 |
| A8 | 合成数据 0/10/20/40% | 测量 t2i 反哺拐点 |

所有实验使用固定 Golden Set、相同 seed 和统一报告模板。

---

## 12. 最终决策清单

本设计已给出的默认决策：

1. MC→HD 使用现有模型，完全不训练；
2. 主监督采用“现有 MC→HD 输出 + 原始真实 MC target”；
3. Path B 只作 baseline/弱正则；
4. Stylizer 独立保存，但初始化自 Stage 3；
5. 首版 target 32×32 RGBA，reference 64×64 RGBA；
6. 首版采用 16×16 对齐的零初始化 spatial adapter；
7. 使用 premultiplied RGBA；
8. 按 lineage 统一 split；
9. 只有 tileable 样本使用 tile loss；
10. 通过 Stylizer Gate 后才生成 t2i 合成数据；
11. t2i 合成数据从 10%–20% 占比开始消融；
12. 先修仓库 P0，再开始正式图生图训练。

仍需项目侧提供或确认：

- 计划使用的现有 MC→HD 模型及其输出许可；
- 运行位置、吞吐和成本；
- 首批 target 选择范围；
- 是否同时支持 block 和 item，还是先从其中一类开始；
- 可接受的人工 Golden Set 审核规模。

---

## 13. 推荐的近期任务顺序

1. 修复 mmap 读取和 `split_by`；
2. 审计并重建 Stage 3 global split，排除 replay 泄漏；
3. 将 dataset batch 改为结构化字段并修复 masked tile loss；
4. 固定 premultiplied RGBA 规范；
5. 建立 512 个 Img2Img Golden Set；
6. 运行现有 MC→HD 模型构建首批 10k target/20k–40k pair；
7. 实现 deterministic baseline；
8. 实现 reference encoder + zero spatial adapter；
9. 完成 Phase 1 消融和 Gate 1；
10. 再决定扩量、有限解冻和 t2i 数据增强。

这一路线把最昂贵的外部生成和大规模训练放在可行性验证之后，能较早发现“模型只学会像素化”“reference 被忽略”“教师结构漂移”或“RGBA/数据泄漏导致虚假提升”等问题。
