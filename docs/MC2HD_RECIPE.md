# MC → HD 配方（定型 v2，2026-09-22）

本项目**只保留这一条** MC→HD 路径（早期 Tier-D 的 `prompt→HD→降采样/SDEdit` 路径已废弃）。
同一份 MC 用**两个生成器**各产一张 HD，作为 Stylizer 的多生成器参考多样性来源：

| 生成器 | 许可证 | 用途 |
|---|---|---|
| FLUX.2-klein-4B | Apache-2.0 | 量产 + 生产数据 |
| Qwen-Image-2.1 | **Qwen Research License（非商用）** | 研究/评测 + 训练多样性 |

## 共用前处理

```
真实 MC 纹理（32px，原生 16×16 网格）
  → NEAREST 放大到 384×384          # 保持 16 网格（每格 24px）
  → 按生成器分档低通（见下表）
  → <生成器> image-edit
      prompt = "<Stage-3 精标描述，**剔除 "block" 一词**>。photorealistic
                high-end game asset, physically based materials, realistic
                lighting and shading, fine surface detail, crisp anti-aliased
                edges. Keep the same object, orientation, proportions,
                silhouette and colors. Remove all pixelation and voxel
                stair-step edges."
      （FLUX 对 item 额外前缀："single item centered on a plain white background, "）
  → HD 384×384（写实游戏素材）
```

- 参数在 512 分辨率下调优；换分辨率时**模糊参数按比例缩放**（`scale = size/512`）。

## 两条配方

### A. FLUX.2-klein-4B（8 步，Apache-2.0）

```
NEAREST 384 → bilateral(d≈19, σ_color=σ_space=60) + Gaussian(σ≈12) → FLUX edit 8 步
```

- 耗时 ≈ 1.0 s/张（单卡 A100，384 分辨率）；
- 产物：`pairs/mchd_stage3_{a,b}` + `pairs/mchd_fill_{a,b}`（满幅 17,085 中的 6,019）+ `pairs/mchd_cutout_flux_{a,b}`（抠图 11,066，带 item 前缀重生成）。

### B. Qwen-Image-2.1（30 步，Research License）

```
NEAREST 384 → Gaussian(σ=28) → Qwen-Image-2.1 edit，30 步，true_cfg=1.0（无 negative prompt）
```

- 耗时 ≈ 3.2–3.5 s/张（单卡 A100，384 分辨率，`torch.compile` max-autotune-no-cudagraphs）；
- 产物：`pairs/mchd_qwen21_{a,b}`（满幅 6,019）+ `pairs/mchd_qwen21_cut_{a,b}`（抠图 11,066，软 alpha 重生成）；
- `true_cfg=1.0` 时 Qwen **忽略 negative prompt**（日志有 warning），CFG 关闭也去除了体素伪影 → 与 FLUX 需要 neg prompt 不同；
- 8-bit 量化不可用：A100 无 FP8 tensor core，`torchao int8_weight_only` 实测**更慢**（bf16 3.30s vs int8 4.41s），已卸载。

## 关键经验（都经过 A/B 实测）

| 项 | 结论 |
|---|---|
| 输入低通 | 必需。nearest 输入的输出台阶比 61.6 → bil+gauss12 后 ~1.5 |
| bilateral / meanshift 单独用 | **无效**（保边算子会保留台阶），只能作为 gauss 的前置去噪 |
| gauss28 | 最平滑但结构损失大；**Qwen 路径选 gauss28**（CFG 已关，靠输入平滑去台阶）；FLUX 路径 **bil+gauss12** |
| prompt 写实措辞 | 必需。"flat / round off stair-steps" 会得到扁平卡通；"photorealistic PBR / studio render" 才是照片级 |
| prompt 里的 `block` | **必须删**，否则模型画成立方体（苔藓地毯→3D 方块）|
| 额外加"平滑"从句 | **有害**，每加一句都让输出变糊 → 只用原始 REALISM 措辞（P1）|
| 步数 | FLUX 8 步足够；Qwen 30 步 |
| FLUX latent img2img（加噪去噪）| **不可用**：蒸馏模型只从纯噪声起步，t0=0.2~0.8 全是"磨糊" |
| Qwen-Image-Edit（2511）| 写实强但重写布局、爱加场景；分辨率下限 256，128 会崩坏 → 已被 Qwen-Image-2.1 替代 |
| 分辨率 | 384 是质量/成本平衡点；256 可用；128 对简单物品可用 |

## 抠图（transparent cutout）处理

**问题**：17,085 对里 11,066（64.8%）是透明抠图（床侧板、矿粒、物品精灵）。把它们合成白底后，
两个模型都"填满画面"，输出变成被裁切的放大 3D 物件。

**修复分三步**：

1. **FLUX**：给 item 加前缀 `"single item centered on a plain white background, "` → 输出回到物体本身
   （代价：把 `olive_leaves` 这类**图案型**抠图过度简化成单枝树叶）。
2. **alpha-aware ref**（`preprocess_ref` in `build_mchd_pairs.py`）：抠图传 RGBA（透明背景），满幅传白底 RGB。
   满幅行两个生成器都不再有"填满画面"问题。
3. **软 alpha（关键）**：抠图分支原先只模糊 RGB、**保留原 NEAREST alpha 硬边** → 模型把锯齿轮廓当像素画复刻，
   输出仍有方块感。改为对**整个 RGBA 做 gauss28**（`gauss()` 本身对 alpha 用 `σ/2`），方块感消失。

**紫色背景假象**：RGBA 在 alpha=0 处的 RGB 是任意值（实测 (141,57,180)），用 `convert("RGB")` 直接看会
显示成紫色 → **不是模型幻觉**。显示/合成时必须 alpha-over-white。仓库里多处旧脚本仍用裸 `convert("RGB")`。

## 产物

| 目录 | 内容 | 数量 |
|---|---|---|
| `pairs/mchd_stage3_{a,b}` + `mchd_fill_{a,b}` | FLUX 满幅 + 旧抠图 | 17,085（其中抠图 11,066 已废弃）|
| `pairs/mchd_cutout_flux_{a,b}` | FLUX 抠图重生成（item 前缀）| 11,066 |
| `pairs/mchd_qwen21_{a,b}` | Qwen 满幅 + 旧硬 alpha 抠图 | 17,085（其中抠图 11,066 已废弃）|
| `pairs/mchd_qwen21_cut_{a,b}` | Qwen 抠图重生成（软 alpha）| 11,066 |
| `pairs/stylizer_v4/` | 多生成器合并 Stylizer 训练对 | 满幅 6,019×2 + 抠图 11,066×2 |

- 脚本：`scripts/build_mchd_pairs.py`（FLUX）、`scripts/build_mchd_pairs_qwen21.py`（Qwen）；
- HD 文件 `hd/row{07d}.png`，manifest 记录 `row / prompt / subject / asset_type / project_id /
  prompt_source / 许可证 / 预处理 / ref_mode / out_mode / size / steps / seed / 耗时`；
- `config.json` 指纹防串味（配方、行数、`rows_sha`），换配方必须换 `--out`。

## 已知限制

- 长尾/模糊文件名的精标描述可能不准（例：`881685` 木板→HD 生成铲子、`367339` 深蓝门→亮蓝），量产时建议加自动过滤（VLM 同一性 + 颜色直方图距离 + alpha IoU）；
- 极细结构（玻璃瓶壁等）在高斯平滑后会变软，HD 侧细节有限；
- FLUX item 前缀对**图案型**抠图（`olive_leaves` 等 39% 透明）会过度简化 → 这类目前靠 Qwen 侧补位；
- Qwen-Image-2.1 许可证限制：产出的 HD 只能用于研究/评测，**不能进生产数据**（生产链路只用 FLUX）。
