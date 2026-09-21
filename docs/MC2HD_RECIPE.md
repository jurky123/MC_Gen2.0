# MC → HD 配方（定型 v1，2026-09-20）

本项目**只保留这一条** MC→HD 路径（早期 Tier-D 的 `prompt→HD→降采样/SDEdit` 路径已废弃）。

## 配方

```
真实 MC 纹理（32px，原生 16×16 网格）
  → NEAREST 放大到 384×384          # 保持 16 网格（每格 24px）
  → bilateral(d≈19, σ_color=σ_space=60) + Gaussian(σ≈9)   # 去台阶、保留低频结构与色彩
  → FLUX.2-klein-4B image-edit，8 步
      prompt = "<Stage-3 精标描述，**剔除 "block" 一词**>。photorealistic
                high-end game asset, physically based materials, realistic
                lighting and shading, fine surface detail, crisp anti-aliased
                edges. Keep the same object, orientation, proportions,
                silhouette and colors. Remove all pixelation and voxel
                stair-step edges."
      （item 额外前缀："single item centered on a plain white background, "）
  → HD 384×384（写实游戏素材）
```

- 耗时 ≈ 1.0 s/张（单卡 A100，8 步，384 分辨率）；
- 许可证：FLUX.2-klein-4B = Apache-2.0；
- 参数在 512 分辨率下调优；换分辨率时**模糊参数按比例缩放**（`scale = size/512`）。

## 关键经验（都经过 A/B 实测）

| 项 | 结论 |
|---|---|
| 输入低通 | 必需。nearest 输入的输出台阶比 61.6 → bil+gauss12 后 ~1.5 |
| bilateral / meanshift 单独用 | **无效**（保边算子会保留台阶），只能作为 gauss 的前置去噪 |
| gauss28 | 最平滑但结构损失大；gauss16 备选；**推荐 bil+gauss12** |
| prompt 写实措辞 | 必需。"flat / round off stair-steps" 会得到扁平卡通；"photorealistic PBR / studio render" 才是照片级 |
| prompt 里的 `block` | **必须删**，否则模型画成立方体（苔藓地毯→3D 方块）|
| 步数 | 8 步足够，16/24 无差别 |
| FLUX latent img2img（加噪去噪）| **不可用**：蒸馏模型只从纯噪声起步，t0=0.2~0.8 全是"磨糊" |
| Qwen-Image-Edit | 写实强但重写布局、爱加场景；分辨率下限 256，128 会崩坏 → 暂不用于本路径 |
| 分辨率 | 384 是质量/成本平衡点；256 可用；128 对简单物品可用 |

## 产物

- 脚本：`scripts/build_mchd_pairs.py`（分层抽样、断点续跑、原子写入、`config.json` 指纹、`manifest.jsonl` provenance）；
- 数据：`pairs/mchd_v1`（200 验收）、`pairs/mchd_stage3_{a,b}` + `pairs/mchd_fill_{a,b}`（合计 **17,085** 对 = Stage-3 精标全子集）；
- HD 文件 `hd/row{07d}.png`，manifest 记录 `row / prompt / subject / asset_type / project_id / prompt_source / 许可证 / 预处理 / size / steps / seed / 耗时`。

## 已知限制

- 长尾/模糊文件名的精标描述可能不准（例：`881685` 木板→HD 生成铲子、`367339` 深蓝门→亮蓝），量产时建议加自动过滤（VLM 同一性 + 颜色直方图距离 + alpha IoU）；
- 极细结构（玻璃瓶壁等）在高斯平滑后会变软，HD 侧细节有限。
