# MC Gen 2.0 配对数据发布说明

这两份发布服务于不同的数据用途。v5 是通用配对合集的高清质量修订版，v3 是用于学习装备和物品变体的专项合集。表中的数量按各 ZIP 自己的配对索引统计。跨包合并训练时，应按来源和图像哈希重新去重，再分配训练与验证集。

| 发布 | 用途 | 配对规模 | 下载 |
| --- | --- | --- | --- |
| 通用合集 · HDRepair v5 | 通用 MC/HD 配对训练 | 主集 3,592 对；含历史试验共 3,685 对 | [ZIP](https://huggingface.co/datasets/Risposta/MC_Gen/resolve/8c45179b167b084eea9a00b8c2d8a7d0d6ead050/releases/MCGEN_Final_20261120_HDRepair_v5.zip) |
| 变体矩阵 · HDRepair v3 | 根据参考图与文字学习类别内的设计变体 | 970 对（首批 198 + 追加批 772） | [ZIP](https://huggingface.co/datasets/Risposta/MC_Gen/resolve/034180d30058e993454c333797d1c99ffa747a04/releases/release_combined_HDRepair_v3.zip) |

## 通用合集 · HDRepair v5

- 对原合集 3,691 张 HD 完成初筛，448 张进入二级筛查。1,387 个修复候选经实际看图复核，375 张重生成并独立复审通过，1,006 张保留，6 张隔离。
- 修复针对 MC 网格残留的大尺度阶梯和方块感。重生成后重新核对物体、数量、朝向、主色和主要结构，并重写 HD 与 MC 各自的中英精标注。修复保留原 MC 目标字节。
- 主索引用 `metadata/pairs_main_3592.jsonl`，完整索引用 `metadata/pairs_all.jsonl`。完整索引包括 93 对历史试验记录，其中 16 对为历史非主集格式变体；如需严格 16×16 RGBA 训练，使用主索引。
- 已纳入另一个抽样生产批次中通过验收的 593 对。该批目标为 1,000 对，发布时还有 407 对未纳入。

## 变体矩阵 · HDRepair v3

- 从 MC Gen 素材中抽取同类不同设计的装备和物品，覆盖材质、部件颜色、拼色、花纹、镶嵌、轮廓及部件造型等可叠加变体。适用于参考图与文字联合条件训练。每条索引绑定一对 MC/HD 和一张生成参考。
- 追加批从原 800 对开始，v2 已隔离 8 对。v3 在剩余 792 对上放大复查外缘、主要开孔和内部饰纹，再发现 60 个问题。40 张重新生成并通过独立复审，20 张退出训练索引（17 张已纠正过一次、2 张生成请求被拒绝、1 张与同批设计近似重复）。追加批最终保留 772 对。
- 主索引为 `metadata/pairs.jsonl`，也提供 `metadata/pairs.csv`。组件数量见 `metadata/matrix_audit.json`，来源及类别信息见 `metadata/provenance.jsonl`。新版审核、修复与隔离证据在 `audit/release_additional_800_HDRepair_v3/`。

## 如何读取

从上述配对索引取每对的文件路径与哈希。 `mc/` 保存 MC 目标，`hd/` 保存高清原生图，`references/` 保存生成参考。MC→HD 的 r16 高斯模糊只施加于放大的生成参考，训练目标仍使用原始 MC 图。中英标注见 `annotations/fine_captions.jsonl`，与配对记录一起读取。训练应以配对索引为准，`audit/` 内的失败图、旧版图和隔离证据用于追溯。

| ZIP | 字节数 | SHA-256 |
| --- | ---: | --- |
| `MCGEN_Final_20261120_HDRepair_v5.zip` | 6,605,541,743 | `5f5dd0d493e7ae1322c40cca23de566dadc1c2fef8fa11d11f2b006044566253` |
| `release_combined_HDRepair_v3.zip` | 1,230,314,207 | `22318691748d932ed030948eb1a192732562e8154cfeebb9a4b3e24c450d5e31` |

两份 ZIP 已通过 CRC 校验，且远端大小和 SHA-256 均与本地一致。素材使用和归属信息参见数据仓库说明及包内来源记录。 [Hugging Face 数据仓库](https://huggingface.co/datasets/Risposta/MC_Gen)
