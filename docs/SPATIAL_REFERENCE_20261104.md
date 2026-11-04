# 独立视觉 Token 参考图实验

目标是让相同类别随参考图呈现不同的配色、装饰和轮廓，同时保持 MC 风格与物品可辨认性。

保留 4128 步文生图底座及原文本条件通路。参考图经过冻结的 Qwen3-VL-Embedding-8B **视觉塔**，提取 **64 个空间 token**，经投影后在第 **3、6、9 层**新增独立 Cross-Attention。Query 来自生成器特征，Key/Value 来自参考图；新增结果通过零初始化的 tanh 门控加入生成器。

本轮无需运行 Qwen 语言模型来编码参考图，不使用图文融合全局向量，也不另设全局图像分支。参考图保留完整物品，等比例缩放并居中补白至 256×256；使用官方视觉 patch merger 的 8×8×4096 特征，按单图协议缓存。原文本继续使用已验证的原始 embedding 缓存。

## 训练

- 只更新视觉投影、三个图像 Attention 和门控，共 **5,259,264** 参数；原生成器和视觉塔冻结。
- 沿用 **5,716 个训练目标 / 639 个验证目标**，分别有 11,432 / 1,278 张高清配对。均匀抽目标，再随机抽 FLUX/Qwen 参考版本。
- **80% 图文、20% 纯参考图**，文本使用原文件信息标签；保持原 Flow Matching MSE。
- 首轮不做颜色与几何增强，用于观察独立视觉特征通路的实际效果。
- AdamW，学习率 1e-4，50 步预热后余弦衰减。首轮预算约为每个目标 60 次条件曝光，在 10、30、60 次曝光保存检查点。
- 两张 GPU 已有推理服务；本轮仅使用剩余显存。GPU 0 实测 batch 32–192 后选择 **192**，峰值 reserved 约 **9.40 GiB**，约 **479 张/秒**。计划 **1,787 步**；该速度受共享 GPU 负载影响。

## 验证

每次验证检查原生成器全部张量摘要，以及无参考图生成与独立载入的原底座是否逐元素一致。正确参考、替换参考、无参考对照分别在低/中/高噪声上计算误差；替换参考时保持文字不变。

保存训练内、验证集的普通生图对照，同时固定 `helmet`、`sword`、`shield`、`axe` 文字和种子，替换同类参考图，并使用第二个种子检查主要设计是否稳定。生成原始 16×16 RGBA，不进行轮廓或配色修饰。

验证误差最优的图文与纯参考图权重分别保留。它们属于实验检查点，装饰保留和 MC 风格仍需看图判断；本轮不使用测试集挑选模型。高清参考新增、但目标中不存在的装饰，可能仍被现有配对监督忽略。

实现检查：195 项回归测试通过。16 个真实配对经过 80 次更新后，固定噪声误差由 0.1414 降到 0.1298，投影和门控获得有效梯度，原权重与纯文本生成保持一致。重新加载视觉塔编码四个跨缓存位置的样本，与训练缓存逐元素一致；这些检查只验证通路与实现，不代表装饰泛化已经改善。

正式训练已完成 **1,787 步**，约 **60 次目标曝光**，含生图验证共约 **18.6 分钟**。图文及纯图像的验证误差均在最后一步最低；最佳与最后检查点的全部模型张量相同。各次验证中，原生成器张量及无参考图输出保持一致。

随后完成训练内、训练未见各64个目标及16组同文字不同参考图的双种子对照，共1,664张对照输出。新版图文联合更接近配对MC目标，纯参考图弱于旧版；装饰多样性尚未稳定改善。旧版约145次目标曝光、新版60次，任务比例也不同，属于版本效果比较。[完整结论与指标](SPATIAL_REFERENCE_EVALUATION_20261104.md)。

## 运行

输出目录：`outputs/spatial_reference_20261104/`。

```bash
PYTHONPATH=src python scripts/cache_visual_reference.py \
  --out outputs/spatial_reference_20261104 --device cuda:1

# 新实验仅执行一次；恢复时保留现有 config.yaml 和 batch_budget.json。
PYTHONPATH=src python scripts/train_visual_reference.py prepare \
  --out outputs/spatial_reference_20261104

PYTHONPATH=src python scripts/queue_visual_reference.py \
  --out outputs/spatial_reference_20261104 --device cuda:0
```

`cache_progress.json` 表示缓存进度；`queue_status.json` 表示队列阶段；`status.json` 表示训练步数与时间估算。`training/validation.jsonl` 保存验证，`generation_checks/` 保存普通对照及固定文字换参考图的拼图。SIGTERM 会保存当前模型和优化器；重启队列后从 `latest.pt` 恢复。

新检查点可由原 Runtime/CLI/API 加载。有参考图时自动使用独立视觉 token，也可显式指定 `--conditioning-mode visual_tokens`；`--text-scale` 与 `--ref-scale` 分别控制文字和参考图。无参考图直接沿用原文生图路径。旧检查点及其推理协议保持兼容。

已有本机数据和缓存时，可用 `scripts/evaluate_spatial_reference.py --out outputs/new_comparison --device cuda:1` 复现本轮版本对比；该脚本需要两版实验目录和旧固定评测，默认路径可通过参数调整。公开发布仅含实验权重及必要校验，原始图片、特征缓存与完整评估保留本地。
