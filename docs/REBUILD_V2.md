# MCGen 重构说明（2026-11-01）

## 范围与状态

用户要求全面重构，并明确文字生成和参考图生成两个功能都要保留。
本次采用独立 `mcgen` 包重新组织可执行核心；旧源码及未提交工作完整保留。
重构前源码快照位于 `outputs/rebuild_snapshot_20261101/source_before.tar.gz`。

完成工程实现不等于证明模型画质。新版模型需要重新训练，旧权重不能映射为新版 checkpoint。
初次重构验收时正式语义文本塔缺少本地权重；当时的流程测试使用明确标记的 byte 后端。
用户随后授权恢复 Qwen3-8B，恢复记录见本文末尾。未启动长时间 GPU 训练、未覆盖旧评测。

## 旧问题如何在新架构中消除

| 旧问题 | 新约束 |
|---|---|
| 配对 val 与文本 replay train 重叠 | 所有任务从一个 catalog 读取；训练入口审计 lineage/hash/split |
| 每个阶段各自划分 | 项目与重复目标组成连通分量，只划分一次；衍生记录继承 |
| 字符串 false 被当作 True | 数据边界严格解析三态，目录内只保存 bool/null |
| toroidal 默认开启 | 新数据层没有隐式增强；仅明确 tileable 样本可参加可选 seam loss |
| alpha 被丢弃或重复反预乘 | 磁盘 straight uint8；训练 premultiplied float；导出只 decode 一次 |
| 16px 目标被当作任意 32px 图 | 训练输出原生像素；32px 展示是显式 nearest export |
| 文本与参考 null 路径分叉 | 同一个 context 构造器，显式 mask/presence；不以零图表示缺失 |
| 双零初始化死通路 | 残差门零初始化但内部投影随机；测试参考通路梯度 |
| API/CLI/评测各自采样 | 全部调用 Runtime → sampling，API 与 runtime 输出逐像素比较 |
| MSE 最优被等同于质量最优 | 文件名明确 best_flow；双任务生成评测与人工复核独立 |
| epoch 验证随机变化 | 固定样本顺序、噪声与时间点；训练按 step 定位随机数 |
| 旧配置拼写错误被忽略 | dataclass 严格字段校验；未知选项直接失败 |

## 数据规范

每个样本提供 id、lineage、target_hash、prompt、asset_type、tileable、target、可选 reference。
target/reference 是只读 mmap 的 path/shape/index，整个源文件有 SHA256。
目标的 alpha=0 区域 RGB 规范化为零再计算内容哈希，避免无意义隐藏颜色绕过去重。
native16 迁移只接受精确 2×2 重复的 32px 目标，不对不合格目标偷偷降采样。

按项目和精确重复内容联合分组会减少独立组数，有时导致大连通分量。
这是需要报告的数据性质，不能为了看似均匀的 90/5/5 比例拆开重复组。
比例是分组哈希的期望，不保证精确记录数。数据增加后分组可能变化，必须发布新 catalog。
近重复/语义近重复仍需后续人工与专门检索审计，本版只保证精确内容和来源项目隔离。

迁移优先从 Stage-3 canonical prompt 读取文本，而不是用 MC→HD 的写实增强 prompt 训练 MC 文生图。
全量文本扩展通过 `--include-text` 在同一划分中加入，禁止在训练期直接读取旧 replay。
joint 训练的 `paired_fraction` 显式控制配对与纯文本来源比例；每个来源先均匀抽目标，再抽变体，
避免两个生成器或大量重复副本改变目标权重。

## 模型决策

首个可比较的新基线是原生像素 Flow Transformer：默认 16×16、256 token、256 hidden、8 blocks。
文本特征只投影一次，在所有 block 共享；参考图卷积逐步降至原生 token 网格，保留空间位置。
时间条件通过 AdaLN 门控进入 self attention、cross attention 和 MLP。
文本与参考各有独立 null sentinel 和类型标记，dropout 发生在明确的条件 mask 上。

该结构是可训练基线，不预先断言优于旧 130M 模型。宽度、深度、原生分辨率可通过配置调整。
不继续默认冻结一个已受旧数据问题影响的主干，不默认启用 EMA，不默认增加像素平铺损失。
palette 是可选导出变体；没有把量化后的图片冒充模型原始输出。
离散调色板建模、量化损失、专门的透明轮廓建模应在干净基线后独立消融。

## 训练和恢复

训练梯度累计按完整 optimizer step 执行；没有 epoch 尾 batch 的缩放歧义。
按 step 派生采样、噪声、dropout 随机数，模型没有未记录的随机 dropout。
checkpoint 包含严格版本、完整配置、数据指纹、optimizer、step、best_flow 指标。
本地文本塔的配置、tokenizer、权重文件内容指纹同时进入 checkpoint；HF 路径也有离线微型模型集成测试。
恢复要求配置和数据指纹一致，不静默裁剪权重、不跳过不匹配参数。
CPU 精确恢复有回归测试；CUDA 数值确定性与吞吐需要在实际训练硬件上独立验证。

每次验证分别输出 text/reference/joint flow MSE；生成评测使用相同 seed 比较各条件。
训练代码先采用可审计的同步数据与文本编码，没有重新引入有共享 hook 状态的后台线程。
大规模吞吐优化应在正确性稳定后添加，并保留同步实现作为对照。

## 评测与验收

自动报告包含：黑/白合成 RGB MAE、alpha MAE/IoU、边缘 MAE、可见颜色数、目标色板距离、
仅 tileable 样本的 seam MAE。IoU 的双方全空情况定义为 1。
输出原始 PNG、明确的 box/box+16 色基线、参考打乱对照以及分页人工复核表。
参考打乱必须换成不同 target_hash 的参考，不仅换一个数组下标。

所有颜色/轮廓指标只是辅助。文本生成不应因没还原某张唯一 target 就被判定失败，
因此语义、像素风格、可用性由人工复核表明确保留，不自动用像素误差替代。
当前 catalog test 是同源管线 holdout，不等同于任意自然图像 OOD；外部参考集需独立审计。

模型正式晋级前需要：

1. 固定、人工核验的双任务 Golden Set，覆盖透明 item、tile、长尾语义和组合属性。
2. 分任务与资产类别对比，不把 joint 改善当作 text 改善。
3. 学习式参考转换在结构、像素风格、人工可用性上超过确定性基线。
4. 新合成数据在等训练预算下改善文生图；旧报告不能作为对照数字。
5. 实际 GPU、正式文本塔和部署环境的性能/精度验证。

本次自动测试及短程训练不代替以上画质验收。

## 本次实际验证与产物

- 完整测试运行：72 passed；随后新增导入/合并与远端 revision 校验，增量回归 22 passed。
  当前测试集合共 74 项（旧 48、新 26），API 测试需在允许跨线程事件循环的环境运行。
- 本地微型 HF 模型完成冻结编码、短程训练、checkpoint 加载、推理、文本塔文件变更拒绝测试。
- 原有 196 个源码、脚本、配置、测试文件逐字节与重构前快照核对，未改动。
- 配对 catalog：`data/catalogs/native16_v2`，34,170 条；train/val/test = 30,568/2,076/1,526。
- 全量 catalog：`data/catalogs/joint16_v2`，1,051,142 条；train/val/test = 1,027,890/13,526/9,726。
  全量唯一目标数分别为 837,932/12,212/9,240；全量源文件 SHA256 与全部 split 已复核。
- 全量实际数据 CPU 4-step 流程验证：`outputs/rebuild_realdata_smoke_20261101`。
  此权重使用 byte 后端与微型模型，只是集成测试产物。
- 6 个真实 holdout 目标的双任务/打乱参考/确定性基线验证：
  `outputs/rebuild_realdata_eval_20261101/report.json`，含原始 PNG、分页图、人工复核 CSV。
  CLI 文生图、纯参考图均成功，PNG 附带请求 JSON。
- 旧污染审计、数据桶统计和环境版本：`outputs/rebuild_snapshot_20261101/`。
- wheel 构建与 `git diff --check` 通过；未向现有环境安装新依赖。

## Qwen3-8B 恢复与真实 GPU 验证

用户授权下载后，官方 `Qwen/Qwen3-8B` 的版本
`b968826d9c46dd6066d109eabc6255188de91218` 已恢复到 `/home/iflab/models/Qwen3-8B`。
14 个文件中包含 5 个权重分片，共 16,381,516,776 字节权重；分片大小、SHA256 和索引引用均通过校验。
小文件来自官方 Hugging Face；权重经 hf-mirror.com 传输，内容按官方 API 的 LFS SHA256 验证。
`configs/v2/joint_native16.yaml` 已固定本地路径与上述版本。

真实验证使用两张空闲 A100：文本塔位于 cuda:1，原生像素模型位于 cuda:0。
第 9/18/27 层特征拼接输出为 `[2, 18, 12288]`，有效 token 数为 16/18；
特征、模型前向损失和反向梯度均有限，文本塔保持冻结，权重更新次数为 0。
峰值分配显存：生成模型侧约 0.234 GiB，文本塔侧约 14.119 GiB。
这验证了真实文本塔接入，不代表已完成生成模型训练。

记录：`outputs/qwen3_restore/download_verified.json`、`integration_verified.json`。
复现命令：

```bash
python scripts/restore_qwen3_8b.py --transport mirror
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/check_qwen3_8b.py
```
