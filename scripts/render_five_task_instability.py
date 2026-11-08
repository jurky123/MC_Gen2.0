"""Publish finite-gradient collapse diagnostics through the private gallery."""
import json
from pathlib import Path
import shutil
import statistics

from mcgen.catalog import atomic_json
from prepare_reference_recovery import COMMON
from train_five_task_reference import draw_sheet

RUN = Path('outputs/reference_five_task_20261108')
ROOT = Path('outputs/five_task_instability_20261108')
GALLERY = Path('outputs/native16_data_review_20261107')


def main():
    destination = GALLERY/'five-task-diagnosis'
    (destination/'panels').mkdir(parents=True, exist_ok=True)
    root_panels = ROOT/'panels'
    root_panels.mkdir(exist_ok=True)
    phases = [4000, 8000, 16000]
    entries = [([RUN/'generation_checks'/f'{s:06d}'/f'common_{j:02d}.png' for s in phases], prompt,
                'Same prompt and Gaussian noise; FP32; CFG2') for j, prompt in enumerate(COMMON)]
    draw_sheet(root_panels/'checkpoint_text.png', 'Five-task training | healthy intermediate snapshots vs failed final',
               ['Best balanced4000', 'Best text8000', 'Failed final16000'], entries)
    rows = json.loads((RUN/'generation_checks/008000/panel.json').read_text())
    entries = []
    for j in range(24, 32):
        row = rows[j]
        entries.append(([row['reference_file'], row['target_file']]+[RUN/'generation_checks'/f'{s:06d}'/f'{j:02d}_joint_detail.png' for s in phases],
                        row['captions'][0], 'TRAIN sample | fixed detailed caption and noise'))
    draw_sheet(root_panels/'checkpoint_train.png', 'Training targets also deteriorate | not just unseen-design generalization',
               ['HD reference', 'MC target', 'Balanced4000', 'Text8000', 'Failed16000'], entries)
    for name in ('common_causal', 'joint_causal'):
        shutil.copy2(ROOT/'causal'/f'{name}.png', root_panels/f'{name}.png')
    for path in root_panels.glob('*.png'):
        shutil.copy2(path, destination/'panels'/path.name)
    body = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五任务训练失稳诊断</title><style>body{font:16px/1.7 system-ui;background:#111827;color:#e5e7eb;max-width:1080px;margin:24px auto;padding:0 16px}a{color:#93c5fd}img{max-width:100%}article,details{background:#1f2937;padding:18px;border-radius:10px;margin:18px 0}table{border-collapse:collapse}td,th{padding:6px 12px;border-bottom:1px solid #53647c;text-align:left}summary{cursor:pointer}</style><p><a href="/five-task/">训练阶段</a> · <a href="/five-task-evaluation/">完整训练集外评估</a></p><h1>五任务训练：后期失稳诊断</h1><p><b>最终16000步已经退化，不建议使用或继续训练这个版本。</b>本轮综合最佳4000步、文字最佳8000步均完整保留，生产权重未被本次诊断修改。</p><p>约9000步开始出现梯度增长，约11000步起裁剪频繁，12000步后所有任务的训练损失都变差。峰值裁剪前梯度范数约29480，全部参数仍为有限值，所以原来的NaN检查没有触发。独立FP32评估也复现破碎。</p><h2>低噪声阶段先失稳</h2><p>固定8个训练目标、同一输入噪声，下面为纯文字条件的FP32前向/反向。t=0接近干净图，t=1接近纯噪声。</p><table><tr><th>版本</th><th>t=0.05误差</th><th>t=0.05梯度范数</th><th>t=0.99误差</th></tr><tr><td>本轮8000</td><td>0.0987</td><td>0.92</td><td>0.1006</td></tr><tr><td>最终16000</td><td>0.3530</td><td>27.40</td><td>0.0863</td></tr></table><p>时间MLP在低噪声时的输出幅值约增长5倍，经AdaLN调制后部分自注意力分数放大到约86万。高噪声预测仍能表达物品，后面的去噪过程却会破坏轮廓。BF16进一步放大梯度，FP32下也存在异常。</p><h2>保留版本与最终版</h2><img loading="lazy" src="panels/checkpoint_text.png"><h2>训练样本也受到破坏</h2><img loading="lazy" src="panels/checkpoint_train.png"><details><summary>因果对照：仅在内存中替换时间/调制分支</summary><p>最后一列将异常时间分支放入健康模型，直接生成彩色噪声。把健康时间与调制分支换回异常模型可明显减轻破碎，但不等于完整修复。所有替换组合仅供定位，未保存成可部署权重。</p><img loading="lazy" src="panels/common_causal.png"><img loading="lazy" src="panels/joint_causal.png"></details><h2>原方案与冻结时间MLP的短续训对照</h2><p>两组均恢复本轮8000步的完整模型和优化器，使用相同配对、条件与随机种子，上限各4000次更新。唯一训练差别是是否冻结时间MLP；两组都启用失稳停训保护，不使用测试集。</p>'''
    trials = {}
    for arm, title in [('control', '原方案'), ('freeze_time', '冻结时间MLP')]:
        trial = ROOT/'continuation'/arm
        status = json.loads((trial/'status.json').read_text()) if (trial/'status.json').exists() else {}
        metrics = [json.loads(x) for x in (trial/'metrics.jsonl').read_text().splitlines()] if (trial/'metrics.jsonl').exists() else []
        tail = metrics[-200:]
        report = {'status': status}
        body += f'<article><h3>{title}</h3><p>当前更新：{status.get("step", "初始化")}，完成：{status.get("complete", False)}</p>'
        if tail:
            report['recent_median_gradient'] = statistics.median(x['gradient_norm'] for x in tail)
            report['recent_mean_loss'] = statistics.mean(x['loss'] for x in tail)
            body += f'<p>最近200步：梯度中位数 {report["recent_median_gradient"]:.4f}，平均训练误差 {report["recent_mean_loss"]:.4f}</p>'
        if status.get('halt'):
            body += '<p>已因持续梯度与损失恶化触发停训保护。</p>'
        if status.get('time_weights_bitwise_unchanged'):
            body += '<p>时间MLP权重逐元素保持不变。</p>'
        evaluations = sorted(p for p in trial.glob('eval_*') if (p/'metrics.json').exists())
        if evaluations:
            latest = evaluations[-1]
            report['evaluation'] = json.loads((latest/'metrics.json').read_text())
            report['evaluation_step'] = int(latest.name.split('_')[-1])
            for stem in ('common', 'paired'):
                name = f'{arm}_{stem}.png'
                shutil.copy2(latest/f'{stem}.png', destination/'panels'/name)
                body += f'<img loading="lazy" src="panels/{name}">'
        body += '</article>'
        trials[arm] = report
    body += '<h2>已补的训练保护</h2><p>对持续的训练损失上升、梯度增长和高裁剪比例联合检测；单次尖峰不误停。历史日志回放会在约11500步停下，而不是继续跑到16000。触发时跳过下一次优化器更新，保留best和latest，另存仅供诊断的halted权重，拒绝静默续训。原始训练代码已按当时的哈希归档。</p><p>以上短对照结果是诊断，不自动替换现有推理模型。</p>'
    temporary = destination/'index.html.tmp'
    temporary.write_text(body)
    temporary.replace(destination/'index.html')
    atomic_json(ROOT/'continuation_summary.json', trials)
    banner = '<p id="instability-banner" style="padding:14px;background:#7f1d1d;border-radius:8px">本轮后期训练已失稳，最终16000步不建议使用。4000步综合最佳与8000步文字最佳均保留。<a href="/five-task-diagnosis/">查看原因与修复对照</a></p>'
    for directory in ('', 'five-task', 'five-task-evaluation'):
        p = GALLERY/directory/'index.html'
        content = p.read_text()
        if 'id="instability-banner"' not in content:
            p.write_text(content.replace('<h1>', banner+'<h1>', 1))
    print('PUBLISHED http://localhost:8766/five-task-diagnosis/')


if __name__ == '__main__':
    main()
