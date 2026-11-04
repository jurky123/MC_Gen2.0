#!/usr/bin/env python3
"""Read-only live training monitor. Standard library only; Ctrl+C never stops training."""
import argparse
from collections import deque
import csv
import datetime as dt
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
STAGES = {
    'waiting_for_verified_model': '等待模型下载及校验',
    'encoding_and_batch_probe': '计算 Embedding / 准备 batch 实测',
    'batch_tuning': '实测 batch 与显存',
    'finishing_embedding_cache': '计算 Embedding 缓存',
    'features': '准备文本缓存', 'smoke': '训练短测', 'train': '正式训练',
    'evaluate': '生图评估', 'complete': '已完成', 'failed': '失败', 'stopped': '已停止',
    'waiting_for_first_cache_chunk': '等待首批图像特征',
    'waiting_for_full_cache': '计算图像 / 图文特征缓存',
    'controls': '准备固定文字、替换参考图的验证对照',
    'probe': '实测 batch 与显存',
}


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def scalar_config(run):
    """Prefer the exact training JSON; otherwise read the flat YAML train/model scalars."""
    config = (read_json(run / 'run.json').get('config') or
              read_json(run / 'training/run.json').get('config'))
    if config:
        return config
    result, section = {}, None
    try:
        lines = (run / 'config.yaml').read_text().splitlines()
    except OSError:
        return result
    for line in lines:
        if re.fullmatch(r'[a-zA-Z_]+:', line):
            section = line[:-1]
            result.setdefault(section, {})
        match = re.match(r'^  ([a-zA-Z_]+):\s*([^#]+?)\s*(?:#.*)?$', line)
        if section and match:
            key, value = match.groups()
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            result[section][key] = value
    return result


class Metrics:
    """Incremental JSONL reader: leave a partially written last line for the next refresh."""
    def __init__(self, path):
        self.path = path
        self.reset()

    def reset(self):
        self.offset = 0
        self.inode = None
        self.last = {}
        self.losses = deque(maxlen=100)
        self.best = self.validation = None
        self.validation_count = 0
        self.bad_lines = 0
        self.session_start_step = None
        self.elapsed = None

    def update(self):
        try:
            info = self.path.stat()
            if self.inode != info.st_ino or info.st_size < self.offset:
                self.reset()
                self.inode = info.st_ino
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b'\n'):
                        break
                    self.offset = stream.tell()
                    try:
                        row = json.loads(line)
                        step = int(row['step'])
                    except (ValueError, KeyError, TypeError):
                        self.bad_lines += 1
                        continue
                    self.last = row
                    elapsed = row.get('seconds')
                    if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0:
                        if self.elapsed is None or elapsed < self.elapsed:
                            self.session_start_step = step - 1
                        self.elapsed = elapsed
                    if isinstance(row.get('loss'), (int, float)):
                        self.losses.append(row['loss'])
                    scores = {k: v for k, v in row.items() if k.endswith('_flow_mse') and isinstance(v, (int, float))}
                    if scores:
                        score = sum(scores.values()) / len(scores)
                        record = {'step': step, 'score': score, 'scores': scores, 'details': row}
                        self.validation = record
                        self.validation_count += 1
                        if self.best is None or score < self.best['score']:
                            self.best = record
            return info.st_mtime
        except OSError:
            return None


def gpu_status():
    if not shutil.which('nvidia-smi'):
        return {'available': False, 'reason': '未找到 nvidia-smi'}
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
        if result.returncode:
            return {'available': False, 'reason': '当前环境无法读取 GPU 状态'}
        rows = []
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) == 5:
                rows.append({'index': row[0].strip(), 'name': row[1].strip(),
                             'used_mib': float(row[2]), 'total_mib': float(row[3]),
                             'utilization_percent': float(row[4])})
        return {'available': True, 'devices': rows}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {'available': False, 'reason': 'GPU 查询失败或超时'}


def find_run():
    candidates = []
    for run in (PROJECT / 'outputs').glob('*'):
        files = [p for p in (run / 'queue_status.json', run / 'training/metrics.jsonl',
                            run / 'cache_progress.json', run / 'features/progress.json') if p.is_file()]
        if files:
            queue, status = read_json(run / 'queue_status.json'), read_json(run / 'status.json')
            stage = queue.get('phase', queue.get('stage'))
            active = bool(stage and stage not in ('complete', 'failed', 'stopped')
                          and not status.get('complete') and not status.get('stopped'))
            candidates.append((active, max(p.stat().st_mtime for p in files), run))
    return max(candidates, key=lambda item: item[:2])[2] if candidates else None


def duration(value):
    if value is None or not math.isfinite(value):
        return '待估算'
    value = max(0, int(value))
    hours, rest = divmod(value, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours}小时{minutes:02d}分' if hours else f'{minutes}分{seconds:02d}秒'


def number(value, digits=6):
    return '—' if value is None else f'{value:.{digits}g}'


def snapshot(run, metrics, samples, query_gpu, validation=None):
    now = time.time()
    modified = metrics.update()
    queue = read_json(run / 'queue_status.json')
    status = read_json(run / 'status.json')
    if validation is not None:
        validation.update()
    config = scalar_config(run)
    train = config.get('train', {})
    budget = read_json(run / 'batch_budget.json') or read_json(run / 'batch_and_budget.json')
    stage = queue.get('phase', queue.get('stage', 'unknown'))
    if stage != 'failed':
        if status.get('complete'):
            stage = 'complete'
        elif status.get('stopped'):
            stage = 'stopped'
    step = max(metrics.last.get('step', 0), status.get('step', 0))
    total = budget.get('steps', train.get('steps', status.get('planned_steps')))
    phase_key = (stage, queue.get('updated_unix'))
    if samples and (samples[-1][2] != phase_key or step < samples[-1][1]):
        samples.clear()
    samples.append((now, step, phase_key))
    rate, rate_source, eta = None, None, None
    if stage == 'train' and total and step < total:
        if len(samples) > 1 and now - samples[0][0] >= 15:
            change = step - samples[0][1]
            if change > 0:
                rate = change / (now - samples[0][0])
                rate_source = '近期窗口（含验证、保存）'
        if rate is None:
            if metrics.elapsed and metrics.session_start_step is not None:
                rate = (step - metrics.session_start_step) / metrics.elapsed
                rate_source = '本次训练均速（含已记录的验证、保存）'
            elif budget.get('chosen', {}).get('seconds_per_step'):
                rate = 1 / budget['chosen']['seconds_per_step']
                rate_source = 'batch 实测值（不含验证、保存）'
        if rate:
            eta = max(0, total - step) / rate
    feature = read_json(run / 'cache_progress.json')
    if feature:
        feature = {'cache': '图像 / 图文', **feature,
                   'complete': read_json(run / 'features/metadata.json').get('complete', False)}
        try:
            feature['age_seconds'] = max(0, now - (run / 'cache_progress.json').stat().st_mtime)
        except OSError:
            pass
    for folder in ([] if feature else ['features', 'delta_features']):
        progress = read_json(run / folder / 'progress.json')
        meta = read_json(run / folder / 'metadata.json')
        if progress or meta:
            feature = {'cache': folder, **progress, 'complete': meta.get('complete', False)}
            feature_total = meta.get('rows') or (meta.get('shape') or [None])[0]
            if feature_total:
                feature.setdefault('total', feature_total)
                if feature['complete']:
                    feature.setdefault('done', feature_total)
            break
    diagnostics = sorted((run / 'diagnostics').glob('*.json'))
    diagnostic = read_json(diagnostics[-1]) if diagnostics else {}
    val_source = validation if validation is not None and validation.validation else metrics
    best = val_source.best
    latest_panels = sorted((run / 'generation_checks').glob('*/comparison_*.png'))
    next_validation = ((step // train['eval_every'] + 1) * train['eval_every']
                       if train.get('eval_every') and stage == 'train' else None)
    return {
        'time': dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'run': str(run),
        'stage': stage, 'stage_label': STAGES.get(stage, stage), 'error': queue.get('error'),
        'failed_stage': queue.get('stage') if stage == 'failed' else None,
        'returncode': queue.get('returncode'), 'budget_measured': bool(budget),
        'queue_pid': read_json(run / 'queue_process.json').get('pid'),
        'step': step, 'total_steps': total, 'fraction': step / total if total else None,
        'batch_size': train.get('batch_size'), 'accumulation': train.get('accumulation', 1),
        'width': config.get('model', {}).get('width'), 'depth': config.get('model', {}).get('depth'),
        'loss': metrics.last.get('loss'),
        'mean_recent_loss': sum(metrics.losses) / len(metrics.losses) if metrics.losses else None,
        'loss_window': len(metrics.losses), 'learning_rate': metrics.last.get('lr'),
        'last_validation': val_source.validation, 'best_validation': best,
        'validation_count': val_source.validation_count, 'cache': metrics.last.get('text_cache', {}),
        'next_validation_step': min(total, next_validation) if total and next_validation else next_validation,
        'smoke': read_json(run / 'smoke.json'),
        'latest_comparison': str(latest_panels[-4]) if len(latest_panels) >= 4 else (str(latest_panels[-1]) if latest_panels else None),
        'metric_age_seconds': max(0, now - modified) if modified else None,
        'updates_per_second': rate, 'rate_source': rate_source,
        'eta_seconds': eta, 'estimated_finish': dt.datetime.fromtimestamp(now + eta).astimezone().isoformat(timespec='minutes') if eta is not None else None,
        'features': feature, 'diagnostic': diagnostic,
        'malformed_metric_lines': metrics.bad_lines + (validation.bad_lines if validation else 0),
        'gpu': gpu_status() if query_gpu else {'available': False, 'reason': '已关闭 GPU 查询'},
    }


def progress_bar(done, total, width=28):
    fraction = min(1., max(0., done / total)) if total else 0.
    filled = int(width * fraction)
    return f'[{"█" * filled}{"░" * (width - filled)}] {fraction:6.1%}  {done:,} / {total:,}'


def trend(values, width=44):
    values = [v for v in values if math.isfinite(v)][-width:]
    if len(values) < 2:
        return ''
    low, high = min(values), max(values)
    return ''.join('▁▂▃▄▅▆▇█'[min(7, int((v - low) / (high - low) * 7))] if high > low else '▄'
                   for v in values)


def render(data):
    rule = '─' * 74
    lines = [rule, f'MCGen 训练监看                         {data["time"]}', rule,
             f'实验目录  {data["run"]}', f'当前阶段  {data["stage_label"]}']
    if data['error']:
        lines.append(f'错误：{data["error"]}')
    if data['failed_stage']:
        lines.append(f'失败位置  {STAGES.get(data["failed_stage"], data["failed_stage"])}  |  退出码 {data["returncode"]}')
    feature = data['features']
    if feature:
        done, total = feature.get('done', 0), feature.get('total')
        lines.append('')
        lines.append('特征缓存  ' + (progress_bar(done, total) if total else '等待进度记录'))
        seconds = feature.get('seconds', feature.get('seconds_this_run'))
        if seconds and done:
            lines.append(f'          已用 {duration(seconds)}  |  平均 {done / seconds:.2f} 条/秒')
        cache_eta = feature.get('estimated_remaining_seconds')
        if data['stage'] in ('waiting_for_first_cache_chunk', 'waiting_for_full_cache',
                             'finishing_embedding_cache', 'features') and not feature.get('complete'):
            if cache_eta is not None:
                lines.append(f'          缓存预计还需 {duration(cache_eta)}（仅当前阶段）')
            age = feature.get('age_seconds')
            if age is not None:
                lines.append(f'          最近更新 {duration(age)}前' + ('；暂未收到新进度，请检查 cache.log' if age > 180 else ''))
        elif feature.get('complete'):
            lines.append('          已完成')
    lines.append('')
    started = data['step'] > 0 or data['stage'] in ('train', 'complete', 'stopped')
    if started and data['total_steps']:
        lines.append('正式训练  ' + progress_bar(data['step'], data['total_steps']))
    elif not started:
        lines.append('正式训练  尚未开始；任务已失败，请检查对应日志' if data['stage'] == 'failed'
                     else '正式训练  尚未开始；缓存和准备阶段结束后自动进入')
    if data['width']:
        suffix = '' if data['budget_measured'] or started else '（预设，待 batch 实测）'
        lines.append(f'          {data["width"]} 宽 × {data["depth"]} 层  |  batch {data["batch_size"]} × 累积 {data["accumulation"]}{suffix}')
    if data['loss'] is not None:
        lines.append(f'当前损失  {number(data["loss"])}  |  近 {data["loss_window"]} 步均值 {number(data["mean_recent_loss"])}  |  LR {number(data["learning_rate"])}')
        if data.get('loss_trend'):
            lines.append(f'损失走势  {data["loss_trend"]}  左旧 → 右新')
        age = data['metric_age_seconds']
        lines.append('指标更新  ' + (duration(age) + '前' if age is not None else '等待首条指标'))
        if age is not None and age > 300 and data['stage'] == 'train':
            lines.append('          较久未更新，可能在验证或保存；可查看 train.log')
    if data['updates_per_second']:
        rate = data['updates_per_second']
        lines.append(f'训练速度  {1 / rate:.2f} 秒/步  |  剩余约 {duration(data["eta_seconds"])}')
        lines.append(f'预计完成  {data["estimated_finish"]}  |  {data["rate_source"]}')
    elif data['stage'] == 'train' and not data['step']:
        lines.append('训练速度  等待首批指标；可能正在加载、检查缓存或首次验证')
    if data['next_validation_step'] is not None:
        lines.append(f'下次验证  第 {data["next_validation_step"]:,} 步')
    lines.extend(['', '验证与原能力保留'])
    for field, title in [('last_validation', '最近验证'), ('best_validation', '最低验证误差')]:
        record = data[field]
        lines.append(f'{title}：第 {record["step"]:,} 步，Flow MSE {record["score"]:.6f}' if record else f'{title}：暂无记录')
    latest = data['last_validation']
    if latest:
        scores, details = latest['scores'], latest['details']
        labels = {'reference_flow_mse': '纯图像', 'joint_flow_mse': '图文联合', 'text_flow_mse': '纯文本'}
        lines.append('任务误差  ' + '  |  '.join(f'{labels.get(k, k)} {v:.6f}' for k, v in scores.items()))
        for name, key in [('原生成器权重', 'backbone_exact'), ('原文生图输出', 'text_generation_exact')]:
            if key in details:
                lines.append(f'{name}  {"[通过] 完全一致" if details[key] else "[异常] 发生变化"}')
        for mode, label in [('reference', '纯图像'), ('joint', '图文联合')]:
            high_noise = details.get(mode + '_t0.9', {})
            if high_noise:
                lines.append(f'高噪声收益  {label} vs 错误参考 {number(high_noise.get("gain_over_wrong_reference"))}（正值更好）')
    if data['smoke']:
        lines.append(f'学习检查  {"[通过]" if data["smoke"].get("passed") else "[未通过]"}')
    if data['latest_comparison']:
        lines.append(f'生成对照  {data["latest_comparison"]}')
    diag = data['diagnostic'].get('noise_level_scores', {}).get('0.9', {})
    if diag:
        lines.append(f'高噪声文本收益：vs 空文本 {number(diag.get("gain_over_null"))}，vs 打乱文本 {number(diag.get("gain_over_shuffle"))}（正值更好）')
    gpu = data['gpu']
    lines.extend(['', '设备状态（整张卡，包含其他任务）'])
    if gpu['available']:
        for row in gpu['devices']:
            lines.append(f'GPU {row["index"]}：{row["used_mib"] / 1024:.1f}/{row["total_mib"] / 1024:.1f} GiB  |  利用率 {row["utilization_percent"]:.0f}%  |  {row["name"]}')
    else:
        lines.append('GPU：' + gpu['reason'])
    if data['malformed_metric_lines']:
        lines.append(f'提示：跳过了 {data["malformed_metric_lines"]} 条格式异常的指标记录。')
    lines.extend([rule, '每次刷新重读状态；Ctrl+C 仅退出监看，训练继续。'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='实验目录；默认自动选择最近的实验')
    parser.add_argument('--interval', type=float, default=5, help='刷新间隔秒数，默认 5')
    parser.add_argument('--once', action='store_true', help='只显示一次')
    parser.add_argument('--json', action='store_true', help='输出 JSON，每次刷新一行')
    parser.add_argument('--no-gpu', action='store_true', help='不调用 nvidia-smi')
    parser.add_argument('--no-clear', action='store_true', help='保留历史输出，不清屏')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval 必须是正数')
    run = args.run.expanduser().resolve() if args.run else find_run()
    if run is None or not run.is_dir():
        parser.error('未找到实验目录；请用 --run 指定')
    metrics = Metrics(run / 'training/metrics.jsonl')
    validation = Metrics(run / 'training/validation.jsonl')
    samples = deque(maxlen=120)
    interactive = sys.stdout.isatty() and not args.no_clear and not args.json and not args.once
    if interactive:
        print('\033[?1049h\033[?25l', end='', flush=True)
    try:
        while True:
            data = snapshot(run, metrics, samples, not args.no_gpu, validation)
            data['loss_trend'] = trend(metrics.losses)
            if args.json:
                print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)
            else:
                if interactive:
                    print('\033[H', end='')
                print(render(data) + ('\033[J' if interactive else ''), flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n已退出监控，训练继续运行。', file=sys.stderr)
    except BrokenPipeError:
        pass
    finally:
        if interactive:
            print('\033[?25h\033[?1049l', end='', flush=True)


if __name__ == '__main__':
    main()
