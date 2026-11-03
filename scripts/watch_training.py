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
}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def scalar_config(run):
    """Prefer the exact training JSON; otherwise read the flat YAML train/model scalars."""
    config = read_json(run / 'training/run.json').get('config')
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
                    if isinstance(row.get('loss'), (int, float)):
                        self.losses.append(row['loss'])
                    scores = {k: v for k, v in row.items() if k.endswith('_flow_mse') and isinstance(v, (int, float))}
                    if scores:
                        score = sum(scores.values()) / len(scores)
                        record = {'step': step, 'score': score, 'scores': scores}
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
        files = [p for p in (run / 'queue_status.json', run / 'training/metrics.jsonl') if p.is_file()]
        if files:
            candidates.append((max(p.stat().st_mtime for p in files), run))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def duration(value):
    if value is None or not math.isfinite(value):
        return '待估算'
    value = max(0, int(value))
    hours, rest = divmod(value, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours}小时{minutes:02d}分' if hours else f'{minutes}分{seconds:02d}秒'


def number(value, digits=6):
    return '—' if value is None else f'{value:.{digits}g}'


def snapshot(run, metrics, samples, query_gpu):
    now = time.time()
    modified = metrics.update()
    queue = read_json(run / 'queue_status.json')
    config = scalar_config(run)
    train = config.get('train', {})
    budget = read_json(run / 'batch_and_budget.json')
    stage = queue.get('stage', 'unknown')
    step = metrics.last.get('step', 0)
    total = budget.get('steps', train.get('steps'))
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
            start = queue.get('updated_unix', now)
            baseline = read_json(run / 'smoke.json').get('step', 0)
            if now > start and step > baseline:
                rate = (step - baseline) / (now - start)
                rate_source = '阶段均速（含启动、验证、保存）'
        if rate:
            eta = max(0, total - step) / rate
    feature = {}
    for folder in ['features', 'delta_features']:
        progress = read_json(run / folder / 'progress.json')
        meta = read_json(run / folder / 'metadata.json')
        if progress or meta:
            feature = {'cache': folder, **progress, 'complete': meta.get('complete', False)}
            break
    diagnostics = sorted((run / 'diagnostics').glob('*.json'))
    diagnostic = read_json(diagnostics[-1]) if diagnostics else {}
    return {
        'time': dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'run': str(run),
        'stage': stage, 'stage_label': STAGES.get(stage, stage), 'error': queue.get('error'),
        'queue_pid': read_json(run / 'queue_process.json').get('pid'),
        'step': step, 'total_steps': total, 'fraction': step / total if total else None,
        'batch_size': train.get('batch_size'), 'accumulation': train.get('accumulation', 1),
        'width': config.get('model', {}).get('width'), 'depth': config.get('model', {}).get('depth'),
        'loss': metrics.last.get('loss'),
        'mean_recent_loss': sum(metrics.losses) / len(metrics.losses) if metrics.losses else None,
        'loss_window': len(metrics.losses), 'learning_rate': metrics.last.get('lr'),
        'last_validation': metrics.validation, 'best_validation': metrics.best,
        'validation_count': metrics.validation_count, 'cache': metrics.last.get('text_cache', {}),
        'metric_age_seconds': max(0, now - modified) if modified else None,
        'updates_per_second': rate, 'rate_source': rate_source,
        'eta_seconds': eta, 'estimated_finish': dt.datetime.fromtimestamp(now + eta).astimezone().isoformat(timespec='minutes') if eta is not None else None,
        'features': feature, 'diagnostic': diagnostic, 'malformed_metric_lines': metrics.bad_lines,
        'gpu': gpu_status() if query_gpu else {'available': False, 'reason': '已关闭 GPU 查询'},
    }


def render(data):
    lines = [f'MCGen 训练监控  {data["time"]}', f'目录：{data["run"]}',
             f'阶段：{data["stage_label"]}  |  队列 PID：{data["queue_pid"] or "—"}']
    if data['error']:
        lines.append(f'错误：{data["error"]}')
    if data['total_steps']:
        fraction = min(1, max(0, data['fraction']))
        bar = '#' * int(28 * fraction) + '-' * (28 - int(28 * fraction))
        lines.append(f'进度：[{bar}] {data["step"]:,} / {data["total_steps"]:,}  ({fraction:.1%})')
    if data['width']:
        lines.append(f'模型：{data["width"]} 宽 × {data["depth"]} 层  |  batch：{data["batch_size"]} × 累积 {data["accumulation"]}')
    lines.append(f'损失：{number(data["loss"])}  |  最近 {data["loss_window"]} 步均值：{number(data["mean_recent_loss"])}  |  LR：{number(data["learning_rate"])}')
    for field, title in [('last_validation', '最近验证'), ('best_validation', '最低验证误差')]:
        record = data[field]
        lines.append(f'{title}：第 {record["step"]:,} 步，Flow MSE {record["score"]:.6f}' if record else f'{title}：暂无记录')
    if data['cache']:
        hits = data["cache"].get("hits")
        hits = f"{hits:,}" if isinstance(hits, (int, float)) else "—"
        lines.append(f'文本缓存：命中 {hits}  |  缺失 {data["cache"].get("misses", "—")}')
    age = data['metric_age_seconds']
    lines.append('最近指标更新：' + (duration(age) + '前' if age is not None else '等待首条训练指标'))
    if age is not None and age > 300 and data['stage'] == 'train':
        lines.append('提示：较久没有新指标，可能在验证/保存；也请检查 train.log 和 queue.log。')
    if data['updates_per_second']:
        rate = data['updates_per_second']
        lines.append(f'速度：{1 / rate:.2f} 秒/步  |  剩余约 {duration(data["eta_seconds"])}  |  预计完成 {data["estimated_finish"]}')
        lines.append(f'估算依据：{data["rate_source"]}；评估及速度波动可能使时间延后。')
    feature = data['features']
    if feature:
        lines.append(f'Embedding 缓存：{"已完成" if feature.get("complete") else "计算中"}  {feature.get("done", "—")} / {feature.get("total", "—")}')
    diag = data['diagnostic'].get('noise_level_scores', {}).get('0.9', {})
    if diag:
        lines.append(f'高噪声文本收益：vs 空文本 {number(diag.get("gain_over_null"))}，vs 打乱文本 {number(diag.get("gain_over_shuffle"))}（正值更好）')
    gpu = data['gpu']
    if gpu['available']:
        for row in gpu['devices']:
            lines.append(f'GPU {row["index"]}：{row["used_mib"] / 1024:.1f}/{row["total_mib"] / 1024:.1f} GiB  |  利用率 {row["utilization_percent"]:.0f}%  |  {row["name"]}')
    else:
        lines.append('GPU：' + gpu['reason'])
    if data['malformed_metric_lines']:
        lines.append(f'提示：跳过了 {data["malformed_metric_lines"]} 条格式异常的指标记录。')
    lines.extend(['', '只读监控；Ctrl+C 仅退出本脚本，不会停止训练。'])
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
    samples = deque(maxlen=120)
    try:
        while True:
            data = snapshot(run, metrics, samples, not args.no_gpu)
            if args.json:
                print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)
            else:
                if sys.stdout.isatty() and not args.no_clear:
                    print('\033[2J\033[H', end='')
                print(render(data), flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n已退出监控，训练继续运行。', file=sys.stderr)
    except BrokenPipeError:
        pass


if __name__ == '__main__':
    main()
