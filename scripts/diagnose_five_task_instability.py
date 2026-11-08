"""Read-only weight, attention and gradient comparison across saved checkpoints."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.five_task_reference import FiveTaskData
from mcgen.model import PixelFlow
from mcgen.runtime import load_checkpoint

RUN = Path('outputs/reference_five_task_20261108')
DEFAULT = Path('outputs/five_task_instability_20261108')


def weight_report(weights, baseline):
    rows = []
    groups = defaultdict(lambda: [0., 0., 0.])
    for name, value in weights.items():
        old = baseline[name]
        current = value.float()
        norm = current.square().sum().item()
        delta = (current-old.float()).square().sum().item()
        previous = old.float().square().sum().item()
        prefix = '.'.join(name.split('.')[:2]) if name.startswith('blocks.') else name.split('.')[0]
        for j, x in enumerate((norm, delta, previous)):
            groups[prefix][j] += x
        rows.append({'name': name, 'norm': norm**.5, 'relative_drift': (delta/max(previous, 1e-20))**.5,
                     'max_abs': current.abs().max().item(), 'finite': bool(torch.isfinite(current).all())})
    return {'all_finite': all(x['finite'] for x in rows),
        'groups': {k: {'norm': x[0]**.5, 'drift_norm': x[1]**.5, 'relative_drift': (x[1]/max(x[2], 1e-20))**.5} for k, x in groups.items()},
        'largest_relative_drifts': sorted(rows, key=lambda x: x['relative_drift'], reverse=True)[:30]}


def hooks(model, record):
    handles = []

    def norm_hook(name):
        def pre(module, args):
            with torch.no_grad():
                x = args[0].detach().float()
                variance = x.var(-1, unbiased=False)
                record[name] = {'input_rms': x.square().mean().sqrt().item(),
                    'variance_min': variance.min().item(), 'variance_median': variance.median().item(),
                    'max_inverse_std': (variance+module.eps).rsqrt().max().item()}
        return pre

    def attention_hook(name):
        def pre(module, args):
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                w = module.embed_dim
                query, key = args[0][:2].detach().float(), args[1][:2].detach().float()
                bias = module.in_proj_bias
                q = F.linear(query, module.in_proj_weight[:w].float(), None if bias is None else bias[:w].float())
                k = F.linear(key, module.in_proj_weight[w:2*w].float(), None if bias is None else bias[w:2*w].float())
                q = q.reshape(len(q), -1, module.num_heads, w//module.num_heads).transpose(1, 2)
                k = k.reshape(len(k), -1, module.num_heads, w//module.num_heads).transpose(1, 2)
                logits = (q @ k.transpose(-1, -2))/(w//module.num_heads)**.5
                probabilities = logits.softmax(-1)
                record[name] = {'q_rms': q.square().mean().sqrt().item(), 'k_rms': k.square().mean().sqrt().item(),
                    'logit_abs_max': logits.abs().max().item(), 'logit_std': logits.std().item(),
                    'attention_entropy': -(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1).mean().item()}
        return pre

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            handles.append(module.register_forward_pre_hook(norm_hook(name)))
        elif isinstance(module, torch.nn.MultiheadAttention):
            handles.append(module.register_forward_pre_hook(attention_hook(name)))
    return handles


def gradient_report(model):
    groups = defaultdict(float)
    individual = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        squared = parameter.grad.detach().float().square().sum().item()
        prefix = '.'.join(name.split('.')[:3]) if name.startswith('blocks.') else name.split('.')[0]
        groups[prefix] += squared
        individual.append({'name': name, 'norm': squared**.5})
    return {'total_norm': sum(groups.values())**.5,
        'groups': dict(sorted(((k, v**.5) for k, v in groups.items()), key=lambda x: x[1], reverse=True)),
        'top_parameters': sorted(individual, key=lambda x: x['norm'], reverse=True)[:20]}


def main(out, device):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.45, device)
    print('LOAD_DATA', flush=True)
    data = FiveTaskData(RUN)
    cfg = data.cfg
    run = json.loads((RUN/'run.json').read_text())
    ids = run['generation_panel_indices'][-8:]
    assert all(data.rows[i]['split'] == 'train' for i in ids)
    source = Path(json.loads((RUN/'initialization.json').read_text())['checkpoint'])
    baseline = load_checkpoint(source)['model']
    paths = {'source': source, 'balanced4000': RUN/'training/best_balanced.pt',
             'text8000': RUN/'training/best_text.pt', 'final16000': RUN/'training/latest.pt'}
    result = {'checkpoints': {}, 'training_ids': ids, 'readonly': True, 'weights_updated': False}
    for label, path in paths.items():
        print('CHECKPOINT', label, flush=True)
        digest = file_hash(path)
        state = load_checkpoint(path)
        report = {'path': str(path.resolve()), 'step': state['step'], 'file_sha256': digest,
                  'weights': weight_report(state['model'], baseline), 'measurements': []}
        model = PixelFlow(Experiment.from_dict(state['config']).model).to(device).eval()
        model.load_state_dict(state['model'], strict=True)
        del state
        gc.collect()
        record = {}
        handles = hooks(model, record)
        for task in ('text', 'joint', 'reference'):
            target, text, mask, refs, present = data.batch(ids, [task]*len(ids), [1]*len(ids), device)
            if task == 'text':
                refs, present = None, None
            for level in (.001, .05, .5, .99):
                noise = torch.randn(target.shape, device=device, generator=torch.Generator(device=device).manual_seed(2026110901))
                x = target*(1-level)+noise*level
                predictions = {}
                for precision in ('fp32', 'bf16'):
                    model.zero_grad(set_to_none=True)
                    record.clear()
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                        context = model.context(text, mask, refs, present)
                        predicted = model(x, torch.full((len(ids),), level, device=device), context)
                        loss = (predicted.float()-(noise-target)).square().mean()
                    loss.backward()
                    gradients = gradient_report(model)
                    predictions[precision] = predicted.detach().float()
                    report['measurements'].append({'task': task, 't': level, 'precision': precision,
                        'loss': loss.item(), 'prediction_rms': predicted.detach().float().square().mean().sqrt().item(),
                        'gradients': gradients, 'activations': dict(record)})
                    print(label, task, level, precision, 'loss', round(loss.item(), 6),
                          'grad', round(gradients['total_norm'], 4),
                          'top', gradients['top_parameters'][0]['name'], flush=True)
                report['measurements'][-1]['fp32_bf16_prediction_mae'] = (predictions['fp32']-predictions['bf16']).abs().mean().item()
        for handle in handles:
            handle.remove()
        assert file_hash(path) == digest
        atomic_json(out/(label+'.json'), report)
        result['checkpoints'][label] = {'step': report['step'], 'path': report['path'], 'file_sha256': digest}
        del model, report
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(out/'verification.json', result)
    print('COMPLETE READONLY DIAGNOSIS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    main(args.out, args.device)
