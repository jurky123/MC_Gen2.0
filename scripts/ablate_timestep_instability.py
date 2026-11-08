"""Read-only causal swaps of the timestep MLP and AdaLN modulation weights."""
import argparse
import gc
import json
from pathlib import Path

import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.five_task_reference import FiveTaskData
from mcgen.images import decode
from mcgen.model import PixelFlow
from mcgen.runtime import load_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from PIL import Image
from prepare_reference_recovery import COMMON
from train_five_task_reference import draw_sheet, fixed_noise
from diagnose_five_task_instability import gradient_report

ROOT = Path('outputs/reference_five_task_20261108')
DEFAULT = Path('outputs/five_task_instability_20261108/causal')


def main(out, device):
    out.mkdir(parents=True, exist_ok=False)
    (out/'native').mkdir()
    data = FiveTaskData(ROOT)
    ids = json.loads((ROOT/'run.json').read_text())['generation_panel_indices'][-8:]
    paths = {'healthy': ROOT/'training/best_text.pt', 'failed': ROOT/'training/latest.pt'}
    digests = {k: file_hash(p) for k, p in paths.items()}
    states = {k: load_checkpoint(p) for k, p in paths.items()}
    cfg = Experiment.from_dict(states['healthy']['config'])
    weights = {k: state['model'] for k, state in states.items()}
    del states
    gc.collect()
    variants = [('healthy', 'healthy', None, None), ('failed', 'failed', None, None),
                ('failed_time_healthy', 'failed', 'healthy', None),
                ('failed_modulation_healthy', 'failed', None, 'healthy'),
                ('failed_both_healthy', 'failed', 'healthy', 'healthy'),
                ('healthy_time_failed', 'healthy', 'failed', None)]
    report = {'diagnostic_only': True, 'production_weights_modified': False,
              'source_files_sha256': digests, 'variants': {}}
    common_h, common_mask = data.encoder.encode(conditioned_prompts(COMMON, data.cfg), device)
    for name, base, time_source, modulation_source in variants:
        print('VARIANT', name, flush=True)
        state = dict(weights[base])
        for key in state:
            if time_source and key.startswith('time.'):
                state[key] = weights[time_source][key]
            if modulation_source and '.modulation.' in key:
                state[key] = weights[modulation_source][key]
        model = PixelFlow(cfg.model).to(device).eval()
        model.load_state_dict(state, strict=True)
        observations = {}

        def time_hook(module, args, output):
            observations['time_rms'] = output.detach().float().square().mean().sqrt().item()

        def modulation_hook(label):
            def callback(module, args, output):
                values = output.detach().float().chunk(9, -1)
                observations[label] = {'self_scale_abs_max': (1+values[1]).abs().max().item(),
                    'self_shift_abs_max': values[0].abs().max().item(),
                    'self_gate_saturated_fraction': (values[2].tanh().abs() > .99).float().mean().item()}
            return callback

        handles = [model.time.register_forward_hook(time_hook)]
        for j in (0, 4, 8):
            handles.append(model.blocks[j].modulation.register_forward_hook(modulation_hook(f'block{j}')))
        measurements = []
        for task in ('text', 'joint'):
            target, text, mask, refs, present = data.batch(ids, [task]*len(ids), [1]*len(ids), device)
            for level in (.05, .99):
                eps = torch.randn(target.shape, generator=torch.Generator(device=device).manual_seed(2026110901), device=device)
                x = (1-level)*target+level*eps
                model.zero_grad(set_to_none=True)
                prediction = model(x, torch.full((len(ids),), level, device=device),
                    model.context(text, mask, refs if task == 'joint' else None, present if task == 'joint' else None))
                loss = (prediction-(eps-target)).square().mean()
                loss.backward()
                grad = gradient_report(model)
                measurements.append({'task': task, 't': level, 'loss': loss.item(),
                    'gradient_norm': grad['total_norm'], 'activation': dict(observations)})
                print(name, task, level, 'loss', round(loss.item(), 5), 'gradient', round(grad['total_norm'], 4),
                      'time_rms', round(observations['time_rms'], 4), flush=True)
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        common = sample(model, common_h, common_mask, seeds=[2026110841]*len(COMMON), steps=32,
            text_scale=2., initial_noise=fixed_noise(len(COMMON), 2026110841, device))
        for j, value in enumerate(common):
            Image.fromarray(decode(value)).save(out/'native'/f'{name}_common_{j:02d}.png')
        target, text, mask, refs, present = data.batch(ids, ['joint']*len(ids), [1]*len(ids), device)
        joint = sample(model, text, mask, refs, present, seeds=[2026110907]*len(ids), steps=32,
            text_scale=1., ref_scale=1., initial_noise=fixed_noise(len(ids), 2026110907, device))
        for j, value in enumerate(joint):
            Image.fromarray(decode(value)).save(out/'native'/f'{name}_joint_{j:02d}.png')
        report['variants'][name] = {'base': base, 'time_source': time_source,
                                   'modulation_source': modulation_source, 'measurements': measurements}
        atomic_json(out/'metrics.json', report)
        del model, prediction, loss, common, joint
        gc.collect()
        torch.cuda.empty_cache()
    labels = ['Healthy8000', 'Failed16000', 'Failed + old time', 'Failed + old AdaLN', 'Failed + both', 'Healthy + bad time']
    for kind in ('common', 'joint'):
        entries = []
        for j in range(8):
            entries.append(([out/'native'/f'{v[0]}_{kind}_{j:02d}.png' for v in variants],
                            COMMON[j] if kind == 'common' else data.rows[ids[j]]['captions'][0],
                            'Diagnostic parameter swaps only; same inputs and noise'))
        draw_sheet(out/f'{kind}_causal.png', 'DIAGNOSTIC ONLY | timestep / AdaLN weight swaps | FP32', labels, entries)
    assert all(file_hash(paths[k]) == digest for k, digest in digests.items())
    atomic_json(out/'verification.json', {'complete': True, 'read_only': True, 'diagnostic_only': True,
        'all_source_checkpoint_hashes_unchanged': True, 'native_images': len(list((out/'native').glob('*.png')))})
    print('COMPLETE CAUSAL SWAPS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(.45, args.device)
    main(args.out, args.device)
