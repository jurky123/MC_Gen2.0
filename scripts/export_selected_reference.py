"""Export a selected reference checkpoint with exact weights and honest provenance."""
import argparse
import gc
import json
from pathlib import Path

import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.runtime import load_checkpoint, save_checkpoint


def export(source, out, name, frozen_time=False):
    source = source.resolve()
    if out.exists():
        raise ValueError('Use a fresh staging directory')
    state = load_checkpoint(source)
    cfg = Experiment.from_dict(state['config'])
    if not cfg.model.visual_adapter or cfg.model.size != 16 or state.get('target_leakage'):
        raise ValueError('Expected a native16 spatial-reference model without target leakage')
    if state.get('arm') == 'control':
        raise ValueError('Diverging control is not a release candidate')
    health_path = source.parent.parent/'training_health.json'
    if health_path.exists():
        health = json.loads(health_path.read_text())
        if any(x.get('unsafe') and Path(x['path']).resolve() == source for x in health['models']):
            raise ValueError('This checkpoint is explicitly marked unsafe')
    parent_step = None
    frozen_updates = 0
    if frozen_time:
        protocol = json.loads((source.parent/'protocol.json').read_text())
        status = json.loads((source.parent/'status.json').read_text())
        if (state.get('arm') != 'freeze_time' or not status['complete']
                or not status['time_weights_bitwise_unchanged']):
            raise ValueError('Frozen-time provenance is not verified')
        parent = load_checkpoint(protocol['source'])
        if file_hash(protocol['source']) != protocol['source_sha256']:
            raise ValueError('Parent checkpoint changed')
        assert all(torch.equal(v, parent['model'][k]) for k, v in state['model'].items() if k.startswith('time.'))
        parent_step = protocol['source_step']
        frozen_updates = state['step']-parent_step
        del parent
        gc.collect()
    elif state.get('arm') == 'freeze_time':
        raise ValueError('A frozen-time checkpoint must be labelled with --frozen-time')
    model = PixelFlow(cfg.model)
    model.load_state_dict(state['model'], strict=True)
    assert all(bool(torch.isfinite(x).all()) for x in state['model'].values())
    out.mkdir(parents=True)
    filename = f'mcgen-{name}.pt'
    fields = ('config', 'model', 'step', 'text_identity', 'visual_identity', 'training_protocol', 'diagnostic_only')
    payload = {k: state[k] for k in fields if k in state}
    payload['release'] = {'name': name, 'experimental': True, 'time_mlp_frozen': frozen_time,
        'parent_step': parent_step, 'frozen_continuation_updates': frozen_updates,
        'source_checkpoint_sha256': file_hash(source)}
    save_checkpoint(out/filename, payload)
    exported = load_checkpoint(out/filename)
    assert exported['config'] == state['config'] and 'optimizer' not in exported
    assert all(torch.equal(value, exported['model'][key]) for key, value in state['model'].items())
    metadata = {'name': name, 'artifact_type': 'experimental_inference_checkpoint', 'step': state['step'],
        'time_mlp_frozen': frozen_time, 'parent_step': parent_step, 'frozen_continuation_updates': frozen_updates,
        'source_checkpoint_sha256': payload['release']['source_checkpoint_sha256'],
        'sha256': file_hash(out/filename), 'bytes': (out/filename).stat().st_size,
        'filename': filename, 'model_parameters': sum(x.numel() for x in model.parameters()),
        'visual_adapter_parameters': sum(x.numel() for x in model.visual_adapter.parameters()),
        'text_identity': state['text_identity'], 'visual_identity': state['visual_identity'],
        'native_size': 16, 'source_diagnostic_only': bool(state.get('diagnostic_only')),
        'optimizer_included': False, 'encoder_included': False, 'model_tensors_exact': True,
        'hf_repo': 'Risposta/MC_Gen', 'hf_repo_type': 'dataset', 'hf_prefix': 'checkpoints/'+name}
    atomic_json(out/'metadata.json', metadata)
    (out/'SHA256SUMS.txt').write_text(f'{metadata["sha256"]}  {filename}\n')
    (out/'README.md').write_text('# MCGen reference generation — experimental\n\n'
        f'Selected snapshot: **{name}**, recorded step **{state["step"]}**. '
        +(f'Time MLP frozen for **{frozen_updates}** continuation updates from step **{parent_step}**. '
          if frozen_time else 'Selected from the five-task text/reference training run. ')
        +'The original checkpoint and earlier releases are retained.\n\n'
        '16×16 RGBA; text-only, reference-only and text+reference inference. '
        'Download the frozen Qwen3-VL-Embedding-8B encoder separately and use `MCGEN_EMBEDDING_MODEL` '
        'to locate it. No optimizer or encoder weights are included.\n\n'
        'Complex silhouettes and decoration transfer remain experimental. '
        'See metadata for actual step counts and diagnostic provenance.\n\n'
        '[Code, examples and usage](https://github.com/jurky123/MC_Gen2.0)\n\n'
        'Research and study only, consistent with the project license.\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--frozen-time', action='store_true')
    args = p.parse_args()
    if not args.name or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for x in args.name):
        p.error('--name must contain lowercase ASCII letters, digits, - or _')
    torch.set_num_threads(4)
    export(args.checkpoint, args.out, args.name, args.frozen_time)
