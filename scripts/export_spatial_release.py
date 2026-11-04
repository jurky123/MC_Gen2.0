"""Export inference weights only; no data, evaluation archive, or network upload."""
import argparse
import json
from pathlib import Path

import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.reference_adapter import backbone_digest
from mcgen.runtime import load_checkpoint, save_checkpoint


def prepare(out, run):
    out.mkdir(parents=True, exist_ok=False)
    status = json.loads((run/'status.json').read_text())
    if not status['complete']:
        raise ValueError('Training is incomplete')
    checkpoint = run/'training/best_joint_flow.pt'
    state = load_checkpoint(checkpoint)
    cfg = Experiment.from_dict(state['config'])
    if not cfg.model.visual_adapter:
        raise ValueError('Expected a spatial reference adapter')
    model = PixelFlow(cfg.model)
    model.load_state_dict(state['model'], strict=True)
    if backbone_digest(model) != state['backbone_sha256']:
        raise ValueError('Backbone digest mismatch')
    step = state['step']
    name = f'mcgen-spatial-reference-step{step}.pt'
    payload = {k: v for k, v in state.items() if k not in ('format', 'optimizer')}
    payload['initialization'] = {**payload['initialization'], 'path': 'checkpoints/mcgen-step4128.pt'}
    save_checkpoint(out/name, payload)
    exported = load_checkpoint(out/name)
    assert 'optimizer' not in exported and exported['config'] == state['config']
    assert all(torch.equal(t, exported['model'][k]) for k, t in state['model'].items())
    metadata = {'artifact_type': 'inference_only', 'stage': 'experimental_spatial_reference', 'step': step,
        'base_step': state['initialization']['step'], 'selection': 'validation_joint_flow_minimum',
        'sha256': file_hash(out/name), 'file_size': (out/name).stat().st_size,
        'source_checkpoint_sha256': file_hash(checkpoint),
        'generator_parameters': sum(p.numel() for p in model.parameters()),
        'adapter_parameters': sum(p.numel() for p in model.visual_adapter.parameters()),
        'text_identity': state['text_identity'], 'visual_identity': state['visual_identity'],
        'backbone_sha256': state['backbone_sha256'], 'contains_optimizer': False,
        'contains_frozen_encoder': False, 'batch_size': cfg.train.batch_size,
        'task_mix': cfg.train.task_mix,
        'visual_tokens': [cfg.model.visual_grid**2, cfg.model.visual_dim],
        'injection_blocks_one_based': [i+1 for i in cfg.model.visual_layers],
        'distribution': {'model_only': True, 'original_images_uploaded': False,
                         'complete_evaluation_uploaded': False}}
    atomic_json(out/'metadata.json', metadata)
    (out/'SHA256SUMS.txt').write_text(f'{metadata["sha256"]}  {name}\n')
    (out/'README.md').write_text('# MCGen spatial reference adapter — experimental\n\n'
        'Generator and adapter inference weights only. Download Qwen3-VL-Embedding-8B separately. '
        'No optimizer, frozen encoder, original images, or full evaluation archive is included.\n\n'
        'Use `--conditioning-mode visual_tokens` with independent `--text-scale` and `--ref-scale`. '
        'Image+text improves paired-target similarity in the current version comparison; image-only is weaker '
        'than the previous adapter. Stable decoration diversity remains unproven.\n\n'
        '[Usage and evaluation summary](https://github.com/jurky123/MC_Gen2.0)\n\n'
        'Research and study only, consistent with the project license.\n')
    print('EXPORTED', metadata['file_size'], 'bytes;', metadata['sha256'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=Path('outputs/spatial_reference_20261104'))
    args = parser.parse_args()
    prepare(args.out, args.run)
