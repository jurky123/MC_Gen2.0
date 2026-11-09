"""Export the evaluated token checkpoint; optionally upload as a separate experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.runtime import load_checkpoint, save_checkpoint

NAME = 'token-full-hq-step12000'
REPO = 'Risposta/MC_Gen'
PREFIX = 'checkpoints/' + NAME


def export(folder):
    run = Path('outputs/token_full_hq_20261109')
    review = Path('outputs/token_full_hq_protocol_review_20261109')
    verified = json.loads((review/'verification.json').read_text())
    if not verified['complete'] or not verified['source_checkpoints_unchanged']:
        raise ValueError('Evaluation verification is incomplete')
    source = run/'training/best_balanced.pt'
    state = load_checkpoint(source)
    if state['step'] != 12000 or state.get('diagnostic_only') or state.get('stability_halted'):
        raise ValueError('Unexpected or unhealthy checkpoint')
    cfg = Experiment.from_dict(state['config'])
    if cfg.text.backend != 'qwen3_vl_tokens':
        raise ValueError('Expected token text backend')
    model = PixelFlow(cfg.model)
    model.load_state_dict(state['model'], strict=True)
    folder.mkdir(parents=True, exist_ok=False)
    filename = 'mcgen-' + NAME + '.pt'
    payload = {k: v for k, v in state.items() if k not in ('format', 'optimizer')}
    save_checkpoint(folder/filename, payload)
    restored = load_checkpoint(folder/filename)
    exact = restored['config'] == state['config'] and all(
        torch.equal(value, restored['model'][key]) for key, value in state['model'].items())
    if not exact or 'optimizer' in restored:
        raise ValueError('Inference export differs from evaluated weights')
    metadata = {'name': NAME, 'artifact_type': 'experimental_inference_checkpoint',
        'step': state['step'], 'filename': filename, 'sha256': file_hash(folder/filename),
        'bytes': (folder/filename).stat().st_size, 'source_checkpoint_sha256': file_hash(source),
        'text_backend': cfg.text.backend, 'text_identity': state['text_identity'],
        'visual_identity': state['visual_identity'], 'native_size': cfg.model.size,
        'model_tensors_exact': exact, 'optimizer_included': False, 'encoder_included': False,
        'time_mlp_frozen': True, 'hf_repo': REPO, 'hf_prefix': PREFIX,
        'default_release_replaced': False,
        'selection': 'balanced and text best at completed12000; same tensors as final',
        'limitations': 'Paired T2I improves; broad T2I regresses; reference generalization is flat.'}
    atomic_json(folder/'metadata.json', metadata)
    (folder/'SHA256SUMS.txt').write_text(f'{metadata["sha256"]}  {filename}\n')
    shutil.copy2(review/'metrics.json', folder/'evaluation.json')
    shutil.copy2(Path('docs/TOKEN_TEXT_EVALUATION_20261109.md'), folder/'EVALUATION.md')
    for name in ('test_joint_detail_00.png', 'test_text_detail_00.png', 'broad_val_00.png'):
        shutil.copy2(review/'panels'/name, folder/name)
    (folder/'README.md').write_text('# MCGen multi-token text experiment — step 12000\n\n'
        'Native 16×16 RGBA; frozen Qwen3-VL-Embedding-8B caption hidden states and independent visual tokens. '
        'Inference weights only. Download the frozen encoder separately. Requires the matching MC_Gen2.0 token-backend code.\n\n'
        '**Experimental:** paired T2I improves, broad T2I regresses, reference generalization is flat. '
        'This upload preserves the previous selected release. See [metrics](evaluation.json) and '
        '[evaluation report](https://github.com/jurky123/MC_Gen2.0/blob/main/docs/TOKEN_TEXT_EVALUATION_20261109.md).\n\n'
        '![Unseen reference comparison](test_joint_detail_00.png)\n\n'
        '![Broad text comparison](broad_val_00.png)\n\n'
        'Set `MCGEN_EMBEDDING_MODEL` to the verified Qwen encoder directory. Use `mcgen sample --checkpoint '
        + filename + ' --prompt "blue diamond sword" --out result.png --device cuda:0`. '
        'For reference generation add `--reference reference.png --conditioning-mode visual_tokens`. '
        'The checkpoint selects the token protocol automatically; do not use pooled text caches.\n')
    return metadata


def upload(folder):
    from huggingface_hub import CommitOperationAdd, HfApi
    metadata = json.loads((folder/'metadata.json').read_text())
    if file_hash(folder/metadata['filename']) != metadata['sha256']:
        raise ValueError('Changed export')
    api = HfApi()
    parent = api.dataset_info(REPO).sha
    files = sorted(p for p in folder.iterdir() if p.is_file())
    atomic_json(folder.parent/'upload_plan.json', {'repo': REPO, 'prefix': PREFIX,
        'parent_revision': parent, 'files': [p.name for p in files],
        'default_release_replaced': False, 'delete_operations': False})
    commit = api.create_commit(repo_id=REPO, repo_type='dataset', parent_commit=parent,
        operations=[CommitOperationAdd(path_in_repo=PREFIX+'/'+p.name, path_or_fileobj=p) for p in files],
        commit_message='Archive evaluated multi-token text experiment step12000')
    records = {r.path: r for r in api.get_paths_info(REPO, [PREFIX+'/'+p.name for p in files],
        repo_type='dataset', revision=commit.oid)}
    for p in files:
        record = records[PREFIX+'/'+p.name]
        if record.size != p.stat().st_size:
            raise ValueError('Remote size mismatch: '+p.name)
        if record.lfs:
            valid = record.lfs.sha256 == file_hash(p)
        else:
            data = p.read_bytes()
            valid = record.blob_id == hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if not valid:
            raise ValueError('Remote hash mismatch: '+p.name)
    receipt = {'repo_id': REPO, 'revision': commit.oid, 'commit_url': commit.commit_url,
        'folder_url': f'https://huggingface.co/datasets/{REPO}/tree/{commit.oid}/{PREFIX}',
        'verified_files': [p.name for p in files], 'metadata': metadata,
        'previous_releases_preserved': True, 'default_release_replaced': False}
    atomic_json(folder.parent/'upload_receipt.json', receipt)
    atomic_json(Path('docs/models')/(NAME+'_hf_upload.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('outputs/token_release_20261109')/NAME)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    if not args.folder.exists():
        export(args.folder)
    if args.upload:
        upload(args.folder)
