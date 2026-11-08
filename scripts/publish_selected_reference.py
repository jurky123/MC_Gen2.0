"""Upload one verified checkpoint directory without modifying earlier releases."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from mcgen.catalog import atomic_json, file_hash


def publish(folder):
    metadata = json.loads((folder/'metadata.json').read_text())
    checkpoint = folder/metadata['filename']
    if not metadata['model_tensors_exact'] or file_hash(checkpoint) != metadata['sha256']:
        raise ValueError('Unverified or changed inference checkpoint')
    repo, prefix = metadata['hf_repo'], metadata['hf_prefix']
    api = HfApi()
    info = api.dataset_info(repo)
    files = sorted(p for p in folder.iterdir() if p.is_file())
    operations = [CommitOperationAdd(path_in_repo=prefix+'/'+p.name, path_or_fileobj=p) for p in files]
    remote_card = hf_hub_download(repo, 'README.md', repo_type='dataset', revision=info.sha)
    text = Path(remote_card).read_text()
    start, end = '<!-- selected-reference-20261108:start -->', '<!-- selected-reference-20261108:end -->'
    section = (start+'\n## Selected reference generation checkpoint\n\n'
        f'[{metadata["name"]}]({prefix}/README.md) — native16 RGBA; recorded step {metadata["step"]}. '
        +(f'Time MLP frozen for {metadata["frozen_continuation_updates"]} continuation updates. '
          if metadata['time_mlp_frozen'] else 'Five-task text/reference experiment. ')
        +'Earlier model and dataset releases are retained. Complex geometry remains experimental.\n'+end)
    card = folder.parent/'hf_dataset_card.md'
    if start in text:
        if end not in text:
            raise ValueError('Malformed existing card marker')
        body = re.sub(re.escape(start)+r'.*?'+re.escape(end), lambda _: section, text, flags=re.S)
    else:
        body = text.rstrip()+'\n\n'+section+'\n'
    card.write_text(body)
    operations.append(CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card))
    plan = {'repo': repo, 'prefix': prefix, 'parent_revision': info.sha,
        'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in files],
        'delete_operations': False}
    atomic_json(folder.parent/'upload_plan.json', plan)
    receipt = api.create_commit(repo_id=repo, repo_type='dataset', revision='main', parent_commit=info.sha,
        operations=operations, commit_message='Preserve selected MCGen reference checkpoint: '+metadata['name'])
    records = api.get_paths_info(repo, [prefix+'/'+p.name for p in files], repo_type='dataset', revision=receipt.oid)
    lookup = {r.path: r for r in records}
    verified = []
    for p in files:
        record = lookup[prefix+'/'+p.name]
        assert record.size == p.stat().st_size
        if record.lfs:
            assert record.lfs.sha256 == file_hash(p)
        else:
            b = p.read_bytes()
            assert record.blob_id == hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        verified.append(p.name)
    result = {'repo_id': repo, 'repo_type': 'dataset', 'revision': receipt.oid,
        'commit_url': receipt.commit_url, 'folder_url': f'https://huggingface.co/datasets/{repo}/tree/{receipt.oid}/{prefix}',
        'verified_files': verified, 'checkpoint_sha256': metadata['sha256'], 'metadata': metadata,
        'previous_releases_preserved': True, 'repo_visibility_unchanged': True}
    atomic_json(folder.parent/'upload_receipt.json', result)
    atomic_json(Path('docs/models')/(metadata['name']+'_hf_upload.json'), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    args = p.parse_args()
    publish(args.folder.resolve())
