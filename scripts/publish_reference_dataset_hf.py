"""Publish only the verified reference-data export and a small dataset-card update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from huggingface_hub import HfApi, hf_hub_download

from mcgen.catalog import atomic_json, file_hash

REPO='Risposta/MC_Gen'
PREFIX='data/reference-20261107'
START='<!-- reference-data-20261107:start -->'
END='<!-- reference-data-20261107:end -->'


def update_card(body):
    section=f'''{START}
## Reference image pairs · 2026-11-07

[Data and labels]({PREFIX}/README.md) · [Files](https://huggingface.co/datasets/{REPO}/tree/main/{PREFIX})

1,000 user-curated high-quality HD/MC pairs, plus reviewed legacy pairs. Captions, original filenames, provenance and existing splits are preserved.
The current **lossless16** version has **1,064 training / 1,350 validation / 56 test pairs** (864 / 711 / 56 distinct targets).
The other 500 genuine32×32 targets are archived separately and excluded from16×16 training; no lossy32→16 candidates are included.
Download the high-quality-native16 and reviewed-legacy archives for the current training version. See the data card for extraction and per-source attribution.

![Reference data and actual training labels]({PREFIX}/preview.png)
{END}'''
    if START in body:
        return re.sub(re.escape(START)+r'.*?'+re.escape(END),lambda _:section,body,flags=re.S)
    if '## Contents' in body:
        return body.replace('## Contents',section+'\n\n## Contents',1)
    return body.rstrip()+'\n\n'+section+'\n'


def verify_remote(api,root,revision,files):
    records=api.get_paths_info(REPO,[p.relative_to(root/'hf').as_posix() for p in files],
                               revision=revision,repo_type='dataset')
    by_path={r.path:r for r in records}
    verified=[]
    for path in files:
        relative=path.relative_to(root/'hf').as_posix()
        remote=by_path[relative]
        if remote.size!=path.stat().st_size:
            raise ValueError('Remote file size differs: '+relative)
        digest=file_hash(path)
        lfs=getattr(remote,'lfs',None)
        if lfs:
            remote_digest=lfs.sha256
            if remote_digest!=digest:raise ValueError('Remote LFS SHA256 differs: '+relative)
            method='remote_lfs_sha256'
        else:
            payload=path.read_bytes()
            blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
            if remote.blob_id!=blob:raise ValueError('Remote Git blob differs: '+relative)
            method='remote_git_blob'
        verified.append({'file':relative,'bytes':remote.size,'sha256':digest,'verification':method})
    return verified


def publish(root):
    verification=json.loads((root/'export_verification.json').read_text())
    if not verification['passed'] or verification['summary']['lossy_native16_pairs']!=0:
        raise ValueError('Export is not verified')
    directory=root/'hf'/PREFIX
    checksum=(directory/'SHA256SUMS.txt').read_text().splitlines()
    for line in checksum:
        digest,relative=line.split('  ',1)
        if file_hash(directory/relative)!=digest:raise ValueError('Export changed: '+relative)
    api=HfApi()
    info=api.dataset_info(REPO)
    original=hf_hub_download(REPO,'README.md',repo_type='dataset',revision=info.sha,
                             cache_dir=str(root/'remote_cache'))
    body=Path(original).read_text()
    card=root/'hf/README.md'
    card.write_text(update_card(body))
    files=sorted(p for p in (root/'hf').rglob('*') if p.is_file())
    if any(p.is_symlink() for p in files):raise ValueError('No symlinks in upload staging')
    if any(p.relative_to(root/'hf').as_posix()!='README.md' and
           not p.relative_to(root/'hf').as_posix().startswith(PREFIX+'/') for p in files):
        raise ValueError('Unexpected path in upload staging')
    plan={'repo_id':REPO,'repo_type':'dataset','existing_private':info.private,'parent_revision':info.sha,
          'transport':'legacy_http_lfs' if os.environ.get('HF_HUB_DISABLE_XET')=='1' else 'xet',
          'direct_storage_route':'amazonaws.com' in os.environ.get('NO_PROXY',''),
          'files':[{'file':p.relative_to(root/'hf').as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in files],
          'total_bytes':sum(p.stat().st_size for p in files),'delete_operations':False,
          'source_data_or_checkpoint_mutations':False,'started_unix':time.time()}
    atomic_json(root/'upload_plan.json',plan)
    atomic_json(root/'upload_status.json',{'phase':'uploading','started_unix':time.time(),
                                          'files':len(files),'bytes':plan['total_bytes'],
                                          'transport':plan['transport']})
    print('UPLOADING',len(files),'files,',plan['total_bytes'],'bytes;',REPO,PREFIX,flush=True)
    receipt=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=root/'hf',
        path_in_repo='',revision='main',parent_commit=info.sha,
        allow_patterns=[p.relative_to(root/'hf').as_posix() for p in files],
        commit_message='Add curated reference data and lossless native16 training splits',
        commit_description='1000 primary HQ pairs plus reviewed legacy data. Native16 excludes every lossy conversion. Preserve captions, source attribution and splits; existing data and checkpoints are untouched.')
    revision=receipt.oid
    atomic_json(root/'upload_status.json',{'phase':'verifying','revision':revision,'files':len(files)})
    verified=verify_remote(api,root,revision,files)
    report={'repo_id':REPO,'repo_type':'dataset','revision':revision,'commit_url':receipt.commit_url,
        'folder_url':f'https://huggingface.co/datasets/{REPO}/tree/{revision}/{PREFIX}',
        'current_folder_url':f'https://huggingface.co/datasets/{REPO}/tree/main/{PREFIX}',
        'dataset_card_url':f'https://huggingface.co/datasets/{REPO}/blob/{revision}/{PREFIX}/README.md',
        'previous_revision':info.sha,'repo_visibility_unchanged':True,'uploaded_files':verified,
        'total_bytes':plan['total_bytes'],'all_remote_files_verified':True,
        'primary_hq_pairs':1000,'native16_pairs':2470,'lossy_native16_pairs':0,
        'models_or_feature_caches_uploaded':False,'completed_unix':time.time()}
    report['transport']=plan['transport']
    report['direct_storage_route']=plan['direct_storage_route']
    atomic_json(root/'upload_receipt.json',report)
    output=Path('docs/data/reference_20261107_hf_upload.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,report)
    atomic_json(root/'upload_status.json',{'phase':'complete','revision':revision,'all_remote_files_verified':True})
    print('VERIFIED',len(verified),'remote files;',report['current_folder_url'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/hf_reference_release_20261107'))
    args=parser.parse_args()
    try:publish(args.out.resolve())
    except Exception as exc:
        atomic_json(args.out/'upload_status.json',{'phase':'failed','error':str(exc)})
        raise
