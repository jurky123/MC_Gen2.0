"""Package a verified inference checkpoint, public comparisons and small Git samples.

This prepares local files only. Upload the hf/ directory to Risposta/MC_Gen after
review; optimizer states, frozen Qwen weights and feature caches are excluded.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image,ImageDraw,ImageFont
import torch
import yaml

from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.reference_adapter import backbone_digest
from mcgen.runtime import load_checkpoint,save_checkpoint
from expand_reference_comparisons import render

RUN=Path('outputs/reference_adapter_20261104')
EVAL=Path('outputs/reference_adapter_expanded_20261104')
RELEASE='reference-step2151'
PUBLIC_FIELDS=('case_id','base_case_id','split','category','prompt','source_filename','source_mod',
               'lineage','split_group','target_hash','reference_generator','reference_sha256',
               'generator_license','prompt_seen_in_base_training','selection_seed','generation_seed')


def public_row(row,root):
    result={k:row[k] for k in PUBLIC_FIELDS if k in row}
    ident=row['case_id'];base=row.get('base_case_id',ident)
    result['reference_file']=f'inputs/{ident}_reference.png'
    result['target_file']=f'inputs/{base}_target.png'
    result['exported_reference_sha256']=file_hash(root/result['reference_file'])
    result['exported_target_sha256']=file_hash(root/result['target_file'])
    return result


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    checkpoint=RUN/'training/best_flow.pt';state=load_checkpoint(checkpoint)
    if state['step']!=2151:raise ValueError('expected validation-selected adapter step2151')
    cfg=Experiment.from_dict(state['config']);model=PixelFlow(cfg.model);model.load_state_dict(state['model'],strict=True)
    assert backbone_digest(model)==state['backbone_sha256']
    payload={k:v for k,v in state.items() if k not in ('format','optimizer')}
    weights=out/'hf/checkpoints'/RELEASE;weights.mkdir(parents=True)
    name='mcgen-reference-step2151.pt';save_checkpoint(weights/name,payload)
    exported=load_checkpoint(weights/name)
    assert 'optimizer' not in exported and exported['config']==state['config']
    assert all(torch.equal(t,exported['model'][k]) for k,t in state['model'].items())
    audit=json.load(open(RUN/'data_audit.json'));comparison=json.load(open(EVAL/'results.json'))
    assert comparison['text_generation_exact'] and comparison['backbone_exact']
    metadata={'artifact_type':'inference_only','stage':'experimental_reference_conditioning','step':2151,
              'base_step':4128,'selection':'validation_mean_flow_minimum; qualitative quality remains experimental',
              'source_checkpoint_sha256':file_hash(checkpoint),'sha256':file_hash(weights/name),
              'file_size':(weights/name).stat().st_size,'generator_parameters':sum(p.numel() for p in model.parameters()),
              'adapter_parameters':sum(p.numel() for p in model.image_adapter.parameters()),'backbone_frozen':True,
              'text_identity':state['text_identity'],'multimodal_identity':state['multimodal_identity'],
              'backbone_sha256':state['backbone_sha256'],'contains_optimizer':False,'contains_frozen_encoder':False,
              'training_pairs':audit['canonical_pairs'],'target_counts':audit['unique_targets'],
              'trained_steps':2382,'task_mix':{'image_only':.5,'image_text':.5},
              'evaluation':{'new_targets':128,'seen':64,'unseen':64,'seeds':2,'native_outputs':896,
                            'paired_reference_targets':32,'text_generation_exact':True},
              'limitations':['Color and brightness errors','Unstable complex silhouettes','Seed and reference sensitivity'],
              'hf_repo_id':'Risposta/MC_Gen','hf_repo_type':'dataset','hf_path':f'checkpoints/{RELEASE}/{name}',
              'inference_checkpoint':f'checkpoints/{name}',
              'download_url':f'https://huggingface.co/datasets/Risposta/MC_Gen/resolve/main/checkpoints/{RELEASE}/{name}'}
    atomic_json(weights/'metadata.json',metadata)
    (weights/'SHA256SUMS.txt').write_text(f'{metadata["sha256"]}  {name}\n')
    (weights/'README.md').write_text('# MCGen reference conditioning — experimental\n\n'
        'Frozen step4128 text-to-image backbone with a trained multimodal embedding adapter. '
        'Supports image-only, image+text and the original text-only path. '
        'Color and brightness errors and unstable complex shapes remain.\n\n'
        f'Generator parameters: {metadata["generator_parameters"]:,}; adapter: {metadata["adapter_parameters"]:,}. '
        'The Qwen3-VL-Embedding-8B encoder must be downloaded separately. '
        'This inference checkpoint contains no optimizer or encoder weights.\n\n'
        '[Code and usage](https://github.com/jurky123/MC_Gen2.0) · '
        '[Comparisons](../../evaluations/reference-step2151/README.md)\n\n'
        'Use `--conditioning-mode multimodal_embedding` with a reference image. '
        '`--ref-scale` guides the complete image/text embedding; text-only calls use the original path.\n\n'
        'Research and study only, consistent with the project license.\n')
    # A portable copy of all results: keep provenance, remove local mmap paths.
    public=out/'evaluation';public.mkdir()
    for folder in ('inputs','native'):shutil.copytree(EVAL/folder,public/folder)
    for name_ in ('README.md','results.json','generation_manifest.jsonl','verification.json'):
        shutil.copy2(EVAL/name_,public/name_)
    rows=json.load(open(EVAL/'samples.json'));alternates=json.load(open(EVAL/'alternate_samples.json'))
    safe_rows=[public_row(r,public) for r in rows];safe_alt=[public_row(r,public) for r in alternates]
    atomic_json(public/'samples.json',safe_rows);atomic_json(public/'alternate_samples.json',safe_alt)
    protocol=json.load(open(EVAL/'protocol.json'))
    protocol['original_samples_sha256']=protocol['samples_sha256'];protocol['samples_sha256']=file_hash(public/'samples.json')
    protocol['public_manifest']='Local mmap locations removed; original input hashes and exported PNG hashes retained.'
    atomic_json(public/'protocol.json',protocol)
    with (public/'samples.csv').open('w',newline='') as f:
        fields=['case_id','split','category','prompt','source_filename','reference_generator','reference_file','target_hash']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(safe_rows)
    render(public)
    evaluation=out/'hf/evaluations'/RELEASE;evaluation.mkdir(parents=True)
    archive=evaluation/'evaluation.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(public.rglob('*')):
            if path.is_file():z.write(path,str(path.relative_to(public)))
    for source,dest in [('train_overview_01.png','train_overview.png'),('test_overview_01.png','test_overview.png'),
                        ('test_references_01.png','reference_comparison.png')]:shutil.copy2(public/source,evaluation/dest)
    (evaluation/'README.md').write_text('# MCGen reference conditioning comparisons\n\n'
        '64 fresh seen targets and64 targets held out by image and project, two seeds, '
        'text/image/image+text conditions;32 targets also use a second reference. '
        '896 raw16×16 RGBA generations. No output-quality filtering.\n\n'
        '[Download the complete gallery and provenance](evaluation.zip). '
        'Extract it and open `index.html`; `README.md` contains all overview panels.\n\n'
        'Color/brightness errors, complex-shape failures and seed sensitivity remain. '
        'These filename-stratified examples do not estimate population accuracy.\n\n'
        '![Seen targets](train_overview.png)\n\n![Held-out targets](test_overview.png)\n\n'
        '![Two reference versions](reference_comparison.png)\n')
    (evaluation/'SHA256SUMS.txt').write_text(f'{file_hash(archive)}  evaluation.zip\n')
    # Keep just twelve fixed examples in Git, including the failed examples.
    small=Path('examples/reference_generation');small.mkdir(parents=True,exist_ok=False)
    (small/'inputs').mkdir();(small/'generated').mkdir()
    selected=[]
    for split in ('train','test'):selected.extend([r for r in safe_rows if r['split']==split][:6])
    for r in selected:
        ident=r['case_id'];files={}
        for suffix in ('reference','target'):
            path=f'inputs/{ident}_{suffix}.png';shutil.copy2(public/path,small/path);files[suffix]={'file':path,'sha256':file_hash(small/path)}
        for mode,seed in [('text',0),('image',0),('joint',0),('image',1),('joint',1)]:
            filename=f'{ident}_{mode}_s{seed}.png';path=f'generated/{filename}';shutil.copy2(public/'native'/filename,small/path)
            files[f'{mode}_seed{seed+1}']={'file':path,'sha256':file_hash(small/path)}
        r['files']=files
    figure=Path('docs/images/reference_generation');figure.mkdir(parents=True,exist_ok=False)
    preview=Image.new('RGB',(1080,970),'white');draw=ImageDraw.Draw(preview)
    normal=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',17)
    draw.text((20,14),'MCGen reference conditioning — experimental',font=bold,fill='#172033')
    for block,title in enumerate(['(a) Seen during training','(b) Held out by project']):
        draw.text((20+block*536,45),title,font=bold,fill='#172033')
        for col,label in enumerate(['HD reference','MC target','Text only','Image only','Image + text']):
            draw.text((20+block*536+col*104,76),label,font=normal,fill='#475569')
    for i,r in enumerate(selected):
        block=i//6;line=i%6;y=102+line*137
        for col,key in enumerate(['reference','target','text_seed1','image_seed1','joint_seed1']):
            with Image.open(small/r['files'][key]['file']) as im:
                im=im.convert('RGBA');bg=Image.new('RGBA',im.size,(239,241,245,255));bg.alpha_composite(im)
                pic=bg.convert('RGB').resize((96,96),Image.Resampling.LANCZOS if col==0 else Image.Resampling.NEAREST)
            preview.paste(pic,(20+block*536+col*104,y))
        draw.text((20+block*536,y+101),r['prompt'],font=normal,fill='#334155')
    draw.text((20,935),'Fixed first six samples per split; adapter step2151; native16x16 RGBA; same seed; no color or silhouette correction.',font=normal,fill='#475569')
    preview.save(figure/'preview.png')
    atomic_json(small/'manifest.json',{'checkpoint_step':2151,'base_step':4128,'checkpoint_sha256':metadata['sha256'],
               'selection':'First six samples per split from the pre-generation fixed expanded sample; failures retained.',
               'preview':{'file':'../../docs/images/reference_generation/preview.png','sha256':file_hash(figure/'preview.png')},'items':selected})
    (small/'README.md').write_text('# 参考图生成抽样\n\n训练内、训练外各六个固定样本，保留失败结果。'
        '列顺序：参考图、MC 原图、纯文本、纯参考图、图文联合。\n\n'
        '![真实对照](../../docs/images/reference_generation/preview.png)\n\n'
        '`manifest.json` 保留文件名标签、项目、参考图来源和文件哈希；`generated/` 是原始16×16 RGBA输出。'
        '两种种子的参考图及图文结果一并保留。完整128个新物品的896张对照通过模型仓库分发。\n\n'
        '[模型与完整对照](https://huggingface.co/datasets/Risposta/MC_Gen/tree/main/checkpoints/reference-step2151)\n')
    atomic_json(Path('docs/models/reference-step2151.json'),metadata)
    Path('docs/models/reference_training_config.yaml').write_text('# Archived reference adapter training configuration.\n'+yaml.safe_dump(state['config'],sort_keys=False))
    atomic_json(out/'packaging.json',{'model':metadata,'small_git_examples':len(selected),
                'evaluation_zip_sha256':file_hash(archive),'evaluation_zip_bytes':archive.stat().st_size,
                'inference_export_model_tensors_exact':True,'optimizer_excluded':True})
    print('PREPARED',metadata['file_size'],'model bytes;',archive.stat().st_size,'evaluation zip bytes;',len(selected),'Git samples',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();prepare(args.out)
