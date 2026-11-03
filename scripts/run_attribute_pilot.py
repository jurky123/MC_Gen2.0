"""Two-stage development experiment; never reuses the previous locked test.

Run with PYTHONPATH=src, the mcgen environment and offline HF flags.
Stages are explicit so failed runs do not silently overwrite earlier artifacts.
"""
import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageDraw
import torch

from mcgen.catalog import read_catalog, write_catalog, atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.dataset import CatalogDataset
from mcgen.evaluation import metrics
from mcgen.images import native_grid, composite
from mcgen.model import PixelFlow
from mcgen.runtime import Runtime, load_checkpoint
from mcgen.training import Trainer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/attribute_prior_20261102"
PREVIOUS = ROOT / "outputs/generalization_v3_20261101"
CATALOG = ROOT / "data/catalogs/native16_v3/catalog"
BASE = PREVIOUS / "train/best_flow.pt"
CONFIGS = ROOT / "configs/v3/experiments"
FAMILIES = ("pickaxe", "sword", "axe", "shovel", "helmet", "shield", "key",
            "potion", "apple", "flower", "gem", "book", "door", "ingot")


def family(prompt):
    for f in FAMILIES:
        pattern = r"\b(?:gem|gemstone|diamond|emerald|ruby|sapphire)\b" if f == "gem" else rf"\b{f}\b"
        if re.search(pattern, prompt.lower()):
            return f
    return "other"


def prepare():
    records, meta = read_catalog(CATALOG)
    items = [r for r in records if r["asset_type"] == "item" and r["prompt"].strip()]
    paired_hashes = {r["target_hash"] for r in items if r.get("reference")}
    rng = random.Random(20261102)
    unique = {}
    for r in items:
        if r["split"] == "train" and r["target_hash"] not in paired_hashes:
            unique.setdefault(r["target_hash"], r)
    groups = defaultdict(list)
    for r in sorted(unique.values(), key=lambda x: x["id"]):
        groups[family(r["prompt"])].append(r)
    extra = []
    for f in FAMILIES:
        rng.shuffle(groups[f]); extra.extend(groups[f][:500])
    chosen = {r["target_hash"] for r in extra}
    remaining = sorted((r for h, r in unique.items() if h not in chosen), key=lambda x: x["id"])
    rng.shuffle(remaining); extra += remaining[:8000-len(extra)]
    train = [r for r in items if r["split"] == "train" and r["target_hash"] in paired_hashes] + extra
    unique_val = {}
    for r in items:
        if r["split"] == "val":
            h = r["target_hash"]
            if h not in unique_val or r.get("reference"):
                unique_val[h] = r
    val_paired = sorted((r for r in unique_val.values() if r.get("reference")), key=lambda x:x["id"])
    val_broad = sorted((r for h, r in unique_val.items() if h not in paired_hashes), key=lambda x:x["id"])
    rng.shuffle(val_paired); rng.shuffle(val_broad)
    val = val_paired[:128] + val_broad[:256]
    dev = {"paired": val_paired[128:224], "broad": val_broad[256:352]}
    assert all(len(rows) == 96 for rows in dev.values())
    assert not {r['target_hash'] for r in val} & {r['target_hash'] for rows in dev.values() for r in rows}
    out_meta = write_catalog(OUT / "catalog", train + val, {
        "native_size": 16, "parent_fingerprint": meta["fingerprint"],
        "policy": "Inherited train/val splits; all paired training targets plus 8000 nonpaired targets. Up to 500 per caption-derived family, remainder random. No test records.",
        "family_labels": "weak caption heuristics for selection only, not ground truth shape labels"})
    for name, rows in dev.items():
        (OUT / f"dev_{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    atomic_json(OUT / "protocol.json", {
        "purpose": "development pilot, not new final holdout or full reconstruction completion",
        "initial_checkpoint": str(BASE), "initial_sha256": file_hash(BASE),
        "catalog_fingerprint": out_meta["fingerprint"], "catalog_audit": out_meta["audit"],
        "extra_nonpaired_targets": len(extra), "extra_family_counts": dict(Counter(family(r['prompt']) for r in extra)),
        "selection": "minimum fixed validation mean flow MSE per stage, before development generation",
        "stage1": "1600 text-only transfer steps from previous paired checkpoint, batch64, lr5e-5",
        "stage2": "1000 attribute-adapter-only steps, batch64, frozen backbone, lr3e-4",
        "generation": {"seed": 20261102, "steps": 40, "solver": "heun", "text_scale": 2., "ref_scale": 2.},
        "dev_hashes": {f"dev_{name}.jsonl": file_hash(OUT / f"dev_{name}.jsonl") for name in dev},
        "config_hashes": {n: file_hash(CONFIGS/n) for n in ("attribute_prior_text.yaml", "attribute_prior_adapter.yaml")},
        "limits": ["One training seed; modest pilot exposure", "Existing captions are noisy; no manually verified vanilla anchors", "Development val projects, some previously inspected; no untouched test claim", "Orderless color statistics do not encode spatial material detail", "Image-only defaults to inherited shape path; no category classifier yet"]})
    print(json.dumps(out_meta["audit"]), flush=True)


def train(stage, device):
    config = Experiment.load(CONFIGS / f"attribute_prior_{stage}.yaml")
    initialize = BASE if stage == "text" else OUT / "text/best_flow.pt"
    trainer = Trainer(config, OUT / "catalog", OUT / stage, device, device, initialize=initialize)
    trainer.run()
    if stage == "adapter":
        before = load_checkpoint(initialize)["model"]
        after = load_checkpoint(OUT / "adapter/best_flow.pt")["model"]
        assert all(torch.equal(v, after[k]) for k, v in before.items()), "frozen backbone changed"
        atomic_json(OUT / "backbone_verification.json", {"all_inherited_tensors_bitwise_identical": True,
                    "tensors": len(before), "parent_sha256": file_hash(initialize)})


def checker(a):
    y, x = np.indices(a.shape[:2]); bg = np.where(((x//2+y//2)%2)[...,None], 190, 225)
    alpha = a[...,3:4].astype(np.float32)/255
    return Image.fromarray(np.uint8(np.clip(a[...,:3]*alpha+bg*(1-alpha),0,255))).resize((96,96),Image.Resampling.NEAREST)


def evaluate(device):
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name,h in protocol["dev_hashes"].items():
        assert file_hash(OUT / name) == h
    rows = {s:[json.loads(x) for x in (OUT/f"dev_{s}.jsonl").read_text().splitlines()] for s in ("paired","broad")}
    datasets = {s:CatalogDataset(r,16,128,"val",asset_type="item") for s,r in rows.items()}
    # Keep file order; CatalogDataset only supplies source access here.
    def raw(r, key):
        return datasets['paired'].array(r[key])
    out = OUT / "evaluation"; out.mkdir(exist_ok=False); (out/"images").mkdir()
    runtime = Runtime(BASE, device, device)
    checkpoints = {"base":BASE,"prior":OUT/"text/best_flow.pt","adapter":OUT/"adapter/best_flow.pt"}
    results=[]; images={}; selection={}
    semantic=[json.loads(x) for x in (PREVIOUS/'semantic_cases.jsonl').read_text().splitlines()]
    for name, checkpoint in checkpoints.items():
        state=load_checkpoint(checkpoint);cfg=Experiment.from_dict(state['config'])
        runtime.config=cfg;runtime.model=PixelFlow(cfg.model).to(device)
        runtime.model.load_state_dict(state['model']);runtime.model.eval()
        assert runtime.encoder.identity==state['text_identity']
        selection[name]={"path":str(checkpoint),"sha256":file_hash(checkpoint),"step":state['step']}
        for split, cases in rows.items():
            modes=['text']
            if split=='paired' and name!='prior':
                modes += ['joint','rotated','shuffled','shape','shape_rotated','reference'] if name=='adapter' else ['joint','rotated','reference']
            for mode in modes:
                for start in range(0,len(cases),32):
                    chunk=cases[start:start+32];refs=None
                    if mode!='text':
                        refs=[raw(cases[(start+i+1)%len(cases)] if mode=='shuffled' else r,'reference') for i,r in enumerate(chunk)]
                        if mode in ('rotated','shape_rotated'):refs=[np.rot90(r).copy() for r in refs]
                    prompts=['' if mode=='reference' else r['prompt'] for r in chunk]
                    options={}
                    if name=='adapter' and mode!='text':options['reference_mode']='shape' if mode in ('shape','shape_rotated','reference') else 'attributes'
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        generated=runtime.generate(prompts,refs,seeds=[20261102+start+i for i in range(len(chunk))],steps=40,
                            text_scale=0. if mode=='reference' else 2.,ref_scale=0. if mode=='text' else 2.,solver='heun',**options)
                    for i,(r,a) in enumerate(zip(chunk,generated)):
                        index=start+i;key=f'{name}_{split}_{mode}_{index:03d}';path=f'images/{key}.png'
                        Image.fromarray(a).save(out/path);images[name,split,mode,index]=a
                        results.append({'model':name,'split':split,'mode':mode,'id':r['id'],'lineage':r['lineage'],
                                        'image':path,'metrics':metrics(a,native_grid(raw(r,'target'),16))})
                print('EVAL',name,split,mode,flush=True)
        # Previously authored prompts are development diagnostics, not unseen tests.
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs=runtime.generate([r['prompt'] for r in semantic],seeds=[20261102]*len(semantic),steps=40,
                                     text_scale=2.,ref_scale=0.,solver='heun')
        sheet=Image.new('RGB',(4*104,12*124+24),'white');draw=ImageDraw.Draw(sheet)
        for j,color in enumerate(('red','blue','green','yellow')):draw.text((j*104,2),color,fill='black')
        for i,(r,a) in enumerate(zip(semantic,outputs)):
            Image.fromarray(a).save(out/'images'/f'{name}_semantic_{r["id"]}.png')
            if i%4==0:draw.text((0,24+(i//4)*124),r['id'].rsplit('_',1)[0],fill='black')
            sheet.paste(checker(a),((i%4)*104,44+(i//4)*124))
        sheet.save(out/f'{name}_semantic.png')
    summary={}
    for name in checkpoints:
        for split in rows:
            for mode in sorted({r['mode'] for r in results if r['model']==name and r['split']==split}):
                chosen=[r for r in results if r['model']==name and r['split']==split and r['mode']==mode]
                summary[f'{name}/{split}/{mode}']={}
                for metric in ('composite_mae','alpha_iou','palette_mae'):
                    projects=defaultdict(list)
                    for r in chosen:
                        if r['metrics'][metric] is not None:projects[r['lineage']].append(r['metrics'][metric])
                    summary[f'{name}/{split}/{mode}'][metric]=float(np.mean([np.mean(v) for v in projects.values()])) if projects else None
    rotation={}
    for name,a,b in [('base','joint','rotated'),('adapter','joint','rotated'),('adapter','shape','shape_rotated')]:
        diffs=[float(np.abs(images[name,'paired',a,i].astype(float)-images[name,'paired',b,i].astype(float)).mean()) for i in range(96)]
        rotation[f'{name}/{a}']={'mean_rgba_change_0_255':float(np.mean(diffs)),'max_case_mean_rgba_change':max(diffs)}
    # Fixed ordered pages; never select best-looking cases.
    cols=[('base','joint'),('base','rotated'),('prior','text'),('adapter','joint'),('adapter','rotated'),('adapter','shape_rotated')]
    for page in range(12):
        sheet=Image.new('RGB',(8*104,8*130+24),'white');draw=ImageDraw.Draw(sheet)
        for j,label in enumerate(['target','reference']+[f'{a}/{b}' for a,b in cols]):draw.text((j*104,2),label,fill='black')
        for k in range(8):
            i=page*8+k;r=rows['paired'][i];draw.text((0,24+k*130),r['prompt'][:120].encode('ascii','replace').decode(),fill='black')
            arrays=[native_grid(raw(r,'target'),16),raw(r,'reference')]+[images[a,'paired',b,i] for a,b in cols]
            for j,a in enumerate(arrays):sheet.paste(checker(a),(j*104,44+k*130))
        sheet.save(out/f'paired_{page:02d}.png')
    for split in rows:
        assert all(np.array_equal(images['prior',split,'text',i],images['adapter',split,'text',i]) for i in range(96)), 'text output changed under frozen adapter'
    for page in range(12):
        sheet=Image.new('RGB',(4*104,8*130+24),'white');draw=ImageDraw.Draw(sheet)
        for j,label in enumerate(('target','base/text','prior/text','adapter/text')):draw.text((j*104,2),label,fill='black')
        for k in range(8):
            i=page*8+k;r=rows['broad'][i];draw.text((0,24+k*130),r['prompt'][:70].encode('ascii','replace').decode(),fill='black')
            arrays=[native_grid(raw(r,'target'),16)]+[images[n,'broad','text',i] for n in checkpoints]
            for j,a in enumerate(arrays):sheet.paste(checker(a),(j*104,44+k*130))
        sheet.save(out/f'broad_{page:02d}.png')
    (out/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    atomic_json(out/'report.json',{'selection':selection,'project_mean_metrics':summary,'rotation':rotation,
        'text_prior_preserved_bitwise':True,'limits':protocol['limits'], 'interpretation':'Rotation invariance alone cannot prove MC style or useful reference control. Compare joint to text and shuffled references.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','text','adapter','evaluate'));parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    if args.stage=='prepare':prepare()
    elif args.stage=='evaluate':evaluate(args.device)
    else:train(args.stage,args.device)
