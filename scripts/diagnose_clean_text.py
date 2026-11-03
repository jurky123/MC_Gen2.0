"""Bounded learnability and caption-length ablation; never promotes a model."""
import argparse, gc, json, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml
from mcgen.catalog import atomic_json, file_hash, write_catalog
from mcgen.config import Experiment
from mcgen.training import Trainer
from mcgen.runtime import Runtime, load_checkpoint
from mcgen.images import composite, encode
from mcgen.evaluation import metrics, summarize
from mcgen.tasks import conditioned_prompts
from audit_item_captions import array

SOURCE=Path('outputs/text_caption_scale1152_20261102')
INITIAL=Path('outputs/attribute_prior_20261102/text/best_flow.pt')
TRAIN={2:'glow berry ice cream bowl',6:'direwolf chestplate',7:'chorus flower dead boots',8:'emerald shovel',9:'bone',15:'crystal shard',16:'helmet guide',18:'mithril ingot',20:'basalt top pickaxe',23:'acacia pickaxe',25:'stacked iron axe',28:'ruby gem',29:'emerald sword',33:'lead apple',40:'pupsat boots',43:'enderium basic shield'}
VAL={51:'fire core',52:'batet pickaxe',53:'frozen apple',55:'nickel rapier',57:'rusted iron ingot',58:'fossilized sword',59:'copper diamond pickaxe',60:'tin pickaxe'}

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    cases={r['id']:r for r in map(json.loads,(SOURCE/'vision/cases.jsonl').read_text().splitlines())}
    review={r['index']:r for r in map(json.loads,(SOURCE/'fixed_review/records.jsonl').read_text().splitlines())}
    ratings={r['id']:r['new_rating'] for r in json.loads((SOURCE/'quality_review.json').read_text())['ratings']}
    selected=[]
    for index,name in {**TRAIN,**VAL}.items():
        r=review[index];c=cases[r['id']]
        assert ratings[r['id']]=='plausible' and c['split']==('train' if index in TRAIN else 'val')
        selected.append({**c,'reference':None,'short':name,'long':r['new_caption'],'review_index':index})
    # Compact immutable storage preserves pixel hashes and project splits.
    target=out/'targets.mmap';np.stack([array(r) for r in selected]).tofile(target)
    for i,r in enumerate(selected):r['target']={'path':str(target.resolve()),'shape':[len(selected),16,16,4],'index':i}
    atomic_json(out/'cases.json',selected)
    cfg=yaml.safe_load(Path('configs/v3/experiments/text_caption_ab1152.yaml').read_text())
    cfg['train'].update(steps=800,lr=2e-4,eval_every=200,val_samples=8,seed=20261109)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    for arm in ['long','mixed']:
        rows=[]
        for r in selected:
            variants=['short'] if r['split']=='val' else ['long','short'] if arm=='mixed' else ['long']
            for variant in variants:rows.append({**r,'id':r['id']+':'+variant,'prompt':r[variant]})
        write_catalog(out/'catalogs'/arm,rows,{'native_size':16,'diagnostic_only':True,'arm':arm,'selection':'16 visually inspected complete train items; 8 existing validation items. No split changes.'})
    atomic_json(out/'protocol.json',{'initialization':str(INITIAL),'initialization_sha256':file_hash(INITIAL),'config_sha256':file_hash(out/'config.yaml'),'training_targets':16,'validation_targets':8,'steps':800,'primary_sampling':{'seeds':[20261109,20261110],'steps':32,'solver':'heun','cfg':1},'scope':'Learnability diagnostic, not population generalization. Mixed arm samples 50% short / 50% long per target. Same images, target-uniform sampler, config and initialization. No checkpoint selection.'})
    return selected

@torch.no_grad()
def probe(runtime,rows):
    result={}
    for split in ['train','val']:
        subset=[r for r in rows if r['split']==split]
        x=torch.stack([encode(array(r)) for r in subset]).to(runtime.device)
        gen=torch.Generator(device=runtime.device).manual_seed(8675309)
        noise=torch.randn(x.shape,generator=gen,device=runtime.device)
        for kind in ['short','long']:
            text,mask=runtime.encoder.encode(conditioned_prompts([r[kind] for r in subset],runtime.config),runtime.device)
            contexts={'correct':runtime.model.context(text,mask,None,None),'shuffled':runtime.model.context(text.roll(1,0),mask.roll(1,0),None,None),'null':runtime.model.context(text,torch.zeros_like(mask),None,None)}
            for t in [.5,.9,1.]:
                noisy=(1-t)*x+t*noise
                values={k:float((runtime.model(noisy,torch.full((len(x),),t,device=runtime.device),c)-(noise-x)).square().mean()) for k,c in contexts.items()}
                result[f'{split}:{kind}:t={t}']=values
    return result

def evaluate(out,rows,device):
    dest=out/'evaluation';dest.mkdir();(dest/'images').mkdir()
    runtime=Runtime(INITIAL,device,device)
    generated={};scores=[];probes={};cases=[]
    for seed in [20261109,20261110]:
        for kind in ['short','long']:
            for r in rows:cases.append({**r,'kind':kind,'seed':seed})
    atomic_json(dest/'cases.json',cases)
    train=[r for r in rows if r['split']=='train'];bank=torch.stack([encode(array(r)) for r in train]);bank_ids=[r['id'] for r in train]
    for arm in ['initial','long','mixed']:
        if arm!='initial':
            state=load_checkpoint(out/'training'/arm/'latest.pt')
            assert state['step']==800 and state['text_identity']==runtime.encoder.identity
            assert state['config']['model']==runtime.config.to_dict()['model']
            runtime.model.load_state_dict(state['model']);del state
        probes[arm]=probe(runtime,rows);images=[]
        for start in range(0,len(cases),8):
            batch=cases[start:start+8]
            images.extend(runtime.generate([r[r['kind']] for r in batch],seeds=[r['seed'] for r in batch],steps=32,solver='heun',text_scale=1))
            print(json.dumps({'stage':'generate','arm':arm,'done':len(images),'total':len(cases)}),flush=True)
        for i,(r,img) in enumerate(zip(cases,images)):
            Image.fromarray(img).save(dest/'images'/f'{arm}_{i:03d}.png')
            score=metrics(img,array(r));score['seam_mae']=None
            score['no_foreground']=float(not (img[...,3]>=128).any())
            score['correct_train_retrieval']=float(bank_ids[int((bank-encode(img)).square().mean((1,2,3)).argmin())]==r['id']) if r['split']=='train' else None
            scores.append({'id':r['id'],'lineage':r['lineage'],'mode':f'{arm}:{r["split"]}:{r["kind"]}','seed':r['seed'],'metrics':score})
        generated[arm]=images
        # Unconditional images and fixed CFG check are diagnostics, not a tuned sampler.
        uncond=runtime.generate(['']*16,seeds=list(range(100,116)),steps=32,solver='heun',text_scale=0)
        sheet=Image.new('RGB',(4*128,4*128),(128,128,128))
        for i,img in enumerate(uncond):sheet.paste(Image.fromarray(composite(img,128).round().astype('uint8')).resize((128,128),Image.Resampling.NEAREST),((i%4)*128,(i//4)*128))
        sheet.save(dest/f'unconditional_{arm}.png')
        checks=[rows[i] for i in [0,3,11,12,16,17,18,23]]
        cfg2=runtime.generate([r['short'] for r in checks],seeds=[20261109]*8,steps=32,solver='heun',text_scale=2)
        for i,img in enumerate(cfg2):Image.fromarray(img).save(dest/'images'/f'{arm}_cfg2_{i}.png')
        atomic_json(dest/f'cfg2_{arm}.json',[{'id':r['id'],'metrics':{**metrics(img,array(r)),'seam_mae':None}} for r,img in zip(checks,cfg2)])
    atomic_json(dest/'probes.json',probes);atomic_json(dest/'summary.json',summarize(scores))
    (dest/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores))
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    for start in range(0,len(cases),8):
        sheet=Image.new('RGB',(1080,8*148+32),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,4),'Prompt / target / initial / long only / mixed lengths',font=font,fill='black')
        for k,r in enumerate(cases[start:start+8]):
            y=32+k*148;words=f'{r["split"]} {r["kind"]}: '+r[r['kind']];lines=[];line=''
            for char in words:
                if font.getlength(line+char)>350:lines.append(line);line=char
                else:line+=char
            lines.append(line);draw.multiline_text((8,y),'\n'.join(lines),font=font,fill='black')
            for x,img in [(380,array(r))]+[(550+j*170,generated[arm][start+k]) for j,arm in enumerate(['initial','long','mixed'])]:
                sheet.paste(Image.fromarray(composite(img,128).round().astype('uint8')).resize((128,128),Image.Resampling.NEAREST),(x,y))
        sheet.save(dest/f'comparison_{start//8:02d}.png')
    del runtime;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args()
    torch.set_num_threads(4)
    rows=prepare(a.out)
    while torch.cuda.mem_get_info(a.device)[0]<23*1024**3:time.sleep(30)
    free,total=torch.cuda.mem_get_info(a.device);torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),a.device)
    for arm in ['long','mixed']:
        atomic_json(a.out/'status.json',{'stage':'training','arm':arm})
        trainer=Trainer(Experiment.load(a.out/'config.yaml'),a.out/'catalogs'/arm,a.out/'training'/arm,a.device,a.device,initialize=INITIAL)
        trainer.run();del trainer;gc.collect();torch.cuda.empty_cache()
    atomic_json(a.out/'status.json',{'stage':'evaluating'})
    evaluate(a.out,rows,a.device)
    atomic_json(a.out/'status.json',{'stage':'complete','production_promoted':False})
