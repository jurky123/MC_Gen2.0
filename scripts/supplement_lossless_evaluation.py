"""Use the idle GPU for the second-seed joint shard, ahead of the main worker."""
import json
from pathlib import Path
import time

from PIL import Image
import torch

from mcgen.catalog import atomic_json
from mcgen.images import decode
from mcgen.sampling import sample
from evaluate_lossless_native16 import load_model, noise_for, SEEDS


@torch.inference_mode()
def main():
    out=Path('outputs/native16_lossless_eval_20261107')
    device='cuda:0'
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.30,device)
    features=torch.load(out/'features.pt',weights_only=True)
    rows=json.loads((out/'samples.json').read_text())
    diverse=json.loads((out/'diversity.json').read_text())
    model=load_model('prior32',out,device)
    seed=SEEDS[1]
    done=0
    started=time.monotonic()
    def generate(ids,key,names,reference_ids=None):
        nonlocal done
        # Main worker is still on seed0. This shard only covers late seed1
        # tasks; both use identical features, FP32 solver and CPU noise.
        chosen=[j for j,name in enumerate(names) if not (out/'native'/name).exists()]
        if not chosen:
            return
        h=features[key][[ids[j] for j in chosen]].to(device)
        refs=None if reference_ids is None else features['reference'][[reference_ids[j] for j in chosen]].to(device)
        result=sample(model,h,torch.ones(h.shape[:2],dtype=torch.bool,device=device),refs,
            seeds=[seed]*len(chosen),steps=32,initial_noise=noise_for([seed]*len(chosen),32,device))
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite supplement output')
        for value,j in zip(result,chosen):
            path=out/'native'/names[j]
            if not path.exists():
                Image.fromarray(decode(value)).save(path)
                done+=1
        atomic_json(out/'progress_supplement.json',{'phase':'running','done':done,'seconds':time.monotonic()-started})
    for start in range(0,len(rows),12):
        ids=list(range(start,min(start+12,len(rows))))
        generate(ids,'short',[f'{rows[i]["case_id"]}_prior32_joint_s1.png' for i in ids],ids)
    for start in range(0,len(diverse),12):
        ids=list(range(start,min(start+12,len(diverse))))
        generate(ids,'fixed_text',[f'{diverse[i]["case_id"]}_prior32_joint_s1.png' for i in ids],[i+len(rows) for i in ids])
    ids=list(range(0,len(diverse),4))
    generate(ids,'fixed_text',[f'fixedtext_{diverse[i]["fixed_prompt"]}_prior32_s1.png' for i in ids])
    ids=list(range(len(features['common_prompts'])))
    generate(ids,'common_text',[f'common_{i:02d}_prior32_s1.png' for i in ids])
    atomic_json(out/'progress_supplement.json',{'phase':'complete','outputs_written':done,'seconds':time.monotonic()-started})
    print('SUPPLEMENT COMPLETE',done,flush=True)


if __name__=='__main__':
    main()
