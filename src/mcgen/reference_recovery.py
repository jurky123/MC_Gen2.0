"""Reference-heavy condition sampling and an explicit update-budget schedule."""
import math
import re

import torch

from .curated_training import MixedData, MixedSampler
from .tasks import conditioned_prompts

NOUNS = {
    'helmet':'helmet','helm':'helmet','sword':'sword','axe':'axe','battleaxe':'axe',
    'waraxe':'axe','greataxe':'axe','pickaxe':'pickaxe','shield':'shield',
    'chestplate':'chestplate','cuirass':'chestplate','leggings':'leggings',
    'boots':'boots','boot':'boots','bow':'bow','crossbow':'crossbow','arrow':'arrow',
    'wand':'wand','staff':'staff','scythe':'scythe','mace':'mace','hammer':'hammer',
    'warhammer':'hammer','spear':'spear','ring':'ring','crown':'crown','glove':'glove',
    'gauntlet':'glove','necklace':'necklace','charm':'charm','gem':'gem',
    'crystal':'crystal','ingot':'ingot','potion':'potion','bottle':'bottle',
    'lantern':'lantern','torch':'torch','candy':'candy','bed':'bed','book':'book',
    'key':'key','seed':'seed','shovel':'shovel','hoe':'hoe',
}
WARM_MIX = {'joint':.70,'reference':.30,'text':0.,'unconditional':0.}
JOINT_MIX = {'joint':.60,'reference':.20,'text':.18,'unconditional':.02}


def coarse_caption(caption):
    text=caption.lower().replace('_',' ').replace('-',' ')
    if 'bow tie' in text:return 'item'
    for word in reversed(re.findall(r'[a-z]+',text)):
        if word in NOUNS:return NOUNS[word]
    return 'item'


def phase_for(update,recipe):
    return 'visual' if update<recipe['visual_warmup_steps'] else 'joint'


def learning_rates(update,recipe):
    if phase_for(update,recipe)=='visual':
        return 0.,recipe['visual_lr']*min(1.,(update+1)/recipe['warmup_steps'])
    local=update-recipe['visual_warmup_steps']
    base=recipe['base_lr']*min(1.,(local+1)/recipe['warmup_steps'])
    late=max(0.,(update-recipe['decay_start'])/max(1,recipe['steps']-recipe['decay_start']))
    floor=recipe.get('lr_floor',.3)
    factor=floor+(1-floor)*(1+math.cos(math.pi*min(1.,late)))/2
    return base*factor,recipe['visual_lr']*factor


def trainable_phase(model,phase):
    if phase not in ('visual','joint'):raise ValueError('Unknown training phase')
    model.requires_grad_(phase=='joint')
    model.visual_adapter.requires_grad_(True)
    model.eval()


class RecoverySampler(MixedSampler):
    def sample(self,n,rng):
        ids,tasks,variants=super().sample(n,rng)
        for j,task in enumerate(tasks):
            if task=='joint':
                variants[j]=2 if torch.rand((),generator=rng)<.7 else 0
        return ids,tasks,variants


class RecoveryData(MixedData):
    def __init__(self,root):
        super().__init__(root)
        self.categories=[coarse_caption(row['captions'][0]) for row in self.rows]
        self.coarse_text,self.coarse_mask=self.encoder.encode(conditioned_prompts(self.categories,self.cfg),'cpu')
        self.samplers={phase:RecoverySampler(self.rows,mix) for phase,mix in [('visual',WARM_MIX),('joint',JOINT_MIX)]}
        self.cached=None

    def cache_device(self,device):
        import numpy as np
        reference=torch.empty(self.features.shape,dtype=torch.bfloat16,device=device)
        for start in range(0,len(self.rows),128):
            reference[start:start+128]=torch.from_numpy(np.array(self.features[start:start+128],copy=True)).view(torch.bfloat16).to(device)
        values=[('target',self.targets)]
        if not getattr(getattr(self,'encoder',None),'is_ragged',False):
            values += [('text',self.text),('mask',self.mask),('coarse',self.coarse_text),('coarse_mask',self.coarse_mask)]
        self.cached={name:value.to(device) for name,value in values}
        self.cached['reference']=reference

    def batch(self,ids,tasks,variants,device):
        indices=torch.tensor(ids,dtype=torch.long,device=device)
        if getattr(getattr(self,'encoder',None),'is_ragged',False):
            captions=[self.categories[i] if v==2 else self.rows[i]['captions'][0 if v==0 else -1]
                      for i,v in zip(ids,variants)]
            text,mask=self.encoder.encode(conditioned_prompts(captions,self.cfg),device)
            if self.cached is None:
                import numpy as np
                refs=torch.from_numpy(np.array(self.features[ids],copy=True)).view(torch.bfloat16).to(device)
                target=self.targets[ids].to(device)
            else:
                refs=self.cached['reference'][indices];target=self.cached['target'][indices]
            mask=mask & torch.tensor([t in ('joint','text') for t in tasks],device=device)[:,None]
            present=torch.tensor([t in ('joint','reference') for t in tasks],device=device)
            return target,text,mask,refs,present
        labels=torch.tensor(variants,dtype=torch.long,device=device)
        if self.cached is None:
            text=self.text[ids,[min(v,1) for v in variants]].to(device)
            mask=self.mask[ids,[min(v,1) for v in variants]].to(device)
            coarse=self.coarse_text[ids].to(device)
            coarse_mask=self.coarse_mask[ids].to(device)
            import numpy as np
            refs=torch.from_numpy(np.array(self.features[ids],copy=True)).view(torch.bfloat16).to(device)
            target=self.targets[ids].to(device)
        else:
            text=self.cached['text'][indices,labels.clamp_max(1)]
            mask=self.cached['mask'][indices,labels.clamp_max(1)]
            coarse=self.cached['coarse'][indices]
            coarse_mask=self.cached['coarse_mask'][indices]
            refs=self.cached['reference'][indices]
            target=self.cached['target'][indices]
        replace=labels==2
        text=torch.where(replace[:,None,None],coarse,text)
        mask=torch.where(replace[:,None],coarse_mask,mask)
        mask=mask & torch.tensor([task in ('joint','text') for task in tasks],device=device)[:,None]
        present=torch.tensor([task in ('joint','reference') for task in tasks],device=device)
        return target,text,mask,refs,present
