"""Generate native MC textures with a single API-edition checkpoint and DashScope."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from PIL import Image, ImageOps
import torch
from torch import nn
from mcgen.config import Experiment
from mcgen.images import decode
from mcgen.model import PixelFlow
from mcgen.sampling import sample

PROTOCOL = 'dashscope-qwen3-vl-embedding-2560-independent-white256-v1'
URL = 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding'

def vector(kind, value, cache):
    dimension = 2560
    if kind == 'image':
        with Image.open(value) as im:
            rgba = ImageOps.exif_transpose(im).convert('RGBA')
            rgba.thumbnail((256, 256), Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', rgba.size, 'white')
            canvas.paste(rgba, mask=rgba.getchannel('A'))
            stream = io.BytesIO()
            canvas.save(stream, 'PNG')
        value = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
    protocol = PROTOCOL
    key = hashlib.sha256((protocol+kind+value).encode()).hexdigest()
    path = cache/f'{key}.pt'
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    token = os.environ.get('DASHSCOPE_API_KEY')
    if not token:
        raise SystemExit('Set DASHSCOPE_API_KEY for uncached embeddings.')
    payload = dict(model='qwen3-vl-embedding',input={'contents':[{kind:value}]},
                   parameters=dict(dimension=dimension,enable_fusion=False))
    req = urllib.request.Request(URL,data=json.dumps(payload).encode(),headers={
        'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as resp:
        item = json.load(resp)['output']['embeddings'][0]
    result = torch.tensor(item['embedding'],dtype=torch.float32)
    if item['type'] not in (kind,'vl') or result.shape!=(dimension,) or not result.isfinite().all() or not result.norm()>0:
        raise ValueError(f'Unexpected DashScope {kind} result: type={item.get("type")!r}, shape={tuple(result.shape)}')
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
        torch.save(result,tmp)
    Path(tmp.name).replace(path)
    return result


class GlobalConditions(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.text = nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,width),nn.GELU(),nn.Linear(width,4*width))
        self.image = nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,width),nn.GELU(),nn.Linear(width,8*width))


class Transfer(PixelFlow):
    def context(self,text,text_mask,reference,ref_present,reference_mode=None):
        b=len(text)
        tokens=self.conditions.text(text).reshape(b,4,-1)+self.type_embedding[0]
        mask=text_mask.reshape(b,1).expand(-1,4)
        ctx=(torch.cat((self.text_null.expand(b,-1,-1),tokens),1),
             torch.cat((torch.ones_like(mask[:,:1]),mask),1))
        if reference is not None:
            ref=self.conditions.image(reference).reshape(b,8,-1)
            return (*ctx,(ref,ref_present))
        return ctx



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/mcgen-api-step10516.pt'))
    parser.add_argument('--prompt', default='')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('.cache/mcgen-api'))
    args = parser.parse_args()
    if not args.prompt.strip() and not args.reference:
        parser.error('Provide an English item name or a reference image.')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state.get('protocol') != PROTOCOL or state.get('format') != 'mcgen.api-conditions.v1':
        parser.error('Expected an API-edition inference checkpoint.')
    cfg = Experiment.from_dict(state['config'])
    model = Transfer(cfg.model)
    model.conditions = GlobalConditions(cfg.model.width)
    model.load_state_dict(state['model'])
    model.to(args.device).eval()
    has_text = bool(args.prompt.strip())
    text = vector('text',args.prompt,args.cache) if has_text else torch.zeros(2560)
    ref = vector('image',args.reference,args.cache)[None].to(args.device) if args.reference else None
    mask = torch.tensor([[has_text]],device=args.device)
    image = sample(model,text[None].to(args.device),mask,ref,torch.ones_like(mask[:,0]),
                   seeds=[args.seed],steps=args.steps,text_scale=1.,ref_scale=1.)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(decode(image[0])).save(args.out)

if __name__ == '__main__':
    main()
