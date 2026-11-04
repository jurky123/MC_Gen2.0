"""Residual embedding alignment with an immutable pretrained pixel generator."""
import hashlib
import torch
from .model import PixelFlow


def backbone_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('image_adapter.'):continue
        h.update(name.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def initialize_adapter(config,state,device):
    after=config.to_dict()['model'].copy()
    # Fill defaults for historical checkpoints before comparing contracts.
    from .config import Experiment
    before=Experiment.from_dict(state['config']).to_dict()['model']
    before.pop('multimodal_adapter');after.pop('multimodal_adapter')
    if before!=after or not config.model.multimodal_adapter:raise ValueError('adapter requires the exact pretrained backbone architecture')
    model=PixelFlow(config.model).to(device)
    loaded=model.load_state_dict(state['model'],strict=False)
    expected={n for n in model.state_dict() if n.startswith('image_adapter.')}
    if set(loaded.missing_keys)!=expected or loaded.unexpected_keys:raise ValueError('unexpected adapter initialization keys')
    model.requires_grad_(False);model.image_adapter.requires_grad_(True)
    # The backbone has no dropout; keep inference behavior while allowing
    # gradients through its computations into the new condition adapter.
    model.eval()
    return model
