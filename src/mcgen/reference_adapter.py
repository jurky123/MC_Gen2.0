"""Residual embedding alignment with an immutable pretrained pixel generator."""
import hashlib
import torch
from .model import PixelFlow


def backbone_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith(('image_adapter.', 'visual_adapter.')):continue
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


class SpatialReferenceAdapter(torch.nn.Module):
    """Independent residual attention to a grid of frozen visual features."""
    def __init__(self, cfg):
        super().__init__()
        from .model import position
        w = cfg.width
        self.projection = torch.nn.Sequential(torch.nn.LayerNorm(cfg.visual_dim),
                                               torch.nn.Linear(cfg.visual_dim, w))
        self.register_buffer('position', position(cfg.visual_grid, w))
        self.attention = torch.nn.ModuleDict({str(i): torch.nn.MultiheadAttention(w, cfg.heads,
                                                batch_first=True) for i in cfg.visual_layers})
        self.norm = torch.nn.LayerNorm(w, elementwise_affine=False)
        self.gates = torch.nn.ParameterDict({str(i): torch.nn.Parameter(torch.zeros(w))
                                            for i in cfg.visual_layers})

    def project(self, features):
        return self.projection(features.float()) + self.position

    def inject(self, layer, tokens, reference, present):
        key = str(layer)
        if key not in self.attention or not bool(present.any()):
            return tokens
        query = self.norm(tokens)
        delta = self.attention[key](query, reference, reference, need_weights=False)[0]
        return tokens + delta * self.gates[key].tanh() * present[:, None, None]


def initialize_visual_adapter(config, state, device):
    from .config import Experiment
    before = Experiment.from_dict(state['config']).to_dict()['model']
    after = config.to_dict()['model'].copy()
    if before['visual_adapter'] or before['multimodal_adapter'] or not after['visual_adapter']:
        raise ValueError('visual adaptation requires an unmodified text backbone')
    for key in ('visual_adapter', 'visual_dim', 'visual_grid', 'visual_layers'):
        before.pop(key)
        after.pop(key)
    if before != after:
        raise ValueError('visual adapter requires the exact pretrained backbone architecture')
    model = PixelFlow(config.model).to(device)
    loaded = model.load_state_dict(state['model'], strict=False)
    expected = {n for n in model.state_dict() if n.startswith('visual_adapter.')}
    if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
        raise ValueError('unexpected visual adapter initialization keys')
    model.requires_grad_(False)
    model.visual_adapter.requires_grad_(True)
    model.eval()
    return model
