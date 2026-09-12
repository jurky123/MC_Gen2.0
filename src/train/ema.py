import torch
import torch.nn as nn


class EMA:
    def __init__(self, model, decay=0.999, update_every=1, device="cpu"):
        self.decay = decay
        self.update_every = update_every
        self.device = device
        self.shadow = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.detach().float().cpu().clone()
        self.updates = 0

    def step(self, model):
        self.updates += 1
        if self.updates % self.update_every != 0:
            return False
        with torch.no_grad():
            sd = model.state_dict()
            for k in self.shadow:
                t = sd[k].detach().float().cpu()
                self.shadow[k].mul_(self.decay).add_(t, alpha=1.0 - self.decay)
        return True

    def copy_to(self, model):
        sd = model.state_dict()
        for k in self.shadow:
            sd[k].copy_(self.shadow[k].to(device=sd[k].device, dtype=sd[k].dtype))

    def state_dict(self):
        return {"decay": self.decay, "update_every": self.update_every, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, sd):
        # Keep decay / update_every from the current config so that changing
        # them takes effect on resume; only restore the accumulated state.
        self.updates = sd["updates"]
        self.shadow = sd["shadow"]