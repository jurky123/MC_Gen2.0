"""Strict checkpoint loading and one runtime for every consumer."""
from pathlib import Path
import os
import torch

from .conditioning import TextEncoder
from .config import Experiment
from .images import decode, reference
from .model import PixelFlow
from .sampling import sample

FORMAT = "mcgen.native-flow.v3"


def save_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save({"format": FORMAT, **payload}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError("not a native-flow v3 checkpoint; earlier weights require their matching implementation")
    Experiment.from_dict(state["config"])
    return state


class Runtime:
    def __init__(self, checkpoint, device="cpu", text_device=None):
        state = load_checkpoint(checkpoint)
        self.config = Experiment.from_dict(state["config"])
        self.device = device
        self.model = PixelFlow(self.config.model).to(device)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.encoder = TextEncoder(self.config.text, self.config.model, text_device or device)
        if self.encoder.identity != state["text_identity"]:
            raise ValueError("text encoder artifacts changed since training")
        self.metadata = {"format": FORMAT, "step": state["step"],
                         "catalog_fingerprint": state["catalog_fingerprint"],
                         "text_identity": state["text_identity"],
                         "text_backend": self.config.text.backend,
                         "task": self.config.task, "asset_type": self.config.asset_type,
                         "style": self.config.style, "alpha_mode": self.config.alpha_mode,
                         "reference_encoding": self.config.model.reference_encoding,
                         "reference_routing": "text:attributes; image-only:shape" if self.config.model.reference_encoding == "dual" else "shape"}

    def generate(self, prompts, references=None, seeds=None, **sampling):
        if not prompts:
            raise ValueError("at least one prompt is required (empty string supports image-only)")
        seeds = list(range(len(prompts))) if seeds is None else seeds
        if references is not None and len(references) != len(prompts):
            raise ValueError("one reference or None required per prompt")
        if references is not None and any(r is not None for r in references) and self.config.task == "text":
            raise ValueError("text-only checkpoint was not trained for reference conditioning")
        if self.config.task == "reference" and (references is None or any(r is None for r in references)):
            raise ValueError("reference-only checkpoint requires a reference for every sample")
        from .tasks import conditioned_prompts
        text, mask = self.encoder.encode(conditioned_prompts([""] * len(prompts) if self.config.task == "reference" else prompts, self.config), self.device)
        mask = mask & torch.tensor([bool(p.strip()) for p in prompts], device=self.device)[:, None]
        if self.config.task == "reference":
            mask[:] = False
        ref, present = None, None
        if references is not None:
            size = self.config.model.ref_size
            ref = torch.stack([reference(r, size) if r is not None else torch.zeros(4, size, size)
                               for r in references]).to(self.device)
            present = torch.tensor([r is not None for r in references], device=self.device)
        with torch.inference_mode():
            if self.config.model.reference_encoding == "dual" and sampling.get("reference_mode") is None:
                # Image-only still needs geometry until a category predictor is trained.
                sampling["reference_mode"] = ["attributes" if p.strip() and self.config.task != "reference"
                                               and sampling.get("text_scale", 2.) > 0 else "shape" for p in prompts]
            tensors = sample(self.model, text, mask, ref, present, seeds=seeds, **sampling)
        return [decode(x) for x in tensors]  # Raw output; alpha policy is an explicit export operation.
