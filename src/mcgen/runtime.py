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
        if self.config.model.multimodal_adapter and state.get("multimodal_identity") != self.encoder.embedding.multimodal_identity:
            raise ValueError("multimodal encoder protocol changed since adapter training")
        self.visual_identity = state.get('visual_identity')
        self.visual_encoder = None
        if self.config.model.visual_adapter and not self.visual_identity:
            raise ValueError('visual adapter checkpoint has no visual encoder identity')
        self.metadata = {"format": FORMAT, "step": state["step"],
                         "catalog_fingerprint": state["catalog_fingerprint"],
                         "text_identity": state["text_identity"],
                         "text_backend": self.config.text.backend,
                         "task": self.config.task, "asset_type": self.config.asset_type,
                         "style": self.config.style, "alpha_mode": self.config.alpha_mode,
                         "conditioning_input": self.config.conditioning_input,
                         "multimodal_adapter": self.config.model.multimodal_adapter,
                         "visual_adapter": self.config.model.visual_adapter,
                         "reference_encoding": self.config.model.reference_encoding,
                         "reference_routing": "visual_tokens" if self.config.model.visual_adapter else ("pooled_embedding" if self.config.model.multimodal_adapter else ("text:attributes; image-only:shape" if self.config.model.reference_encoding == "dual" else "shape"))}

    def generate(self, prompts, references=None, seeds=None, conditioning_mode="native", **sampling):
        if not prompts:
            raise ValueError("at least one prompt is required (empty string supports image-only)")
        seeds = list(range(len(prompts))) if seeds is None else seeds
        if references is not None and len(references) != len(prompts):
            raise ValueError("one reference or None required per prompt")
        if conditioning_mode not in ("native", "multimodal_embedding", "visual_tokens"):
            raise ValueError("unknown conditioning mode")
        has_reference = references is not None and any(r is not None for r in references)
        if conditioning_mode == 'visual_tokens' or (self.config.model.visual_adapter and has_reference):
            if conditioning_mode == 'multimodal_embedding':
                raise ValueError('visual adapter uses independent spatial reference tokens')
            return self._generate_visual(prompts, references, seeds, sampling)
        if conditioning_mode == "multimodal_embedding":
            return self._generate_multimodal(prompts, references, seeds, sampling)
        if self.config.model.multimodal_adapter and references is not None and any(r is not None for r in references):
            raise ValueError("use conditioning_mode=multimodal_embedding for this adapter checkpoint")
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

    def _generate_visual(self, prompts, references, seeds, sampling):
        if not self.config.model.visual_adapter:
            raise ValueError('checkpoint has no spatial visual adapter')
        if references is None or not any(r is not None for r in references):
            raise ValueError('visual token mode requires a reference')
        if sampling.get('reference_mode') is not None:
            raise ValueError('visual tokens do not use attributes/shape modes')
        from .tasks import conditioned_prompts
        from .visual_tokens import VisualTokenEncoder
        text, mask = self.encoder.encode(conditioned_prompts(prompts, self.config), self.device)
        # Frozen encoders can return inference tensors; do not mutate them
        # outside their inference_mode scope.
        mask = mask & torch.tensor([bool(p.strip()) for p in prompts], device=self.device)[:, None]
        if self.visual_encoder is None:
            self.visual_encoder = VisualTokenEncoder(self.config.text, self.device)
            if self.visual_encoder.identity != self.visual_identity:
                self.visual_encoder = None
                raise ValueError('visual encoder protocol changed since adapter training')
        ids = [i for i, ref in enumerate(references) if ref is not None]
        encoded = self.visual_encoder.encode([references[i] for i in ids], self.device)
        features = encoded.new_zeros((len(prompts), *encoded.shape[1:]))
        features[ids] = encoded
        present = torch.tensor([r is not None for r in references], device=self.device)
        with torch.inference_mode():
            tensors = sample(self.model, text, mask, features, present, seeds=seeds, **sampling)
        return [decode(x) for x in tensors]

    def _generate_multimodal(self, prompts, references, seeds, sampling):
        """Pooled multimodal conditioning, optionally using a trained adapter.

        ref_scale guides the complete image/text embedding. There are no separate
        image and text guidance terms and no spatial-reference branch in this mode.
        """
        if sampling.get('guidance_order', 'text_first') != 'text_first':
            raise ValueError('reference_first requires separate text and reference conditions')
        if (self.config.text.backend != "qwen3_vl_embedding" or
                self.config.model.conditioning != "text_only" or self.config.task not in ("text", "joint")):
            raise ValueError("multimodal embedding transfer requires a Qwen VL text-only checkpoint")
        if references is None or any(r is None for r in references):
            raise ValueError("multimodal embedding requires a reference for every sample")
        if sampling.get("reference_mode") is not None:
            raise ValueError("reference_mode applies only to spatial reference checkpoints")
        from .tasks import conditioned_prompts
        texts = conditioned_prompts(prompts, self.config)
        text, mask = self.encoder.encode_multimodal(texts, references, self.device)
        sampling.pop("reference_mode", None)
        sampling["text_scale"] = sampling.pop("ref_scale", 2.)
        if self.config.model.multimodal_adapter:
            sampling["embedding_adapter"] = True
        with torch.inference_mode():
            tensors = sample(self.model, text, mask, seeds=seeds, **sampling)
        return [decode(x) for x in tensors]
