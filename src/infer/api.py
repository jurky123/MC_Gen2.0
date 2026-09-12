import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel, Field

from config import ModelConfig
from model.mc_flow_dit import MCFlowDiT
from eval.seam import make_tiled_preview
from .solver import sample, to_uint8
from .sample import encode_prompts

app = FastAPI(title="MC-FlowDiT Texture Generation API")

_MODEL = None
_MODEL_CFG = None
_CKPT = os.environ.get("MC_CKPT", "")


class GenerateRequest(BaseModel):
    prompt: str
    seed: int = Field(default=0)
    steps: int = Field(default=20, ge=1, le=64)
    cfg: float = Field(default=2.0, ge=0.0, le=8.0)
    solver: str = "heun"
    text_encoder: str = ""
    encoder_type: str = ""
    instruction: str = ""


def _load_model():
    global _MODEL, _MODEL_CFG
    if _MODEL is None:
        if not _CKPT:
            raise RuntimeError("set MC_CKPT env var to a checkpoint path")
        sd = torch.load(_CKPT, map_location="cpu")
        cfg = ModelConfig.from_dict(sd["model_cfg"])
        model = MCFlowDiT(cfg)
        model.load_state_dict(sd["model"])
        model.eval()
        _MODEL = model
        _MODEL_CFG = cfg
    return _MODEL, _MODEL_CFG


def _png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok", "model": _MODEL_CFG.name if _MODEL_CFG else "not loaded"}


@app.post("/generate")
def generate(req: GenerateRequest):
    model, cfg = _load_model()
    device = next(model.parameters()).device
    embs = torch.from_numpy(
        encode_prompts(
            [req.prompt],
            text_dim=cfg.text_dim,
            max_tokens=cfg.max_text_tokens,
            model_name=req.text_encoder,
            encoder_type=req.encoder_type,
            instruction=req.instruction,
        )
    ).to(device)
    uncond = torch.zeros_like(embs)
    g = torch.Generator(device=device).manual_seed(int(req.seed))
    z = torch.randn(1, cfg.in_channels, cfg.image_size, cfg.image_size, generator=g, device=device)
    with torch.no_grad():
        x = sample(model, z, embs, steps=req.steps, cfg=req.cfg, text_uncond=uncond, solver=req.solver)
    u8 = to_uint8(x[0]).permute(1, 2, 0).cpu().numpy()
    tiled = make_tiled_preview(u8)
    return {
        "texture_png": _png_bytes(u8),
        "tiled_png": _png_bytes(tiled),
        "seed": req.seed,
        "model_version": cfg.name,
        "sampling_config": {"steps": req.steps, "cfg": req.cfg, "solver": req.solver},
    }


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()