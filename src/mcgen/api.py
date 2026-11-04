"""Application factory: model configuration is server-owned, not request-controlled."""
import base64
from contextlib import asynccontextmanager
import io
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from .images import export, apply_alpha
from .runtime import Runtime


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", max_length=4096)
    reference_png: str | None = Field(default=None, max_length=12_000_000)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    steps: int = Field(default=20, ge=1, le=100)
    text_scale: float = Field(default=2., ge=0, le=8)
    ref_scale: float = Field(default=2., ge=0, le=8)
    reference_mode: Literal["attributes", "shape"] | None = None
    conditioning_mode: Literal["native", "multimodal_embedding"] = "native"
    solver: str = "heun"
    export_scale: int = Field(default=1, ge=1, le=16)
    palette_colors: int = Field(default=0, ge=0, le=256)


def create_app(checkpoint, device="cpu", text_device=None):
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = Runtime(checkpoint, device, text_device)
        yield
        del app.state.runtime

    app = FastAPI(title="MCGen native pixel generation", lifespan=lifespan)

    @app.get("/health")
    def health():
        ready = hasattr(app.state, "runtime")
        return {"ready": ready, "model": app.state.runtime.metadata if ready else None}

    @app.post("/generate")
    def generate(req: GenerateRequest):
        if not hasattr(app.state, "runtime"):
            raise HTTPException(503, "model not loaded")
        ref = None
        try:
            if req.reference_png:
                raw = base64.b64decode(req.reference_png, validate=True)
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width * im.height > 16_000_000:
                        raise ValueError("reference exceeds 16 megapixels")
                    ref = im.convert("RGBA")
            if not req.prompt.strip() and ref is None:
                raise ValueError("provide a prompt or reference")
            with lock:
                result = app.state.runtime.generate([req.prompt], [ref] if ref else None,
                    seeds=[req.seed], steps=req.steps, text_scale=req.text_scale,
                    ref_scale=req.ref_scale, solver=req.solver, reference_mode=req.reference_mode,
                    conditioning_mode=req.conditioning_mode)[0]
            image = export(apply_alpha(result, app.state.runtime.config.alpha_mode), req.export_scale, req.palette_colors)
        except (ValueError, UnidentifiedImageError) as exc:
            raise HTTPException(422, str(exc)) from exc
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {"png_base64": base64.b64encode(buf.getvalue()).decode(),
                "native_size": app.state.runtime.config.model.size, "seed": req.seed,
                "export_scale": req.export_scale, "palette_colors": req.palette_colors,
                "conditioning_mode": req.conditioning_mode,
                "model": app.state.runtime.metadata}

    return app
