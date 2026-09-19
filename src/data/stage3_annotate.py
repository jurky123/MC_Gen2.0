"""Stage-3 fine annotation of a curated subset with the local 27B VLM.

Rules (strict):
  * every source label word must be kept verbatim (orientation and abstract
    words included);
  * ``short_prompt`` must contain all label words and the explicit ``block`` /
    ``item`` type word;
  * colour / pattern / surface / shape must come from the images only;
  * no invented materials, objects or lore; ``unknown`` when unsure;
  * no "pixel art" / "minecraft" / "texture" boilerplate.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from data.filename_prompts import fallback_prompt, validate_prompt
from data.vlm_caption import (
    HttpBackend, _Tee, build_views, extract_json, load_record_image, record_has_mmap,
)
from data.work_queue import run_queued

STAGE3_FIELDS = [
    "name", "material", "form", "state", "dominant_colors", "pattern", "surface",
    "shape", "symmetry", "tileable", "emissive", "transparency",
    "short_prompt", "detailed_prompt", "uncertainty",
]

_SCHEMA = """{
  "name": "string (short canonical name; must contain every label word)",
  "material": "string or unknown",
  "form": "string or unknown (sword, brick, ore, door, ingot, gem, ...)",
  "state": ["string"],
  "dominant_colors": ["up to 3 colour words"],
  "pattern": "string (uniform, stripes, grid, mottled, checker, organic, ...)",
  "surface": "string (smooth, rough, grainy, glossy, ...)",
  "shape": "string (e.g. diagonal blade, round gem, full tile, thin bar, ...)",
  "symmetry": "symmetric | vertical | horizontal | rotational | asymmetric | unknown",
  "tileable": "yes | no | unknown",
  "emissive": false,
  "transparency": "none | partial | cutout | unknown",
  "short_prompt": "string (<=14 words; must contain every label word AND block/item)",
  "detailed_prompt": "string (<=32 words)",
  "uncertainty": 0.0
}"""

DEFAULT_SYSTEM_PROMPT = """You annotate one Minecraft / voxel block or item texture for a text-to-image dataset.

You receive the source label words, the asset type (block or item) and two images: View A is the texture upscaled with nearest-neighbour, View B is a 4x4 tiling of it.

Hard rules:
- The label words are authoritative. Keep EVERY one of them verbatim (same spelling), including orientation words (top, side, front, ...) and abstract words (lava, mossy, overthium, ...). The "name" and "short_prompt" must contain all of them.
- The asset type is authoritative: the word "block" or "item" must appear in "short_prompt" exactly as given.
- Describe only what is visible: colour, pattern, surface, shape, symmetry, transparency, emissive. Infer colour from meaning when needed (lava -> red) but prefer what the images show.
- Never add materials, objects, creatures, characters, brands or lore the label does not imply. Use "unknown" instead of guessing.
- Never use "pixel art", "minecraft", "texture", "sprite" or a resolution.
- Lowercase, no trailing punctuation. Return valid JSON only."""


@dataclass
class Stage3Config:
    model: str = ""
    backend: str = "http"
    endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    endpoints: list = field(default_factory=list)
    api_key: str = "EMPTY"
    served_model_name: str = ""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 320
    seed: int = 0
    chat_template_kwargs: dict = field(default_factory=dict)
    response_format: dict = field(default_factory=dict)

    target: int = 512
    repeat: int = 4
    max_retries: int = 3
    retry_backoff: float = 2.0
    timeout: float = 180.0
    concurrency: int = 64
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_dict(cls, d):
        cfg = dict(d.get("annotator", d))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in known})

    @classmethod
    def from_yaml(cls, path):
        from config import load_yaml

        return cls.from_dict(load_yaml(path))


def build_messages(cfg, record, tokens, type_word):
    path = None
    image = load_record_image(record, root=None)
    views = build_views(image, target=cfg.target, repeat=cfg.repeat, mode="nearest",
                        views=["single", "tiled"])
    label = " ".join(tokens)
    text = (
        f"label words (all must appear in name and short_prompt): {label}\n"
        f"asset type (must appear literally in short_prompt): {type_word}\n\n"
        "Return ONE JSON object with exactly these keys:\n"
        f"{_SCHEMA}"
    )
    content = [{"type": "image", "image": img} for _, img in views]
    content.append({"type": "text", "text": text})
    return [{"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": content}]


def parse_annotation(raw):
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        raise ValueError("annotation must be a JSON object")
    out = {}
    for key in STAGE3_FIELDS:
        val = obj.get(key)
        if key in ("state", "dominant_colors"):
            if isinstance(val, str):
                val = [val] if val else []
            elif not isinstance(val, (list, tuple)):
                val = []
            out[key] = [str(v).strip().lower() for v in val if str(v).strip()]
        elif key == "emissive":
            out[key] = bool(val) if not isinstance(val, str) else val.lower() in ("true", "yes", "1")
        elif key == "uncertainty":
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                out[key] = 0.0
        else:
            out[key] = "" if val is None else str(val).strip()
    return out


async def annotate_one(backend, cfg, record):
    started = time.time()
    tokens = record["tokens"]
    type_word = record["type_word"]
    result = {"index": record["index"], "label": " ".join(tokens),
              "type": type_word, "annotation": None, "raw": None,
              "error": None, "latency_s": None}
    messages = build_messages(cfg, record, tokens, type_word)
    last = None
    for attempt in range(cfg.max_retries + 1):
        try:
            raw = await backend.generate(messages)
            ann = parse_annotation(raw)
            if not validate_prompt(ann.get("short_prompt", ""), tokens, type_word):
                raise ValueError(f"short_prompt missing label/type words: {ann.get('short_prompt')!r}")
            if not all(t in ann["name"].lower() for t in tokens):
                ann["name"] = (ann["name"] + " " + " ".join(tokens)).strip()
            result["annotation"] = ann
            result["raw"] = raw
            last = None
            break
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if attempt < cfg.max_retries:
                await asyncio.sleep(cfg.retry_backoff ** attempt)
    if result["annotation"] is None:
        result["annotation"] = {"short_prompt": fallback_prompt(tokens, type_word),
                                "name": " ".join(tokens), "fallback": True}
    result["error"] = last
    result["latency_s"] = round(time.time() - started, 3)
    return result


def read_done(path):
    done = set()
    p = Path(path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("index") is not None and r.get("annotation"):
                done.add(r["index"])
    return done


async def run_stage3(cfg, records, out_path, resume=True, num_shards=1, shard_index=0,
                     log_file="", progress_every=50):
    log = _Tee(log_file)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out_path) if resume else set()
    pending = [r for r in records if r["index"] not in done]
    if num_shards > 1:
        pending = [r for i, r in enumerate(pending) if i % num_shards == shard_index]
    log(f"records={len(records)} pending={len(pending)} shard={shard_index}/{num_shards} model={cfg.model}")
    if not pending:
        log("nothing to do"); log.close(); return {"pending": 0, "written": 0}

    from dataclasses import replace
    endpoints = list(cfg.endpoints or []) or [cfg.endpoint]
    backends = [HttpBackend(replace(cfg, endpoint=ep)) for ep in endpoints]
    for b in backends:
        await b.astart()
    stats = {"written": 0}

    def is_ok(r):
        return r.get("annotation") is not None

    def write_result(r):
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["written"] += 1

    async def worker(backend, record):
        return await annotate_one(backend, cfg, record)

    await run_queued(pending, backends, cfg.concurrency, worker, is_ok, write_result,
                     log, progress_every=progress_every)
    log(f"done written={stats['written']} -> {out_path}")
    log.close()
    return {"pending": len(pending), "written": stats["written"]}
