"""Local VLM fine-annotation for MC / pixel-art textures.

This module annotates individual texture tiles with structured JSON captions
using a local vision-language model. It implements plan sections 19-29:

* two nearest-neighbour views (single upscaled + 4x4 tiled), no bilinear;
* supplied metadata is treated as ground truth and never guessed;
* a fixed system prompt, ``temperature=0`` and JSON-only output;
* batched, resumable, concurrency-limited annotation over a JSONL manifest.

The annotator is *not* the frozen text encoder used for prompt conditioning
(``data.embed_text`` / ``configs/text_encoder.yaml``). It only writes captions.

Backends:
    http         OpenAI-compatible chat completions (vLLM; see scripts/serve_vlm.py)
    transformers in-process generation (single GPU, no server)
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from PIL import Image

from data.work_queue import run_queued

DEFAULT_SYSTEM_PROMPT = """You are annotating low-resolution voxel-game block textures.

Use supplied metadata as ground truth when available.
Describe only visually observable properties.
Do not invent a Minecraft item or block name.
If the material cannot be determined, output "unknown".
Do not infer gameplay function, lore, rarity or provenance.
Preserve uncertainty.
Return valid JSON only.
"""

# Background used when flattening RGBA tiles for annotation views / VLM input.
# White (not black) so dropped alpha is not described as part of the texture.
ANNOTATION_BG = (255, 255, 255, 255)

_ANNOTATION_KEYS = [
    "material", "form", "state", "dominant_colors", "pattern", "surface",
    "directionality", "details", "emissive", "tileability",
    "short_caption", "detailed_caption", "uncertainty",
]

_LIST_KEYS = {"state", "dominant_colors", "details"}

# JSON example value per field, used to render the requested schema.
_FIELD_SPEC = {
    "material": "string",
    "form": "string",
    "state": ["string"],
    "dominant_colors": ["string"],
    "pattern": "string",
    "surface": "string",
    "directionality": "string",
    "details": ["string"],
    "emissive": False,
    "tileability": "string",
    "short_caption": "string",
    "detailed_caption": "string",
    "uncertainty": 0.0,
}


def _normalize_local_path(path) -> Path:
    """Manifests built on Windows may store ``data\\processed\\...``."""
    return Path(str(path).replace("\\", "/"))


def resolve_image_path(record, root=None):
    path = record.get("path") or record.get("image") or record.get("file")
    if not path:
        return None
    p = _normalize_local_path(path)
    if not p.is_absolute() and root is not None:
        p = Path(root) / p
    return p


def load_image(path) -> Image.Image:
    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, ANNOTATION_BG)
        background.alpha_composite(image)
        image = background
    return image.convert("RGB")


def record_has_mmap(record) -> bool:
    return bool(record.get("mmap")) and "index" in record


def image_ref(record, root=None):
    """Human-readable image identifier for the output record."""
    if record_has_mmap(record):
        return f"{record['mmap']}#{record['index']}"
    path = resolve_image_path(record, root=root)
    return str(path) if path else None


def load_record_image(record, root=None) -> Image.Image:
    """Load a tile from either an individual PNG or a uint8 ``(N,H,W,C)`` mmap.

    Large built datasets (Stage 1/2) are headerless mmaps; annotating them
    directly avoids materialising millions of small PNGs.
    """
    if record_has_mmap(record):
        import numpy as np

        path = _normalize_local_path(record["mmap"])
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        shape = tuple(int(v) for v in record["shape"])
        index = int(record["index"])
        element = int(np.prod(shape))
        count = path.stat().st_size // element
        if index < 0 or index >= count:
            raise IndexError(f"index {index} out of range for {count} tiles")
        array = np.memmap(path, dtype=np.uint8, mode="r", shape=(count,) + shape)
        tile = np.asarray(array[index])
        mode = "RGBA" if shape[-1] == 4 else "RGB"
        image = Image.fromarray(tile, mode=mode)
        if image.mode == "RGBA":
            # Composite transparent pixels onto a neutral background so the VLM
            # does not mistake dropped alpha (black) for part of the texture.
            background = Image.new("RGBA", image.size, ANNOTATION_BG)
            background.alpha_composite(image)
            image = background
        return image.convert("RGB")
    path = resolve_image_path(record, root=root)
    return load_image(path)


def build_views(image: Image.Image, target: int = 512, repeat: int = 4,
                mode: str = "nearest", views=("single", "tiled")):
    """Plan section 23: nearest-neighbour only; never bilinear/bicubic."""
    if mode != "nearest":
        raise ValueError(f"only nearest-neighbour views are allowed, got {mode!r}")
    resample = Image.Resampling.NEAREST
    out = []
    if "single" in views:
        out.append(("single", image.resize((target, target), resample)))
    if "tiled" in views:
        tiled = Image.new("RGB", (image.width * repeat, image.height * repeat))
        for row in range(repeat):
            for col in range(repeat):
                tiled.paste(image, (col * image.width, row * image.height))
        out.append(("tiled", tiled.resize((target, target), resample)))
    return out


def image_to_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


_META_ORDER = [
    "block_id", "display_name", "namespace", "type", "texture_name",
    "texture_size", "primary_colors", "secondary_colors",
    "pattern_description", "overall_texture_description", "weak_prompt",
    "material", "form", "state",
]


def format_metadata(record, image_path=None) -> str:
    """Render known metadata as ground-truth bullet lines (plan section 24)."""
    meta = dict(record.get("metadata") or {})
    for key in ("namespace", "block_id", "display_name", "weak_prompt",
                "material", "form", "state"):
        value = record.get(key)
        if value not in (None, "", [], {}):
            meta.setdefault(key, value)

    lines = []
    if image_path:
        name = Path(image_path).name
    else:
        name = meta.get("file_name") or ""
    if name:
        lines.append(f"- filename: {name}")

    seen = set()
    for key in _META_ORDER:
        value = meta.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- {key}: {value}")
        seen.add(key)

    for key, value in meta.items():
        if key in seen or key == "filename":
            continue
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def reference_label(record):
    """Best available source-derived weak label (filename / weak_prompt / text)."""
    meta = dict(record.get("metadata") or {})
    for key in ("weak_prompt", "reference_label", "text"):
        value = record.get(key) or meta.get(key)
        if value:
            return str(value).strip()
    name = meta.get("file_name") or record.get("file_name")
    if name:
        stem = Path(str(name)).stem
        return re.sub(r"[_\-]+", " ", stem).strip()
    return ""


def _view_notes(view_names) -> str:
    notes = []
    for i, name in enumerate(view_names):
        label = chr(ord("A") + i)
        if name == "single":
            notes.append(f"View {label}: the texture upscaled with nearest-neighbour.")
        elif name == "tiled":
            notes.append(
                f"View {label}: a 4x4 tiling of the texture upscaled with "
                "nearest-neighbour; use it to judge pattern, seams, orientation "
                "and repetition."
            )
        else:
            notes.append(f"View {label}: {name}.")
    return "\n".join(notes)


def build_prompt_prefix(task_description, view_names, output_fields=None) -> str:
    """Invariant prompt prefix (shared by every request -> prefix-cacheable)."""
    fields = output_fields or _ANNOTATION_KEYS
    schema = json.dumps({field: _FIELD_SPEC.get(field, "string") for field in fields},
                        ensure_ascii=False)
    parts = []
    if task_description:
        parts.append(task_description.strip())
    parts.append("Annotate the texture in the provided images.")
    parts.append(_view_notes(view_names))
    parts.append("Return ONLY a single JSON object with exactly these keys:\n" + schema)
    return "\n\n".join(parts)


def build_prompt_suffix(record, image_path) -> str:
    """Per-sample tail: known metadata + weak reference label (varies)."""
    metadata = format_metadata(record, image_path) or "- filename: (unknown)"
    reference = reference_label(record)
    parts = ["Known metadata (ground truth, do not contradict):\n" + metadata]
    if reference:
        parts.append(
            "Reference label extracted from the source filename/metadata "
            "(weak, possibly noisy or incomplete -- use it as a hint, but trust "
            "the image and correct it if it disagrees):\n"
            f"- {reference}"
        )
    return "\n\n".join(parts)


def build_user_text(record, image_path, view_names, task_description="",
                    output_fields=None) -> str:
    """Combined text (metadata first); used by dry-run/tests."""
    return "\n\n".join([
        build_prompt_prefix(task_description, view_names, output_fields),
        build_prompt_suffix(record, image_path),
    ])


def build_messages(record, image_path, views, system_prompt, task_description="",
                   output_fields=None):
    """Return chat messages.

    Layout: invariant text prefix first, then the images, then the per-sample
    metadata. Keeping every per-sample token after the images lets vLLM's prefix
    cache reuse the system + task + schema prefix across requests.
    """
    view_names = [name for name, _ in views]
    prefix = build_prompt_prefix(task_description, view_names, output_fields)
    content = [{"type": "text", "text": prefix}]
    content += [{"type": "image", "image": img} for _, img in views]
    content.append({"type": "text", "text": build_prompt_suffix(record, image_path)})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _missing_closers(text):
    """Return the closing brackets needed to balance ``text``, or None if the
    text ends inside an unterminated string (unrepairable by closing alone)."""
    stack = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if stack:
                stack.pop()
    if in_string:
        return None
    return "".join(reversed(stack))


def repair_json(text):
    """Best-effort recovery of JSON truncated by ``max_tokens``.

    Chops trailing partial content until the prefix parses once its open
    brackets are closed, so a cut-off response still yields the fields that were
    emitted before the limit."""
    start = text.find("{")
    if start == -1:
        return None
    candidate = text[start:].rstrip()
    for cut in range(len(candidate), 0, -1):
        prefix = candidate[:cut].rstrip()
        closers = _missing_closers(prefix)
        if closers is None:
            continue
        try:
            return json.loads(prefix + closers)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def extract_json(text: str) -> dict:
    if text is None:
        raise ValueError("empty VLM response")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass
    repaired = repair_json(cleaned)
    if repaired is not None:
        return repaired
    raise ValueError(f"no JSON object found in response: {text[:200]!r}")


def normalize_annotation(raw: dict) -> dict:
    """Coerce a parsed object into the stable annotation schema (plan section 25)."""
    if not isinstance(raw, dict):
        raise ValueError("annotation must be a JSON object")
    out = {}
    for key in _ANNOTATION_KEYS:
        value = raw.get(key)
        if key in _LIST_KEYS:
            if value is None:
                value = []
            elif isinstance(value, str):
                value = [value] if value else []
            elif not isinstance(value, (list, tuple)):
                value = [str(value)]
            out[key] = [str(v).strip() for v in value if str(v).strip()]
        elif key == "emissive":
            if isinstance(value, str):
                out[key] = value.strip().lower() in ("true", "yes", "1")
            else:
                out[key] = bool(value)
        elif key == "uncertainty":
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = 0.0
        else:
            out[key] = "" if value is None else str(value).strip()
    for key in ("short_caption",):
        if not out[key] and out.get("material"):
            out[key] = " ".join(
                part for part in [out.get("state", [None])[0] if out.get("state") else None,
                                  out.get("form"), out.get("material")] if part
            )
    return out


@dataclass
class AnnotatorConfig:
    model: str = "Qwen/Qwen3.8-VL-27B"
    revision: str = ""
    backend: str = "http"
    endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    # Extra endpoints for data-parallel shared-queue mode (one per GPU).
    endpoints: list = field(default_factory=list)
    api_key: str = "EMPTY"
    served_model_name: str = ""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: int = 0
    # Passed to the chat template, e.g. {"enable_thinking": False} for Qwen3.x.
    chat_template_kwargs: dict = field(default_factory=dict)
    # Optional OpenAI response_format, e.g. {"type": "json_object"} for vLLM
    # guided decoding (greatly reduces malformed JSON on bulk runs).
    response_format: dict = field(default_factory=dict)
    # Which annotation fields the model must emit (shorter = faster decoding).
    output_fields: list = field(default_factory=lambda: list(_ANNOTATION_KEYS))

    image: dict = field(default_factory=lambda: {
        "target": 512, "repeat": 4, "mode": "nearest", "views": ["single", "tiled"],
    })

    max_retries: int = 3
    retry_backoff: float = 2.0
    timeout: float = 120.0
    concurrency: int = 8
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    task_description: str = ""
    profile: str = ""
    profiles: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        cfg = dict(d.get("annotator", d))
        profiles = cfg.get("profiles") or {}
        profile = cfg.get("profile") or ""
        overrides = profiles.get(profile) if profile else None
        if overrides:
            nested = dict(overrides)
            if isinstance(nested.get("image"), dict):
                nested["image"] = {**(cfg.get("image") or {}), **nested["image"]}
            cfg = {**cfg, **nested}
        known = {f for f in cls.__dataclass_fields__}
        obj = cls(**{k: v for k, v in cfg.items() if k in known})
        obj.profiles = profiles
        obj.profile = profile
        return obj

    @classmethod
    def from_yaml(cls, path):
        from config import load_yaml

        return cls.from_dict(load_yaml(path))


class HttpBackend:
    """OpenAI-compatible `/v1/chat/completions` (vLLM server)."""

    name = "http"

    def __init__(self, cfg: AnnotatorConfig):
        self.cfg = cfg
        self.model = cfg.served_model_name or cfg.model
        self._client = None

    async def astart(self):
        import httpx

        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        self._client = httpx.AsyncClient(timeout=self.cfg.timeout, headers=headers)

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _to_openai_content(self, content):
        out = []
        for part in content:
            if part["type"] == "text":
                out.append({"type": "text", "text": part["text"]})
            else:
                out.append({"type": "image_url",
                            "image_url": {"url": image_to_data_uri(part["image"])}})
        return out

    async def generate(self, messages) -> str:
        payload_messages = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = await asyncio.to_thread(self._to_openai_content, content)
            payload_messages.append({"role": message["role"], "content": content})

        payload = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
            "seed": self.cfg.seed,
        }
        if self.cfg.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.cfg.chat_template_kwargs
        if self.cfg.response_format:
            payload["response_format"] = self.cfg.response_format
        response = await self._client.post(self.cfg.endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class TransformersBackend:
    """In-process Qwen-VL generation via `transformers` (no server)."""

    name = "transformers"

    def __init__(self, cfg: AnnotatorConfig, device="cuda", dtype="bfloat16"):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoModel
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoModel

        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if cfg.revision:
            kwargs["revision"] = cfg.revision
        self.processor = AutoProcessor.from_pretrained(
            cfg.model, revision=cfg.revision or None, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(cfg.model, **kwargs).to(device)
        self.model.eval()
        self.device = device
        self.torch = torch

    async def astart(self):
        return None

    async def aclose(self):
        return None

    def _generate_sync(self, messages) -> str:
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            **(self.cfg.chat_template_kwargs or {}),
        )
        inputs = inputs.to(self.device)
        do_sample = self.cfg.temperature > 0
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_tokens,
                do_sample=do_sample,
                temperature=self.cfg.temperature if do_sample else None,
                top_p=self.cfg.top_p if do_sample else None,
            )
        prompt_len = inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )
        return decoded[0]

    async def generate(self, messages) -> str:
        return await asyncio.to_thread(self._generate_sync, messages)


def make_backend(cfg: AnnotatorConfig, device="cuda", dtype="bfloat16"):
    backend = (cfg.backend or "http").lower()
    if backend in ("http", "openai", "vllm"):
        return HttpBackend(cfg)
    if backend in ("transformers", "hf", "local"):
        return TransformersBackend(cfg, device=device, dtype=dtype)
    raise ValueError(f"unknown annotator backend {cfg.backend!r}")


def load_manifest(path, limit=0, start=0):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if line_no < start:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def read_done_sources(out_path):
    done = set()
    out_path = Path(out_path)
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = record.get("source")
            if source and record.get("annotation"):
                done.add(source)
    return done


async def annotate_one(backend, cfg: AnnotatorConfig, record, root=None,
                       image_path=None) -> dict:
    started = time.time()
    path = Path(image_path) if image_path else resolve_image_path(record, root=root)
    result = {
        "source": record.get("source"),
        "image": image_ref(record, root=root),
        "sha256_rgba": record.get("sha256_rgba"),
        "model": cfg.model,
        "profile": cfg.profile,
        "backend": backend.name,
        "system_prompt": cfg.system_prompt,
        "annotation": None,
        "raw": None,
        "error": None,
        "latency_s": None,
    }
    try:
        if path:
            image = await asyncio.to_thread(load_image, path)
        else:
            image = await asyncio.to_thread(load_record_image, record, root)
        views = await asyncio.to_thread(
            build_views,
            image,
            target=int(cfg.image.get("target", 512)),
            repeat=int(cfg.image.get("repeat", 4)),
            mode=cfg.image.get("mode", "nearest"),
            views=cfg.image.get("views", ["single", "tiled"]),
        )
    except Exception as exc:  # image missing / unreadable
        result["error"] = f"image: {exc}"
        result["latency_s"] = round(time.time() - started, 3)
        return result

    messages = build_messages(record, path, views, cfg.system_prompt,
                              cfg.task_description, cfg.output_fields)
    last_error = None
    for attempt in range(cfg.max_retries + 1):
        try:
            raw = await backend.generate(messages)
            result["raw"] = raw
            result["annotation"] = normalize_annotation(extract_json(raw))
            last_error = None
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < cfg.max_retries:
                await asyncio.sleep(cfg.retry_backoff ** attempt)
    result["error"] = last_error
    result["latency_s"] = round(time.time() - started, 3)
    return result


class _Tee:
    """Timestamped progress to stdout and an optional append-only log file."""

    def __init__(self, path=None):
        self.handle = None
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("a", encoding="utf-8")

    def __call__(self, message=""):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        if self.handle is not None:
            self.handle.write(line + "\n")
            self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


async def run_annotation(cfg: AnnotatorConfig, manifest, out_path, root=None,
                         limit=0, start=0, resume=True, dry_run=False,
                         device="cuda", dtype="bfloat16", progress_every=25,
                         log_file="", num_shards=1, shard_index=0):
    log = _Tee(log_file)
    records = load_manifest(manifest, limit=limit, start=start)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = read_done_sources(out_path) if resume else set()
    pending_all = [r for r in records if r.get("source") not in done]
    already_done = len(records) - len(pending_all)
    pending = pending_all
    if num_shards > 1:
        pending = [r for i, r in enumerate(pending_all) if i % num_shards == shard_index]

    log(f"manifest={manifest} records={len(records)} already_done={already_done} "
        f"pending={len(pending)} shard={shard_index}/{num_shards} backend={cfg.backend} "
        f"profile={cfg.profile or '-'} model={cfg.model}")
    if not pending:
        log("nothing to do")
        log.close()
        return {"records": len(records), "annotated": 0, "failed": 0, "skipped": len(records)}

    if dry_run:
        record = pending[0]
        path = resolve_image_path(record, root=root)
        image = load_image(path) if path else load_record_image(record, root=root)
        views = build_views(image, target=int(cfg.image.get("target", 512)),
                            repeat=int(cfg.image.get("repeat", 4)),
                            mode=cfg.image.get("mode", "nearest"),
                            views=cfg.image.get("views", ["single", "tiled"]))
        messages = build_messages(record, path, views, cfg.system_prompt,
                                  cfg.task_description, cfg.output_fields)
        for message in messages:
            log(f"[{message['role']}]")
            content = message["content"]
            if isinstance(content, str):
                log(content)
                continue
            for part in content:
                if part["type"] == "text":
                    log(part["text"])
                else:
                    log(f"<image {part['image'].size}>")
        log(f"[dry-run] {len(pending)} pending records; no requests sent")
        log.close()
        return {"records": len(records), "annotated": 0, "failed": 0,
                "skipped": len(records) - len(pending)}

    endpoints = list(getattr(cfg, "endpoints", []) or []) or [cfg.endpoint]
    if len(endpoints) > 1 and (cfg.backend or "").lower() not in ("http", "openai", "vllm"):
        log(f"multi-endpoint requires the http backend; falling back to {endpoints[0]}")
        endpoints = endpoints[:1]
    log(f"endpoints={endpoints} concurrency/index={cfg.concurrency}/per-endpoint (shared queue)")
    backends = [make_backend(replace(cfg, endpoint=ep), device=device, dtype=dtype)
                for ep in endpoints]
    for backend in backends:
        await backend.astart()

    def is_ok(result):
        return result["annotation"] is not None and not result["error"]

    def write_result(result):
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        if is_ok(result):
            annotation = result["annotation"]
            preview = f"{annotation.get('material', '')} | {annotation.get('short_caption', '')}"
            log(f"ok   {result['source']} {result['latency_s']}s :: {preview[:80]}")
        else:
            log(f"FAIL {result['source']} :: {result['error']}")

    async def worker(backend, record):
        return await annotate_one(backend, cfg, record, root=root)

    started = time.time()
    stats = await run_queued(pending, backends, cfg.concurrency, worker, is_ok,
                             write_result, log, progress_every=progress_every)

    summary = {
        "records": len(records),
        "annotated": stats["ok"],
        "failed": stats["failed"],
        "skipped": len(records) - len(pending),
    }
    log(f"done annotated={summary['annotated']} failed={summary['failed']} "
        f"skipped={summary['skipped']} elapsed={(time.time() - started) / 60:.1f}min -> {out_path}")
    log.close()
    return summary
