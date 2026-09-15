"""LLM-based prompt diversification for texture annotations.

The coarse/fine VLM annotation describes colours and shapes objectively and the
source labels are full of domain boilerplate ("pixel art", "minecraft",
"texture", "16x16"). Both are poor conditioning for the generator, which should
learn that pixel style is implicit and that inputs can be short, evocative,
game-item-style names.

This module takes the structured annotation + a cleaned source label and asks a
text LLM to write K prompts per texture at different abstraction levels
(factual / evocative / thematic / descriptive). Domain boilerplate is stripped
from every view as a safety net.

Output record:
    {"source": ..., "prompts": [{"style": "factual", "text": ...}, ...], ...}
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from data.vlm_caption import HttpBackend, _Tee, extract_json
from data.work_queue import run_queued

DEFAULT_STYLES = ["factual", "evocative", "thematic", "descriptive"]


@dataclass
class RewriterConfig:
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    backend: str = "http"
    endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    endpoints: list = field(default_factory=list)
    api_key: str = "EMPTY"
    served_model_name: str = ""

    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 260
    seed: int = 0
    chat_template_kwargs: dict = field(default_factory=dict)
    response_format: dict = field(default_factory=dict)

    styles: list = field(default_factory=lambda: list(DEFAULT_STYLES))
    strip_phrases: list = field(default_factory=list)

    max_retries: int = 3
    retry_backoff: float = 2.0
    timeout: float = 120.0
    concurrency: int = 256
    system_prompt: str = ""
    task_description: str = ""

    @classmethod
    def from_dict(cls, d):
        cfg = dict(d.get("rewriter", d))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in known})

    @classmethod
    def from_yaml(cls, path):
        from config import load_yaml

        return cls.from_dict(load_yaml(path))


def _compile_strip(phrases):
    compiled = []
    for phrase in phrases or []:
        phrase = str(phrase).strip()
        if phrase:
            compiled.append(re.compile(re.escape(phrase), re.IGNORECASE))
    return compiled


def strip_domain(text, compiled) -> str:
    value = str(text or "")
    for pattern in compiled:
        value = pattern.sub(" ", value)
    value = re.sub(r"[,;|/]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    for _ in range(2):
        value = re.sub(r"^(a|an|the|of)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(a|an|the|of|and|with|or|in|on)$", "",
                   value, flags=re.IGNORECASE)
    return value.strip(" .,;:-").strip()


def clean_view(text, compiled) -> str:
    return strip_domain(text, compiled).lower()


def source_label(record) -> str:
    """Best source-derived label: weak_prompt > text > filename stem."""
    meta = dict(record.get("metadata") or {})
    for key in ("weak_prompt", "text"):
        value = record.get(key) or meta.get(key)
        if value:
            return str(value)
    file_name = meta.get("file_name") or record.get("file_name")
    if file_name:
        stem = Path(str(file_name)).stem
        return re.sub(r"[_\-]+", " ", stem)
    return ""


def format_attributes(annotation, clean_ref: str) -> str:
    lines = []
    if clean_ref:
        lines.append(f"- source label (hint): {clean_ref}")
    for key, label in [
        ("material", "material"), ("form", "form"), ("state", "state"),
        ("dominant_colors", "dominant colors"), ("pattern", "pattern"),
        ("surface", "surface"), ("tileability", "tileability"),
    ]:
        value = annotation.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- {label}: {value}")
    if "emissive" in annotation:
        lines.append(f"- emissive: {'yes' if annotation.get('emissive') else 'no'}")
    for key, label in [("short_caption", "initial short caption"),
                       ("detailed_caption", "initial detailed caption")]:
        value = annotation.get(key)
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def build_rewrite_messages(cfg: RewriterConfig, annotation, clean_ref):
    body = format_attributes(annotation or {}, clean_ref)
    user = (
        f"{cfg.task_description.strip()}\n\n"
        f"Attributes:\n{body}\n\n"
        f"Return JSON only with keys: {', '.join(cfg.styles)}."
    )
    return [
        {"role": "system", "content": cfg.system_prompt.strip()},
        {"role": "user", "content": user},
    ]


def parse_views(raw, styles, compiled):
    obj = extract_json(raw)
    views = []
    seen = set()
    for style in styles:
        text = clean_view(obj.get(style, ""), compiled)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        views.append({"style": style, "text": text})
    return views


def pad_views(views, styles):
    if not views:
        return None
    by_style = {v["style"]: v for v in views}
    filled = []
    for style in styles:
        filled.append(by_style.get(style, {"style": style, "text": views[0]["text"]}))
    return filled


async def rewrite_one(backend, cfg: RewriterConfig, record, clean_ref, compiled) -> dict:
    started = time.time()
    annotation = record.get("annotation") or {}
    result = {
        "source": record.get("source"),
        "model": cfg.model,
        "styles": list(cfg.styles),
        "prompts": None,
        "raw": None,
        "error": None,
        "latency_s": None,
    }
    messages = build_rewrite_messages(cfg, annotation, clean_ref)
    last_error = None
    for attempt in range(cfg.max_retries + 1):
        try:
            raw = await backend.generate(messages)
            result["raw"] = raw
            result["prompts"] = pad_views(parse_views(raw, cfg.styles, compiled), cfg.styles)
            last_error = None
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < cfg.max_retries:
                await asyncio.sleep(cfg.retry_backoff ** attempt)
    result["error"] = last_error
    result["latency_s"] = round(time.time() - started, 3)
    return result


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
            if record.get("source") and record.get("prompts"):
                done.add(record["source"])
    return done


def load_refs(manifest_path):
    """source -> cleaned source label."""
    refs = {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            refs[record.get("source")] = source_label(record)
    return refs


def load_annotations(paths, done, limit=0):
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                source = record.get("source")
                if source in done or not record.get("annotation"):
                    continue
                records.append(record)
                if limit and len(records) >= limit:
                    return records
    return records


async def run_rewrite(cfg: RewriterConfig, annotation_paths, manifest, out_path,
                      limit=0, resume=True, dry_run=False, num_shards=1,
                      shard_index=0, log_file="", progress_every=50):
    log = _Tee(log_file)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compiled = _compile_strip(cfg.strip_phrases)

    done = read_done_sources(out_path) if resume else set()
    refs = load_refs(manifest)
    records = load_annotations(annotation_paths, done, limit=limit)
    if num_shards > 1:
        records = [r for i, r in enumerate(records) if i % num_shards == shard_index]

    log(f"annotations={[str(p) for p in annotation_paths]} refs={len(refs)} "
        f"pending={len(records)} shard={shard_index}/{num_shards} model={cfg.model}")
    if not records:
        log("nothing to do")
        log.close()
        return {"pending": 0, "rewritten": 0, "failed": 0}

    if dry_run:
        record = records[0]
        clean_ref = strip_domain(refs.get(record["source"], ""), compiled)
        messages = build_rewrite_messages(cfg, record.get("annotation") or {}, clean_ref)
        for message in messages:
            log(f"[{message['role']}]\n{message['content']}")
        log(f"[dry-run] {len(records)} pending; no requests sent")
        log.close()
        return {"pending": len(records), "rewritten": 0, "failed": 0}

    endpoints = list(getattr(cfg, "endpoints", []) or []) or [cfg.endpoint]
    log(f"endpoints={endpoints} concurrency/index={cfg.concurrency}/per-endpoint (shared queue)")
    backends = [HttpBackend(replace(cfg, endpoint=ep)) for ep in endpoints]
    for backend in backends:
        await backend.astart()

    def is_ok(result):
        return bool(result["prompts"]) and not result["error"]

    def write_result(result):
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        if is_ok(result):
            preview = " | ".join(p["text"] for p in result["prompts"])
            log(f"ok {result['source']} {result['latency_s']}s :: {preview[:110]}")
        else:
            log(f"FAIL {result['source']} :: {result['error']}")

    async def worker(backend, record):
        clean_ref = strip_domain(refs.get(record["source"], ""), compiled)
        return await rewrite_one(backend, cfg, record, clean_ref, compiled)

    started = time.time()
    stats = await run_queued(records, backends, cfg.concurrency, worker, is_ok,
                             write_result, log, progress_every=progress_every)
    summary = {"pending": len(records), "rewritten": stats["ok"], "failed": stats["failed"]}
    log(f"done rewritten={summary['rewritten']} failed={summary['failed']} "
        f"elapsed={(time.time() - started) / 60:.1f}min -> {out_path}")
    log.close()
    return summary
