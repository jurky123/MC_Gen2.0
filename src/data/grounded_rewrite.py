"""Grounded label -> prompt rewrite (text-only, no image).

Turns a source filename label into ONE natural prompt under strict rules:
every label word (including orientation and abstract words) must be kept, the
block/item type must appear, and colour/shape may only be inferred from the
label's own meaning (``lava`` -> red, unless the label says ``green lava``).

A post-hoc word-boundary check enforces the rules; violations fall back to a
deterministic concatenation so no row is ever left without a valid prompt.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from data.filename_prompts import fallback_prompt, validate_prompt
from data.vlm_caption import HttpBackend, _Tee, extract_json
from data.work_queue import run_queued

DEFAULT_SYSTEM_PROMPT = """You turn a Minecraft asset's source label into ONE short, natural English prompt.

Hard rules:
- Keep EVERY label word exactly, unchanged. Orientation words (top, side, front,
  back, inner, outer, ...) and abstract words (lava, mossy, ethereal, crystal, ...)
  must all appear.
- The given asset type (block or item) must appear literally in the prompt.
- You may add colour and shape/pattern words that follow naturally from the
  label's meaning (lava -> red/orange molten; mossy -> green; iron -> grey metal;
  copper -> orange brown). If the label itself states a colour (e.g. "green lava"),
  that explicit colour must win.
- Never add materials, objects, creatures, characters, brands or lore that the
  label does not imply.
- Never use "pixel art", "minecraft", "texture", resolution or boilerplate.
- Lowercase, no trailing punctuation, at most 18 words.

Return valid JSON only: {"prompt": "..."}"""


@dataclass
class GroundedRewriteConfig:
    model: str = ""
    backend: str = "http"
    endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    endpoints: list = field(default_factory=list)
    api_key: str = "EMPTY"
    served_model_name: str = ""

    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 96
    seed: int = 0
    chat_template_kwargs: dict = field(default_factory=dict)
    response_format: dict = field(default_factory=dict)

    max_retries: int = 3
    retry_backoff: float = 2.0
    timeout: float = 120.0
    concurrency: int = 256
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_dict(cls, d):
        cfg = dict(d.get("rewriter", d))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in known})

    @classmethod
    def from_yaml(cls, path):
        from config import load_yaml

        return cls.from_dict(load_yaml(path))


def build_messages(cfg, tokens, type_word):
    label = " ".join(tokens)
    user = (
        "Rewrite the label into ONE prompt under the system rules.\n\n"
        f"label words (all must appear verbatim): {label}\n"
        f"asset type (must appear literally): {type_word}\n\n"
        'Return JSON: {"prompt": "..."}'
    )
    return [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": user},
    ]


def parse_prompt(raw):
    obj = extract_json(raw)
    return str(obj.get("prompt") or obj.get("text") or "").strip()


async def rewrite_one(backend, cfg, record, tokens, type_word):
    started = time.time()
    result = {
        "source": record.get("source"),
        "index": record.get("index"),
        "label": " ".join(tokens),
        "type": type_word,
        "prompt": None,
        "valid": False,
        "raw": None,
        "error": None,
        "latency_s": None,
    }
    messages = build_messages(cfg, tokens, type_word)
    last_error = None
    for attempt in range(cfg.max_retries + 1):
        try:
            raw = await backend.generate(messages)
            prompt = parse_prompt(raw)
            if validate_prompt(prompt, tokens, type_word):
                result["prompt"] = prompt
                result["valid"] = True
                last_error = None
                break
            last_error = f"validation failed: {prompt!r}"
            if not result.get("raw"):
                result["raw"] = raw
            if attempt < cfg.max_retries:
                await asyncio.sleep(cfg.retry_backoff ** attempt)
                continue
            result["prompt"] = fallback_prompt(tokens, type_word)
            last_error = None
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < cfg.max_retries:
                await asyncio.sleep(cfg.retry_backoff ** attempt)
    if result["prompt"] is None:
        result["prompt"] = fallback_prompt(tokens, type_word)
    result["error"] = last_error
    result["latency_s"] = round(time.time() - started, 3)
    return result


def read_done_indices(out_path):
    done = set()
    p = Path(out_path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("index") is not None and r.get("prompt"):
                done.add(r["index"])
    return done


async def run_grounded_rewrite(cfg, records, out_path, limit=0, resume=True,
                               dry_run=False, num_shards=1, shard_index=0,
                               log_file="", progress_every=100):
    log = _Tee(log_file)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = read_done_indices(out_path) if resume else set()
    pending = [r for r in records if r.get("index") not in done]
    if limit:
        pending = pending[:limit]
    if num_shards > 1:
        pending = [r for i, r in enumerate(pending) if i % num_shards == shard_index]

    log(f"records={len(records)} already_done={len(records) - len(pending)} pending={len(pending)} "
        f"shard={shard_index}/{num_shards} model={cfg.model}")
    if not pending:
        log("nothing to do")
        log.close()
        return {"pending": 0, "written": 0, "valid": 0}
    if dry_run:
        r = pending[0]
        for m in build_messages(cfg, r["tokens"], r["type_word"]):
            log(f"[{m['role']}]\n{m['content']}")
        log(f"[dry-run] {len(pending)} pending")
        log.close()
        return {"pending": len(pending), "written": 0, "valid": 0}

    endpoints = list(cfg.endpoints or []) or [cfg.endpoint]
    from dataclasses import replace
    backends = [HttpBackend(replace(cfg, endpoint=ep)) for ep in endpoints]
    for backend in backends:
        await backend.astart()

    stats = {"written": 0, "valid": 0}
    lock = asyncio.Lock()

    def is_ok(result):
        return bool(result.get("valid")) and not result.get("error")

    def write_result(result):
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        stats["written"] += 1
        if result.get("valid"):
            stats["valid"] += 1

    async def worker(backend, record):
        return await rewrite_one(backend, cfg, record, record["tokens"], record["type_word"])

    summary = await run_queued(pending, backends, cfg.concurrency, worker, is_ok,
                               write_result, log, progress_every=progress_every)
    log(f"done written={stats['written']} valid={stats['valid']} "
        f"fallback_or_failed={stats['written'] - stats['valid']} -> {out_path}")
    log.close()
    return {"pending": len(pending), "written": stats["written"], "valid": stats["valid"]}
