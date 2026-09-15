import hashlib
import re

import numpy as np


def _ngrams(text, k=(1, 2, 3)):
    s = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not s:
        return ["<blank>"]
    grams = []
    words = s.split()
    grams.extend(words)
    for i in range(1, 3):
        grams.extend("_".join(words[j : j + i + 1]) for j in range(len(words) - i))
    for n in k:
        grams.extend(text.lower()[i : i + n] for i in range(len(text) - n + 1))
    return grams


def hash_text_embed(texts, dim=768, max_tokens=64, token_dim=768, seed=42):
    rng = np.random.RandomState(seed)
    proj = rng.randn(8 * 768).reshape(-1)
    offsets = rng.randint(0, 2**31 - 1, size=len(_ngrams("", (1,))))

    def embed_one(text):
        grams = _ngrams(text)
        vec = np.zeros(token_dim, dtype=np.float32)
        for g in grams:
            h = int(hashlib.sha256(g.encode()).hexdigest(), 16)
            idx = h % proj.shape[0]
            vec[idx % token_dim] += 1.0
            vec[(idx * 7 + h) % token_dim] += 0.5
        n = max(len(grams), 1)
        vec = vec / np.sqrt(n + 1e-6)
        vec = vec * 0.1
        out = np.zeros((max_tokens, token_dim), dtype=np.float32)
        out[0] = vec
        return out

    if isinstance(texts, str):
        texts = [texts]
    return np.stack([embed_one(t) for t in texts])


def siglip2_embed(texts, model_name, max_tokens=64, token_dim=768):
    from transformers import AutoTokenizer, AutoModel

    import torch

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    out = np.zeros((len(texts), max_tokens, token_dim), dtype=np.float32)
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = tok([t], padding="max_length", truncation=True, max_length=max_tokens, return_tensors="pt")
            hidden = model(**enc).last_hidden_state[0].numpy()
            out[i, : hidden.shape[0]] = hidden
    return out


_ENCODER_CACHE = {}
_POOL_CACHE = {}

_DTYPE_MAP = {"float16": "float16", "fp16": "float16", "bfloat16": "bfloat16", "bf16": "bfloat16",
              "float32": "float32", "fp32": "float32"}


def _normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def _as_tokens(embeddings, text_dim):
    """Return (N, 1, text_dim) from a pooled (N, D) matrix."""
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[None, :]
    if emb.shape[-1] != text_dim:
        raise ValueError(f"text encoder produced dim {emb.shape[-1]}, expected {text_dim}")
    return emb[:, None, :]


def qwen3vl_embed(
    texts,
    model_name="Qwen/Qwen3-VL-Embedding-2B",
    instruction="Represent the user's input.",
    text_dim=2048,
    max_tokens=1,
    device="cuda",
    devices=None,
    dtype="float16",
    batch_size=16,
    normalize=True,
    revision=None,
    trust_remote_code=True,
    max_length=8192,
    chunk_size=0,
):
    """Frozen Qwen3-VL-Embedding text encoder.

    Returns a single pooled, optionally L2-normalized token per text:
    shape ``(len(texts), 1, text_dim)`` (fp32). Prefers ``sentence-transformers``
    (which ships the pooler/normalizer modules); falls back to raw ``transformers``
    with last-token pooling.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = ["" if t is None else str(t) for t in texts]

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None

    if SentenceTransformer is not None:
        multi_device = len([d for d in (devices or []) if d]) > 1
        target_devices = [d for d in (devices or []) if d]
        load_device = "cpu" if multi_device else device
        key = ("st", model_name, revision, load_device, tuple(target_devices))
        model = _ENCODER_CACHE.get(key)
        if model is None:
            kwargs = {"device": load_device}
            if revision:
                kwargs["revision"] = revision
            if trust_remote_code:
                kwargs["trust_remote_code"] = True
            if dtype and dtype != "float32":
                kwargs["model_kwargs"] = {"torch_dtype": _DTYPE_MAP.get(dtype, dtype)}
            model = SentenceTransformer(model_name, **kwargs)
            model.eval()
            _ENCODER_CACHE[key] = model

        if multi_device:
            # Process pool + chunked map = dynamic work queue across GPUs: a
            # faster card pulls more chunks, so both finish together instead of
            # one idling after its static half is done.
            pool = _POOL_CACHE.get(key)
            if pool is None:
                pool = model.start_multi_process_pool(target_devices=target_devices)
                _POOL_CACHE[key] = pool
            emb = model.encode_multi_process(
                texts, pool,
                prompt=instruction or None,
                batch_size=batch_size,
                chunk_size=int(chunk_size) if chunk_size else max(batch_size * 16, 256),
                normalize_embeddings=bool(normalize),
                show_progress_bar=False,
            )
            return _as_tokens(np.asarray(emb, dtype=np.float32), text_dim)

        encode_kwargs = {
            "batch_size": batch_size,
            "normalize_embeddings": bool(normalize),
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if instruction:
            encode_kwargs["prompt"] = instruction
        emb = model.encode(texts, **encode_kwargs)
        emb = np.asarray(emb, dtype=np.float32)
        if normalize:
            emb = _normalize_rows(emb)
        return _as_tokens(emb, text_dim)

    import torch
    from transformers import AutoModel, AutoTokenizer

    key = ("hf", model_name, revision, device)
    cached = _ENCODER_CACHE.get(key)
    if cached is None:
        torch_dtype = getattr(torch, _DTYPE_MAP.get(dtype, "float16"), torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=trust_remote_code
        )
        model = AutoModel.from_pretrained(
            model_name, revision=revision, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
        ).to(device)
        model.eval()
        cached = (tokenizer, model)
        _ENCODER_CACHE[key] = cached
    tokenizer, model = cached

    if instruction and getattr(tokenizer, "chat_template", None):
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": t}]}],
            add_generation_prompt=False, tokenize=False,
        ) for t in texts]
    else:
        prompts = texts

    chunks = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"]
            idx = mask.sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
            chunks.append(pooled.float().cpu().numpy())
    emb = np.concatenate(chunks, axis=0)
    if normalize:
        emb = _normalize_rows(emb)
    return _as_tokens(emb, text_dim)


def encode_texts(
    texts,
    encoder_type="hash",
    model_name="",
    instruction="",
    text_dim=768,
    max_tokens=64,
    device="cuda",
    devices=None,
    dtype="float16",
    batch_size=16,
    normalize=True,
    revision=None,
    chunk_size=0,
):
    """Route to the configured offline text encoder.

    ``encoder_type`` is one of ``hash`` (deterministic placeholder), ``siglip2``
    (token-level hidden states) or ``qwen3vl`` (single pooled token).
    """
    encoder_type = (encoder_type or "hash").lower()
    if encoder_type in ("qwen", "qwen3vl", "qwen3-vl", "qwen3vl-embedding"):
        return qwen3vl_embed(
            texts, model_name=model_name or "Qwen/Qwen3-VL-Embedding-2B",
            instruction=instruction or "Represent the user's input.",
            text_dim=text_dim, max_tokens=max_tokens, device=device, devices=devices,
            dtype=dtype, batch_size=batch_size, normalize=normalize, revision=revision,
            chunk_size=chunk_size,
        )
    if encoder_type == "siglip2":
        return siglip2_embed(texts, model_name, max_tokens=max_tokens, token_dim=text_dim)
    return hash_text_embed(texts, dim=text_dim, max_tokens=max_tokens, token_dim=text_dim)