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