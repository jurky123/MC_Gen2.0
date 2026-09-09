import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from config import ModelConfig, TrainConfig, load_yaml
from data.embed_text import hash_text_embed, siglip2_embed
from data.build_mmap import write_text_mmap


def precompute(config_path, out_dir, encoder="", col="weak_prompt", split="train"):
    data = load_yaml(config_path).get("dataset", {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_dim = int(data.get("text_dim", 768))
    max_tokens = int(data.get("max_text_tokens", 64))

    import pandas as pd

    meta = pd.read_parquet(data["metadata"])
    texts = meta[col].fillna("").tolist() if col in meta.columns else [""] * len(meta)
    if encoder:
        embs = siglip2_embed(texts, encoder, max_tokens=max_tokens, token_dim=text_dim)
    else:
        embs = hash_text_embed(texts, dim=text_dim, max_tokens=max_tokens)
    path = out_dir / "text_embeddings.f32.mmap"
    write_text_mmap(embs, path)
    print(f"wrote text embeddings {embs.shape} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/stage_c.yaml")
    ap.add_argument("--out", default="")
    ap.add_argument("--encoder", default="", help="SigLIP2 HF model name; empty = offline hash embedder")
    ap.add_argument("--col", default="weak_prompt")
    args = ap.parse_args()
    out = args.out or str(Path(args.config).resolve().parent.parent / "data" / "build" / "mc_c")
    precompute(args.config, out, encoder=args.encoder, col=args.col)


if __name__ == "__main__":
    main()