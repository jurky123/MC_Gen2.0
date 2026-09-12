import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from config import load_yaml
from data.embed_text import encode_texts
from data.build_mmap import write_text_mmap

ENCODER_CFG = ROOT / "configs" / "text_encoder.yaml"


def load_encoder_cfg(path=""):
    path = Path(path) if path else ENCODER_CFG
    if not path.exists():
        return {}
    return load_yaml(path).get("text_encoder", {}) or {}


def precompute(
    config_path,
    out_dir,
    encoder="",
    col="weak_prompt",
    encoder_type="",
    encoder_cfg="",
    device="",
    dtype="",
    batch_size=0,
):
    data = load_yaml(config_path).get("dataset", {})
    ecfg = load_encoder_cfg(encoder_cfg)

    encoder_type = encoder_type or ecfg.get("type", "hash")
    encoder = encoder or ecfg.get("name", "")
    instruction = ecfg.get("instruction", "")
    text_dim = int(data.get("text_dim", ecfg.get("text_dim", 768)))
    max_tokens = int(data.get("max_text_tokens", ecfg.get("max_text_tokens", 64)))
    device = device or ecfg.get("device", "cuda")
    dtype = dtype or ecfg.get("dtype", "float16")
    batch_size = int(batch_size or ecfg.get("batch_size", 16))
    normalize = bool(ecfg.get("normalize", True))
    revision = ecfg.get("revision") or None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    meta = pd.read_parquet(data["metadata"])
    texts = meta[col].fillna("").tolist() if col in meta.columns else [""] * len(meta)
    print(
        f"encoding {len(texts)} texts with type={encoder_type} "
        f"model={encoder or '(offline hash)'} dim={text_dim} tokens={max_tokens} device={device}"
    )

    embs = encode_texts(
        texts,
        encoder_type=encoder_type,
        model_name=encoder,
        instruction=instruction,
        text_dim=text_dim,
        max_tokens=max_tokens,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        normalize=normalize,
        revision=revision,
    )
    embs = np.asarray(embs, dtype=np.float32)
    path = out_dir / "text_embeddings.f32.mmap"
    write_text_mmap(embs, path)
    print(f"wrote text embeddings {embs.shape} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/stage_c.yaml")
    ap.add_argument("--out", default="")
    ap.add_argument("--encoder", default="", help="HF model name; empty = use text_encoder.yaml")
    ap.add_argument("--encoder-type", default="", help="hash | siglip2 | qwen3vl")
    ap.add_argument("--encoder-config", default="", help="path to text_encoder.yaml")
    ap.add_argument("--col", default="weak_prompt")
    ap.add_argument("--device", default="")
    ap.add_argument("--dtype", default="")
    ap.add_argument("--batch-size", type=int, default=0)
    args = ap.parse_args()

    data = load_yaml(args.config).get("dataset", {})
    build = load_yaml(args.config).get("build", {})
    if args.out:
        out = args.out
    elif build.get("out_dir"):
        out = build["out_dir"]
    else:
        out = str(Path(args.config).resolve().parent.parent / "data" / "build" / "mc_c")

    precompute(
        args.config,
        out,
        encoder=args.encoder,
        col=args.col,
        encoder_type=args.encoder_type,
        encoder_cfg=args.encoder_config,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
