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
    views_parquet="",
    views=0,
    num_shards=0,
    shard_index=0,
    init=False,
    devices=None,
):
    data = load_yaml(config_path).get("dataset", {})
    ecfg = load_encoder_cfg(encoder_cfg)

    encoder_type = encoder_type or ecfg.get("type", "hash")
    encoder = encoder or ecfg.get("name", "")
    instruction = ecfg.get("instruction", "")
    text_dim = int(data.get("text_dim", ecfg.get("text_dim", 768)))
    max_tokens = int(data.get("max_text_tokens", ecfg.get("max_text_tokens", 64)))
    device = device or ecfg.get("device", "cuda")
    devices = [d for d in (devices or []) if d]
    if devices:
        device = devices[0]
    dtype = dtype or ecfg.get("dtype", "float16")
    batch_size = int(batch_size or ecfg.get("batch_size", 16))
    normalize = bool(ecfg.get("normalize", True))
    revision = ecfg.get("revision") or None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    path = out_dir / "text_embeddings.f32.mmap"

    if views_parquet:
        # Multi-view conditioning: <views_parquet> has prompt_0..prompt_{K-1}
        # columns aligned to the dataset rows. Write a headerless
        # (N, K, text_dim) fp32 mmap; MmapImageTextDataset(text_views=K) picks
        # one view at random per step.
        #
        # Multi-GPU: run --init once, then N workers with --num-shards N
        # --shard-index i --device cuda:i; each writes disjoint rows (i::N) of
        # the same preallocated mmap (mode "r+").
        vdf = pd.read_parquet(views_parquet)
        prompt_cols = sorted(
            (c for c in vdf.columns if c.startswith("prompt_")),
            key=lambda c: int(c.split("_")[1]),
        )
        k = int(views or len(prompt_cols))
        prompt_cols = prompt_cols[:k]
        if len(prompt_cols) != k:
            raise SystemExit(f"requested {k} views but found {len(prompt_cols)} prompt_* columns")
        n = len(vdf)
        if init:
            np.memmap(path, dtype=np.float32, mode="w+", shape=(n, k, text_dim)).flush()
            print(f"initialized {path} shape ({n}, {k}, {text_dim})")
            return
        sharded = num_shards and num_shards > 1
        progress_path = Path(str(path) + ".progress")
        start_pos = 0
        if not sharded and progress_path.exists():
            try:
                start_pos = int(progress_path.read_text().strip())
            except ValueError:
                start_pos = 0
        resume = start_pos > 0 and path.exists()
        mode = "r+" if (sharded or resume) else "w+"
        out = np.memmap(path, dtype=np.float32, mode=mode, shape=(n, k, text_dim))
        texts_df = vdf[prompt_cols].fillna("").astype(str)
        row_ids = (list(range(shard_index, n, num_shards)) if sharded else list(range(n)))
        chunk = int(ecfg.get("chunk_rows", 2048))
        total = len(row_ids)
        print(f"encoding {total} rows x {k} views (shard {shard_index}/{num_shards or 1}, "
              f"resume_from={start_pos}) type={encoder_type} "
              f"model={encoder or '(offline hash)'} dim={text_dim} device={device} devices={devices or [device]}")
        for start in range(start_pos, total, chunk):
            block = row_ids[start:start + chunk]
            flat = texts_df.iloc[block].to_numpy().reshape(-1).tolist()
            embs = encode_texts(
                flat, encoder_type=encoder_type, model_name=encoder,
                instruction=instruction, text_dim=text_dim, max_tokens=max_tokens,
                device=device, devices=devices or None, dtype=dtype, batch_size=batch_size,
                normalize=normalize, revision=revision,
            )
            out[block] = np.asarray(embs, dtype=np.float32).reshape(len(block), k, text_dim)
            out.flush()
            if not sharded:
                progress_path.write_text(str(min(start + chunk, total)))
            if (start // chunk) % 20 == 0:
                print(f"  {min(start + chunk, total)}/{total}", flush=True)
        if not sharded and progress_path.exists():
            progress_path.unlink()
        print(f"wrote text embeddings ({n}, {k}, {text_dim}) shard {shard_index} -> {path}")
        return

    meta = pd.read_parquet(data["metadata"])
    texts = meta[col].fillna("").tolist() if col in meta.columns else [""] * len(meta)
    n = len(texts)
    if init:
        np.memmap(path, dtype=np.float32, mode="w+", shape=(n, 1, text_dim)).flush()
        print(f"initialized {path} shape ({n}, 1, {text_dim})")
        return
    sharded = num_shards and num_shards > 1
    mode = "r+" if sharded else "w+"
    out = np.memmap(path, dtype=np.float32, mode=mode, shape=(n, 1, text_dim))
    row_ids = (list(range(shard_index, n, num_shards)) if sharded else list(range(n)))
    chunk = int(ecfg.get("chunk_rows", 2048))
    print(
        f"encoding {len(row_ids)} texts (shard {shard_index}/{num_shards or 1}) type={encoder_type} "
        f"model={encoder or '(offline hash)'} dim={text_dim} tokens={max_tokens} device={device}"
    )
    for start in range(0, len(row_ids), chunk):
        block = row_ids[start:start + chunk]
        embs = encode_texts(
            [texts[i] for i in block],
            encoder_type=encoder_type,
            model_name=encoder,
            instruction=instruction,
            text_dim=text_dim,
            max_tokens=max_tokens,
            device=device,
            devices=devices or None,
            dtype=dtype,
            batch_size=batch_size,
            normalize=normalize,
            revision=revision,
        )
        out[block] = np.asarray(embs, dtype=np.float32).reshape(len(block), 1, text_dim)
        out.flush()
        if (start // chunk) % 20 == 0:
            print(f"  {min(start + chunk, len(row_ids))}/{len(row_ids)}", flush=True)
    print(f"wrote text embeddings ({n}, 1, {text_dim}) shard {shard_index} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/stage_c.yaml")
    ap.add_argument("--out", default="")
    ap.add_argument("--encoder", default="", help="HF model name; empty = use text_encoder.yaml")
    ap.add_argument("--encoder-type", default="", help="hash | siglip2 | qwen3vl")
    ap.add_argument("--encoder-config", default="", help="path to text_encoder.yaml")
    ap.add_argument("--col", default="weak_prompt")
    ap.add_argument("--views-parquet", default="",
                    help="parquet with prompt_0..prompt_{K-1} columns for K-view conditioning")
    ap.add_argument("--views", type=int, default=0, help="K views to encode (0 = all prompt_* columns)")
    ap.add_argument("--num-shards", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--init", action="store_true",
                    help="preallocate the output mmap and exit (for multi-process sharding)")
    ap.add_argument("--device", default="")
    ap.add_argument("--devices", default="",
                    help="comma list e.g. cuda:0,cuda:1 -> dynamic multi-GPU pool")
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
        views_parquet=args.views_parquet,
        views=args.views,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        init=args.init,
        devices=[d.strip() for d in args.devices.split(",") if d.strip()],
    )


if __name__ == "__main__":
    main()
