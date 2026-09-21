"""Build MC -> HD pairs with the validated recipe (the only MC2HD path now).

    real MC (32px)
      -> NEAREST upscale to `--size` (384 default)
      -> bilateral + gaussian (stair-step removal, low-frequency structure kept)
      -> FLUX.2-klein edit, 8 steps
         prompt = "<fine description, word 'block' removed>. photorealistic
                   high-end game asset, PBR, realistic lighting, fine detail,
                   keep object/orientation/proportions/silhouette/colors,
                   remove pixelation"
      -> HD image (the reference for the pair)

The MC side of the pair is the real texture itself, so the pair is
(ref_HD, prompt, real_MC) with real MC as supervision.

Resumable, atomic, provenance in manifest.jsonl + config.json fingerprint.

    python scripts/build_mchd_pairs.py --n 200 --size 384 --out pairs/mchd_v1
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from try_antialias_filters import bilateral, gauss  # noqa: E402

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"
FLUX_LICENSE = "Apache-2.0 (black-forest-labs/FLUX.2-klein-4B)"

REALISM = ("photorealistic high-end game asset, physically based materials, "
           "realistic lighting and shading, fine surface detail, crisp "
           "anti-aliased edges. Keep the same object, orientation, "
           "proportions, silhouette and colors. Remove all pixelation and "
           "voxel stair-step edges.")


def preprocess(mc_rgba, size):
    """32px MC -> smoothed `size` input (bil+gauss12 tuned at 512)."""
    pil = Image.fromarray(np.asarray(mc_rgba, dtype=np.uint8), "RGBA")
    bg = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil.convert("RGB"), mask=pil.split()[3])
    large = np.asarray(bg.resize((size, size), Image.Resampling.NEAREST))
    scale = size / 512.0
    d = max(3, int(round(25 * scale)) | 1)
    smoothed = gauss(bilateral(np.dstack([large, np.full((size, size), 255, np.uint8)]),
                               d=d, sc=60, ss=60), max(1.0, 12 * scale))
    return smoothed[..., :3].astype(np.uint8)


def clean_subject(text):
    t = re.sub(r"\bblocks?\b", " ", str(text), flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([,.])", r"\1", t)
    return t.strip(" ,.")


def stratified_rows(build, n, seed=0):
    """50/50 block/item, covering form/material/colour/state."""
    import pandas as pd
    from collections import Counter
    from data.filename_prompts import clean_tokens
    from scripts.select_stage3_subset import COLORS, FORMS, MATERIALS, STATES

    meta = pd.read_parquet(build / "metadata.parquet")
    s3 = json.loads((build / "stage3_splits.json").read_text())
    pool = sorted(set(s3["train"]) | set(s3["val"]))
    rng = np.random.RandomState(seed)
    rng.shuffle(pool)
    per_type = n // 2
    chosen, used = [], set()
    for want_type in ("block", "item"):
        cand = [i for i in pool if str(meta.at[i, "type"]) == want_type]
        # round-robin over (form, material) buckets, then fill
        buckets = {}
        for i in cand:
            # sorted(): set iteration over strings is not deterministic across
            # processes (hash randomisation), which previously made shards
            # overlap and leave ~23% of rows unprocessed.
            toks = sorted(set(clean_tokens(str(meta.at[i, "file_name"]), keep_parts=True,
                                           keep_anim=True, keep_generic=False)))
            form = next((w for w in toks if w in FORMS), "other")
            mat = next((w for w in toks if w in MATERIALS), "other")
            buckets.setdefault((form, mat), []).append(i)
        keys = sorted(buckets)
        rng.shuffle(keys)
        picked = []
        while len(picked) < per_type:
            progressed = False
            for k in keys:
                if buckets[k] and len(picked) < per_type:
                    picked.append(buckets[k].pop(0))
                    progressed = True
            if not progressed:
                break
        chosen.extend(picked)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", default="", help="explicit rows (comma separated)")
    ap.add_argument("--device", default="cuda:1", help="GPU for FLUX")
    ap.add_argument("--shard", default="0/1", help="K/N shard of the row list")
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--out", default=str(ROOT / "pairs/mchd_v1"))
    args = ap.parse_args()

    import pandas as pd
    from diffusers import Flux2KleinPipeline

    build = Path(args.build)
    out = Path(args.out)
    (out / "hd").mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(build / "metadata.parquet")
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))
    s3p = pd.read_parquet(build / "stage3_prompts.parquet", columns=["prompt_0"])
    gp = pd.read_parquet(build / "grounded_prompts.parquet", columns=["prompt_0"])
    s3_rows = set()
    sub = build / "stage3_subset.jsonl"
    if sub.exists():
        s3_rows = {int(json.loads(l)["index"]) for l in sub.read_text().splitlines() if l.strip()}

    if args.rows:
        rows = [int(x) for x in args.rows.split(",") if x.strip()]
    else:
        rows = stratified_rows(build, args.n, seed=args.seed)
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    if shard_n > 1:
        rows = [r for i, r in enumerate(rows) if i % shard_n == shard_k]
        print(f"shard {shard_k}/{shard_n}: {len(rows)} rows", flush=True)

    fingerprint = {"recipe": "mc->nearest->bil+gauss12->flux_edit", "size": args.size,
                   "steps": args.steps, "seed": args.seed, "flux_model": FLUX_MODEL,
                   "realism_prompt": REALISM, "strip_words": ["block", "blocks"],
                   "n_rows": len(rows), "shard": args.shard,
                   "rows_sha": hashlib.sha256(",".join(map(str, rows)).encode()).hexdigest()[:16]}
    fp = out / "config.json"
    if fp.exists():
        old = json.loads(fp.read_text())
        if old != fingerprint:
            raise SystemExit("config fingerprint mismatch; use a fresh --out")
    else:
        fp.write_text(json.dumps(fingerprint, indent=1))

    man_path = out / "manifest.jsonl"
    done = set()
    if man_path.exists():
        for line in man_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (out / rec["hd_png"]).exists():
                done.add(int(rec["row"]))
    man = man_path.open("a", encoding="utf-8")
    if done:
        print(f"resume: {len(done)} done")

    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    rows = [r for r in rows if r not in done]
    t_start = time.time()
    with torch.no_grad():
        for k, row in enumerate(rows):
            subject = clean_subject(s3p["prompt_0"].iloc[row] if row in s3_rows
                                    else gp["prompt_0"].iloc[row])
            pr = f"{subject}. {REALISM}"
            init = preprocess(np.asarray(imgs[row]), args.size)
            g = torch.Generator(device=args.device).manual_seed(args.seed + row)
            t0 = time.time()
            hd = pipe(image=Image.fromarray(init), prompt=pr, height=args.size,
                      width=args.size, num_inference_steps=args.steps, generator=g).images[0]
            el = time.time() - t0
            rel = f"hd/row{row:07d}.png"
            tmp = out / (rel + ".tmp")
            hd.save(tmp, format="PNG")
            os.replace(tmp, out / rel)
            man.write(json.dumps({
                "row": row, "hd_png": rel, "prompt": pr, "subject": subject,
                "asset_type": str(meta.at[row, "type"]),
                "file_name": str(meta.at[row, "file_name"]),
                "project_id": str(meta.at[row, "project_id"]),
                "prompt_source": "stage3" if row in s3_rows else "grounded",
                "mc_source": "real MC target (supervision)",
                "generator_model": "black-forest-labs/FLUX.2-klein-4B",
                "generator_license": FLUX_LICENSE,
                "preproc": "nearest->bil+gauss12", "size": args.size,
                "steps": args.steps, "seed": args.seed + row,
                "sec": round(el, 2),
            }) + "\n")
            man.flush()
            if (k + 1) % 10 == 0:
                el_all = time.time() - t_start
                print(f"{k+1}/{len(rows)}  {el_all/(k+1):.2f}s/img", flush=True)
    print(f"done -> {out} ({len(rows)} new)")


if __name__ == "__main__":
    main()
