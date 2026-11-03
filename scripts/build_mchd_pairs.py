"""Build MC -> HD pairs with the validated recipe (the only MC2HD path now).

    real MC (32px)
      -> NEAREST upscale to `--size` (384 default)
      -> NEAREST upscale, then gauss16 (stair-step removal) or none (nearest)
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


PREPROC_MODES = ("nearest", "g16")
PREPROC_SIGMA = {"g16": 16, "gauss28": 28}
FLUX_PAIR_DIRS = ["pairs/mchd_stage3_a", "pairs/mchd_stage3_b", "pairs/mchd_fill_a", "pairs/mchd_fill_b"]


def preprocess(mc_rgba, size, mode="g16"):
    """32px MC -> `size` model input.

    mode="nearest": plain NEAREST upscale (structure-faithful).
    mode="g16" / "gauss28": NEAREST + gaussian sigma (scaled at 512) -> stair-step removal.
    """
    pil = Image.fromarray(np.asarray(mc_rgba, dtype=np.uint8), "RGBA")
    bg = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil.convert("RGB"), mask=pil.split()[3])
    large = np.asarray(bg.resize((size, size), Image.Resampling.NEAREST))
    if mode == "nearest":
        return large
    if mode == "bil+g12":
        d = max(3, int(round(25 * (size / 512.0))) | 1)
        return gauss(bilateral(np.dstack([large, np.full((size, size), 255, np.uint8)]),
                              d=d, sc=60, ss=60), max(1.0, 12 * (size / 512.0)))[..., :3].astype(np.uint8)
    sigma = PREPROC_SIGMA.get(mode, 16)
    scale = size / 512.0
    smoothed = gauss(np.dstack([large, np.full((size, size), 255, np.uint8)]),
                     max(1.0, sigma * scale))
    return smoothed[..., :3].astype(np.uint8)


def pick_mode(preproc, seed, row):
    """Deterministic per-row mode so resume is stable. 'mix' = nearest/g16 split."""
    if preproc != "mix":
        return preproc
    return PREPROC_MODES[(seed * 1000003 + row) % len(PREPROC_MODES)]


def alpha0_frac(mc_rgba):
    return float(np.mean(np.asarray(mc_rgba)[..., 3] == 0))


def preprocess_ref(mc_rgba, size, mode="gauss28", alpha_thresh=0.05):
    """Alpha-aware reference for edit models.

    64.8% of the MC targets are transparent cutouts (bed sides, nuggets, ...).
    Compositing those on white makes the model fill the frame with a zoomed 3D
    object ("cropped" look). So: cutouts -> keep RGBA (transparent background),
    opaque full-bleed textures -> white composite (RGB ref).

    Cutout alpha is blurred together with the colour (gauss does sigma on RGB and
    sigma/2 on alpha). Keeping the NEAREST alpha edge makes the model read the
    stair-step silhouette as pixel art and reproduce blocky output.

    Returns HxWx3 (RGB) or HxWx4 (RGBA).
    """
    arr = np.asarray(mc_rgba, dtype=np.uint8)
    pil = Image.fromarray(arr, "RGBA")
    large = np.asarray(pil.resize((size, size), Image.Resampling.NEAREST))
    sigma = PREPROC_SIGMA.get(mode, 28)
    sc = size / 512.0
    blur = (lambda x: gauss(np.dstack([x, np.full((size, size), 255, np.uint8)]),
                            max(1.0, sigma * sc))[..., :3].astype(np.uint8))
    if alpha0_frac(arr) > alpha_thresh:
        return large if mode == "nearest" else gauss(large, max(1.0, sigma * sc)).astype(np.uint8)
    bg = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil.convert("RGB"), mask=pil.split()[3])
    large_bg = np.asarray(bg.resize((size, size), Image.Resampling.NEAREST))
    return large_bg if mode == "nearest" else blur(large_bg)


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
    ap.add_argument("--preproc", default="mix", choices=["mix", *PREPROC_MODES, "bil+g12"],
                    help="input preprocessing; 'mix' = per-row nearest/g16")
    ap.add_argument("--cutouts-only", action="store_true",
                    help="only transparent cutout rows (alpha0 > 5%%), e.g. to rebuild the"
                         " 65%% that the old white-composite recipe expanded to full frame")
    ap.add_argument("--rows-from-manifests", action="store_true",
                    help="row list = union of the existing FLUX pair manifests")
    ap.add_argument("--prompt-style", default="auto", choices=["auto", "item"],
                    help="'auto' = item prefix only for type==item; 'item' = always")
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
    elif args.rows_from_manifests:
        rows = []
        for d in FLUX_PAIR_DIRS:
            p = ROOT / d / "manifest.jsonl"
            if p.exists():
                rows += [int(json.loads(l)["row"]) for l in p.read_text().splitlines() if l.strip()]
        rows = sorted(set(rows))
    else:
        rows = stratified_rows(build, args.n, seed=args.seed)
    if args.cutouts_only:
        rows = [r for r in rows if alpha0_frac(np.asarray(imgs[r])) > 0.05]
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    if shard_n > 1:
        rows = [r for i, r in enumerate(rows) if i % shard_n == shard_k]
        print(f"shard {shard_k}/{shard_n}: {len(rows)} rows", flush=True)

    fingerprint = {"recipe": "mc->nearest->{bil+gauss12|none}->flux_edit", "size": args.size,
                   "preproc": args.preproc, "prompt_style": args.prompt_style,
                   "cutouts_only": args.cutouts_only,
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
            is_item = args.prompt_style == "item" or str(meta.at[row, "type"]) == "item" \
                and args.prompt_style == "auto"
            prefix = "single item centered on a plain white background, " if is_item else ""
            pr = f"{prefix}{subject}. {REALISM}"
            mode = pick_mode(args.preproc, args.seed, row)
            init = preprocess(np.asarray(imgs[row]), args.size, mode)
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
                "preproc": {"nearest": "nearest", "g16": "nearest->gauss16",
                            "gauss28": "nearest->gauss28", "bil+g12": "nearest->bil+gauss12"}[mode],
                "size": args.size,
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
