"""MC -> HD anchor pairs with Qwen-Image-2.1 (research licence, non-commercial).

Recipe (decided by A/B): input = g28 (NEAREST + gaussian sigma 28 @512 scale),
prompt = P1 (REALISM), CFG off (true_cfg=1.0, no negative prompt), 30 steps,
384px, native RGBA output, optional torch.compile.

Rows mirror the existing FLUX pairs (union of the mchd_* manifests) so every MC
gets both a FLUX and a Qwen HD reference -> generator diversity.

    python scripts/build_mchd_pairs_qwen21.py --shard 0/2 --device cuda:0 \
        --out pairs/mchd_qwen21_a --compile
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_mchd_pairs import REALISM, alpha0_frac, clean_subject, preprocess_ref  # noqa: E402

QWEN_MODEL = "/home/iflab/models/Qwen-Image-2.1"
QWEN_LICENSE = "Qwen Research License (non-commercial; research/eval only)"
PREPROC = "gauss28"
FLUX_DIRS = ["pairs/mchd_stage3_a", "pairs/mchd_stage3_b", "pairs/mchd_fill_a", "pairs/mchd_fill_b"]


def flux_rows(root):
    rows = set()
    for d in FLUX_DIRS:
        p = root / d / "manifest.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                rows.add(int(json.loads(line)["row"]))
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="", help="comma separated; default = union of FLUX manifests")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--cutouts-only", action="store_true")
    ap.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--out", default=str(ROOT / "pairs/mchd_qwen21"))
    args = ap.parse_args()

    import pandas as pd
    from diffusers import QwenImage21Pipeline

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

    rows = ([int(x) for x in args.rows.split(",") if x.strip()] if args.rows
            else flux_rows(ROOT))
    if args.cutouts_only:
        rows = [r for r in rows if alpha0_frac(np.asarray(imgs[r])) > 0.05]
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    if shard_n > 1:
        rows = [r for i, r in enumerate(rows) if i % shard_n == shard_k]
    print(f"shard {shard_k}/{shard_n}: {len(rows)} rows")

    fingerprint = {"recipe": f"mc->nearest->{PREPROC}->qwen21_edit (alpha-aware ref, soft alpha)", "size": args.size,
                   "steps": args.steps, "seed": args.seed, "model": QWEN_MODEL,
                   "prompt": "P1 (REALISM)", "true_cfg": 1.0, "negative_prompt": None,
                   "preproc": PREPROC, "n_rows": len(rows), "shard": args.shard,
                   "rows_sha": hashlib.sha256(",".join(map(str, rows)).encode()).hexdigest()[:16]}
    fp = out / "config.json"
    if fp.exists():
        if json.loads(fp.read_text()) != fingerprint:
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

    pipe = QwenImage21Pipeline.from_pretrained(QWEN_MODEL, torch_dtype=torch.bfloat16)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    if args.compile:
        print("compiling transformer ...", flush=True)
        t = time.time()
        pipe.transformer.compile(mode=args.compile_mode, fullgraph=False)
        # warm both batch shapes (cfg is off: batch=1) before timing
        with torch.no_grad():
            for r in rows[:1]:
                subj = clean_subject(str(s3p["prompt_0"].iloc[r] if r in s3_rows
                                         else gp["prompt_0"].iloc[r]))
                init = Image.fromarray(preprocess_ref(np.asarray(imgs[r]), args.size, PREPROC))
                pipe(prompt=f"{subj}. {REALISM}", image=[init], height=args.size,
                     width=args.size, num_inference_steps=1, true_cfg_scale=1.0,
                     generator=torch.Generator(device=args.device).manual_seed(r))
        print(f"compiled+warmed in {time.time()-t:.0f}s", flush=True)

    rows = [r for r in rows if r not in done]
    t_start = time.time()
    with torch.no_grad():
        for k, row in enumerate(rows):
            subject = clean_subject(str(s3p["prompt_0"].iloc[row] if row in s3_rows
                                        else gp["prompt_0"].iloc[row]))
            pr = f"{subject}. {REALISM}"
            init = Image.fromarray(preprocess_ref(np.asarray(imgs[row]), args.size, PREPROC))
            ref_mode = "rgba" if alpha0_frac(np.asarray(imgs[row])) > 0.05 else "rgb"
            g = torch.Generator(device=args.device).manual_seed(args.seed + row)
            t0 = time.time()
            hd = pipe(prompt=pr, image=[init], height=args.size, width=args.size,
                      num_inference_steps=args.steps, true_cfg_scale=1.0, generator=g).images[0]
            el = time.time() - t0
            arr = np.asarray(hd)
            alpha0 = float(np.mean(arr[..., 3] == 0)) if arr.shape[2] == 4 else 0.0
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
                "generator_model": "Qwen/Qwen-Image-2.1",
                "generator_license": QWEN_LICENSE,
                "preproc": f"nearest->{PREPROC}", "ref_mode": ref_mode, "size": args.size,
                "steps": args.steps, "seed": args.seed + row,
                "out_mode": hd.mode, "alpha_zero_frac": round(alpha0, 4),
                "sec": round(el, 2),
            }) + "\n")
            man.flush()
            if (k + 1) % 10 == 0:
                print(f"{k+1}/{len(rows)}  {(time.time()-t_start)/(k+1):.2f}s/img", flush=True)
    print(f"done -> {out} ({len(rows)} new)")


if __name__ == "__main__":
    main()
