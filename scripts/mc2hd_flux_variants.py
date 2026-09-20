"""MC -> HD with FLUX edit, comparing input preprocessing variants.

For each MC texture and each input smoothing (nearest baseline, gauss16,
gauss28, bilateral+gauss12), upscale to 512, run the FLUX.2-klein edit and
show the results side by side with the raw MC.

    python scripts/mc2hd_flux_variants.py --rows 10757,486155 --out /tmp/opencode/flux_aa
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from try_antialias_filters import bilateral, gauss, up_nearest  # noqa: E402

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"
EDIT_PROMPT = ("Turn this pixel-art game sprite into a smooth, highly detailed "
               "high-resolution game texture. Keep the same object, the same "
               "orientation, the same proportions, the same silhouette and the "
               "same colors. Round off the pixel stair-steps into smooth "
               "continuous shapes. Add realistic surface detail and clean "
               "anti-aliased edges.")

REALISM = ("Redraw this as a photorealistic high-end game asset: physically "
           "based materials (PBR), realistic lighting and shading, fine surface "
           "detail, crisp anti-aliased edges, professional studio render "
           "quality. Keep the same object, orientation, proportions, silhouette "
           "and colors. Remove all pixelation and voxel stair-step edges.")

def make_preproc(scale=1.0):
    """Blur sizes scale with the working resolution (blur16 was tuned at 512)."""
    d = max(3, int(round(25 * scale)) | 1)
    return {
        "nearest": lambda a: a,
        "gauss16": lambda a: gauss(a, max(1.0, 16 * scale)),
        "gauss28": lambda a: gauss(a, max(1.0, 28 * scale)),
        "bil+gauss12": lambda a: gauss(bilateral(a, d=d, sc=60, ss=60),
                                       max(1.0, 12 * scale)),
    }


PREPROC = make_preproc(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--prompt-parquet", default="",
                    help="per-texture prompt source (stage3/grounded parquet)")
    ap.add_argument("--prompt-col", default="prompt_0")
    ap.add_argument("--instruction", default="smooth",
                    choices=("smooth", "realism"),
                    help="edit instruction style appended to the subject")
    ap.add_argument("--strip-words", default="",
                    help="comma list of words removed from the subject prompt "
                         "(e.g. block)")
    ap.add_argument("--variants", default="",
                    help="comma list of preprocessing variants (default all)")
    ap.add_argument("--out", default="/tmp/opencode/flux_aa")
    args = ap.parse_args()

    import pandas as pd
    from diffusers import Flux2KleinPipeline

    build = Path(args.build)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(build / "metadata.parquet")
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))
    subjects = {}
    if args.prompt_parquet:
        gp = pd.read_parquet(args.prompt_parquet, columns=[args.prompt_col])
        subjects = {int(x): str(gp[args.prompt_col].iloc[int(x)])
                    for x in (args.rows.split(",") if args.rows else [])}
    if args.rows:
        rows = [int(x) for x in args.rows.split(",") if x.strip()]
    else:
        s3 = json.loads((build / "stage3_splits.json").read_text())
        bl = [i for i in s3["val"] if str(meta.at[i, "type"]) == "block"][:args.n // 2]
        it = [i for i in s3["val"] if str(meta.at[i, "type"]) == "item"][:args.n - args.n // 2]
        rows = bl + it
    if args.prompt_parquet and not subjects:
        gp = pd.read_parquet(args.prompt_parquet, columns=[args.prompt_col])
        import re as _re

        strip = [w.strip() for w in args.strip_words.split(",") if w.strip()]

        def clean(t):
            for w in strip:
                t = _re.sub(rf"\b{_re.escape(w)}\b", " ", t, flags=_re.I)
            t = _re.sub(r"\s{2,}", " ", t)
            t = _re.sub(r"\s+([,.])", r"\1", t)
            return t.strip(" ,.")

        subjects = {i: clean(str(gp[args.prompt_col].iloc[i])) for i in rows}

    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda:1")
    pipe.set_progress_bar_config(disable=True)

    preproc = make_preproc(args.size / 512.0)
    results = {}
    with torch.no_grad():
        for row in rows:
            mc = np.asarray(imgs[row])
            pil = Image.fromarray(mc, "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            base = up_nearest(np.dstack([np.asarray(bg), np.full((32, 32), 255, np.uint8)]),
                              args.size)
            per = {}
            variant_names = ([v for v in args.variants.split(",") if v]
                             or list(preproc))
            for name in variant_names:
                fn = preproc[name]
                v = fn(base)
                img = Image.fromarray(v[..., :3].astype(np.uint8), "RGB")
                g = torch.Generator(device="cuda:1").manual_seed(args.seed + row)
                t0 = time.time()
                subject = subjects.get(row, "")
                instr = REALISM if args.instruction == "realism" else EDIT_PROMPT
                pr = (f"{subject}. " if subject else "") + instr
                hd = pipe(image=img, prompt=pr, height=args.size,
                          width=args.size, num_inference_steps=args.steps,
                          generator=g).images[0]
                per[name] = {"input": v[..., :3],
                             "hd": np.asarray(hd.convert("RGB")),
                             "prompt": pr,
                             "sec": round(time.time() - t0, 1)}
                hd.save(out / f"row{row}_{name.replace('+','_')}.png")
                print(f"row{row} {name}: {per[name]['sec']}s", flush=True)
            results[row] = per

    names = list(results[rows[0]])
    S, gap = 190, 6
    sheet = Image.new("RGB", ((len(names) * 2 + 1) * S + (len(names) * 2 + 2) * gap,
                              len(rows) * (S + 22) + gap), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 2), "raw MC | " + " | ".join(
        f"{n} in / out" for n in names), fill=(255, 255, 0))
    for r, row in enumerate(rows):
        y = 24 + r * (S + 22)
        d.text((gap, y), f"row{row} {str(meta.at[row, 'file_name'])[:60]}", fill=(150, 220, 255))
        pil = Image.fromarray(np.asarray(imgs[row]), "RGBA")
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil.convert("RGB"), mask=pil.split()[3])
        sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST), (gap, y + 18))
        for c, n in enumerate(names):
            sheet.paste(Image.fromarray(results[row][n]["input"].astype(np.uint8)).resize(
                (S, S), Image.Resampling.LANCZOS), ((2 * c + 1) * S + (2 * c + 2) * gap, y + 18))
            sheet.paste(Image.fromarray(results[row][n]["hd"]).resize(
                (S, S), Image.Resampling.LANCZOS), ((2 * c + 2) * S + (2 * c + 3) * gap, y + 18))
    sheet.save(out / "flux_aa_sheet.png")
    (out / "manifest.json").write_text(json.dumps(
        {str(k): {n: {"sec": v[n]["sec"]} for n in v} for k, v in results.items()}, indent=1))
    print("saved", out / "flux_aa_sheet.png")


if __name__ == "__main__":
    main()
