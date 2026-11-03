"""MC -> HD comparison: FLUX.2-klein-4B (edit) vs Qwen-Image-2.1 (image cond).

Same preprocessing (NEAREST upscale + bilateral+gaussian) and the same prompt
format (subject with "block" removed + realism instruction) for both models.

NOTE on licensing: Qwen-Image-2.1 is released under the Qwen RESEARCH LICENSE
(non-commercial); FLUX.2-klein-4B is Apache-2.0. Use Qwen-Image-2.1 for research
/ evaluation only and keep production data on the Apache-2.0 model.

    python scripts/mc2hd_compare_flux_qwen21.py --n 8 --size 384
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

from build_mchd_pairs import REALISM, clean_subject, preprocess  # noqa: E402
from data.ref_preprocess import load_alpha_composited  # noqa: E402

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"
QWEN_MODEL = "/home/iflab/models/Qwen-Image-2.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--qwen-steps", type=int, default=30)
    ap.add_argument("--true-cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--out", default="/tmp/opencode/flux_vs_qwen21")
    args = ap.parse_args()

    import pandas as pd

    build = Path(args.build)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(build / "metadata.parquet")
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))
    s3p = pd.read_parquet(build / "stage3_prompts.parquet", columns=["prompt_0"])
    if args.rows:
        rows = [int(x) for x in args.rows.split(",") if x.strip()]
    else:
        s3 = json.loads((build / "stage3_splits.json").read_text())
        bl = [i for i in s3["val"] if str(meta.at[i, "type"]) == "block"][:args.n // 2]
        it = [i for i in s3["val"] if str(meta.at[i, "type"]) == "item"][:args.n - args.n // 2]
        rows = bl + it

    from diffusers import Flux2KleinPipeline, QwenImage21Pipeline

    flux = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    flux.to("cuda:1")
    flux.set_progress_bar_config(disable=True)
    qwen = QwenImage21Pipeline.from_pretrained(QWEN_MODEL, torch_dtype=torch.bfloat16)
    qwen.to("cuda:0")
    qwen.set_progress_bar_config(disable=True)

    records, tiles = [], []
    with torch.no_grad():
        for row in rows:
            subject = clean_subject(str(s3p["prompt_0"].iloc[row]))
            prompt = f"{subject}. {REALISM}"
            init = preprocess(np.asarray(imgs[row]), args.size)
            init_pil = Image.fromarray(init)

            g1 = torch.Generator(device="cuda:1").manual_seed(args.seed + row)
            t0 = time.time()
            hd_flux = flux(image=init_pil, prompt=prompt, height=args.size,
                           width=args.size, num_inference_steps=args.steps,
                           generator=g1).images[0]
            s_flux = time.time() - t0

            g2 = torch.Generator(device="cuda:0").manual_seed(args.seed + row)
            t0 = time.time()
            hd_qwen = qwen(prompt=prompt, image=[init_pil], height=args.size,
                           width=args.size, num_inference_steps=args.qwen_steps,
                           true_cfg_scale=args.true_cfg, generator=g2).images[0]
            s_qwen = time.time() - t0

            hd_flux.save(out / f"row{row:07d}_flux.png")
            hd_qwen.save(out / f"row{row:07d}_qwen21.png")
            records.append({"row": row, "prompt": prompt, "size": args.size,
                            "flux_sec": round(s_flux, 2), "qwen21_sec": round(s_qwen, 2),
                            "flux_model": FLUX_MODEL, "flux_license": "Apache-2.0",
                            "qwen_model": QWEN_MODEL,
                            "qwen_license": "Qwen RESEARCH LICENSE (non-commercial)",
                            "qwen_mode": hd_qwen.mode})
            tiles.append((subject, np.asarray(imgs[row]), hd_flux, hd_qwen))
            print(f"row{row} flux {s_flux:.1f}s  qwen21 {s_qwen:.1f}s mode={hd_qwen.mode}",
                  flush=True)

    S, gap = 260, 8
    sheet = Image.new("RGB", (3 * S + 4 * gap, len(tiles) * (S + 30) + gap + 12), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 2), "raw MC (32) | FLUX.2-klein edit (384) | Qwen-Image-2.1 (384)", fill=(255, 255, 0))
    for r, (subj, mc, hf, hq) in enumerate(tiles):
        y = 24 + r * (S + 30)
        d.text((gap, y), subj[:100], fill=(150, 220, 255))
        pil = Image.fromarray(mc, "RGBA")
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil.convert("RGB"), mask=pil.split()[3])
        sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST), (gap, y + 26))
        sheet.paste(hf.convert("RGB").resize((S, S), Image.Resampling.LANCZOS),
                    (gap * 2 + S, y + 26))
        sheet.paste(hq.convert("RGB").resize((S, S), Image.Resampling.LANCZOS),
                    (gap * 3 + S * 2, y + 26))
    sheet.save(out / "compare.png")
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    print("saved", out / "compare.png")


if __name__ == "__main__":
    main()
