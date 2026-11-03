"""Build the OOD reference benchmark for the HD->MC Stylizer.

Groups
  G1 paired / same generator : FLUX-edit HD from unseen MC rows (real MC target)
  G2 unpaired / t2i          : FLUX.2-klein text-to-image HD (no MC target)
  G3 paired / other generator: Qwen-Image-Edit (MC -> HD) HD (real MC target)
  G4 perturbation matrix     : derived from G1 refs at eval time (crop / shift /
                               scale / blur / jpeg / colour drift)

    python scripts/build_ood_benchmark.py --n-paired 24 --n-t2i 16 --n-qwen 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_mchd_pairs import REALISM, clean_subject, preprocess  # noqa: E402
from data.ref_preprocess import preprocess_reference  # noqa: E402

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"
QWEN_MODEL = "/home/iflab/models/Qwen-Image-Edit-2511"

QWEN_INSTRUCTION = (
    "Turn this pixel-art game sprite into a smooth, highly detailed "
    "high-resolution game texture with realistic materials. Keep the same "
    "object, the same orientation, the same proportions, the same silhouette "
    "and the same colors. Round off the pixel stair-steps into smooth "
    "continuous shapes.")
QWEN_NEG = ("voxel, blocky, cubes, minecraft, pixel art, pixelated, jagged "
            "stair-step edges, mosaic, grid, aliasing, background scene, text, "
            "watermark")

T2I_PROMPTS = [
    "a medieval iron door with rivets and a small barred window",
    "a copper teapot with a curved spout and wooden handle",
    "a bundle of dried herbs tied with twine",
    "a wooden barrel with iron bands and a cork",
    "a blue crystal shard cluster on dark stone",
    "a red mushroom with white spots",
    "a rusty iron ladder with worn rungs",
    "a golden pocket watch with a white face",
    "a stack of leather books with gold spines",
    "a woven basket filled with green apples",
    "a stone brick wall covered in ivy",
    "a glass jar containing glowing blue liquid",
    "a ceramic tile with a blue floral pattern",
    "a snowy pine sapling in a clay pot",
    "a crimson fabric banner with a gold emblem",
    "a cracked clay pot with dirt spilling out",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "pairs/stylizer_v3"))
    ap.add_argument("--hd-dirs", default="pairs/mchd_stage3_a,pairs/mchd_stage3_b,"
                                        "pairs/mchd_fill_a,pairs/mchd_fill_b")
    ap.add_argument("--n-paired", type=int, default=24)
    ap.add_argument("--n-t2i", type=int, default=16)
    ap.add_argument("--n-qwen", type=int, default=8)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--t2i-size", type=int, default=512)
    ap.add_argument("--qwen-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--flux-device", default="cuda:1")
    ap.add_argument("--out", default=str(ROOT / "pairs/ood_benchmark"))
    args = ap.parse_args()

    import pandas as pd

    out = Path(args.out)
    for g in ("g1_flux_edit", "g2_flux_t2i", "g3_qwen_edit"):
        (out / "refs" / g).mkdir(parents=True, exist_ok=True)
    build = ROOT / "data/build/mc_text2image32_wl"
    meta = pd.read_parquet(build / "metadata.parquet")
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))
    df = pd.read_parquet(Path(args.pairs) / "metadata.parquet")
    splits = json.loads((Path(args.pairs) / "splits.json").read_text())
    hd_map = {}
    for d in args.hd_dirs.split(","):
        for p in Path(d).glob("hd/row*.png"):
            hd_map[int(p.stem[3:])] = p

    val_idx = splits["val"]
    rows = [int(df["row"].iloc[i]) for i in val_idx if int(df["row"].iloc[i]) in hd_map]
    rows = rows[:args.n_paired]
    manifest = []

    # ---- G1: copy the FLUX-edit HDs (already generated) ----
    for r in rows:
        dst = out / "refs" / "g1_flux_edit" / f"row{r:07d}.png"
        if not dst.exists():
            Image.open(hd_map[r]).convert("RGB").save(dst)
        manifest.append({"id": f"g1_{r}", "group": "g1_flux_edit", "row": r,
                         "ref_png": str(dst.relative_to(out)),
                         "prompt": str(df.loc[df["row"] == r, "prompt"].iloc[0])})
    print(f"G1: {len(rows)} refs", flush=True)

    # ---- G2: FLUX text-to-image (unpaired) ----
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.to(args.flux_device)
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        for k, subj in enumerate(T2I_PROMPTS[:args.n_t2i]):
            dst = out / "refs" / "g2_flux_t2i" / f"{k:03d}.png"
            if not dst.exists():
                g = torch.Generator(device=args.flux_device).manual_seed(args.seed + k)
                txt = f"{clean_subject(subj)}. {REALISM}"
                hd = pipe(image=None, prompt=txt, height=args.t2i_size,
                          width=args.t2i_size, num_inference_steps=args.steps,
                          generator=g).images[0]
                hd.save(dst)
                print(f"G2 {k}: {subj[:50]}", flush=True)
            manifest.append({"id": f"g2_{k}", "group": "g2_flux_t2i", "row": None,
                             "ref_png": str(dst.relative_to(out)),
                             "prompt": f"{clean_subject(subj)}. {REALISM}"})

    # ---- G3: Qwen-Image-Edit MC -> HD (other generator, paired) ----
    if args.n_qwen:
        from diffusers import QwenImageEditPlusPipeline

        qpipe = QwenImageEditPlusPipeline.from_pretrained(QWEN_MODEL,
                                                          torch_dtype=torch.bfloat16)
        qpipe.to(args.device)
        qpipe.set_progress_bar_config(disable=True)
        qrows = rows[:args.n_qwen]
        with torch.no_grad():
            for k, r in enumerate(qrows):
                dst = out / "refs" / "g3_qwen_edit" / f"row{r:07d}.png"
                if not dst.exists():
                    init = Image.fromarray(preprocess(np.asarray(imgs[r]), args.qwen_size))
                    g = torch.Generator(device=args.device).manual_seed(args.seed + r)
                    hd = qpipe(image=[init], prompt=QWEN_INSTRUCTION,
                               negative_prompt=QWEN_NEG, height=args.qwen_size,
                               width=args.qwen_size, num_inference_steps=25,
                               true_cfg_scale=4.0, generator=g).images[0]
                    hd.save(dst)
                    print(f"G3 row{r}", flush=True)
                manifest.append({"id": f"g3_{r}", "group": "g3_qwen_edit", "row": r,
                                 "ref_png": str(dst.relative_to(out)),
                                 "prompt": str(df.loc[df["row"] == r, "prompt"].iloc[0])})

    (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest))
    print(f"benchmark built -> {out}  ({len(manifest)} refs)")


if __name__ == "__main__":
    main()
