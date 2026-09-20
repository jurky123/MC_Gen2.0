"""SDEdit-style MC-ification: HD (or real MC) -> nearest 32px -> noise to t0 ->
denoise with our MC t2i model -> MC-style output.

Two input modes:
  1. --rows: mmap row indices + prompts from a parquet (self-reconstruction
     ceiling test on real MC; the downsample step is the identity here).
  2. --hd-dir: external HD images (Tier-D prototype: prompt -> FLUX HD -> MC);
     each HD is nearest-downsampled to 32px first (bicubic-blurred input is
     OOD for the MC model).

Text prompts are passed through unchanged, so Tier-D pairs keep the original
HD-generation text as their annotation.

    python scripts/sample_sdedit.py --ckpt checkpoints/stage_3_frozen_v2/best.pt \\
        --rows 156911,474710,62332 --t0 0.3,0.5,0.7 --out /tmp/opencode/sdedit
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

from infer.sample import load_model_from_checkpoint  # noqa: E402
from infer.solver import to_uint8  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402


def sdedit_euler(model, xt0, t0, text, steps=20, cfg=2.0, text_uncond=None,
                 text_mask=None, text_uncond_mask=None):
    """Integrate the reverse ODE from t=t0 down to 0 (euler)."""
    from infer.solver import _velocity_at

    x = xt0
    dt = t0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), t0 - i * dt, device=x.device, dtype=x.dtype)
        v = _velocity_at(model, x, t, text, cfg, text_uncond,
                         text_mask=text_mask, text_uncond_mask=text_uncond_mask)
        x = x - dt * v
    return x.clamp(-1.0, 1.0)


def load_init_from_rows(rows, build):
    import pandas as pd

    meta = pd.read_parquet(build / "metadata.parquet")
    n = len(meta)
    img = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                    shape=(n, 32, 32, 4))
    return [np.asarray(img[i]).copy() for i in rows]


def load_init_from_hd(hd_dir, size=32):
    inits = []
    for p in sorted(Path(hd_dir).glob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        im = Image.open(p).convert("RGBA")
        # NEAREST: bicubic-blurred 32px input is OOD for the MC model.
        im = im.resize((size, size), Image.Resampling.NEAREST)
        inits.append((p.stem, np.asarray(im, dtype=np.uint8)))
    return inits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/stage_3_frozen_v2/best.pt"))
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--rows", default="",
                    help="comma-separated mmap rows (mode 1: real-MC self test)")
    ap.add_argument("--prompt-parquet", default="",
                    help="prompt source for --rows (default stage3_prompts)")
    ap.add_argument("--hd-dir", default="", help="mode 2: external HD images")
    ap.add_argument("--prompts", default="",
                    help="mode 2: JSON list of prompts aligned with --hd-dir files")
    ap.add_argument("--t0", default="0.3,0.5,0.7")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/opencode/sdedit")
    args = ap.parse_args()

    build = Path(args.build)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0s = [float(x) for x in args.t0.split(",") if x.strip()]
    device = args.device
    tower_dev = "cuda:1" if torch.cuda.device_count() > 1 else device

    model, _, manifest = load_model_from_checkpoint(args.ckpt, device, use_ema=False)
    premultiplied = bool(manifest.get("rgba_mode") == "premultiplied")
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device=tower_dev,
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])

    if args.rows:
        import pandas as pd

        rows = [int(x) for x in args.rows.split(",") if x.strip()]
        pp = args.prompt_parquet or str(build / "stage3_prompts.parquet")
        gp = pd.read_parquet(pp, columns=["prompt_0"])
        prompts = [str(gp["prompt_0"].iloc[i]) for i in rows]
        names = [f"row{i}" for i in rows]
        inits = load_init_from_rows(rows, build)
    else:
        items = load_init_from_hd(args.hd_dir)
        names = [n for n, _ in items]
        inits = [a for _, a in items]
        prompts = json.loads(args.prompts) if args.prompts else [""] * len(items)
        assert len(prompts) == len(items), "prompts must align with hd-dir files"

    h, mask = enc.encode(prompts)
    null_h = torch.zeros_like(h[:1])
    null_m = torch.zeros_like(mask[:1])

    results, metrics = {}, []
    with torch.no_grad():
        for j, (name, init, prompt) in enumerate(zip(names, inits, prompts)):
            x0 = torch.from_numpy(init).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1.0
            if premultiplied and x0.shape[1] == 4:
                from data.rgba import premultiply_rgba_torch

                x0 = premultiply_rgba_torch(x0)
            cols = {"init": init}
            for t0 in t0s:
                g = torch.Generator(device=device).manual_seed(args.seed)
                z = torch.randn_like(x0)
                xt0 = (1.0 - t0) * x0 + t0 * z
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    xhat = sdedit_euler(
                        model, xt0, t0, h[j:j + 1].to(device),
                        steps=args.steps, cfg=args.cfg,
                        text_uncond=null_h.to(device),
                        text_mask=mask[j:j + 1].to(device),
                        text_uncond_mask=null_m.to(device))
                arr = to_uint8(xhat[0], premultiplied=premultiplied).permute(1, 2, 0).cpu().numpy()
                cols[f"t0={t0}"] = arr
                metrics.append({"name": name, "t0": t0,
                                "l2": float(np.abs(arr.astype(int) - init.astype(int)).mean()),
                                "alpha_iou": alpha_iou(arr, init)})
            results[name] = cols
            print(f"{name}: " + " ".join(
                f"t0={t['t0']} L2={t['l2']:.1f} aIoU={t['alpha_iou']:.3f}"
                for t in metrics if t["name"] == name))

    S, gap = 128, 6
    keys = ["init"] + [f"t0={t}" for t in t0s]
    sheet = Image.new("RGB", (len(keys) * S + (len(keys) + 1) * gap,
                              len(results) * S + (len(results) + 1) * gap), (30, 30, 30))
    for r, (name, cols) in enumerate(results.items()):
        for c, k in enumerate(keys):
            pil = Image.fromarray(cols[k], "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST),
                        (gap + c * (S + gap), gap + r * (S + gap)))
    sheet.save(out / "sdedit_sheet.png")
    (out / "sdedit_metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "sdedit_prompts.json").write_text(
        json.dumps([{"name": n, "prompt": p} for n, p in zip(names, prompts)], indent=1))
    print(f"saved {out / 'sdedit_sheet.png'}")


def alpha_iou(a, b, thr=127):
    ma = a[..., 3] > thr
    mb = b[..., 3] > thr
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / max(union, 1))


if __name__ == "__main__":
    main()
