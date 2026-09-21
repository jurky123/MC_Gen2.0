"""Probe the Stylizer's reference robustness (alignment shortcut test).

Takes paired val samples (so a real MC target exists) and perturbs the
reference in ways a truly domain-robust stylizer should tolerate:
    crop 5%, shift 4px, scale 0.95, blur, jpeg, plus cross-domain-ish variants.
Reports L2 / alpha-IoU / edge vs the target for each perturbation and for a
couple of reference CFG strengths. If a 4px shift or 5% crop destroys the
output, the model is relying on pixel alignment rather than semantics.

    python scripts/probe_reference_robustness.py --n 16
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ModelConfig  # noqa: E402
from data.pair_dataset import MmapPairDataset  # noqa: E402
from data.rgba import unpremultiply_rgba_np  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402
from eval_stylizer import alpha_iou, edges, metrics  # noqa: E402
from infer.solver import to_uint8  # noqa: E402
from model.mc_flow_dit import MCFlowDiT  # noqa: E402


def load(ckpt, dev):
    sd = torch.load(ckpt, map_location="cpu")
    m = MCFlowDiT(ModelConfig.from_dict(sd["model_cfg"]))
    m.load_state_dict(sd["model"], strict=True)
    m.to(dev).eval()
    return m, sd.get("conditioning", {})


def perturb(ref, kind, size):
    """ref: (1,4,H,W) in [-1,1]."""
    if kind == "none":
        return ref
    if kind == "crop5":
        s = int(size * 0.95)
        o = (size - s) // 2
        return F.interpolate(ref[..., o:o + s, o:o + s], size=(size, size),
                             mode="nearest")
    if kind == "shift4":
        out = torch.roll(ref, shifts=4, dims=-1)
        return out
    if kind == "scale95":
        s = int(size * 0.90)
        v = F.interpolate(ref, size=(s, s), mode="nearest")
        pad = size - s
        return F.pad(v, (pad // 2, pad - pad // 2, pad // 2, pad - pad // 2), value=1.0)
    # pil-based
    arr = ((ref[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    im = Image.fromarray(arr[..., :3])
    if kind == "blur":
        im = im.filter(__import__("PIL.ImageFilter", fromlist=["ImageFilter"]).GaussianBlur(2.0))
    elif kind == "jpeg":
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=60)
        im = Image.open(buf)
    elif kind == "color":
        a = np.asarray(im).astype(np.float32)
        a = np.clip(a * np.array([1.08, 0.96, 0.9]) + 6, 0, 255)
        im = Image.fromarray(a.astype(np.uint8))
    out = torch.from_numpy(np.asarray(im).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
    return torch.cat([out, ref[0, 3:4].cpu()], dim=0)[None].to(ref.device)


def sample_ref(model, z, text, tmask, ref, nh, nm, steps=20, st=2.5, sr=2.5):
    x = z
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=x.dtype)
        v_nn = model(x, t, nh, text_mask=nm, reference=None)
        v_tn = model(x, t, text, text_mask=tmask, reference=None)
        v_tt = model(x, t, text, text_mask=tmask, reference=ref)
        x = x - dt * (v_nn + st * (v_tn - v_nn) + sr * (v_tt - v_tn))
    return x.clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/stylizer_v3/best.pt"))
    ap.add_argument("--pairs", default=str(ROOT / "pairs/stylizer_v3"))
    ap.add_argument("--ref-size", type=int, default=128)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--st", type=float, default=2.5)
    ap.add_argument("--srs", default="1.0,2.5")
    ap.add_argument("--kinds", default="none,crop5,shift4,scale95,blur,jpeg,color")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/opencode/probe")
    args = ap.parse_args()

    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    pairs = Path(args.pairs)
    ds = MmapPairDataset(str(pairs / "ref.uint8.mmap"), str(pairs / "target.uint8.mmap"),
                         str(pairs / "metadata.parquet"), str(pairs / "splits.json"),
                         split="val", ref_size=args.ref_size, target_size=32,
                         rgba_mode="premultiplied")
    idxs = list(range(min(args.n, len(ds))))
    prompts = [str(ds.df["prompt"].iloc[ds.index[i]]) for i in idxs]
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device="cuda:1",
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])
    h, hm = enc.encode(prompts)
    nh, nm = torch.zeros_like(h).to(dev), torch.zeros_like(hm).to(dev)
    model, man = load(args.ckpt, dev)
    pm = bool(man.get("rgba_mode") == "premultiplied")

    kinds = args.kinds.split(",")
    srs = [float(x) for x in args.srs.split(",")]
    acc = {(k, sr): [] for k in kinds for sr in srs}
    with torch.no_grad():
        for k, i in enumerate(idxs):
            x_t, _p, aux = ds[i]
            ref0 = aux["reference"][None].to(dev)
            tgt = unpremultiply_rgba_np(to_uint8(x_t, premultiplied=pm).permute(1, 2, 0).cpu().numpy())
            for kind in kinds:
                ref = perturb(ref0, kind, args.ref_size)
                for sr in srs:
                    g = torch.Generator(device=dev).manual_seed(args.seed + k)
                    z = torch.randn(1, 4, 32, 32, generator=g, device=dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        xs = sample_ref(model, z, h[k:k + 1].to(dev), hm[k:k + 1].to(dev),
                                        ref, nh[k:k + 1], nm[k:k + 1], steps=args.steps,
                                        st=args.st, sr=sr)
                    arr = unpremultiply_rgba_np(
                        to_uint8(xs[0], premultiplied=pm).permute(1, 2, 0).cpu().numpy())
                    acc[(kind, sr)].append(metrics(arr, tgt))
            if (k + 1) % 4 == 0:
                print(f"{k+1}/{len(idxs)}", flush=True)

    print(f"{'perturbation':12s} " + " ".join(f"sr={sr:<4}" for sr in srs)
          + "   (rgb_l2 / alpha_iou, mean over "
          + str(len(idxs)) + ")")
    table = {}
    for kind in kinds:
        cells = []
        for sr in srs:
            v = acc[(kind, sr)]
            l2 = float(np.mean([x["rgb_l2"] for x in v]))
            iou = float(np.mean([x["alpha_iou"] for x in v]))
            cells.append(f"{l2:6.1f}/{iou:.3f}")
            table[f"{kind}@{sr}"] = {"rgb_l2": round(l2, 1), "alpha_iou": round(iou, 4)}
        print(f"{kind:12s} " + " ".join(cells))
    (out / "probe.json").write_text(json.dumps(table, indent=1))
    print("saved", out / "probe.json")


if __name__ == "__main__":
    main()
