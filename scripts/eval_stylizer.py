"""Gate-1 evaluation for the HD->MC Stylizer.

For validation pairs, samples with dual CFG (text + reference) and compares
against the deterministic baseline and two controls:

    columns: reference (HD) | target (MC) | base t2i (no ref)
             | stylizer (dual CFG) | stylizer (shuffled ref)

Metrics per sample: RGB L2 to target, alpha IoU, edge-difference ratio.

    python scripts/eval_stylizer.py --ckpt checkpoints/stylizer_phase1/best.pt \
        --n 16 --out outputs/eval_stylizer
"""
import argparse
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
from data.rgba import premultiply_rgba_torch, unpremultiply_rgba_torch  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402
from infer.solver import to_uint8  # noqa: E402
from model.mc_flow_dit import MCFlowDiT  # noqa: E402


def load(ckpt, device):
    sd = torch.load(ckpt, map_location="cpu")
    mcfg = ModelConfig.from_dict(sd["model_cfg"])
    model = MCFlowDiT(mcfg)
    model.load_state_dict(sd["model"], strict=True)
    model.to(device).eval()
    return model, mcfg, sd.get("conditioning", {})


def dual_cfg_step(model, x, t, text, tmask, ref, st, sr, plain=None):
    v_nn = None
    if plain is None:
        def f(ref_arg, txt, msk):
            return model(x, t, txt, text_mask=msk, reference=ref_arg)
        null_t = torch.zeros_like(text)
        null_m = torch.zeros_like(tmask)
        v_tt = f(ref, text, tmask)
        v_tn = f(None, text, tmask)
        v_nn = f(None, null_t, null_m)
        v_nt = f(ref, null_t, null_m)
    return v_nn + st * (v_tn - v_nn) + sr * (v_tt - v_tn)


def sample_with_ref(model, z, text, tmask, ref, steps=16, st=2.0, sr=2.0):
    x = z
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=x.dtype)
        v = dual_cfg_step(model, x, t, text, tmask, ref, st, sr)
        x = x - dt * v
    return x.clamp(-1.0, 1.0)


def metrics(out_arr, tgt_arr):
    o = out_arr.astype(np.float32)
    t = tgt_arr.astype(np.float32)
    l2 = float(np.abs(o[..., :3] - t[..., :3]).mean())
    io = alpha_iou(o, t)
    eo = edges(o[..., :3])
    et = edges(t[..., :3])
    edge = float(np.abs(eo - et).mean() / max(et.mean(), 1e-3))
    return {"rgb_l2": round(l2, 1), "alpha_iou": round(io, 4), "edge_rel": round(edge, 3)}


def alpha_iou(a, b, thr=127):
    ma, mb = a[..., 3] > thr, b[..., 3] > thr
    return float(np.logical_and(ma, mb).sum() / max(np.logical_or(ma, mb).sum(), 1))


def edges(rgb):
    g = rgb.mean(-1)
    return np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/stylizer_phase1/best.pt"))
    ap.add_argument("--base", default=str(ROOT / "checkpoints/stage_3_frozen_v2/best.pt"))
    ap.add_argument("--pairs", default=str(ROOT / "pairs/stylizer"))
    ap.add_argument("--ref-size", type=int, default=64)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--st", type=float, default=2.0)
    ap.add_argument("--sr", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "outputs/eval_stylizer"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    ds = MmapPairDataset(
        ref_mmap=str(Path(args.pairs) / "ref.uint8.mmap"),
        target_mmap=str(Path(args.pairs) / "target.uint8.mmap"),
        metadata=str(Path(args.pairs) / "metadata.parquet"),
        splits=str(Path(args.pairs) / "splits.json"), split="val",
        ref_size=args.ref_size, target_size=32, rgba_mode="premultiplied")
    idxs = list(range(min(args.n, len(ds))))
    prompts = [str(ds.df["prompt"].iloc[ds.index[i]]) for i in idxs]
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device="cuda:1",
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])
    h, hmask = enc.encode(prompts)
    null_h, null_m = torch.zeros_like(h).to(dev), torch.zeros_like(hmask).to(dev)

    model, mcfg, man = load(args.ckpt, dev)
    premult = bool(man.get("rgba_mode") == "premultiplied")
    base, base_cfg, _ = load(args.base, dev)

    rows, agg = [], {"base": [], "styl": [], "shuf": []}
    for k, i in enumerate(idxs):
        x_tgt, _p, aux = ds[i]
        ref = aux["reference"][None].to(dev)
        blk = ref.clone()  # blocky input shown = reference (64px) upscaled
        g = torch.Generator(device=dev).manual_seed(args.seed + k)
        z = torch.randn(1, mcfg.in_channels, 32, 32, generator=g, device=dev)
        tgt_arr = to_uint8(x_tgt, premultiplied=premult).permute(1, 2, 0).cpu().numpy()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x_base = sample_plain(base, z, h[k:k + 1].to(dev), hmask[k:k + 1].to(dev),
                                      null_h[k:k + 1], null_m[k:k + 1], steps=args.steps)
                x_styl = sample_with_ref(model, z, h[k:k + 1].to(dev), hmask[k:k + 1].to(dev),
                                         ref, steps=args.steps, st=args.st, sr=args.sr)
                ref_shuf = ds[int(ds.index[(k + 7) % len(ds.index)])][2]["reference"][None].to(dev)
                x_shuf = sample_with_ref(model, z, h[k:k + 1].to(dev), hmask[k:k + 1].to(dev),
                                         ref_shuf, steps=args.steps, st=args.st, sr=args.sr)
        base_arr = to_uint8(x_base[0], premultiplied=premult).permute(1, 2, 0).cpu().numpy()
        styl_arr = to_uint8(x_styl[0], premultiplied=premult).permute(1, 2, 0).cpu().numpy()
        shuf_arr = to_uint8(x_shuf[0], premultiplied=premult).permute(1, 2, 0).cpu().numpy()
        base_arr = unpremultiply_uint8_if(base_arr, premult)
        styl_arr = unpremultiply_uint8_if(styl_arr, premult)
        shuf_arr = unpremultiply_uint8_if(shuf_arr, premult)
        tgt22 = unpremultiply_uint8_if(tgt_arr, premult)
        rows.append((blk, tgt22, base_arr, styl_arr, shuf_arr))
        for name, arr in (("base", base_arr), ("styl", styl_arr), ("shuf", shuf_arr)):
            agg[name].append(metrics(arr, tgt22))
        print(f"[{k}] {prompts[k][:50]} "
              f"base {agg['base'][-1]} styl {agg['styl'][-1]} shuf {agg['shuf'][-1]}",
              flush=True)

    S, gap = 200, 8
    sheet = Image.new("RGB", (5 * S + 6 * gap, len(rows) * (S + 22) + gap), (25, 25, 25))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    d.text((gap, 2), "ref64 (HD) | target MC | base t2i (no ref) | stylizer (dual CFG) | stylizer (shuffled ref)",
           fill=(255, 255, 0))
    for r, (blk, tgt, b, stl, sh) in enumerate(rows):
        y = 24 + r * (S + 22)
        d.text((gap, y), f"{prompts[r][:70]}", fill=(150, 220, 255))
        refshow = to_uint8(blk[0], premultiplied=False).permute(1, 2, 0).cpu().numpy()
        for c, arr in enumerate((refshow, tgt, b, stl, sh)):
            pil = Image.fromarray(arr[..., :4] if arr.shape[-1] == 4 else
                                  np.dstack([arr[..., :3], np.full(arr.shape[:2], 255, np.uint8)]), "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST),
                        (gap + c * (S + gap), y + 18))
    sheet.save(out / "stylizer_gate1.png")
    report = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in agg.items()}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def unpremultiply_uint8_if(arr, premult):
    if not premult:
        return arr
    from data.rgba import unpremultiply_rgba_np

    return unpremultiply_rgba_np(arr)


def sample_plain(model, z, text, tmask, null_h, null_m, steps=16, cfg=2.0):
    x = z
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=x.dtype)
        v_c = model(x, t, text, text_mask=tmask)
        v_u = model(x, t, null_h, text_mask=null_m)
        x = x - dt * (v_u + cfg * (v_c - v_u))
    return x.clamp(-1.0, 1.0)


if __name__ == "__main__":
    main()
