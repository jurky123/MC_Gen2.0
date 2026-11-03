"""Evaluate the Stylizer on the OOD benchmark (see build_ood_benchmark.py).

Paired groups (g1 same-generator, g3 other-generator, g4 perturbations) give
L2 / alpha-IoU / edge vs the real MC target.
Unpaired groups (g2 t2i) give MC-ness statistics plus structure preservation
(correlation between reference and output layouts) and palette distance to the
real MC distribution.

    python scripts/eval_ood.py --ckpt checkpoints/stylizer_v3/best.pt
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
from data.ref_preprocess import preprocess_reference  # noqa: E402
from data.rgba import unpremultiply_rgba_np  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402
from eval_stylizer import metrics  # noqa: E402
from infer.solver import to_uint8  # noqa: E402
from model.mc_flow_dit import MCFlowDiT  # noqa: E402


def load(ckpt, dev):
    sd = torch.load(ckpt, map_location="cpu")
    m = MCFlowDiT(ModelConfig.from_dict(sd["model_cfg"]))
    try:
        m.load_state_dict(sd["model"], strict=True)
    except RuntimeError as exc:
        # legacy checkpoint without the learned null-reference token
        missing = [k for k in m.state_dict() if k not in sd["model"]]
        if any("ref_null" in k for k in missing):
            m.load_state_dict(sd["model"], strict=False)
            m.legacy_null = True
            print(f"legacy checkpoint: {missing} left at init")
        else:
            raise
    m.to(dev).eval()
    return m, sd.get("conditioning", {}), ModelConfig.from_dict(sd["model_cfg"])


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


def perturb(t, kind):
    """t: (1,4,S,S) tensor on device."""
    S = t.shape[-1]
    if kind == "none":
        return t
    if kind == "crop5":
        s = int(S * 0.95)
        o = (S - s) // 2
        return F.interpolate(t[..., o:o + s, o:o + s], size=(S, S), mode="nearest")
    if kind == "shift4":
        return torch.roll(t, shifts=4, dims=-1)
    if kind == "scale90":
        s = int(S * 0.90)
        v = F.interpolate(t, size=(s, s), mode="nearest")
        pad = S - s
        return F.pad(v, (pad // 2, pad - pad // 2, pad // 2, pad - pad // 2), value=1.0)
    arr = ((t[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    im = Image.fromarray(arr[..., :3])
    if kind == "blur":
        from PIL import ImageFilter

        im = im.filter(ImageFilter.GaussianBlur(2.0))
    elif kind == "jpeg":
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=60)
        im = Image.open(buf)
    elif kind == "color":
        a = np.asarray(im).astype(np.float32)
        im = Image.fromarray(np.clip(a * np.array([1.08, 0.96, 0.9]) + 6, 0, 255).astype(np.uint8))
    out = torch.from_numpy(np.asarray(im).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
    return torch.cat([out, t[0, 3:4].cpu()], dim=0)[None].to(t.device)


def mc_stats(arr):
    """MC-ness of an RGBA uint8 image + how MC-like it is vs the real distribution."""
    vis = arr[..., 3] > 8
    rgb = arr[..., :3][vis]
    if len(rgb) == 0:
        return {"colors": 0, "blocks2": 0.0, "sat": 0.0, "edge": 0.0}
    uq = len(np.unique(rgb, axis=0))
    g = arr[..., :3].astype(float).mean(-1)
    blocks2 = float(((np.abs(g[0::2, 0::2] - g[0::2, 1::2])
                      + np.abs(g[0::2, 0::2] - g[1::2, 0::2])) == 0).mean())
    mx, mn = rgb.max(1).astype(float), rgb.min(1).astype(float)
    sat = float(((mx - mn) / np.maximum(mx, 1)).mean())
    edge = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
    return {"colors": uq, "blocks2": round(blocks2, 3), "sat": round(sat, 3),
            "edge": round(edge, 1)}


def structure_corr(ref_pil, out_arr, size=16):
    """Pearson correlation of white-composited 16x16 grayscale layouts."""
    r = np.asarray(ref_pil.convert("RGB").resize((size, size), Image.Resampling.LANCZOS))
    r = r.mean(-1).astype(np.float32)
    pil = Image.fromarray(out_arr, "RGBA")
    bg = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil.convert("RGB"), mask=pil.split()[3])
    o = np.asarray(bg.resize((size, size), Image.Resampling.LANCZOS)).mean(-1).astype(np.float32)
    r = r - r.mean()
    o = o - o.mean()
    d = float(np.sqrt((r ** 2).sum() * (o ** 2).sum()) + 1e-6)
    return float((r * o).sum() / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/stylizer_v3/best.pt"))
    ap.add_argument("--bench", default=str(ROOT / "pairs/ood_benchmark"))
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--pairs", default=str(ROOT / "pairs/stylizer_v3"))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--st", type=float, default=2.5)
    ap.add_argument("--sr", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--perturb", default="crop5,shift4,scale90,blur,jpeg,color")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/opencode/ood_eval")
    args = ap.parse_args()

    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = Path(args.bench)
    man = [json.loads(l) for l in (bench / "manifest.jsonl").read_text().splitlines() if l.strip()]
    build = Path(args.build)
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(pd.read_parquet(build / "metadata.parquet")), 32, 32, 4))
    model, cond, mcfg = load(args.ckpt, args.device)
    pm = bool(cond.get("rgba_mode") == "premultiplied")
    ref_size = int(getattr(mcfg, "ref_size", 64))
    print(f"ref_size from checkpoint = {ref_size}")
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device="cuda:1",
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])

    # real MC reference statistics
    s3 = json.loads((build / "stage3_splits.json").read_text())
    real = [np.asarray(imgs[i]) for i in s3["val"][:64]]
    real_stats = {k: round(float(np.mean([mc_stats(a)[k] for a in real])), 3)
                  for k in ("blocks2", "sat", "edge")}
    real_stats["colors"] = round(float(np.mean([mc_stats(a)["colors"] for a in real])), 1)

    kinds = ["none"] + [k for k in args.perturb.split(",") if k] if args.perturb else ["none"]
    # per-group sampling so every paired group is represented
    g1 = [m for m in man if m["group"] == "g1_flux_edit"][:args.n]
    g3 = [m for m in man if m["group"] == "g3_qwen_edit"][:max(2, args.n // 3)]
    rows_paired = g1 + g3
    print(f"paired: g1={len(g1)} g3={len(g3)}")
    prompts = [m["prompt"] for m in rows_paired]
    if prompts:
        h, hm = enc.encode(prompts)
        nh, nm = torch.zeros_like(h).to(args.device), torch.zeros_like(hm).to(args.device)
    unpaired = [m for m in man if m["group"] == "g2_flux_t2i"]
    if unpaired:
        hu, hmu = enc.encode([m["prompt"] for m in unpaired])
        nhu, nmu = torch.zeros_like(hu).to(args.device), torch.zeros_like(hmu).to(args.device)

    results = {"paired": {}, "unpaired": []}
    with torch.no_grad():
        for k, m in enumerate(rows_paired):
            ref0 = preprocess_reference(bench / m["ref_png"], ref_size)[None].to(args.device)
            tgt = unpremultiply_rgba_np(np.asarray(imgs[m["row"]])) if m["row"] is not None else None
            for kind in kinds:
                ref = perturb(ref0, kind)
                g = torch.Generator(device=args.device).manual_seed(args.seed + k)
                z = torch.randn(1, 4, 32, 32, generator=g, device=args.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    xs = sample_ref(model, z, h[k:k + 1].to(args.device), hm[k:k + 1].to(args.device),
                                    ref, nh[k:k + 1], nm[k:k + 1], steps=args.steps,
                                    st=args.st, sr=args.sr)
                arr = unpremultiply_rgba_np(to_uint8(xs[0], premultiplied=pm)
                                            .permute(1, 2, 0).cpu().numpy())
                key = f"{m['group']}:{kind}"
                entry = results["paired"].setdefault(key, {"m": [], "corr": [], "mc": []})
                if tgt is not None:
                    entry["m"].append(metrics(arr, tgt))
                entry["corr"].append(structure_corr(Image.open(bench / m["ref_png"]), arr))
                entry["mc"].append(mc_stats(arr))
                if kind == "none":
                    Image.fromarray(arr, "RGBA").save(out / f"{m['id']}_none.png")
            if (k + 1) % 8 == 0:
                print(f"paired {k+1}/{len(rows_paired)}", flush=True)

        for j, m in enumerate(unpaired):
            ref0 = preprocess_reference(bench / m["ref_png"], ref_size)[None].to(args.device)
            g = torch.Generator(device=args.device).manual_seed(args.seed + 1000 + j)
            z = torch.randn(1, 4, 32, 32, generator=g, device=args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xs = sample_ref(model, z, hu[j:j + 1].to(args.device), hmu[j:j + 1].to(args.device),
                                ref0, nhu[j:j + 1], nmu[j:j + 1], steps=args.steps,
                                st=args.st, sr=args.sr)
            arr = unpremultiply_rgba_np(to_uint8(xs[0], premultiplied=pm)
                                        .permute(1, 2, 0).cpu().numpy())
            results["unpaired"].append({
                "id": m["id"], "corr": structure_corr(Image.open(bench / m["ref_png"]), arr),
                **mc_stats(arr)})
            Image.fromarray(arr, "RGBA").save(out / f"{m['id']}_none.png")

    print("\n== real MC reference stats ==", json.dumps(real_stats))
    print("\n== paired (rgb_l2 / alpha_iou / edge_rel | layout corr | colors/blocks2/sat/edge) ==")
    table = {}
    for key in sorted(results["paired"]):
        e = results["paired"][key]
        if e["m"]:
            l2 = float(np.mean([x["rgb_l2"] for x in e["m"]]))
            iou = float(np.mean([x["alpha_iou"] for x in e["m"]]))
            edge = float(np.mean([x["edge_rel"] for x in e["m"]]))
        else:
            l2 = iou = edge = float("nan")
        corr = float(np.mean(e["corr"]))
        mc = {kk: round(float(np.mean([x[kk] for x in e["mc"]])), 3)
              for kk in ("colors", "blocks2", "sat", "edge")}
        table[key] = {"rgb_l2": round(l2, 1), "alpha_iou": round(iou, 4),
                      "edge_rel": round(edge, 3), "layout_corr": round(corr, 3), "mc": mc}
        print(f"{key:26s} {l2:6.1f} / {iou:.3f} / {edge:5.2f} | corr {corr:5.3f} | {mc}")
    if results["unpaired"]:
        corr = float(np.mean([x["corr"] for x in results["unpaired"]]))
        mc = {kk: round(float(np.mean([x[kk] for x in results["unpaired"]])), 3)
              for kk in ("colors", "blocks2", "sat", "edge")}
        table["g2_flux_t2i:unpaired"] = {"layout_corr": round(corr, 3), "mc": mc}
        print(f"{'g2_flux_t2i:unpaired':26s}    --  /  --  /  --   | corr {corr:5.3f} | {mc}")
    (out / "report.json").write_text(json.dumps({"real_mc": real_stats, "table": table}, indent=1))
    print("saved", out / "report.json")


if __name__ == "__main__":
    main()
