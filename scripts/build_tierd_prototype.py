"""Tier-D prototype: structured prompt -> FLUX.2-klein-4B HD -> nearest 32px ->
SDEdit with our MC model -> (prompt, MC) pair, reusing the HD prompt verbatim
as the MC annotation.

Provenance (model id, revision, seed, steps, guidance, prompt template,
license) is recorded per sample in manifest.jsonl.

    python scripts/build_tierd_prototype.py --prompts prompts.json --out pairs/tierd_proto
prompts.json: [{"prompt": "high-resolution game texture of ...", "asset_type": "block"}, ...]
"""
import argparse
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

from infer.sample import load_model_from_checkpoint  # noqa: E402
from infer.solver import to_uint8  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402
from sample_sdedit import sdedit_euler  # noqa: E402

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"
FLUX_LICENSE = "Apache-2.0 (black-forest-labs/FLUX.2-klein-4B)"

# Prompt wrapper: smooth, detailed, NON-voxel HD game art. The HD must NOT
# already look like Minecraft: no cubes, no voxels, no pixelation, otherwise
# the HD stage adds no value (downsampling blocky HD trivially looks "MC").
HD_SUFFIX_BLOCK = (", seamless flat game texture, orthographic front view, "
                   "smooth detailed realistic PBR materials, NOT voxel, NOT "
                   "minecraft style, NOT made of cubes, NOT pixelated, "
                   "no background scene, no text, no watermark")
HD_SUFFIX_ITEM = (", single centered smooth AAA game item render on a pure "
                  "white background, detailed realistic PBR materials, NOT "
                  "voxel, NOT minecraft style, NOT made of cubes, NOT "
                  "pixelated, no scene, no text, no watermark")
HD_SUFFIX = (", game texture asset filling the whole frame, flat front view, "
             "no background scene, no character, no text, no watermark")


def hd_suffix(asset_type):
    return HD_SUFFIX_ITEM if asset_type == "item" else HD_SUFFIX_BLOCK


def whitekey_alpha(hd_rgb, thresh=242):
    """Transparent background for item HDs: near-white pixels connected to the
    image border become transparent (flood fill, so white *inside* the item
    survives). Returns an RGBA array."""
    from scipy.ndimage import binary_propagation

    arr = np.asarray(hd_rgb.convert("RGB"), dtype=np.uint8)
    white = (arr >= thresh).all(axis=-1)
    seeds = np.zeros_like(white)
    seeds[0, :] = seeds[-1, :] = seeds[:, 0] = seeds[:, -1] = True
    bg = binary_propagation(seeds, mask=white)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    out = np.dstack([arr, alpha])
    return out


def load_flux(device):
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.to(device)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def read_prompts(path):
    """Accept a JSON array (*.json) or JSONL (*.jsonl): a list of
    {prompt, asset_type, ...}. Validates required fields up front (P0-1)."""
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty prompt file: {path}")
    try:
        items = json.loads(text)
        if isinstance(items, dict):
            items = items.get("prompts", [])
    except json.JSONDecodeError:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(items, list) or not items:
        raise ValueError(f"no prompts parsed from {path}")
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not str(it.get("prompt", "")).strip():
            raise ValueError(f"prompt entry {i} missing required field 'prompt'")
        it.setdefault("asset_type", "block")
        it.setdefault("bucket", -1)
        it.setdefault("novelty", 0)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="pairs/tierd_proto")
    ap.add_argument("--mc-ckpt", default=str(ROOT / "checkpoints/stage_3_frozen_v2/best.pt"))
    ap.add_argument("--hd-size", type=int, default=512)
    ap.add_argument("--flux-steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--t0", type=float, default=0.5)
    ap.add_argument("--sde-steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flux-device", default="cuda:1")
    ap.add_argument("--flux-batch", type=int, default=4)
    ap.add_argument("--sde-batch", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "hd").mkdir(parents=True, exist_ok=True)
    (out / "mc").mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(args.prompts)
    device, flux_device = args.device, args.flux_device

    flux = load_flux(flux_device)
    try:
        rev = json.load(open(f"{FLUX_MODEL}/model_index.json")).get("_diffusers_version", "?")
    except Exception:
        rev = "?"
    model, _, manifest = load_model_from_checkpoint(args.mc_ckpt, device, use_ema=False)
    premultiplied = bool(manifest.get("rgba_mode") == "premultiplied")
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device=device,
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])

    man_path = out / "manifest.jsonl"
    FB, SB = int(args.flux_batch), int(args.sde_batch)

    import queue as _queue
    import threading as _threading

    # ---- config fingerprint + resume (P1-10) ----
    fingerprint = {
        "flux_model": FLUX_MODEL, "flux_steps": args.flux_steps,
        "guidance": args.guidance, "hd_size": args.hd_size,
        "flux_batch": FB, "sde_batch": SB, "seed": args.seed,
        "mc_ckpt": args.mc_ckpt, "t0": args.t0, "sde_steps": args.sde_steps,
        "cfg": args.cfg, "hd_suffix_block": HD_SUFFIX_BLOCK,
        "hd_suffix_item": HD_SUFFIX_ITEM,
        "n_prompts": len(prompts),
        "prompts_sha": __import__("hashlib").sha256(
            "\n".join(p["prompt"] for p in prompts).encode()).hexdigest()[:16],
    }
    fp_path = out / "config.json"
    if fp_path.exists():
        old = json.loads(fp_path.read_text())
        if old != fingerprint:
            raise SystemExit(
                f"config fingerprint mismatch in {out}; use a fresh --out or "
                f"remove it. Diff: { {k: (old.get(k), fingerprint.get(k)) for k in fingerprint if old.get(k) != fingerprint.get(k)} }")
    else:
        fp_path.write_text(json.dumps(fingerprint, indent=1))
    done_ids = set()
    if man_path.exists():
        for line in man_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            j = int(rec.get("id", -1))
            hd_ok = (out / "hd" / f"{j:05d}.png").exists()
            mc_ok = (out / "mc" / f"{j:05d}.png").exists()
            if hd_ok and mc_ok:
                done_ids.add(j)
    man = man_path.open("a", encoding="utf-8")
    if done_ids:
        print(f"resume: skipping {len(done_ids)} completed ids")

    def atomic_save(pil_img, path):
        tmp = Path(str(path) + ".tmp")
        pil_img.save(tmp, format="PNG")
        os.replace(tmp, path)

    q = _queue.Queue(maxsize=2)
    stop = _threading.Event()
    err = {}

    def producer():
        try:
            for s in range(0, len(prompts), FB):
                if stop.is_set():
                    break
                if all(s + k in done_ids for k in range(min(FB, len(prompts) - s))):
                    continue  # whole batch done (P1-10 resume)
                batch = prompts[s:s + FB]
                # P0-5: one generator per sample so flux_seed is reproducible.
                gens = [torch.Generator(device=flux_device).manual_seed(args.seed + s + k)
                        for k in range(len(batch))]
                t0 = time.time()
                hds = flux(image=None,
                           prompt=[p["prompt"] + hd_suffix(p.get("asset_type", "block")) for p in batch],
                           height=args.hd_size, width=args.hd_size,
                           num_inference_steps=args.flux_steps,
                           guidance_scale=args.guidance,
                           generator=gens).images
                el = time.time() - t0
                while not stop.is_set():
                    try:
                        q.put((s, batch, list(hds), el / max(len(batch), 1)), timeout=1.0)
                        break
                    except _queue.Full:
                        continue
        except BaseException as e:
            err["producer"] = e
        finally:
            try:
                q.put(None, timeout=5)
            except Exception:
                pass

    th = _threading.Thread(target=producer, daemon=True)
    th.start()
    try:
        with torch.no_grad():
            while True:
                item = q.get()
                if item is None:
                    if err.get("producer") is not None:
                        raise err["producer"]
                    break
                s, batch, hds, hd_t = item
                # save HDs + build inits first (fast, CPU)
                inits, hd_paths = [], []
                for k, (spec, hd) in enumerate(zip(batch, hds)):
                    j = s + k
                    if j in done_ids:
                        inits.append(None)
                        hd_paths.append(out / "hd" / f"{j:05d}.png")
                        continue
                    hd_path = out / "hd" / f"{j:05d}.png"
                    atomic_save(hd, hd_path)
                    hd_paths.append(hd_path)
                    rgba = hd.convert("RGBA")
                    if spec.get("asset_type", "block") == "item":
                        rgba = Image.fromarray(whitekey_alpha(hd.convert("RGB")))
                    inits.append(np.asarray(
                        rgba.resize((32, 32), Image.Resampling.NEAREST), dtype=np.uint8))
                # SDEdit in sub-batches of SB; text encoded per sub-batch
                # (P0-2: stream, never cache all hidden states on GPU).
                for b in range(0, len(batch), SB):
                    sub = batch[b:b + SB]
                    idx = [s + b + t for t in range(len(sub))]
                    live = [(t, j) for t, j in enumerate(idx) if j not in done_ids]
                    if not live:
                        continue
                    hb, mb = enc.encode([prompts[j]["prompt"] for _, j in live])
                    null_h = torch.zeros_like(hb[:1])
                    null_m = torch.zeros_like(mb[:1])
                    x0 = torch.stack([
                        torch.from_numpy(inits[b + t]).permute(2, 0, 1)
                        for t, _ in live]).float().to(device) / 127.5 - 1.0
                    if premultiplied:
                        from data.rgba import premultiply_rgba_torch

                        x0 = premultiply_rgba_torch(x0)
                    # P0-5: explicit per-sample noise generators.
                    zs = []
                    for _, j in live:
                        gj = torch.Generator(device=device).manual_seed(args.seed + 1000003 + j)
                        zs.append(torch.randn(x0[:1].shape, generator=gj, device=device))
                    z = torch.cat(zs, dim=0)
                    xt0 = (1.0 - args.t0) * x0 + args.t0 * z
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        xhat = sdedit_euler(
                            model, xt0, args.t0, hb.to(device), steps=args.sde_steps,
                            cfg=args.cfg, text_uncond=null_h.expand(len(live), -1, -1).to(device),
                            text_mask=mb.to(device),
                            text_uncond_mask=null_m.expand(len(live), -1).to(device))
                    for p, (t, j) in enumerate(live):
                        mc = to_uint8(xhat[p], premultiplied=premultiplied).permute(1, 2, 0).cpu().numpy()
                        mc_path = out / "mc" / f"{j:05d}.png"
                        atomic_save(Image.fromarray(mc, "RGBA"), mc_path)
                        man.write(json.dumps({
                            "id": j, "prompt": prompts[j]["prompt"],
                            "asset_type": prompts[j].get("asset_type", "block"),
                            "bucket": prompts[j].get("bucket"),
                            "novelty": prompts[j].get("novelty", 0),
                            "hd_png": str(hd_paths[b + t]),
                            "mc_png": str(mc_path),
                            "generator_model": "black-forest-labs/FLUX.2-klein-4B",
                            "generator_license": FLUX_LICENSE,
                            "flux_steps": args.flux_steps, "guidance": args.guidance,
                            "hd_size": args.hd_size,
                            "flux_seed": args.seed + j,
                            "sdedit_seed": args.seed + 1000003 + j,
                            "mc_ckpt": args.mc_ckpt, "t0": args.t0,
                            "sde_steps": args.sde_steps, "cfg": args.cfg,
                            "hd_seconds": round(hd_t, 2),
                        }) + "\n")
                    man.flush()
                print(f"[{s}-{s+len(batch)-1}] done", flush=True)
    finally:
        stop.set()
        th.join(timeout=120)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
