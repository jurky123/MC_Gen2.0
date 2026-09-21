"""Open-loop test: my own prompts -> FLUX text-to-image HD -> Stylizer -> MC.

This is the intended Tier-D data production path. The prompts here are NOT from
the dataset (hand-written, including novel combinations), so there is no real MC
target: evaluation is qualitative (does the output look like a Minecraft texture
and match the prompt?).

    python scripts/tierd_t2i_stylizer.py --out /tmp/opencode/tierd_open
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_mchd_pairs import REALISM, clean_subject  # noqa: E402
from data.rgba import unpremultiply_rgba_np  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402
from infer.solver import to_uint8  # noqa: E402
from review_stylizer import load, sample_ref  # noqa: E402

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"

PROMPTS = [
    "ancient mossy stone brick with green moss growing in the cracks",
    "polished black marble tile with thin white veins",
    "glowing orange lava rock with bright cracks between dark plates",
    "dark oak wood planks with wide horizontal grain",
    "ruby gemstone with faceted deep red crystal",
    "steel dagger with a straight gray blade and brown leather handle",
    "golden crown with small red jewels around the band",
    "glass potion bottle with glowing green liquid",
    "frozen ice crystal with pale blue shards",
    "bronze gear with a round toothed wheel",
    "purple amethyst ore with crystals embedded in dark stone",
    "knight steel helmet with a narrow eye slit",
    # novel combinations (not present in the dataset)
    "copper clock with a blue moonstone face and gold trim",
    "obsidian lantern with a cyan flame and gold frame",
    "woven straw mat with a yellow braided pattern",
    "cracked desert sandstone with a pale yellow surface",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/stylizer_phase1_v2/best.pt"))
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--flux-steps", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--st", type=float, default=2.5)
    ap.add_argument("--sr", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flux-device", default="cuda:1")
    ap.add_argument("--out", default="/tmp/opencode/tierd_open")
    args = ap.parse_args()

    from diffusers import Flux2KleinPipeline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.to(args.flux_device)
    pipe.set_progress_bar_config(disable=True)
    model, man = load(args.ckpt, args.device)
    pm = bool(man.get("rgba_mode") == "premultiplied")

    subjects = [clean_subject(p) for p in PROMPTS]
    texts = [f"{s}. {REALISM}" for s in subjects]
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device="cuda:1",
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])
    h, hm = enc.encode(texts)
    nh, nm = torch.zeros_like(h).to(args.device), torch.zeros_like(hm).to(args.device)

    tiles = []
    with torch.no_grad():
        for k, (subj, txt) in enumerate(zip(subjects, texts)):
            g = torch.Generator(device=args.flux_device).manual_seed(args.seed + k)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hd = pipe(image=None, prompt=txt, height=args.size, width=args.size,
                          num_inference_steps=args.flux_steps, generator=g).images[0]
            hd.save(out / f"{k:02d}_hd.png")
            ref = torch.from_numpy(
                np.dstack([np.asarray(hd.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)),
                           np.full((64, 64), 255, np.uint8)])).permute(2, 0, 1)[None].float().to(args.device) / 127.5 - 1.0
            g2 = torch.Generator(device=args.device).manual_seed(args.seed + k)
            z = torch.randn(1, 4, 32, 32, generator=g2, device=args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xs = sample_ref(model, z, h[k:k + 1].to(args.device), hm[k:k + 1].to(args.device),
                                ref, nh[k:k + 1], nm[k:k + 1], steps=args.steps,
                                st=args.st, sr=args.sr)
            mc = to_uint8(xs[0], premultiplied=pm).permute(1, 2, 0).cpu().numpy()
            mc = unpremultiply_rgba_np(mc) if pm else mc
            Image.fromarray(mc, "RGBA").save(out / f"{k:02d}_mc.png")
            tiles.append((subj, hd, mc))
            print(f"[{k}] {subj}", flush=True)

    S, gap = 300, 8
    sheet = Image.new("RGB", (2 * S + 3 * gap, len(tiles) * (S + 30) + gap + 12), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 2), "prompt (mine, unseen) -> FLUX HD (384) -> STYLIZER MC (32)",
           fill=(255, 255, 0))
    for r, (subj, hd, mc) in enumerate(tiles):
        y = 24 + r * (S + 30)
        d.text((gap, y), f"{subj[:120]}", fill=(150, 220, 255))
        sheet.paste(hd.convert("RGB").resize((S, S), Image.Resampling.LANCZOS), (gap, y + 26))
        pil = Image.fromarray(mc, "RGBA")
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil.convert("RGB"), mask=pil.split()[3])
        sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST), (gap * 2 + S, y + 26))
    sheet.save(out / "open_loop.png")
    (out / "prompts.json").write_text(json.dumps(
        [{"subject": s, "text": t} for s, t in zip(subjects, texts)], indent=1))
    print("saved", out / "open_loop.png")


if __name__ == "__main__":
    main()
