"""Evaluate a trained MC-FlowDiT checkpoint on the grounded Stage-2 val set.

Produces (1) real-vs-generated comparison sheets and (2) quantitative metrics:
  - concept recall: VLM caption vs prompt content words (word boundary)
  - colour accuracy: prompts containing a colour word
  - real-vs-generated fidelity: mean-RGB distance + RGB histogram cosine
  - seam score on generated blocks
  - diversity across seeds

    python scripts/eval_generation.py --ckpt checkpoints/stage_2_grounded_k1/best.pt \
        --model configs/model/base_flux2klein.yaml --n 64 --out outputs/eval_grounded
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from config import ModelConfig  # noqa: E402
from data.filename_prompts import COLOR_WORDS  # noqa: E402
from data.text_tower import FrozenTextEncoder  # noqa: E402
from eval.seam import seam_score  # noqa: E402
from infer.solver import sample, to_uint8  # noqa: E402
from model.mc_flow_dit import MCFlowDiT  # noqa: E402

STOP = {"a", "an", "the", "block", "item", "of", "with", "and", "on", "in", "for",
        "top", "bottom", "side", "front", "back", "inner", "outer", "under"}

# common, visually-recognisable concepts for a fair recall metric
COMMON = {
    "stone", "cobblestone", "wood", "wooden", "oak", "birch", "spruce", "log",
    "plank", "brick", "bricks", "iron", "gold", "golden", "copper", "diamond",
    "emerald", "glass", "wool", "sand", "dirt", "grass", "leaves", "leaf", "ore",
    "sword", "axe", "pickaxe", "shovel", "helmet", "boots", "ingot", "nugget",
    "gem", "crystal", "potion", "bottle", "door", "lamp", "lantern", "red",
    "blue", "green", "yellow", "black", "white", "gray", "grey", "purple",
    "orange", "brown", "pink", "cyan", "metal", "metallic",
}


def load_model(ckpt, device="cuda"):
    sd = torch.load(ckpt, map_location="cpu")
    model = MCFlowDiT(ModelConfig.from_dict(sd["model_cfg"]))
    model.load_state_dict(sd["model"])
    return model.eval().to(device), sd.get("step")


def generate(model, enc, prompts, device="cuda", steps=24, cfg=2.5, seeds=(0,)):
    h, mask = enc.encode(prompts)
    null_h = torch.zeros_like(h[:1])
    null_m = torch.zeros_like(mask[:1])
    outs = []
    for i in range(len(prompts)):
        per_seed = []
        for s in seeds:
            g = torch.Generator(device=device).manual_seed(int(s))
            z = torch.randn(1, model.cfg.in_channels, model.cfg.image_size,
                            model.cfg.image_size, generator=g, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = sample(model, z, h[i:i + 1], steps=steps, cfg=cfg,
                           text_uncond=null_h, solver="heun",
                           text_mask=mask[i:i + 1], text_uncond_mask=null_m)
            per_seed.append(to_uint8(x[0]).permute(1, 2, 0).cpu().numpy())
        outs.append(per_seed)
    return outs


def hist_cosine(a, b, bins=4):
    def hist(arr):
        rgb = arr[..., :3][arr[..., 3] > 127]
        if len(rgb) == 0:
            return np.zeros(bins ** 3, dtype=np.float32)
        q = (rgb // (256 // bins)).astype(int)
        idx = q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2]
        h = np.bincount(idx, minlength=bins ** 3).astype(np.float32)
        n = np.linalg.norm(h)
        return h / n if n > 0 else h
    return float(np.dot(hist(a), hist(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/stage_2_grounded_k1/best.pt"))
    ap.add_argument("--model", default=str(ROOT / "configs/model/base_flux2klein.yaml"))
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--text-tower", default="/home/iflab/models/Qwen3-8B")
    ap.add_argument("--vlm", default="/home/iflab/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs/eval_grounded"))
    args = ap.parse_args()

    build = Path(args.build)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    model, step = load_model(args.ckpt, device)
    enc = FrozenTextEncoder(args.text_tower, device=device, dtype="bfloat16",
                            max_length=512, layers=[9, 18, 27])
    gp = pd.read_parquet(build / "grounded_prompts.parquet")
    meta = pd.read_parquet(build / "metadata.parquet", columns=["type"])
    splits = json.load(open(build / "splits.json"))
    N = len(meta)
    img = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                    shape=(N, 32, 32, 4))

    rng = random.Random(0)
    blocks = [i for i in splits["val"] if meta["type"].values[i] == "block"]
    items = [i for i in splits["val"] if meta["type"].values[i] == "item"]
    picks = rng.sample(blocks, args.n // 2) + rng.sample(items, args.n // 2)
    prompts = [str(gp["prompt_0"].iloc[i]) for i in picks]
    print(f"ckpt step={step} samples={len(picks)}")

    outs = generate(model, enc, prompts, device)

    # ---- real vs generated sheet (first 8 block + 8 item) ----
    S, gap = 160, 6
    sheet = Image.new("RGB", (8 * S + 9 * gap, 4 * S + 5 * gap), (245, 245, 245))
    for k, i in enumerate(picks[:16]):
        for row, arr in enumerate((np.asarray(img[i]), outs[k][0])):
            pil = Image.fromarray(arr, "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            r, c = row, k % 8
            sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST),
                        (gap + c * (S + gap), gap + r * (S + gap)))
    sheet.save(out / "real_vs_generated.png")

    # ---- concept recall (VLM) ----
    recall = {}
    captions = [None] * len(picks)
    if not args.no_vlm:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(args.vlm)
        vlm = AutoModelForImageTextToText.from_pretrained(args.vlm, dtype=torch.bfloat16).to(device).eval()
        hits = {"block": [0, 0], "item": [0, 0]}
        for k, i in enumerate(picks):
            arr = outs[k][0]
            words = [w for w in re.findall(r"[a-z0-9]+", prompts[k].lower()) if w not in STOP]
            words = list(dict.fromkeys(words))
            if not words:
                continue
            pil = Image.fromarray(arr, "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            msgs = [{"role": "user", "content": [{"type": "image", "image": bg},
                    {"type": "text", "text": "Name the main object, material and colour in a few words. Answer briefly."}]}]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(device)
            with torch.no_grad():
                o = vlm.generate(**inp, max_new_tokens=24, do_sample=False)
            cap = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].lower()
            captions[k] = cap
            kind = "block" if meta["type"].values[i] == "block" else "item"
            hit = sum(1 for w in words if re.search(rf"\b{re.escape(w)}s?\b", cap))
            hits[kind][0] += hit; hits[kind][1] += len(words)
        recall = {k: round(v[0] / max(v[1], 1), 4) for k, v in hits.items()}

    # ---- colour accuracy + fidelity + seam + diversity ----
    colour_ok = colour_n = 0
    rgb_l2, hist_cos, seams, seams_real = [], [], [], []
    for k, i in enumerate(picks):
        gen = outs[k][0]; real = np.asarray(img[i])
        words = set(re.findall(r"[a-z0-9]+", prompts[k].lower()))
        cols = words & COLOR_WORDS
        if cols:
            gop = gen[..., :3][gen[..., 3] > 127]
            rop = real[..., :3][real[..., 3] > 127]
            if len(gop) and len(rop):
                colour_n += 1
                rgb_l2.append(float(np.linalg.norm(gop.mean(0) - rop.mean(0))))
                hist_cos.append(hist_cosine(gen, real))
                if float(np.linalg.norm(gop.mean(0) - rop.mean(0))) < 90:
                    colour_ok += 1
        if meta["type"].values[i] == "block":
            seams.append(float(seam_score(gen[..., :3])[0]))
            seams_real.append(float(seam_score(real[..., :3])[0]))

    # ---- common-concept recall from captions (if VLM ran) ----
    common_recall = {}
    if all(c is not None for c in captions):
        per = {"block": [], "item": []}
        for k, i in enumerate(picks):
            words = set(re.findall(r"[a-z0-9]+", prompts[k].lower())) & COMMON
            if not words:
                continue
            cap = captions[k]
            hit = sum(1 for w in words if re.search(rf"\b{re.escape(w)}s?\b", cap))
            per["block" if meta["type"].values[i] == "block" else "item"].append(hit / len(words))
        common_recall = {k: round(float(np.mean(v)), 3) for k, v in per.items() if v}

    # ---- retrieval: is the generated image closest to its own real image? ----
    retrieval = None
    try:
        import imagehash

        gh = [imagehash.phash(Image.fromarray(np.asarray(o[0])[..., :3])) for o in outs]
        rh = [imagehash.phash(Image.fromarray(np.asarray(img[i])[..., :3])) for i in picks]
        retrieval = round(sum(1 for k, h in enumerate(gh)
                              if min(range(len(rh)), key=lambda j: h - rh[j]) == k) / len(picks), 3)
    except Exception as exc:
        print("retrieval skipped:", exc)

    div = []
    for k in range(0, 8):
        seeds = generate(model, enc, [prompts[k]], device, seeds=(0, 1, 2, 3))[0]
        stack = np.stack([s[..., :3].astype(np.float32) for s in seeds])
        div.append(float(stack.std(0).mean()))

    # ---- text effect: val flow-MSE real text vs null ----
    text_effect = None
    try:
        from train.flow import rand_timesteps, sample_data_noise
        v_picks = (blocks + items)
        random.Random(2).shuffle(v_picks)
        v_picks = v_picks[:256]
        v_prompts = [str(gp["prompt_0"].iloc[i]) for i in v_picks]
        v_real = np.stack([np.asarray(img[i]) for i in v_picks])
        x = torch.from_numpy(v_real.astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2).to(device)
        th, tm = enc.encode(v_prompts)
        a = torch.nn.functional.mse_loss
        tr = tn = 0.0; nn = 0
        for s in range(0, len(v_picks), 64):
            xb = x[s:s + 64]; hb = th[s:s + 64]; mb = tm[s:s + 64]
            t = rand_timesteps(xb.shape[0], device=device)
            xt, z, tgt = sample_data_noise(xb, t)
            nh = torch.zeros_like(hb); nm = torch.zeros_like(mb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vr = model(xt, t, hb, text_mask=mb).float()
                vn = model(xt, t, nh, text_mask=nm).float()
            b = xb.shape[0]
            tr += a(vr, tgt).item() * b; tn += a(vn, tgt).item() * b; nn += b
        text_effect = round(100 * (tn - tr) / tn, 1)
    except Exception as exc:
        print("text-effect skipped:", exc)

    report = {
        "ckpt_step": step,
        "samples": len(picks),
        "concept_recall_all_words": recall,
        "concept_recall_common": common_recall,
        "colour_prompts": colour_n,
        "colour_match_rate": round(colour_ok / max(colour_n, 1), 3),
        "real_vs_gen_rgb_l2": round(float(np.mean(rgb_l2)), 1) if rgb_l2 else None,
        "real_vs_gen_hist_cosine": round(float(np.mean(hist_cos)), 3) if hist_cos else None,
        "seam_score_mean": round(float(np.mean(seams)), 2) if seams else None,
        "seam_score_real": round(float(np.mean(seams_real)), 2) if seams_real else None,
        "diversity_seed_std": round(float(np.mean(div)), 1),
        "retrieval_accuracy": retrieval,
        "text_effect_pct": text_effect,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("saved ->", out / "real_vs_generated.png")


if __name__ == "__main__":
    main()
