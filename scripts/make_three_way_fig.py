"""Three-way data comparison: T1 anchors vs T2 direct-downsample vs T3 SDEdit.
T2 and T3 share the same 8 HDs (same prompts) for a direct comparison."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "pairs/anch_proto"
D = ROOT / "pairs/tierd_17k"
BUILD = ROOT / "data/build/mc_text2image32_wl"
OUT = Path("/tmp/opencode/three_way.png")

try:
    FONT_MID = ImageFont.load_default(size=22)
    FONT_SML = ImageFont.load_default(size=19)
except TypeError:
    FONT_MID = FONT_SML = ImageFont.load_default()

# Review 2026-09-20 round 2 (smooth de-blocked edit prompt): all 8 kept.
VERDICT_A = ["OK", "OK", "OK", "OK", "OK", "OK", "OK", "OK"]


def tile_rgba(arr, s, bg=(255, 255, 255)):
    pil = Image.fromarray(np.asarray(arr, dtype=np.uint8), "RGBA")
    canvas = Image.new("RGB", pil.size, bg)
    canvas.paste(pil.convert("RGB"), mask=pil.split()[3])
    return canvas.resize((s, s), Image.Resampling.NEAREST)


def main():
    rows_a = [json.loads(l) for l in open(A / "manifest.jsonl")]
    man_d = [json.loads(l) for l in open(D / "manifest.jsonl")]
    # 4 block + 4 item from completed tierd samples
    bl = [m for m in man_d if m.get("asset_type") == "block"][:4]
    it = [m for m in man_d if m.get("asset_type") == "item"][:4]
    sel = bl + it
    meta = pd.read_parquet(BUILD / "metadata.parquet")
    n = len(meta)
    imgs = np.memmap(BUILD / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(n, 32, 32, 4))

    S, gap = 200, 8
    W = 3 * S + 4 * gap
    cap_w = W - 16
    meas = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def wrap_px(text, font):
        words, lines, cur = str(text).split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if meas.textlength(trial, font=font) <= cap_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return "\n".join(lines)

    blocks = []
    a_rows = []
    for m, verdict in zip(rows_a, VERDICT_A):
        mc = tile_rgba(np.asarray(imgs[m["row"]]), S)
        hd = Image.open(A / m["hd_png"]).convert("RGB").resize((S, S), Image.Resampling.LANCZOS)
        rf = Image.open(A / m["ref_png"]).convert("RGB").resize((S, S), Image.Resampling.NEAREST)
        color = (120, 255, 120) if verdict == "OK" else (255, 120, 120)
        a_rows.append({"caption": f"[{verdict}] row{m['row']} :: {m['prompt']}",
                       "color": color, "tiles": [mc, hd, rf]})
    blocks.append({"title": "T1 anchor: real MC target (supervision) -> FLUX edit -> HD reference",
                   "cols": ["real MC (32px)", "FLUX HD (512px)", "ref64 (stylizer input)"],
                   "rows": a_rows})
    c_rows = []
    for m in sel:
        hd = Image.open(m["hd_png"]).convert("RGBA")
        t2 = tile_rgba(np.asarray(hd.resize((32, 32), Image.Resampling.NEAREST)), S,
                       bg=(200, 200, 200))
        mc = Image.open(m["mc_png"]).convert("RGBA")
        bg = Image.new("RGB", mc.size, (200, 200, 200))
        bg.paste(mc.convert("RGB"), mask=mc.split()[3])
        t3 = bg.resize((S, S), Image.Resampling.NEAREST)
        c_rows.append({
            "caption": f"#{m['id']} [{m.get('asset_type')}] :: {m['prompt']}",
            "color": (255, 255, 255),
            "tiles": [hd.convert("RGB").resize((S, S), Image.Resampling.LANCZOS), t2, t3]})
    blocks.append({"title": "T2 vs T3 on the SAME 8 HDs: direct nearest-downsample vs SDEdit (t0=0.5)",
                   "cols": ["FLUX HD", "T2 direct MC", "T3 SDEdit MC"],
                   "rows": c_rows})

    tmp = Image.new("RGB", (W, 10))
    dd = ImageDraw.Draw(tmp)
    prep = []
    for b in blocks:
        title = wrap_px(b["title"], FONT_MID)
        tb = dd.multiline_textbbox((0, 0), title, font=FONT_MID)
        rows = []
        for r in b["rows"]:
            cap = wrap_px(r["caption"], FONT_SML)
            cb = dd.multiline_textbbox((0, 0), cap, font=FONT_SML)
            rows.append((cap, max(S + 8, (cb[3] - cb[1]) + S + 16)))
        prep.append((title, tb[3] - tb[1] + 16, rows))
    col_h = 34
    total = sum(th + col_h + sum(h for _, h in rows) + gap * (len(rows) + 1)
                for _, th, rows in prep) + gap
    sheet = Image.new("RGB", (W, total), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    y = gap
    for (b, (title, th, rows)) in zip(blocks, prep):
        d.rectangle([0, y, W, y + th], fill=(45, 60, 45))
        d.text((gap, y + 8), title, font=FONT_MID, fill=(255, 255, 150))
        y += th + 4
        for i, c in enumerate(b["cols"]):
            d.text((gap + i * (S + gap) + 8, y + 4), c, font=FONT_MID, fill=(150, 220, 255))
        y += col_h
        for (cap, h), r in zip(rows, b["rows"]):
            d.text((gap, y + 2), cap, font=FONT_SML, fill=r["color"])
            ty = y + (h - S)
            for i, t in enumerate(r["tiles"]):
                sheet.paste(t, (gap + i * (S + gap), ty))
            y += h + gap
    sheet.save(OUT)
    print(f"saved {OUT} size={sheet.size}")


if __name__ == "__main__":
    main()
