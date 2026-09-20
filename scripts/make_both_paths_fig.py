"""Combined Path-A / Tier-D review figure (readable layout)."""
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "pairs/anch_proto"
B = ROOT / "pairs/tierd_proto"
BUILD = ROOT / "data/build/mc_text2image32_wl"
OUT = Path("/tmp/opencode/both_paths.png")

try:
    FONT_BIG = ImageFont.load_default(size=26)
    FONT_MID = ImageFont.load_default(size=22)
    FONT_SML = ImageFont.load_default(size=19)
except TypeError:
    FONT_BIG = FONT_MID = FONT_SML = ImageFont.load_default()

# reviewer verdicts from the 2026-09-20 review
VERDICT_A = ["OK", "DRIFT white-out", "DRIFT blue->green", "OK",
             "OK", "OK", "DRIFT blue->gray", "OK"]


def tile_rgba(arr, s, bg=(255, 255, 255)):
    pil = Image.fromarray(np.asarray(arr, dtype=np.uint8), "RGBA")
    canvas = Image.new("RGB", pil.size, bg)
    canvas.paste(pil.convert("RGB"), mask=pil.split()[3])
    return canvas.resize((s, s), Image.Resampling.NEAREST)


def main():
    rows_a = [json.loads(l) for l in open(A / "manifest.jsonl")]
    man_b = [json.loads(l) for l in open(B / "manifest.jsonl")][:8]
    meta = pd.read_parquet(BUILD / "metadata.parquet")
    n = len(meta)
    imgs = np.memmap(BUILD / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(n, 32, 32, 4))

    S, gap = 200, 8
    cap_w = 3 * S + 4 * gap - 16
    W = 3 * S + 4 * gap

    _meas = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def wrap_px(text, font, max_w):
        words, lines, cur = str(text).split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if _meas.textlength(trial, font=font) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return "\n".join(lines)

    def wrapped(text, font, width_chars=52):
        return wrap_px(text, font, cap_w)

    blocks = []
    # ---- Path A ----
    a_rows = []
    for m, verdict in zip(rows_a, VERDICT_A):
        mc = tile_rgba(np.asarray(imgs[m["row"]]), S)
        hd = Image.open(A / m["hd_png"]).convert("RGB").resize((S, S), Image.Resampling.LANCZOS)
        rf = Image.open(A / m["ref_png"]).convert("RGB").resize((S, S), Image.Resampling.NEAREST)
        color = (120, 255, 120) if verdict == "OK" else (255, 120, 120)
        a_rows.append({
            "caption": f"[{verdict}] row{m['row']} :: {m['prompt']}",
            "color": color, "tiles": [mc, hd, rf]})
    blocks.append({
        "title": "PATH A - anchor pairs: real MC target (supervision) -> FLUX edit -> HD reference",
        "cols": ["real MC (32px)", "FLUX HD (512px)", "ref64 (stylizer input)"],
        "rows": a_rows})
    # ---- Tier-D ----
    b_rows = []
    for m in man_b:
        hd = Image.open(m["hd_png"]).convert("RGB")
        ini = hd.convert("RGBA").resize((32, 32), Image.Resampling.NEAREST)
        mc = Image.open(m["mc_png"]).convert("RGBA")
        bg = Image.new("RGB", mc.size, (200, 200, 200))
        bg.paste(mc.convert("RGB"), mask=mc.split()[3])
        b_rows.append({
            "caption": f"#{m['id']} [{m['asset_type']}] :: {m['prompt']}",
            "color": (255, 255, 255),
            "tiles": [hd.resize((S, S), Image.Resampling.LANCZOS),
                      tile_rgba(np.asarray(ini), S),
                      bg.resize((S, S), Image.Resampling.NEAREST)]})
    blocks.append({
        "title": "TIER-D - prompt -> FLUX HD -> SDEdit MC (prompt reused as annotation, t0=0.5)",
        "cols": ["FLUX HD", "nearest 32px init", "SDEdit MC"],
        "rows": b_rows})

    # measure
    tmp = Image.new("RGB", (W, 10))
    dd = ImageDraw.Draw(tmp)
    blocks_wrapped = []
    for b in blocks:
        title = wrap_px(b["title"], FONT_MID, cap_w)
        tb = dd.multiline_textbbox((0, 0), title, font=FONT_MID)
        blocks_wrapped.append((title, tb[3] - tb[1] + 16))
    row_h = []
    for b in blocks:
        hs = []
        for r in b["rows"]:
            tb = dd.multiline_textbbox((0, 0), wrap_px(r["caption"], FONT_SML, cap_w), font=FONT_SML)
            hs.append(max(S + 8, (tb[3] - tb[1]) + S + 16))
        row_h.append(hs)
    col_h = 34
    total = sum(th + col_h + sum(hs) + gap * (len(hs) + 1) for (_, th), hs in zip(blocks_wrapped, row_h)) + gap
    sheet = Image.new("RGB", (W, total), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    y = gap
    for (b, (title, th)), hs in zip(zip(blocks, blocks_wrapped), row_h):
        d.rectangle([0, y, W, y + th], fill=(45, 60, 45))
        d.text((gap, y + 8), title, font=FONT_MID, fill=(255, 255, 150))
        y += th + 4
        for i, c in enumerate(b["cols"]):
            d.text((gap + i * (S + gap) + 8, y + 4), c, font=FONT_MID, fill=(150, 220, 255))
        y += col_h
        for r, h in zip(b["rows"], hs):
            d.text((gap, y + 2), wrap_px(r["caption"], FONT_SML, cap_w), font=FONT_SML,
                   fill=r["color"])
            ty = y + (h - S)
            for i, t in enumerate(r["tiles"]):
                sheet.paste(t, (gap + i * (S + gap), ty))
            y += h + gap
    sheet.save(OUT)
    print(f"saved {OUT} size={sheet.size}")


if __name__ == "__main__":
    main()
