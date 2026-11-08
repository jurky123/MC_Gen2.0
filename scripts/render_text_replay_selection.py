"""Inspect unchanged filename labels and native pixels in the fixed replay set."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

from mcgen.catalog import atomic_json
from mcgen.dataset import CatalogDataset
from mcgen.images import decode


def render(root):
    rows = json.loads((root/'text_replay_selection.json').read_text())
    ds = CatalogDataset(rows, 16, 16, 'train', task='text')
    # Verify every selected image, not only the displayed representatives.
    for i in range(len(ds)):
        item = ds[i]  # Checks native grid and the original pixel-content hash.
        if item['image'].shape != (4, 16, 16) or not decode(item['image'])[..., 3].any():
            raise ValueError('Invalid or empty native16 replay target')
    by_hash = {r['target_hash']: i for i, r in enumerate(ds.records)}
    groups = defaultdict(list)
    for r in rows:
        groups[r['replay_type']].append(r)
    # Two rounds across every text-derived type, with no visual cherry-picking.
    selected = [pool[j] for j in range(2) for _, pool in sorted(groups.items()) if len(pool) > j]
    out = root/'data_review'
    out.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    headings = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    for offset in range(0, len(selected), 64):
        page = selected[offset:offset+64]
        canvas = Image.new('RGB', (8*176, 68+((len(page)+7)//8)*168), '#111827')
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f'Original T2I replay: 2000 unique targets | preview {offset//64+1}', font=headings, fill='white')
        draw.text((16, 38), 'Original filename text unchanged | native16, nearest-neighbor display | fixed type-stratified selection', font=font, fill='#aebbd0')
        for j, row in enumerate(page):
            x, y = (j%8)*176+12, 66+(j//8)*168
            pixels = Image.fromarray(decode(ds[by_hash[row['target_hash']]]['image']))
            bg = Image.new('RGBA', (96, 96), '#d9e0e9')
            bg.alpha_composite(pixels.resize((96, 96), Image.Resampling.NEAREST))
            canvas.paste(bg.convert('RGB'), (x+24, y))
            draw.text((x, y+100), row['replay_type'], font=font, fill='#93c5fd')
            label = textwrap.wrap(row['prompt'], width=23)
            for line, text in enumerate(label[:3]):
                draw.text((x, y+116+line*14), text, font=font, fill='white')
        canvas.save(out/f'preview_{offset//64:02d}.png')
    with (out/'selection.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['replay_type', 'prompt', 'source_filename', 'target_hash'])
        writer.writeheader()
        writer.writerows({k: r.get(k, '') for k in writer.fieldnames} for r in rows)
    atomic_json(out/'summary.json', {'targets': len(rows), 'unique_texts': len({r['prompt'] for r in rows}),
        'types': {k: len(v) for k, v in sorted(groups.items())}, 'preview_images': len(selected),
        'native_size': 16, 'pixel_hashes_verified': len(ds),
        'text_modified': False, 'previews': [p.name for p in sorted(out.glob('preview_*.png'))]})
    print(json.dumps({'review': str(out), 'targets': len(rows), 'types': len(groups),
                      'preview_images': len(selected)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    render(p.parse_args().out)
