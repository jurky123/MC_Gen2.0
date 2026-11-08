"""Restore exact16 grids and compare explicit lossy32-to16 MC target variants."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mcgen.catalog import atomic_json, file_hash
from mcgen.images import content_hash, rgba
from mcgen.target_downsample import METHODS, to_native16


def thumb(value, size=128):
    im = Image.fromarray(rgba(value)) if isinstance(value, np.ndarray) else value.convert('RGBA')
    background = Image.new('RGBA', im.size, '#edf0f4')
    background.alpha_composite(im)
    return background.convert('RGB').resize((size, size), Image.Resampling.NEAREST)


def conversion_metrics(source, output):
    large = output.repeat(2, 0).repeat(2, 1).astype(np.float64)/255
    original = source.astype(np.float64)/255
    a, b = large[..., 3:], original[..., 3:]
    p = np.concatenate((large[..., :3]*a, a), -1)
    q = np.concatenate((original[..., :3]*b, b), -1)
    mask_a, mask_b = a[..., 0] >= .5, b[..., 0] >= .5
    union = np.logical_or(mask_a, mask_b).sum()
    colors_before = {tuple(v) for v in source[source[..., 3] > 0, :3]}
    colors_after = {tuple(v) for v in output[output[..., 3] > 0, :3]}
    return {'premultiplied_mae': float(np.abs(p-q).mean()),
        'alpha_iou': float(np.logical_and(mask_a, mask_b).sum()/union) if union else 1.,
        'opacity_mass_ratio': float(a.sum()/max(1e-12, b.sum())),
        'visible_colors_before': len(colors_before), 'visible_colors_after': len(colors_after),
        'introduced_rgb_colors': len(colors_after-colors_before),
        'visible_pixels16': int((output[..., 3] >= 128).sum()),
        'empty': not bool(output[..., 3].any())}


def write_png(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.png.tmp')
    Image.fromarray(pixels).save(tmp, format='PNG')
    tmp.replace(path)


def prepare(source_run, out, preview):
    out.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    source_pairs = source_run/'pairs.json'
    rows = json.loads(source_pairs.read_text())
    contract = {'source_run': str(source_run.resolve()), 'source_pairs_sha256': file_hash(source_pairs),
                'methods': list(METHODS), 'output_grid': 16, 'protocol': 'native16_derivatives_v1'}
    if (out/'source.json').exists() and json.loads((out/'source.json').read_text()) != contract:
        raise ValueError('Existing derivatives belong to another source or conversion protocol')
    atomic_json(out/'source.json', contract)
    conversions = {}
    for i, row in enumerate(rows):
        key = row['target_hash']
        if key in conversions:
            continue
        with Image.open(row['target_file']) as im:
            source = rgba(im)
        if content_hash(source) != key or source.shape != (32, 32, 4):
            raise ValueError('Source target changed or has unexpected grid')
        variants = {}
        for method in METHODS:
            result, lossless = to_native16(source, method)
            directory = 'lossless' if lossless else method
            path = out/directory/f'{key}.png'
            write_png(path, result)
            if lossless and not np.array_equal(result.repeat(2, 0).repeat(2, 1), source):
                raise ValueError('Exact restoration failed its32px roundtrip')
            variants[method] = {'file': str(path.resolve()), 'hash': content_hash(result),
                'lossless': lossless, 'method': 'exact_repeat_collapse' if lossless else method,
                'metrics': conversion_metrics(source, result)}
        conversions[key] = {'source_file': row['target_file'], 'variants': variants}
        if len(conversions)%250 == 0:
            print('CONVERTED UNIQUE', len(conversions), flush=True)
    manifests = {}
    collision_reports = {}
    for method in METHODS:
        converted = []
        for i, row in enumerate(rows):
            value = conversions[row['target_hash']]['variants'][method]
            item = copy.deepcopy(row)
            item.update(source_pair_index=i, source_target_file32=row['target_file'],
                source_target_hash32=row['target_hash'], target_file=value['file'], target_hash=value['hash'],
                source_split=row['split'], target_conversion={
                    'method': value['method'], 'lossless': value['lossless'], 'source_canvas': [32, 32],
                    'output_canvas': [16, 16], 'status': 'lossless_restored' if value['lossless'] else 'lossy_candidate',
                    'source_quality_tier': row['quality_tier'], 'metrics': value['metrics']})
            converted.append(item)
        path = out/f'pairs_{method}.json'
        atomic_json(path, converted)
        manifests[method] = {'path': str(path.resolve()), 'sha256': file_hash(path)}
        owners = defaultdict(list)
        for row in converted:
            owners[row['target_hash']].append(row)
        collisions = []
        for key, group in owners.items():
            splits = {r['split'] for r in group}
            if len(splits) > 1:
                collisions.append({'target_hash16': key, 'splits': sorted(splits), 'ids': [r['id'] for r in group]})
        collision_reports[method] = collisions
    atomic_json(out/'cross_split_collision_checks.json', collision_reports)
    hq = [r for r in rows if r['quality_tier'] == 'highest_quality']
    if len(hq) != 1000:
        raise ValueError('Expected all1000 primary HQ pairs')
    paired = []
    summaries = {}
    for method in METHODS:
        group = [conversions[r['target_hash']]['variants'][method]['metrics'] for r in hq
                 if not conversions[r['target_hash']]['variants'][method]['lossless']]
        summaries[method] = {k: float(np.mean([x[k] for x in group])) for k in group[0]}
    for row in hq:
        value = conversions[row['target_hash']]
        entry = {'id': row['id'], 'name': row['captions'][0], 'split': row['split'],
            'source_family': row['source_family'], 'original': value['source_file'],
            'lossless': value['variants']['nearest']['lossless'], 'variants': value['variants']}
        paired.append(entry)
        target = preview/'images'/'original'/f'{row["id"]}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(value['source_file'], target)
        for method in METHODS:
            target = preview/'images'/method/f'{row["id"]}.png'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(value['variants'][method]['file'], target)
    atomic_json(preview/'samples.json', paired)
    exact_pairs = sum(conversions[r['target_hash']]['variants']['nearest']['lossless'] for r in rows)
    exact_hq = sum(r['lossless'] for r in paired)
    summary = {**contract, 'complete': True, 'source_pairs': len(rows), 'unique_source_targets': len(conversions),
        'lossless_pairs': exact_pairs, 'lossy_pairs': len(rows)-exact_pairs,
        'hq_pairs': len(paired), 'hq_lossless_restored': exact_hq, 'hq_true32_converted': len(paired)-exact_hq,
        'unique_lossless_targets': sum(v['variants']['nearest']['lossless'] for v in conversions.values()),
        'unique_lossy_targets': sum(not v['variants']['nearest']['lossless'] for v in conversions.values()),
        'lossy_method_metrics': summaries, 'cross_split_collisions': {k: len(v) for k, v in collision_reports.items()},
        'manifests': manifests, 'reference_images_and_captions_unchanged': True,
        'source_splits_unchanged': True, 'raw_files_and_old_training_run_unchanged': True,
        'training_started': False, 'preview': str(preview.resolve()),
        'method_notes': {'nearest': 'center nearest-neighbour; crisp colors but can drop thin parts',
                         'box': '2x2 premultiplied-alpha area mean; preserves coverage but mixes colors and softens edges',
                         'palette': 'most opaque representative original pixel per2x2; keeps thin parts and palette, can thicken edges'},
        'interpretation': 'Lossy outputs are explicit candidates, not lossless native originals. Metrics do not replace visual inspection.'}
    atomic_json(out/'summary.json', summary)
    atomic_json(preview/'summary.json', summary)
    render(preview, paired)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def render(out, rows):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    lossless = [r for r in rows if r['lossless']]
    lossy = [r for r in rows if not r['lossless']]
    rng = random.Random(202611073)
    stress_names = ['Iron Pickaxe', 'Gold Axe', 'Wither Tipped Arrow', 'Fly Agaric Mushroom',
                    'Quartz Cluster', 'Cold Chisel', 'Woven lidded basket', 'Silk bow tie']
    selected = [next(r for r in lossy if r['name'].casefold() == name.casefold()) for name in stress_names]
    selected += rng.sample([r for r in lossy if r not in selected], 16)
    for page, start in enumerate(range(0, len(selected), 8), 1):
        group = selected[start:start+8]
        canvas = Image.new('RGB', (654, 66+164*len(group)), '#111827')
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 10), 'Actual MC targets |32 original ->16 candidates | integer nearest display', font=font, fill='white')
        for c, label in enumerate(['Original32', 'Nearest16', 'Premult area16', 'Palette/line16']):
            draw.text((14+c*160, 34), label, font=font, fill='#cbd5e1')
        for j, row in enumerate(group):
            y = 62+j*164
            paths = [Path(row['original'])]+[Path(row['variants'][method]['file']) for method in METHODS]
            for c, path in enumerate(paths):
                with Image.open(path) as im:
                    canvas.paste(thumb(im), (14+c*160, y))
            draw.text((14, y+132), row['name'], font=font, fill='white')
            draw.text((14, y+148), f'{row["id"]} | {row["split"]} | explicitly lossy', font=font, fill='#aebbd0')
        canvas.save(out/f'compression_{page:02d}.png')
    group = rng.sample(lossless, 8)
    canvas = Image.new('RGB', (494, 66+164*len(group)), '#111827')
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), 'Exact double-pixel recovery | all500 primary gear pairs verified', font=font, fill='white')
    for c, label in enumerate(['Stored32', 'Restored16', '16 repeated to32']):
        draw.text((14+c*160, 34), label, font=font, fill='#cbd5e1')
    for j, row in enumerate(group):
        y = 62+j*164
        with Image.open(row['original']) as im:
            before = rgba(im)
        with Image.open(row['variants']['nearest']['file']) as im:
            after = rgba(im)
        assert np.array_equal(before, after.repeat(2, 0).repeat(2, 1))
        for c, a in enumerate((before, after, after.repeat(2, 0).repeat(2, 1))):
            canvas.paste(thumb(a), (14+c*160, y))
        draw.text((14, y+132), row['name'], font=font, fill='white')
    canvas.save(out/'lossless_restore.png')
    entries = json.dumps([{'id': r['id'], 'name': r['name'], 'split': r['split'], 'lossless': r['lossless']} for r in rows]).replace('<', '\\u003c')
    body = '<!doctype html><meta charset="utf-8"><title>Native16 target preparation</title><style>body{font:16px sans-serif;background:#111827;color:white;max-width:1000px;margin:24px auto}select{padding:8px;margin:8px}a{color:#93c5fd}.controls{position:sticky;top:0;background:#1f2937;padding:12px}article{padding:16px;background:#1f2937;margin:16px 0;border-radius:10px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.grid img{width:128px;height:128px;image-rendering:pixelated;background:#edf0f4}h3{font-size:15px}</style>'
    body += '<h1>MC target32 ->16 candidates</h1><p>Original source files, reference images, captions and splits retained.500 exact restorations;500 genuinely lossy candidates. No training started.</p><div class="controls"><label>View<select id="kind"><option value="lossy">Lossy32 ->16</option><option value="lossless">Exact restorations</option><option value="all">All1000</option></select></label><label>Split<select id="split"><option value="all">All</option><option value="train">Train</option><option value="val">Validation</option><option value="test">Test</option></select></label></div><div id="items"></div>'
    body += '<script>const rows='+entries+';function draw(){const parent=document.getElementById("items");parent.replaceChildren();const kind=document.getElementById("kind").value,split=document.getElementById("split").value;for(const row of rows){if(kind==="lossless"&&!row.lossless||kind==="lossy"&&row.lossless||split!=="all"&&row.split!==split)continue;const card=document.createElement("article"),title=document.createElement("h3");title.textContent=row.name+" | "+row.id+" | "+row.split+(row.lossless?" | exact":" | lossy candidate");card.append(title);const grid=document.createElement("div");grid.className="grid";for(const [dir,label] of [["original","Original32"],["nearest","Nearest16"],["box","Premultiplied area16"],["palette","Palette / thin-part16"]]){const cell=document.createElement("div"),p=document.createElement("p"),img=document.createElement("img");p.textContent=row.lossless&&dir!=="original"?"Exact restored16":label;img.src="images/"+dir+"/"+row.id+".png";img.alt=label;img.loading="lazy";cell.append(p,img);grid.append(cell)}card.append(grid);parent.append(card)}}document.getElementById("kind").addEventListener("change",draw);document.getElementById("split").addEventListener("change",draw);draw();</script>'
    body += '<p>Static previews: <a href="compression_01.png">thin parts and familiar items</a> / <a href="compression_02.png">random sample1</a> / <a href="compression_03.png">random sample2</a> / <a href="lossless_restore.png">lossless checks</a></p>'
    (out/'index.html').write_text(body)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-run', type=Path, default=Path('outputs/reference_mixed_hq_20261107'))
    p.add_argument('--out', type=Path, default=Path('data/processed/reference_mixed_20261107/native16_candidates'))
    p.add_argument('--preview', type=Path, default=Path('outputs/native16_data_review_20261107'))
    args = p.parse_args()
    prepare(args.source_run, args.out, args.preview)
