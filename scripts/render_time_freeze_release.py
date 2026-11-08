"""Build a compact, auditable comparison of8000 and time-frozen12000 snapshots."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from mcgen.catalog import atomic_json, file_hash


def main():
    experiment = Path('outputs/five_task_instability_20261108/continuation/freeze_time')
    protocol = json.loads((experiment/'protocol.json').read_text())
    rows = json.loads(Path('outputs/reference_five_task_20261108/pairs.json').read_text())
    selected = [0, 1, 2, 3, 8, 9]
    example = Path('examples/time_freeze_reference')
    for name in ('inputs', 'generated'):
        (example/name).mkdir(parents=True, exist_ok=True)
    figure = Path('docs/images/time_freeze_reference')
    figure.mkdir(parents=True, exist_ok=True)
    normal = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    label = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
    canvas = Image.new('RGB', (1080, 154+176*len(selected)+46), '#f1f5f9')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1080, 90), fill='#10243a')
    draw.text((24, 17), 'MC Gen  /  Text and reference generation', font=title, fill='white')
    draw.text((24, 53), 'Five-task supervision  |  original8000 vs time-MLP-frozen12000', font=normal, fill='#b9d9e9')
    columns = ['HD reference', 'MC target', 'Text /8000', 'Text /frozen12000', 'Text + ref /8000', 'Text + ref /frozen12000']
    for c, text in enumerate(columns):
        draw.text((26+c*174, 108), text, font=normal, fill='#334155')
    items = []
    for position, j in enumerate(selected):
        row = rows[protocol['panel_indices'][j]]
        ident = f'{row["split"]}_{position:02d}'
        y = 139+position*176
        draw.rounded_rectangle((16, y, 1064, y+165), radius=10, fill='white', outline='#d8e3ed')
        images = []
        with Image.open(row['reference_file']) as im:
            ref = ImageOps.contain(im.convert('RGBA'), (256, 256), Image.Resampling.LANCZOS)
            ref.save(example/'inputs'/f'{ident}_reference.png')
        target = example/'inputs'/f'{ident}_target.png'
        shutil.copy2(row['target_file'], target)
        source32 = example/'inputs'/f'{ident}_source32.png'
        shutil.copy2(row['source_target_file32'], source32)
        with Image.open(target) as im, Image.open(source32) as original:
            assert im.size == (16, 16) and original.size == (32, 32)
            assert np.array_equal(np.array(im.convert('RGBA')).repeat(2, 0).repeat(2, 1), np.array(original.convert('RGBA')))
        paths = [example/'inputs'/f'{ident}_reference.png', target]
        outputs = {}
        for step, mode in [(8000, 'text'), (12000, 'text'), (8000, 'joint'), (12000, 'joint')]:
            path = example/'generated'/f'{ident}_{mode}_step{step}.png'
            shutil.copy2(experiment/f'eval_{step:06d}'/f'{j:02d}_{mode}.png', path)
            paths.append(path)
            outputs[f'{mode}_{step}'] = {'file': path.relative_to(example).as_posix(), 'sha256': file_hash(path)}
        for c, path in enumerate(paths):
            with Image.open(path) as im:
                rgba = im.convert('RGBA')
                bg = Image.new('RGBA', rgba.size, '#edf2f6')
                bg.alpha_composite(rgba)
                pic = ImageOps.contain(bg.convert('RGB'), (128, 128), Image.Resampling.NEAREST if c else Image.Resampling.LANCZOS)
            x = 26+c*174
            canvas.paste(pic, (x+(148-pic.width)//2, y+9+(128-pic.height)//2))
        prefix = 'HELD OUT' if row['split'] == 'val' else 'TRAIN'
        draw.text((30, y+139), prefix+'  /  '+row['captions'][0], font=label, fill='#17374f')
        items.append({'id': ident, 'source_id': row['id'], 'split': row['split'],
            'short_caption': row['captions'][0], 'detail_caption': row['captions'][-1],
            'source_family': row['source_family'], 'target_hash': row['target_hash'],
            'source_reference_sha256': row['reference_sha256'], 'provenance': row.get('provenance', {}),
            'inputs': {name: {'file': p.relative_to(example).as_posix(), 'sha256': file_hash(p)}
                for name, p in [('reference', paths[0]), ('target', target), ('source32', source32)]},
            'outputs': outputs, 'target16_is_exact_2x_collapse': True})
    draw.text((24, canvas.height-33), 'Fixed first4 validation + first2 training cases. Same detailed captions, noise and solver. Native16 RGBA; nearest-neighbour display.', font=normal, fill='#475569')
    canvas.save(figure/'comparison.png')
    atomic_json(example/'manifest.json', {'source_checkpoint_step': 8000, 'time_frozen_checkpoint_step': 12000,
        'time_frozen_updates': 4000, 'seed': 2026110907, 'steps': 24, 'cfg': 1., 'precision': 'float32',
        'selection': 'first4 validation and first2 training monitoring cases; no output-quality filtering',
        'scope': 'held out of reference fine-tuning, not a claim about base pretraining', 'items': items})
    (example/'README.md').write_text('# 五任务与时间MLP冻结对照\n\n'
        '固定取4个验证、2个训练目标，保留失败结果。比较未冻结的8000步与冻结时间MLP后续训4000步、到12000步的版本。\n\n'
        '![生成对照](../../docs/images/time_freeze_reference/comparison.png)\n\n'
        '`manifest.json`包含粗标、精标、来源、划分与文件哈希。`inputs/`保留16格目标、可无损还原的32格来源和缩小展示的高清参考；`generated/`为原始16格输出。所有目标都核验为精确2倍重复，不含有损32→16候选。\n')
    print('BUILT',len(items),'examples;', (figure/'comparison.png').stat().st_size, 'preview bytes', flush=True)


if __name__ == '__main__':
    main()
