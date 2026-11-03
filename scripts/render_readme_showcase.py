"""Render paper-style train/test qualitative panels from verified model outputs."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/images/showcase'


def wrap_prompt(prompt, face, max_width):
    lines, current = [], ''
    for word in prompt.split():
        proposed = (current + ' ' + word).strip()
        width = TextPath((0, 0), proposed, prop=face).get_extents().width
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = proposed
    lines.append(current)
    assert len(lines) <= 3, f'Caption exceeds the three-line figure layout: {prompt}'
    return '\n'.join(lines)


def render():
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    assert manifest['checkpoint_step'] == 4128
    assert len(manifest['items']) == 32
    plt.rcParams.update({'font.family': 'STIXGeneral', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    face = FontProperties(family='STIXGeneral', style='italic', size=14.2)
    text_boxes, image_boxes = [], {}
    for split, start, title in [('train', .035, '(a) Training-set captions'),
                                ('test', .535, '(b) Held-out test captions')]:
        items = [item for item in manifest['items'] if item['split'] == split]
        assert len(items) == 16
        fig.text(start, .973, title, fontsize=20, fontweight='bold', va='top', color='#171717')
        fig.add_artist(Line2D([start, start + .43], [.925, .925],
                             transform=fig.transFigure, color='#777777', linewidth=.7))
        for index, item in enumerate(items):
            row, col = divmod(index, 4)
            center_x = start + .0525 + col * .1085
            top = .904 - row * .214
            image_height = .125
            image_width = image_height * 9 / 16
            axis = fig.add_axes([center_x - image_width / 2, top - image_height,
                                 image_width, image_height])
            path = ASSETS / item['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            with Image.open(path) as raw:
                assert raw.mode == 'RGBA' and raw.size == (16, 16)
                background = Image.new('RGBA', raw.size, (246, 246, 246, 255))
                background.alpha_composite(raw)
                display = np.array(background.convert('RGB'))
            axis.imshow(display, interpolation='nearest', resample=False)
            axis.set_axis_off()
            image_boxes[(split, row, col)] = axis
            label = wrap_prompt(item['prompt'], face, max_width=.102 * 16 * 72)
            text = fig.text(center_x, top - image_height - .011, label,
                            fontproperties=face, ha='center', va='top',
                            linespacing=1.08, color='#202020')
            text_boxes.append((split, row, col, text))
    fig.add_artist(Line2D([.5, .5], [.070, .975], transform=fig.transFigure,
                         color='#d5d5d5', linewidth=.6))
    fig.text(.5, .022, 'All images are generated from exact dataset captions; same checkpoint and seed throughout.',
             fontsize=11, ha='center', va='center', color='#555555')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for split, row, col, text in text_boxes:
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width
        if row < 3:
            next_image = image_boxes[(split, row + 1, col)].get_window_extent(renderer)
            assert bounds.y0 > next_image.y1, f'Caption overlaps the next sample: {text.get_text()}'
        else:
            assert bounds.y0 > .040 * fig.bbox.height
    fig.savefig(ASSETS / 'showcase.png', dpi=180)
    fig.savefig(ASSETS / 'showcase.pdf', metadata={
        'Title': 'MCGen qualitative generation from training and held-out test captions',
        'Creator': 'MCGen paper-style showcase renderer',
    })
    plt.close(fig)


if __name__ == '__main__':
    render()
