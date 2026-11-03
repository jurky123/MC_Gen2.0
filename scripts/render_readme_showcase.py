"""Render an inventory gallery from verified, unmodified model outputs."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/images/showcase'


def font(size, *, chinese=False, bold=False, mono=False):
    if chinese:
        name = 'NotoSansCJK-Bold.ttc' if bold else 'NotoSansCJK-Regular.ttc'
        path = '/usr/share/fonts/opentype/noto/' + name
    else:
        name = 'DejaVuSansMono' if mono else 'DejaVuSans'
        path = '/usr/share/fonts/truetype/dejavu/' + name + ('-Bold' if bold else '') + '.ttf'
    return ImageFont.truetype(path, size)


def spaced(draw, xy, text, *, fill, size=18, spacing=3):
    x, y = xy
    face = font(size)
    for char in text:
        draw.text((x, y), char, font=face, fill=fill)
        x += draw.textlength(char, font=face) + spacing


def slot(draw, x, y, width, height):
    cut = 8
    points = [(x + cut, y), (x + width - cut, y), (x + width, y + cut),
              (x + width, y + height - cut), (x + width - cut, y + height),
              (x + cut, y + height), (x, y + height - cut), (x, y + cut)]
    draw.polygon(points, fill='#17252d', outline='#30414a', width=2)
    draw.line((x + cut + 2, y + 3, x + width - cut - 2, y + 3), fill='#3c4c53')
    draw.line((x + cut, y + height - 3, x + width - cut, y + height - 3), fill='#0c151b', width=2)


def render():
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    assert manifest['checkpoint_step'] == 4128
    groups = manifest['groups']
    assert len(groups) == 4 and len(manifest['items']) == 16
    canvas = Image.new('RGB', (1760, 1536))
    draw = ImageDraw.Draw(canvas)
    for y in range(canvas.height):
        amount = y / canvas.height
        draw.line((0, y, canvas.width, y), fill=tuple(int(a + (b - a) * amount)
                  for a, b in zip((10, 21, 29), (15, 29, 35))))
    for x in range(40, 1760, 40):
        for y in range(40, 1536, 40):
            draw.point((x, y), fill='#223139')
    ink, muted, accent = '#edf4ec', '#8fa5ac', '#b3e9c1'
    spaced(draw, (82, 47), 'MINECRAFT-INSPIRED PIXEL GENERATION', fill=muted, size=17, spacing=2.6)
    draw.text((76, 84), 'MCGEN', font=font(116, bold=True), fill=ink)
    draw.text((82, 229), '一句话，创造一个物品。', font=font(44, chinese=True), fill=ink)
    draw.text((1290, 94), '16 × 16', font=font(64, bold=True), fill=accent)
    spaced(draw, (1298, 181), 'NATIVE PIXEL ART', fill=muted, size=17, spacing=2)
    draw.text((1298, 232), 'TEXT  →  ITEM', font=font(24, mono=True), fill=ink)
    draw.line((80, 315, 1680, 315), fill='#2b3d45', width=2)
    draw.rectangle((80, 312, 172, 317), fill=accent)
    descriptions = ['金属、材质与工具', '穿戴与防御', '用颜色表达想法', '让名字成为灵感']
    colors = ['#b3e9c1', '#91cde1', '#cbb9ec', '#e4c18d']
    for row, group in enumerate(groups):
        y = 356 + row * 262
        color = colors[row]
        draw.rectangle((82, y + 14, 92, y + 36), fill=color)
        draw.text((108, y + 13), f'0{row + 1}', font=font(18, mono=True), fill=color)
        draw.text((80, y + 57), group['title'], font=font(30, chinese=True, bold=True), fill=ink)
        spaced(draw, (83, y + 107), group['id'].upper(), fill=color, size=16, spacing=2.3)
        draw.text((82, y + 151), descriptions[row], font=font(20, chinese=True), fill=muted)
        items = [item for item in manifest['items'] if item['category'] == group['id']]
        assert len(items) == 4
        for col, item in enumerate(items):
            x = 400 + col * 330
            slot(draw, x, y, 296, 222)
            path = ASSETS / item['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            with Image.open(path) as image:
                assert image.size == (16, 16) and image.mode == 'RGBA'
                sprite = image.resize((160, 160), Image.Resampling.NEAREST)
            canvas.paste(sprite, (x + 68, y + 12), sprite)
            face = font(26)
            width = draw.textlength(item['prompt'], font=face)
            draw.text((x + (296 - width) / 2, y + 183), item['prompt'], font=face, fill=ink)
    draw.line((80, 1430, 1680, 1430), fill='#2b3d45', width=2)
    draw.text((82, 1462), '真实模型输出  ·  同一固定种子  ·  原始像素放大展示',
              font=font(20, chinese=True), fill=muted)
    draw.text((1390, 1466), 'SELECTED / 4128', font=font(21, mono=True), fill=accent)
    canvas.save(ASSETS / 'showcase.png', optimize=True)


if __name__ == '__main__':
    render()
