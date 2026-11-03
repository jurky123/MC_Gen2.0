"""Lay out the checked-in model outputs without modifying their pixels."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/images/showcase'


def font(size, chinese=False, bold=False):
    path = ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc' if chinese else
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(path, size)


def render():
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    canvas = Image.new('RGB', (1600, 1160), '#f4f3ed')
    draw = ImageDraw.Draw(canvas)
    ink, muted, green = '#233d31', '#768076', '#3f7754'
    draw.text((60, 40), 'MCGEN', font=font(76, bold=True), fill=ink)
    draw.text((64, 138), '用文字，创造你的物品。', font=font(30, chinese=True), fill=ink)
    draw.text((1110, 75), 'TEXT TO ITEMS', font=font(19, bold=True), fill=green)
    draw.text((1110, 110), 'Minecraft-inspired pixel art', font=font(16), fill=muted)
    for row, col, color in [(0, 1, '#acc39b'), (0, 2, '#d9dfb9'),
                            (1, 0, '#d9dfb9'), (1, 1, '#5c9064'), (1, 2, '#89ad76'),
                            (2, 0, '#acc39b'), (2, 1, '#89ad76')]:
        x, y = 1432 + col * 29, 55 + row * 29
        draw.rectangle((x, y, x + 24, y + 24), fill=color)
    draw.line((60, 202, 1540, 202), fill='#dce0d4', width=2)
    draw.rectangle((60, 200, 132, 204), fill=green)
    categories = [('sword', '剑'), ('gem', '宝石'), ('helmet', '头盔'),
                  ('apple', '苹果'), ('pickaxe', '镐'), ('shield', '盾牌')]
    for col, (category, label) in enumerate(categories):
        x = 60 + col * 250
        draw.text((x + 8, 220), label, font=font(23, chinese=True), fill=ink)
        rows = [r for r in manifest['items'] if r['category'] == category]
        assert len(rows) == 3
        for row, item in enumerate(rows):
            y = 266 + row * 254
            draw.rounded_rectangle((x, y, x + 230, y + 236), radius=18,
                                   fill='#ffffff', outline='#e2e5db', width=1)
            draw.rounded_rectangle((x + 15, y + 15, x + 215, y + 175), radius=12,
                                   fill='#e9ede5')
            path = ASSETS / item['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            sprite = Image.open(path).convert('RGBA')
            assert sprite.size == (16, 16)
            sprite = sprite.resize((128, 128), Image.Resampling.NEAREST)
            canvas.paste(sprite, (x + 51, y + 31), sprite)
            label_font = font(18)
            width = draw.textlength(item['prompt'], font=label_font)
            draw.text((x + (230 - width) / 2, y + 194), item['prompt'], font=label_font, fill=ink)
    draw.line((60, 1051, 1540, 1051), fill='#dce0d4', width=2)
    draw.text((60, 1083), '真实模型输出  ·  仅放大展示，保留原始像素', font=font(20, chinese=True), fill=muted)
    draw.text((1320, 1088), 'MCGen / 2026', font=font(17), fill=muted)
    canvas.save(ASSETS / 'showcase.png', optimize=True)


if __name__ == '__main__':
    render()
