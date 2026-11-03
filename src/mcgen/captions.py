"""Versioned visual-caption candidates. Model confidence never grants approval."""
import json
import re


CAPTION_INSTRUCTION = '''These two images show the SAME native 16x16 Minecraft/mod inventory sprite,
enlarged with nearest-neighbor pixels, on white and dark backgrounds. Backgrounds are not part of the item.
Describe the visible sprite, not its gameplay use, rarity, lore, mod, or an imagined material.
Identify whole objects versus components, and compound objects (a flower pot is not a flower;
an axe head without a handle is not a complete axe). Do not force an unfamiliar shape into a familiar category.
Use a plain common object name only when visually justified. If uncertain, use object="unknown",
certainty="ambiguous", and describe colors and shape without inventing identity.
Do not invent colors, text, glow, transparency, perspective, or tiny details that cannot be seen.
Return only JSON with these fields:
{"object":"plain object name or unknown", "colors":["up to three visible color names"],
 "short_caption":"at most 16 English words: visible object and main appearance",
 "detailed_caption":"at most 45 English words describing visible silhouette, orientation, colors, and major parts",
 "certainty":"clear or ambiguous", "uncertainty":"what cannot be established, or empty string"}'''


def parse_candidate(text):
    start = text.find('{')
    if start < 0:
        raise ValueError('caption response contains no JSON object')
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    expected = {'object','colors','short_caption','detailed_caption','certainty','uncertainty'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('caption fields differ from schema')
    for key in expected - {'colors'}:
        if not isinstance(value[key], str):
            raise ValueError(f'{key} must be a string')
        value[key] = ' '.join(value[key].split())
    if not value['object'] or value['certainty'] not in ('clear','ambiguous'):
        raise ValueError('invalid object or certainty')
    if not isinstance(value['colors'], list) or any(not isinstance(c,str) or not c.strip() for c in value['colors']):
        raise ValueError('colors must be a list of nonempty names')
    warnings = ['many_colors'] if len(value['colors']) > 3 else []
    for key, limit in [('short_caption',16),('detailed_caption',45)]:
        if not value[key]:
            raise ValueError(f'{key} missing')
        if len(value[key].split()) > limit:
            warnings.append(key + '_long')
    flags = []
    joined = ' '.join(value[k] for k in ('short_caption','detailed_caption'))
    if re.search(r'\b(?:legendary|powerful|rare|epic|used for|crafted from|grants|crafting recipe)\b',joined,re.I):
        flags.append('unsupported_function_or_lore')
    if value['object'].lower() == 'unknown' or value['certainty'] == 'ambiguous':
        flags.append('uncertain_identity')
    return {'annotation': value, 'quality_flags': flags, 'format_warnings': warnings,
            'status': 'quarantine' if flags else 'candidate_needs_visual_review',
            'approved_for_training': False}
