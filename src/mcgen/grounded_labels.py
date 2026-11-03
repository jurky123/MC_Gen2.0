"""Independent visual evidence and provenance-aware, fail-closed label policy."""
import json
import re
from pathlib import PurePosixPath
from .semantic_captions import filename_words

VERSION = 'grounded_labels_v1'
VISUAL_INSTRUCTION = """Describe the foreground Minecraft/mod inventory texture in the two transparency previews of the SAME object.
The white/gray surroundings are not part of the object. You have no filename.
Most examples are complete items. Recognize familiar inventory silhouettes when justified:
a helmet is an arch with an opening; paired boots are one complete inventory icon;
a drawn bow includes its string and may have a nocked arrow; diagonal tools can be complete.
A limited palette or coarse grid does not make an object incomplete. Mark component/layer only with visible evidence,
not just unfamiliar identity. Unknown identity can still be a whole object with a safely describable appearance.
Describe only visible main colors, silhouette and major parts. Do not infer material, function, lore or glow.
Do not mention images, views, backgrounds or rendering. If there are only dots/lines, describe those, not an imagined bar.
Return JSON only: {"object":"visible object name or unknown", "role":"whole or component or layer or uncertain",
"caption":"one concise English sentence about actual visible appearance", "certainty":"clear or ambiguous"}."""
NAME_INSTRUCTION = """Extract a concise semantic name from the filename, using resource paths and independent visual evidence.
Keep ALL named-material and abstract concepts: sun, soul, void, sapphire, glowing, celestial, etc.
A name concept is NOT a visual attribute: glowing sapphire does not prove visible glow; soul does not prove purple.
Unknown words can be item names. Keep them verbatim; do not treat unfamiliarity as identity conflict.
Use conflict only for explicitly incompatible object identities. If identity cannot be confirmed, use uncertain,
keep the filename meaning, and explain the uncertainty. Do not invent missing semantics or drop words.
Preserve component context from paths (cleaver/shield, sword/pommel/material, elixir/cap).
Do not rewrite the visual observation. Remove only numeric indices or the literal tokens item/frame/texture/layer,
and record removed tokens. Return JSON only:
{"semantic_name":"readable name preserving filename words", "removed_tokens":[],
"relation":"compatible or uncertain or conflict", "reason":"one brief sentence"}."""

def parse_json(raw, fields):
    start = raw.find('{')
    if start < 0:
        raise ValueError('missing JSON')
    value, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError('invalid label schema')
    return value


def resource_flags(filename, paths=(), emissive_suffix=None):
    words = set(w.lower() for w in filename_words(filename))
    flags = []
    for path in paths:
        p = '/' + path.lower()
        if any(x in p for x in ('/module/', '/tool/cleaver/', '/trims/', '/elixir/cap/', '/elixir/shape/')):
            flags.append('resource_component')
        if any(x in p for x in ('/overlay/', '/overlays/', '/gui/', '/hud/')):
            flags.append('resource_layer')
    if words & {'blade', 'binding', 'handle', 'toolrod', 'toolhead', 'crossguard', 'guard', 'grip'}:
        flags.append('possible_component')
    if words & {'overlay', 'underlay', 'placeholder', 'cursor', 'slot'}:
        flags.append('possible_layer')
    if emissive_suffix and PurePosixPath(filename).stem.endswith(emissive_suffix):
        flags.append('verified_emissive_layer')
    return sorted(set(flags))


def decide(record, visual, semantic):
    """No old description is carried forward. Unknown identity is never auto-repaired by guessing."""
    if visual['role'] not in ('whole', 'component', 'layer', 'uncertain') or visual['certainty'] not in ('clear', 'ambiguous'):
        raise ValueError('invalid visual classification')
    if semantic['relation'] not in ('compatible', 'uncertain', 'conflict'):
        raise ValueError('invalid name relation')
    for value in (visual['object'], visual['caption'], semantic['semantic_name'], semantic['reason']):
        if not isinstance(value, str):raise ValueError('label must be text')
    if not isinstance(semantic['removed_tokens'], list) or any(not isinstance(x,str) for x in semantic['removed_tokens']):
        raise ValueError('invalid removed tokens')
    flags = list(record['resource_flags'])
    if record['visible_pixels'] < 8:flags.append('insufficient_visible_content')
    if visual['role'] != 'whole' or visual['certainty'] != 'clear':flags.append('visual_identity_unresolved')
    caption = ' '.join(visual['caption'].split())
    if re.search(r'\b(?:backgrounds?|previews?|screenshots?|two views|shown in|image)\b',caption,re.I):
        flags.append('annotation_process_in_caption')
    categories = set('pickaxe sword shovel axe hoe helmet chestplate leggings boots shield ingot gem apple book bow'.split())
    file_types = set(w.lower() for w in filename_words(record['source_filename'])) & categories
    visual_types = set(re.findall(r'[a-z]+',visual['object'].lower())) & categories
    if file_types and visual_types and file_types.isdisjoint(visual_types):
        flags.append('visual_filename_identity_conflict')
    if not caption or '_' in caption or len(caption.split()) > 50:flags.append('invalid_visual_caption')
    name = ' '.join(semantic['semantic_name'].split())
    source = set(w.lower() for w in filename_words(record['source_filename']))
    source.update(w.lower() for p in record['resource_paths'] for w in filename_words(p.replace('/', ' ')))
    proposed = set(w.lower() for w in filename_words(name))
    removed = set(w.lower() for t in semantic['removed_tokens'] for w in filename_words(t))
    # Every lost nonnumeric filename token must be explicitly accounted for.
    original = {w.lower() for w in filename_words(record['source_filename']) if not w.isdigit()}
    allowed_removals = {'item', 'frame', 'texture', 'layer'}
    removed_semantics = {w for w in removed if not w.isdigit() and w not in allowed_removals}
    name_ok = bool(name) and '_' not in name and semantic['relation'] != 'conflict' and proposed <= source and original <= proposed | removed and not removed_semantics
    if re.fullmatch(r'[A-Za-z]{1,6}[0-9]+', PurePosixPath(record['source_filename']).stem):name_ok=False
    if not name_ok:flags.append('filename_identity_unverified')
    # Unverified names are suppressed; independently supported visual labels may survive.
    structural = [f for f in flags if f != 'filename_identity_unverified']
    eligible = not structural
    return {'label_version': VERSION, 'file_label': name if eligible and name_ok else None,
            'visual_label': caption if eligible else None, 'visual_observation': visual,
            'semantic_review': semantic, 'quality_flags': sorted(set(flags)),
            'decision': 'eligible_automatic_weak_label' if eligible else 'quarantine',
            'caption_variants': [x for x in (name if eligible and name_ok else None, caption if eligible else None) if x],
            'note': 'Automatic weak labels; provenance and independent observation are evidence, not accuracy certification.'}
