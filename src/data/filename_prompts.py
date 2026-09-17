"""Turn source filenames / metadata into prompt text.

The original resource-pack filenames are the most reliable labels we have
(they are human-authored and describe the asset), so prompts are derived from
them instead of the noisy VLM captions. The transformation is deterministic:
strip orientation/animation/mod noise, keep the informative concept tokens, and
render them as short natural prompt views.

Word-boundary validation keeps concepts honest (e.g. ``lace`` must not match
``palace``).
"""
from __future__ import annotations

import re

# orientation / face / part tokens that are not the object identity
PART_TOKENS = {
    "top", "bottom", "side", "front", "back", "left", "right", "up", "down",
    "north", "south", "east", "west", "inner", "outer", "middle", "center",
    "centre", "corner", "edge", "end", "ends", "upper", "lower", "base",
    "head", "foot", "feet", "tip", "handle", "blade",  # keep? blade is meaningful; overridden below
}
PART_TOKENS.discard("blade")
PART_TOKENS.discard("handle")

# animation / state-frame tokens
ANIM_TOKENS = {
    "on", "off", "lit", "unlit", "active", "inactive", "open", "closed",
    "powered", "unpowered", "stage", "frame", "flow", "flowing", "still",
    "static", "anim", "animated", "idle", "running", "empty", "full",
}

GENERIC_TOKENS = {
    "texture", "textures", "tex", "icon", "icons", "sprite", "sprites", "tile",
    "tiles", "sheet", "atlas", "blank", "empty", "template", "test", "debug",
    "overlay", "mask", "gui", "hud",
}

COLOR_WORDS = {
    "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta",
    "pink", "brown", "black", "white", "gray", "grey", "silver", "gold",
    "golden", "teal", "lime", "olive", "maroon", "navy", "beige", "tan",
    "violet", "indigo", "turquoise", "crimson", "emerald", "amber", "bronze",
    "copper", "light", "dark", "pale", "bright", "deep",
}

# common mod ids that leak into filenames but are not object concepts
MOD_NOISE = {
    "tconstruct", "tinkers", "tcon", "tcompat", "thermal", "mekanism", "create",
    "botania", "ae2", "appliedenergistics", "immersiveengineering", "gtceu",
    "gregtech", "curios", "patchouli", "jei", "jade", "ftb", "forge",
    "fabric", "neoforge", "quark", "ars", "nuveau", "occultism", "aether",
}


def _split(stem: str):
    return [t for t in re.split(r"[_\-\.\s]+", str(stem).lower()) if t]


def clean_tokens(stem: str, extra_noise=(), keep_parts=False, keep_anim=False,
                 keep_generic=False):
    """Informative concept tokens of a filename stem (order preserved).

    ``keep_parts``/``keep_anim`` keep orientation and animation words, which the
    grounded rewriter requires (orientation must be preserved verbatim).
    """
    noise = set(extra_noise)
    out = []
    for token in _split(stem):
        if not keep_parts and token in PART_TOKENS:
            continue
        if not keep_anim and token in ANIM_TOKENS:
            continue
        if not keep_generic and token in GENERIC_TOKENS:
            continue
        if token in noise:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        out.append(token)
    dedup = []
    for token in out:
        if not dedup or dedup[-1] != token:
            dedup.append(token)
    return dedup


def label_tokens(record, extra_noise=()):
    """Strict label for the grounded rewriter: keeps orientation + abstract
    words, drops only mod noise / frame numbers / generic UI words."""
    file_name = record.get("file_name") or ""
    stem = re.sub(r"\.png$", "", str(file_name))
    noise = set(extra_noise)
    for key in ("mod_slug", "project_id"):
        if record.get(key):
            noise |= set(_split(str(record[key])))
    tokens = clean_tokens(stem, noise, keep_parts=True, keep_anim=True, keep_generic=False)
    if not tokens:
        tokens = [t for t in _split(stem) if not re.fullmatch(r"\d+", t)]
    return tokens


def type_word(record):
    return "item" if str(record.get("type", "")).lower().startswith("item") else "block"


def validate_prompt(prompt: str, tokens, type_word_value: str) -> bool:
    """Every label token (word-boundary, optional plural) and the type word must
    appear in the rewritten prompt."""
    text = str(prompt or "").lower()
    for token in tokens:
        if not re.search(rf"\b{re.escape(token)}s?\b", text):
            return False
    if not re.search(rf"\b{re.escape(type_word_value)}s?\b", text):
        return False
    return True


def fallback_prompt(tokens, type_word_value: str) -> str:
    return (" ".join(tokens) + f" {type_word_value}").strip()



def build_prompts(record, extra_noise=(), views=("concept", "phrase", "structured")):
    """Return a list of prompt strings for one record."""
    file_name = record.get("file_name") or ""
    stem = re.sub(r"\.png$", "", str(file_name))
    noise = set(extra_noise)
    if record.get("mod_slug"):
        noise |= set(_split(str(record["mod_slug"])))
    tokens = clean_tokens(stem, noise)
    if not tokens:
        tokens = [t for t in _split(stem) if not re.fullmatch(r"\d+", t)]
    concept = " ".join(tokens).strip()

    prompts = []
    if "concept" in views and concept:
        prompts.append(concept)
    if "phrase" in views and concept:
        article = "an" if concept[:1] in "aeiou" else "a"
        prompts.append(f"{article} {concept}")
    if "structured" in views:
        # material + form + colour, only from tokens that are actually present
        material = next((t for t in tokens if t not in COLOR_WORDS), concept)
        colors = [t for t in tokens if t in COLOR_WORDS]
        form = next((t for t in tokens if t in
                     {"block", "bricks", "brick", "planks", "plank", "log", "door",
                      "ore", "glass", "leaves", "wool", "carpet", "slab", "stairs",
                      "sword", "pickaxe", "axe", "shovel", "hoe", "helmet", "chestplate",
                      "leggings", "boots", "ingot", "nugget", "gem", "crystal", "dust",
                      "rod", "stick", "bow", "arrow", "potion", "food", "seed", "crop"}), None)
        parts = []
        if colors:
            parts.append(" ".join(colors[:2]))
        if material and material not in colors:
            parts.append(material)
        if form and form not in (material,):
            parts.append(form)
        structured = " ".join(parts).strip()
        if structured and structured != concept:
            prompts.append(structured)
    # de-dup, keep order
    seen = set()
    out = []
    for p in prompts:
        p = re.sub(r"\s+", " ", p).strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def contains_concept(text: str, concept: str) -> bool:
    """Word-boundary concept match (avoids 'lace' matching 'palace')."""
    return re.search(rf"\b{re.escape(concept)}\b", str(text).lower()) is not None
