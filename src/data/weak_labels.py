import argparse
import json
import re
from pathlib import Path

MATERIALS = {
    "stone", "cobblestone", "granite", "diorite", "andesite", "deepslate", "slate", "basalt", "blackstone", "calcite",
    "tuff", "dripstone", "marble", "limestone", "sandstone", "red_sandstone", "quartz", "netherrack", "endstone",
    "brick", "clay", "terracotta", "concrete", "plaster", "wood", "oak", "birch", "spruce", "jungle", "acacia",
    "dark_oak", "cherry", "mangrove", "crimson", "warped", "bamboo", "planks", "log", "metal", "iron", "gold",
    "copper", "steel", "tin", "silver", "zinc", "nickel", "lead", "aluminium", "aluminum", "titanium", "brass",
    "bronze", "ore", "ruby", "sapphire", "emerald", "diamond", "amethyst", "crystal", "topaz", "citrine", "amber",
    "glass", "ice", "snow", "mud", "soil", "dirt", "sand", "gravel", "clay", "grass", "moss", "mossy", "leaf",
    "leaves", "fabric", "cloth", "wool", "plastic", "rubber", "ceramic", "porcelain", "tile", "pearl", "obsidian",
    "glowstone", "shroom", "mushroom", "wart", "prismarine", "purpur", "sculk", "paper", "cardboard", "bone",
    "coal", "charcoal", "salt", "silicon", "circuit", "machine", "casing", "gear", "pipe", "vent", "lamp", "light",
}

STATES = {
    "mossy", "weathered", "cracked", "chiseled", "carved", "polished", "cut", "smooth", "rough", "raw", "refined",
    "aged", "ancient", "rusty", "coated", "glazed", "engraved", "striped", "faded", "dirty", "painted", "wet",
    "frozen", "molten", "shiny", "dark", "light", "old", "new", "burning", "glowing", "charged", "reinforced",
    "damaged", "broken", "pressed", "milled", "ground", "crushed",
}

FORMS = {
    "brick", "bricks", "tile", "tiles", "slab", "stair", "stairs", "wall", "block", "plate", "panel", "pillar",
    "column", "beam", "bar", "rod", "wire", "chain", "gate", "fence", "door", "trapdoor", "button", "lever",
    "frame", "cap", "casing", "core", "top", "bottom", "side", "front", "back", "face", "layer", "strip",
}


def parse_block_name(name):
    tokens = [t for t in re.split(r"[_\-\s]+", name) if t]
    material, form, state = None, None, []
    for t in tokens:
        if t in STATES:
            state.append(t)
        elif t in MATERIALS and material is None:
            material = t
        elif t in FORMS and form is None:
            form = t
    state = list(dict.fromkeys(state))
    return {"material": material, "form": form, "state": state, "tokens": tokens}


def build_weak_prompt(meta, name=None):
    parts = []
    if meta.get("state"):
        parts.append(" ".join(meta["state"]))
    if meta.get("form"):
        parts.append(meta["form"])
    if meta.get("material"):
        parts.append(meta["material"].replace("_", " "))
    prompt = " ".join(parts).strip()
    if not prompt and name:
        prompt = name.replace("_", " ")
    if prompt:
        prompt += ", pixel-art block texture"
    return prompt


def load_lang(lang_path):
    if not lang_path or not Path(lang_path).exists():
        return {}
    try:
        return json.load(open(lang_path, encoding="utf-8"))
    except Exception:
        return {}


def scan_texture_dir(root, lang_path="", out_jsonl=None):
    root = Path(root)
    lang = load_lang(lang_path)
    records = []
    for p in root.rglob("*.png"):
        rel = p.relative_to(root)
        parts = rel.parts
        if "block" not in parts and "textures" not in parts:
            continue
        name = p.stem
        ns = parts[0] if len(parts) > 1 else "unknown"
        meta = parse_block_name(name)
        key = f"block.{ns}.{name}"
        display = lang.get(key, lang.get(f"block.{name}", ""))
        rec = {
            "path": str(p),
            "rel_path": str(rel),
            "namespace": ns,
            "block_id": f"{ns}:{name}",
            "display_name": display,
            "material": meta["material"],
            "form": meta["form"],
            "state": ",".join(meta["state"]),
            "weak_prompt": build_weak_prompt(meta, name=name),
        }
        records.append(rec)
    if out_jsonl:
        Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--lang", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    records = scan_texture_dir(args.dir, lang_path=args.lang, out_jsonl=args.out)
    print(f"scanned {len(records)} textures")


if __name__ == "__main__":
    main()