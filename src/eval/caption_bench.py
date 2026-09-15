"""MC-Texture-CaptionBench: build, review and score VLM texture annotations.

Implements the benchmark loop from the plan (sections 20-21): a stratified
sample of textures is annotated by the local VLM, a human reviews/corrects the
fields, and the reviewed set becomes the gold standard used to measure the
model. Everything here is deterministic except the VLM decoding.

The reviewed CSV is the hand-off format (Excel-friendly): every model field is
pre-filled into the matching ``gold_*`` column so the reviewer only edits what
is wrong, then sets ``review_status`` to ``ok`` (accepted) or ``edited``.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ANNOTATION_FIELDS = [
    "material", "form", "state", "dominant_colors", "pattern", "surface",
    "directionality", "details", "emissive", "tileability",
    "short_caption", "detailed_caption", "uncertainty",
]

SCALAR_FIELDS = ["material", "form", "pattern", "surface", "directionality", "tileability"]
SET_FIELDS = ["state", "dominant_colors", "details"]
CAPTION_FIELDS = ["short_caption", "detailed_caption"]

REVIEWED_STATUS = {"ok", "keep", "edited", "done", "yes", "1", "true"}

# Ordered keyword rules -> canonical category (plan section 21). First match wins
# for the primary category; all matches are kept in ``categories``.
CATEGORY_RULES = [
    ("emissive", ("light source", "emissive", "glow", "lamp", "lantern", "torch")),
    ("ore", ("ore",)),
    ("metal", ("metal", "iron", "gold", "copper", "bronze", "silver")),
    ("glass", ("glass",)),
    ("brick", ("brick",)),
    ("wood", ("wood", "log", "planks", "sapling")),
    ("stone", ("stone", "cobble", "deepslate", "basalt", "granite", "diorite",
               "andesite", "obsidian", "quartz", "terracotta", "ceramic")),
    ("sand", ("sand", "gravel", "sandstone")),
    ("soil", ("soil", "dirt", "grass", "mud", "podzol", "farmland", "clay")),
    ("plant", ("plant", "leaves", "flower", "crop", "organic", "aquatic",
               "mushroom", "wart", "vine", "cactus")),
    ("machine", ("utility", "redstone", "machine", "furnace", "crafting", "piston")),
    ("fantasy", ("fantasy", "artifact", "magic", "portal", "nether", "end ",
                 "sculk", "soul")),
    ("decorative", ("decorative",)),
    ("item", ("item -",)),
]

REF_META_KEYS = [
    "texture_name", "type", "texture_size", "primary_colors", "secondary_colors",
    "pattern_description", "texture_style", "lighting_reflection", "symmetry",
    "tileable_direction", "overall_texture_description",
]


def infer_categories(type_str) -> list:
    text = f" {str(type_str or '').lower()} "
    matched = [name for name, keywords in CATEGORY_RULES if any(k in text for k in keywords)]
    return matched or ["other"]


def primary_category(type_str) -> str:
    return infer_categories(type_str)[0]


def stratify(records, size, seed=0, only_blocks=False, category_key="category"):
    """Evenly sample ``size`` records across categories (plan section 21)."""
    pool = list(records)
    if only_blocks:
        pool = [r for r in pool if str((r.get("metadata") or {}).get("type", "")).lower().startswith("block")]
    buckets = defaultdict(list)
    for record in pool:
        buckets[record.get(category_key, "other")].append(record)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    ordered = sorted(buckets, key=lambda c: (-len(buckets[c]), c))
    selected = []
    while len(selected) < min(size, len(pool)):
        progressed = False
        for category in ordered:
            if len(selected) >= size:
                break
            if buckets[category]:
                selected.append(buckets[category].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def _text(value) -> str:
    return str(value or "").strip()


def normalize_text(value) -> str:
    text = _text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(value) -> set:
    if isinstance(value, (list, tuple)):
        tokens = set()
        for item in value:
            tokens |= set(normalize_text(item).split())
        return tokens
    return set(normalize_text(value).split())


def scalar_match(pred, gold) -> int:
    p, g = normalize_text(pred), normalize_text(gold)
    if not p or not g:
        return int(p == g)
    if p == g:
        return 1
    return int(token_set(p) == token_set(g))


def set_f1(pred, gold) -> dict:
    pred_set, gold_set = token_set(pred), token_set(gold)
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def caption_f1(pred, gold) -> float:
    return set_f1(pred, gold)["f1"]


def compare_annotation(pred: dict, gold: dict) -> dict:
    """Field-level agreement between one model annotation and its review."""
    pred, gold = pred or {}, gold or {}
    result = {"scalar": {}, "set_f1": {}, "caption_f1": {}}
    for field in SCALAR_FIELDS:
        result["scalar"][field] = scalar_match(pred.get(field), gold.get(field))
    result["scalar"]["emissive"] = int(bool(pred.get("emissive")) == bool(gold.get("emissive")))
    for field in SET_FIELDS:
        result["set_f1"][field] = set_f1(pred.get(field), gold.get(field))["f1"]
    for field in CAPTION_FIELDS:
        result["caption_f1"][field] = caption_f1(pred.get(field), gold.get(field))
    result["json_exact"] = int(
        all(result["scalar"][f] for f in SCALAR_FIELDS)
        and all(result["set_f1"][f] == 1.0 for f in SET_FIELDS)
    )
    return result


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return float(values[int(pos)])
    return float(values[low] + (values[high] - values[low]) * (pos - low))


def aggregate(rows, latencies=None) -> dict:
    """Aggregate per-row comparisons (see ``compare_annotation``)."""
    n = len(rows)
    if n == 0:
        return {"n_scored": 0}

    scalar = {}
    for field in SCALAR_FIELDS + ["emissive"]:
        scalar[field] = sum(r["scalar"][field] for r in rows) / n
    setf1 = {}
    for field in SET_FIELDS:
        setf1[field] = sum(r["set_f1"][field] for r in rows) / n
    captions = {}
    for field in CAPTION_FIELDS:
        captions[field] = sum(r["caption_f1"][field] for r in rows) / n

    attribute_values = list(scalar.values()) + list(setf1.values())
    summary = {
        "n_scored": n,
        "scalar_accuracy": {k: round(v, 4) for k, v in scalar.items()},
        "set_f1": {k: round(v, 4) for k, v in setf1.items()},
        "caption_token_f1": {k: round(v, 4) for k, v in captions.items()},
        "attribute_macro_f1": round(sum(attribute_values) / len(attribute_values), 4),
        "json_exact_rate": round(sum(r["json_exact"] for r in rows) / n, 4),
    }
    if latencies:
        summary["latency"] = {
            "mean_s": round(sum(latencies) / len(latencies), 3),
            "p50_s": round(_percentile(latencies, 0.5), 3),
            "p95_s": round(_percentile(latencies, 0.95), 3),
        }
        summary["throughput_per_s"] = round(len(latencies) / sum(latencies), 3) if sum(latencies) else None
    return summary


def _join_list(value) -> str:
    if isinstance(value, (list, tuple)):
        return " | ".join(str(v) for v in value)
    return _text(value)


def _split_list(value) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def _bool_str(value) -> str:
    return "true" if bool(value) else "false"


def _parse_bool(value) -> bool:
    return str(value or "").strip().lower() in ("true", "yes", "1", "y", "t")


def review_columns() -> list:
    columns = ["id", "source", "category", "image", "ref_type", "ref_texture_name"]
    columns += [f"ref_{key}" for key in REF_META_KEYS if key not in ("type", "texture_name")]
    columns += [f"model_{field}" for field in ANNOTATION_FIELDS]
    columns += ["gold_" + field for field in ANNOTATION_FIELDS]
    columns += ["human_rating", "human_hallucination", "review_status", "review_notes"]
    return columns


def write_review_csv(records, annotations_by_source, path) -> Path:
    """Pre-fill a reviewer CSV: gold_* starts as the model output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = review_columns()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            meta = dict(record.get("metadata") or {})
            annotation = (annotations_by_source.get(record.get("source")) or {}).get("annotation") or {}
            row = {"id": record.get("id"), "source": record.get("source"),
                   "category": record.get("category"), "image": record.get("path")}
            row["ref_type"] = meta.get("type", "")
            row["ref_texture_name"] = meta.get("texture_name", "")
            for key in REF_META_KEYS:
                if key in ("type", "texture_name"):
                    continue
                row[f"ref_{key}"] = _join_list(meta.get(key, ""))
            for field in ANNOTATION_FIELDS:
                value = annotation.get(field, "")
                row[f"model_{field}"] = _join_list(value) if field in SET_FIELDS else (
                    _bool_str(value) if field == "emissive" else _text(value))
                row[f"gold_{field}"] = row[f"model_{field}"]
            row["human_rating"] = ""
            row["human_hallucination"] = ""
            row["review_status"] = "pending"
            row["review_notes"] = ""
            writer.writerow(row)
    return path


def read_review_csv(path):
    """Return ``(gold_records, human_stats)`` from a reviewer CSV."""
    path = Path(path)
    gold_records = []
    ratings, hallucinations = [], []
    reviewed = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            status = _text(row.get("review_status")).lower()
            if status not in REVIEWED_STATUS:
                continue
            reviewed += 1
            gold = {}
            for field in ANNOTATION_FIELDS:
                key = f"gold_{field}"
                if field in SET_FIELDS:
                    gold[field] = _split_list(row.get(key))
                elif field == "emissive":
                    gold[field] = _parse_bool(row.get(key))
                elif field == "uncertainty":
                    try:
                        gold[field] = float(row.get(key) or 0)
                    except ValueError:
                        gold[field] = 0.0
                else:
                    gold[field] = _text(row.get(key))
            gold_records.append({
                "id": row.get("id"),
                "category": row.get("category"),
                "source": _text(row.get("source")) or row.get("id") or row.get("image"),
                "image": row.get("image"),
                "gold": gold,
            })
            rating = _text(row.get("human_rating"))
            if rating:
                try:
                    ratings.append(float(rating))
                except ValueError:
                    pass
            flag = _text(row.get("human_hallucination")).lower()
            if flag in ("yes", "true", "1", "y", "t", "no", "false", "0", "n", "f"):
                hallucinations.append(1 if flag in ("yes", "true", "1", "y", "t") else 0)
    human = {
        "n_reviewed": reviewed,
        "mean_rating": round(sum(ratings) / len(ratings), 3) if ratings else None,
        "hallucination_rate": round(sum(hallucinations) / len(hallucinations), 4) if hallucinations else None,
    }
    return gold_records, human


def score_gold_records(gold_records, annotations_by_source, human=None, latencies=None) -> dict:
    rows = []
    missing = 0
    for record in gold_records:
        annotation = annotations_by_source.get(record["source"]) or annotations_by_source.get(record["image"])
        if not annotation or not annotation.get("annotation"):
            missing += 1
            continue
        rows.append(compare_annotation(annotation["annotation"], record["gold"]))
    summary = aggregate(rows, latencies=latencies)
    summary["n_gold"] = len(gold_records)
    summary["n_missing_prediction"] = missing
    if human:
        summary["human"] = human
    return summary


def render_markdown(summary, meta=None) -> str:
    lines = ["# MC-Texture-CaptionBench report", ""]
    if meta:
        for key, value in meta.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append(f"- scored samples: {summary.get('n_scored', 0)} / gold {summary.get('n_gold', '?')}")
    if summary.get("n_missing_prediction"):
        lines.append(f"- missing predictions: {summary['n_missing_prediction']}")
    lines.append("")

    def table(title, data, keyfmt="{:.3f}"):
        if not data:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| field | score |")
        lines.append("|---|---:|")
        for key, value in data.items():
            rendered = keyfmt.format(value) if isinstance(value, (int, float)) else str(value)
            lines.append(f"| {key} | {rendered} |")
        lines.append("")

    table("Scalar accuracy", summary.get("scalar_accuracy"))
    table("Set F1", summary.get("set_f1"))
    table("Caption token F1", summary.get("caption_token_f1"))
    if "attribute_macro_f1" in summary:
        lines.append(f"**Attribute macro F1:** {summary['attribute_macro_f1']}")
        lines.append(f"**Full-JSON exact rate:** {summary.get('json_exact_rate')}")
        lines.append("")
    if summary.get("latency"):
        latency = summary["latency"]
        lines.append("## Latency / throughput")
        lines.append("")
        lines.append(f"- mean {latency['mean_s']}s, p50 {latency['p50_s']}s, p95 {latency['p95_s']}s")
        lines.append(f"- throughput {summary.get('throughput_per_s')} samples/s")
        lines.append("")
    if summary.get("human"):
        human = summary["human"]
        lines.append("## Human review")
        lines.append("")
        lines.append(f"- reviewed {human.get('n_reviewed')}")
        lines.append(f"- mean caption rating {human.get('mean_rating')}")
        lines.append(f"- hallucination rate {human.get('hallucination_rate')}")
        lines.append("")
    return "\n".join(lines)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def category_histogram(records, key="category") -> dict:
    return dict(Counter(r.get(key, "other") for r in records).most_common())
