"""Merge rewritten prompt views with a build dataset into an aligned parquet.

Reads ``prompt_views.jsonl`` (source -> K {style,text} prompts) and a build
directory's ``metadata.parquet`` (row order == mmap order, ``source`` is
``"<dataset>#<index>"``) and writes ``prompt_views.parquet`` with
``prompt_0..prompt_{K-1}`` columns aligned to the dataset rows. Rows without a
rewrite fall back to the cleaned ``weak_prompt``/source label so every row has
conditioning text.

    python scripts/merge_prompt_views.py \
        --views  data/build/mc_text2image32_wl/prompt_views.jsonl \
        --build-dir data/build/mc_text2image32_wl \
        --out    data/build/mc_text2image32_wl/prompt_views.parquet

Then encode (N, K, text_dim):

    python scripts/precompute_text.py --config configs/data/stage_2.yaml \
        --views-parquet data/build/mc_text2image32_wl/prompt_views.parquet \
        --encoder /home/iflab/models/Qwen3-VL-Embedding-8B --device cuda:1
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from data.prompt_diversify import RewriterConfig, strip_domain, _compile_strip  # noqa: E402


def load_views(paths):
    views = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                source = record.get("source")
                prompts = record.get("prompts") or []
                if source and prompts:
                    views[source] = prompts
    return views


def fallback_label(row, compiled):
    for key in ("weak_prompt", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return strip_domain(value, compiled) or "unknown"
    file_name = row.get("file_name") or ""
    if file_name:
        stem = re.sub(r"[_\-]+", " ", Path(str(file_name)).stem)
        return strip_domain(stem, compiled) or "unknown"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", nargs="+",
                    default=[str(ROOT / "data" / "build" / "mc_text2image32_wl" / "prompt_views.jsonl")],
                    help="one or more prompt_views JSONL files (e.g. per-shard)")
    ap.add_argument("--build-dir", default=str(ROOT / "data" / "build" / "mc_text2image32_wl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--config", default=str(ROOT / "configs" / "prompt_rewrite.yaml"))
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--dataset-name", default="")
    args = ap.parse_args()

    build_dir = Path(args.build_dir)
    metadata = pd.read_parquet(build_dir / "metadata.parquet")
    dataset = args.dataset_name or build_dir.name

    cfg = RewriterConfig.from_yaml(args.config) if Path(args.config).exists() else RewriterConfig()
    compiled = _compile_strip(cfg.strip_phrases)
    styles = list(cfg.styles)
    k = int(args.k or len(styles))

    positions = metadata["index"].tolist() if "index" in metadata.columns else list(range(len(metadata)))
    sources = [f"{dataset}#{int(p)}" for p in positions]

    views = load_views([Path(p) for p in args.views])
    print(f"rows={len(metadata)} rewritten={len(views)} K={k} styles={styles}")

    rows = metadata.to_dict(orient="records")
    data = {"index": positions, "source": sources}
    for i in range(k):
        data[f"prompt_{i}"] = []
        data[f"style_{i}"] = []
    covered = 0
    for source, row in zip(sources, rows):
        prompts = views.get(source)
        if prompts:
            covered += 1
        texts = [p.get("text", "") for p in (prompts or [])]
        prompt_styles = [p.get("style", styles[i] if i < len(styles) else f"view{i}")
                         for i, p in enumerate(prompts or [])]
        if not texts:
            fallback = fallback_label(row, compiled)
            texts = [fallback] * k
            prompt_styles = styles[:k] if len(styles) >= k else styles + [""] * (k - len(styles))
        for i in range(k):
            data[f"prompt_{i}"].append(texts[i] if i < len(texts) else texts[0])
            data[f"style_{i}"].append(prompt_styles[i] if i < len(prompt_styles) else "")
    out = pd.DataFrame(data)
    out_path = Path(args.out) if args.out else build_dir / "prompt_views.parquet"
    out.to_parquet(out_path, index=False)
    print(f"covered {covered}/{len(metadata)} rows; wrote {out_path}")


if __name__ == "__main__":
    main()
