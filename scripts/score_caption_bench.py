"""Score a reviewed MC-Texture-CaptionBench against the VLM annotations.

Inputs:
    review.csv         produced by scripts/build_caption_bench.py --export-review
                       and edited by a human (review_status = ok/edited)
    annotations.jsonl  raw VLM output (scripts/annotate_textures.py)

Outputs:
    <out>.json  machine-readable metric summary
    <out>.md    human-readable report

    python scripts/score_caption_bench.py \
        --review data/build/caption_bench/review.csv \
        --annotations data/build/caption_bench/annotations.jsonl \
        --out data/build/caption_bench/report
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.caption_bench import (  # noqa: E402
    load_jsonl,
    read_review_csv,
    render_markdown,
    score_gold_records,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", default=str(ROOT / "data" / "build" / "caption_bench" / "review.csv"))
    ap.add_argument("--annotations", default=str(ROOT / "data" / "build" / "caption_bench" / "annotations.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "build" / "caption_bench" / "report"))
    args = ap.parse_args()

    review_path, annotations_path = Path(args.review), Path(args.annotations)
    if not review_path.exists():
        raise SystemExit(f"review CSV not found: {review_path}")
    if not annotations_path.exists():
        raise SystemExit(f"annotations not found: {annotations_path}")

    gold_records, human = read_review_csv(review_path)
    if not gold_records:
        raise SystemExit("no reviewed rows found (set review_status to ok/edited in the CSV)")

    annotations_by_source = {}
    latencies_by_source = {}
    models = set()
    for record in load_jsonl(annotations_path):
        source = record.get("source")
        if not source:
            continue
        annotations_by_source[source] = record
        if record.get("latency_s") is not None:
            latencies_by_source[source] = float(record["latency_s"])
        if record.get("model"):
            models.add(record["model"])

    latencies = [latencies_by_source.get(r["source"]) for r in gold_records]
    latencies = [v for v in latencies if v is not None]

    summary = score_gold_records(gold_records, annotations_by_source, human=human, latencies=latencies)
    meta = {
        "model": ", ".join(sorted(models)) or "unknown",
        "review_csv": str(review_path),
        "annotations": str(annotations_path),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(summary, meta), encoding="utf-8")
    print(render_markdown(summary, meta))
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
