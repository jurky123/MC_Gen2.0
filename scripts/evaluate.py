import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.seam import report_directory as seam_report
from eval.diversity import report_directory as diversity_report
from eval.memorization import report_directory as memorization_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory of generated textures (*.png)")
    ap.add_argument("--mmap", default="", help="dataset mmap for memorization check")
    ap.add_argument("--out", default="outputs/eval_report.json")
    args = ap.parse_args()

    report = {"seam": seam_report(args.dir), "diversity": diversity_report(args.dir)}
    if args.mmap:
        report["memorization"] = memorization_report(args.dir, args.mmap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"evaluation report written: {out}")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "rows"}) for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()