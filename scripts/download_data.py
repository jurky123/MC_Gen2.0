import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(script, args):
    subprocess.run([sys.executable, str(script), *args], check=True, cwd=str(ROOT))


def cmd_ambientcg(args):
    a = ["--out", args.out]
    if args.limit:
        a += ["--limit", str(args.limit)]
    if args.resolution:
        a += ["--resolution", args.resolution]
    if args.dry_run:
        a += ["--dry-run"]
    run(SRC / "data" / "ambientcg_downloader.py", a)


def cmd_mods(args):
    a = ["--out", args.out, "--size", str(args.size)]
    if args.include_review:
        a += ["--include-review"]
    if args.limit:
        a += ["--limit", str(args.limit)]
    if args.token:
        a += ["--token", args.token]
    if args.no_download:
        a += ["--no-download"]
    run(SRC / "data" / "ovaware_builder.py", a)


def cmd_kenney(args):
    a = ["--out", args.out]
    if args.slugs:
        a += ["--slugs"] + args.slugs
    if args.dry_run:
        a += ["--dry-run"]
    if args.no_extract:
        a += ["--no-extract"]
    run(SRC / "data" / "kenney_downloader.py", a)


def cmd_resourcepacks(args):
    a = ["--out", args.out, "--max", str(args.max)]
    if args.dry_run:
        a += ["--dry-run"]
    if args.extract:
        a += ["--extract"]
    run(SRC / "data" / "modrinth_bulk.py", a)


def main():
    ap = argparse.ArgumentParser(description="download data for all (non-finetune) stages")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ambientcg", help="Stage A: ambientCG CC0 basecolor materials")
    p.add_argument("--out", default="data/raw/matsynth")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resolution", default="1K-JPG")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("mods", help="Stage B: OVAWARE 16xModdedMinecraft (mod textures, HF gated)")
    p.add_argument("--out", default="data/build/mc_b")
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--include-review", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--token", default="")
    p.add_argument("--no-download", action="store_true")

    p = sub.add_parser("kenney", help="Stage A.5: Kenney CC0 pixel/tile packs")
    p.add_argument("--out", default="data/raw/kenney")
    p.add_argument("--slugs", nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-extract", action="store_true")

    p = sub.add_parser("resourcepacks", help="Stage B: Modrinth CC0/CC-BY resource packs (bulk)")
    p.add_argument("--out", default="data/raw/mc/downloads")
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extract", action="store_true", help="extract block textures after download")

    args = ap.parse_args()
    if args.cmd == "ambientcg":
        cmd_ambientcg(args)
    elif args.cmd == "mods":
        cmd_mods(args)
    elif args.cmd == "kenney":
        cmd_kenney(args)
    elif args.cmd == "resourcepacks":
        cmd_resourcepacks(args)


if __name__ == "__main__":
    main()