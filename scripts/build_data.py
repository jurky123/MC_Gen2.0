import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

MODULE_FILES = {
    "generic": (SRC / "data" / "matsynth_builder.py", ["--config", "configs/data/stage_a.yaml"]),
    "materialmaker": (SRC / "data" / "materialmaker_builder.py", ["--config", "configs/data/stage_a.yaml"]),
    "pixel": (SRC / "data" / "pixelize.py", ["--config", "configs/data/stage_a5.yaml"]),
    "crawl": (SRC / "data" / "modrinth_crawler.py", []),
    "extract": (SRC / "data" / "mc_extract.py", []),
    "dedupe": (SRC / "data" / "dedupe.py", ["data/raw/mc/manifest.jsonl", "--out", "data/raw/mc/manifest_deduped.jsonl"]),
    "weak": (SRC / "data" / "weak_labels.py", ["data/raw/mc/extracted", "--out", "data/build/mc_b/weak_labels.jsonl"]),
    "text": (SCRIPTS / "precompute_text.py", []),
}


def run(script, args):
    subprocess.run([sys.executable, str(script), *args], check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="MC texture data pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, (_, args) in MODULE_FILES.items():
        sub.add_parser(name, help="run " + name)
    ap.add_argument("extra", nargs="*", help="extra args passed to the underlying script")
    args = ap.parse_args()

    script, default_args = MODULE_FILES[args.cmd]
    run(script, default_args + args.extra)


if __name__ == "__main__":
    main()