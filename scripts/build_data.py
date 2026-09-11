import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

MODULE_FILES = {
    "ambientcg": (SRC / "data" / "ambientcg_downloader.py", ["--out", "data/raw/matsynth"]),
    "kenney": (SRC / "data" / "kenney_downloader.py", ["--out", "data/raw/kenney"]),
    "mods": (SRC / "data" / "ovaware_builder.py", ["--out", "data/build/mc_b"]),
    "resourcepacks": (SRC / "data" / "modrinth_bulk.py", ["--out", "data/raw/mc/downloads", "--extract"]),
    "generic": (SRC / "data" / "matsynth_builder.py", ["--config", "configs/data/stage_a.yaml"]),
    "materialmaker": (SRC / "data" / "materialmaker_builder.py", ["--config", "configs/data/stage_a.yaml"]),
    "pixel": (SRC / "data" / "pixelize.py", ["--config", "configs/data/stage_a5.yaml"]),
    "dedupe": (SRC / "data" / "dedupe.py", ["data/raw/mc/manifest.jsonl", "--out", "data/raw/mc/manifest_deduped.jsonl"]),
    "weak": (SRC / "data" / "weak_labels.py", ["data/raw/mc/extracted", "--out", "data/build/mc_b/weak_labels.jsonl"]),
    "text": (SCRIPTS / "precompute_text.py", []),
    "stage1": (SRC / "data" / "pixel_training_builder.py", [
        "data/processed/kenney_components32/manifest.jsonl",
        "data/processed/itch_components32/manifest.jsonl",
        "data/processed/kaggle_pixel32/manifest.jsonl",
        "data/processed/alucard_sprites32/manifest.jsonl",
        "data/processed/opengameart_2d32/manifest.jsonl",
        "--dataset", "data/build/mc_text2image32",
        "--out", "data/build/stage1_32_rgba",
        "--channels", "4",
    ]),
    "stage2": (SRC / "data" / "pixel_training_builder.py", [
        "--dataset", "data/build/mc_text2image32",
        "--dataset", "data/build/mc_b16",
        "--out", "data/build/stage2_32",
    ]),
    "modrinth-process": (SRC / "data" / "mc_manifest_processor.py", [
        "data/raw/mc/weak_labels.jsonl", "--out", "data/processed/modrinth32",
    ]),
}


def run(script, args):
    subprocess.run([sys.executable, str(script), *args], check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="MC texture data pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    help_text = {
        "ambientcg": "Stage A: download ambientCG CC0 basecolor materials",
        "kenney": "Stage A.5: download Kenney CC0 pixel/tile packs",
        "mods": "Stage B: build mmap from OVAWARE 16xModdedMinecraft (mod textures)",
        "resourcepacks": "Stage B: crawl + extract Modrinth CC0/CC-BY resource packs",
        "generic": "Stage A: crop MatSynth/ambientCG basecolor -> 32x32 mmap",
        "materialmaker": "Stage A: build procedural material mmap",
        "pixel": "Stage A.5: build pixel bridge mmap",
        "dedupe": "Stage B: dedup extraction manifest",
        "weak": "Stage C: build weak labels",
        "text": "Stage C: precompute text embeddings",
        "stage1": "Build unified pixel assets + all MC pretraining dataset",
        "stage2": "Build unified weak-labelled MC dataset",
        "modrinth-process": "Normalize and exactly deduplicate extracted Modrinth textures",
    }
    for name, (_, args) in MODULE_FILES.items():
        sub.add_parser(name, help=help_text.get(name, "run " + name))
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()

    script, default_args = MODULE_FILES[args.cmd]
    run(script, default_args + args.extra)


if __name__ == "__main__":
    main()
