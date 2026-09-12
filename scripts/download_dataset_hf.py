"""Download the MC-Gen 2.0 training datasets into the project tree.

Usage:
    pip install -U "huggingface_hub[hf_xet]"
    python scripts/download_dataset_hf.py                    # everything
    python scripts/download_dataset_hf.py --only stage1_32_rgba

Files are placed at the exact paths the training configs expect, because the
Hugging Face repo mirrors the local ``data/`` layout.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = [
    "data/build/stage1_32_rgba",
    "data/build/mc_text2image32",
    "data/processed/minecraft_16x_finetune32",
    "data/processed/modrinth32",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="Risposta/MC_Gen")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--only", action="append", default=[],
                    help="download only the named artifact(s); repeatable")
    ap.add_argument("--local-dir", default=str(ROOT))
    args = ap.parse_args()

    names = [Path(a).name for a in ARTIFACTS]
    selected = [a for a in ARTIFACTS if not args.only or Path(a).name in args.only]
    if not selected:
        raise SystemExit(f"unknown --only value; choose from {names}")

    allow = [f"{a}/*" for a in selected]
    print(f"downloading {allow}")
    print(f"  from {args.repo_id} ({args.repo_type}@{args.revision})")
    print(f"  into {args.local_dir}")
    path = snapshot_download(
        repo_id=args.repo_id, repo_type=args.repo_type, revision=args.revision,
        allow_patterns=allow, local_dir=args.local_dir,
    )
    print(f"done: {path}")


if __name__ == "__main__":
    main()
