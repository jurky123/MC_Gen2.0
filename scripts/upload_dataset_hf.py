"""Upload the training-ready MC-Gen 2.0 datasets to a Hugging Face dataset repo.

Usage:
    hf auth login
    python scripts/upload_dataset_hf.py --dry-run      # preview sizes
    python scripts/upload_dataset_hf.py                # push everything
    python scripts/upload_dataset_hf.py --only stage1_32_rgba

Large uploads use the Xet backend (``pip install "huggingface_hub[hf_xet]"``),
which is chunked and resumable: re-running only uploads what changed.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

# (local directory relative to the repo root, destination path in the HF repo)
ARTIFACTS = [
    ("data/build/stage1_32_rgba", "data/build/stage1_32_rgba"),
    ("data/build/mc_text2image32", "data/build/mc_text2image32"),
    ("data/processed/minecraft_16x_finetune32", "data/processed/minecraft_16x_finetune32"),
    ("data/processed/modrinth32", "data/processed/modrinth32"),
]

IGNORE_PATTERNS = ["**/__pycache__/**", "**/*.tmp", "**/*.log", "**/.cache/**", "**/.DS_Store"]


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="Risposta/MC_Gen")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-card", action="store_true", help="skip uploading hf/README.md")
    ap.add_argument("--only", action="append", default=[],
                    help="upload only the named artifact(s); repeatable")
    args = ap.parse_args()

    selected = [a for a in ARTIFACTS if not args.only or Path(a[0]).name in args.only]
    if not selected:
        raise SystemExit(f"no artifacts selected; choose from {[Path(a[0]).name for a in ARTIFACTS]}")

    print(f"repo: {args.repo_id} ({args.repo_type}, revision={args.revision})")
    total = 0
    for local_rel, dest in selected:
        local = ROOT / local_rel
        if not local.is_dir():
            print(f"  MISSING  {local_rel}")
            continue
        size = dir_size(local)
        total += size
        print(f"  {local_rel} -> {dest}  ({human(size)})")
    print(f"  total ~{human(total)}")

    if args.dry_run:
        print("dry-run: nothing uploaded")
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type,
                    private=args.private, exist_ok=True)

    card = ROOT / "hf" / "README.md"
    if not args.no_card and card.is_file():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type=args.repo_type,
                        revision=args.revision, commit_message="Update dataset card")
        print("uploaded README.md (dataset card)", flush=True)

    for local_rel, dest in selected:
        local = ROOT / local_rel
        if not local.is_dir():
            continue
        print(f"uploading {local_rel} -> {dest} ...", flush=True)
        api.upload_folder(folder_path=str(local), path_in_repo=dest,
                          repo_id=args.repo_id, repo_type=args.repo_type,
                          revision=args.revision, ignore_patterns=IGNORE_PATTERNS,
                          commit_message=f"Upload {dest}")
        print(f"done {dest}", flush=True)


if __name__ == "__main__":
    main()
