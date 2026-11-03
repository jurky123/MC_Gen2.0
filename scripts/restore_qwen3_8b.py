"""Restore the official text tower at one resolved commit and verify LFS hashes."""
import hashlib
import argparse
import json
import os
import subprocess
from pathlib import Path

# Use resumable HTTP rather than Xet chunk reconstruction, which stalls on
# this host's proxy. Content still comes from the same pinned official repo.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("http", "mirror"), default="http")
    parser.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    args = parser.parse_args()
    repo = "Qwen/Qwen3-8B"
    destination = Path("/home/iflab/models/Qwen3-8B")
    info = HfApi().model_info(repo, revision=args.revision, files_metadata=True)
    print(f"Downloading {repo} at {info.sha} to {destination}", flush=True)
    patterns = ["*.json", "*.txt", "*.model", "*.md", "LICENSE*"]
    if args.transport == "http":
        patterns.append("*.safetensors")
    snapshot_download(repo, revision=info.sha, local_dir=destination, max_workers=4, allow_patterns=patterns)
    if args.transport == "mirror":
        # Mirror is only a transport. File sizes and SHA256 remain authoritative
        # from the official API above; aria2 also checks each official checksum.
        out_dir = Path(__file__).resolve().parents[1] / "outputs/qwen3_restore"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = out_dir / "aria2_weights.txt"
        entries = []
        for item in info.siblings:
            if item.rfilename.endswith(".safetensors"):
                expected = getattr(item.lfs, "sha256", None)
                if not expected:
                    raise ValueError("official weight checksum missing")
                entries.append(f"https://hf-mirror.com/{repo}/resolve/{info.sha}/{item.rfilename}\n"
                               f" dir={destination}\n out={item.rfilename}\n checksum=sha-256={expected}\n")
        manifest.write_text("\n".join(entries))
        env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
        subprocess.run(["aria2c", "--input-file=" + str(manifest), "--max-concurrent-downloads=3",
                        "--disable-ipv6=true", "--connect-timeout=10", "--timeout=30", "--retry-wait=2",
                        "--max-connection-per-server=8", "--split=8", "--min-split-size=16M",
                        "--continue=true", "--auto-file-renaming=false", "--file-allocation=none",
                        "--check-integrity=true", "--show-console-readout=false", "--console-log-level=warn",
                        "--summary-interval=30"], env=env, check=True)
    verified = {}
    for item in info.siblings:
        path = destination / item.rfilename
        if not path.is_file():
            continue
        if item.size is not None and path.stat().st_size != item.size:
            raise ValueError(f"size mismatch: {item.rfilename}")
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                digest.update(chunk)
        expected = getattr(item.lfs, "sha256", None) if item.lfs else None
        if expected and digest.hexdigest() != expected:
            raise ValueError(f"SHA256 mismatch: {item.rfilename}")
        verified[item.rfilename] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest(),
                                    "lfs_hash_verified": bool(expected)}
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if shard not in verified:
            raise ValueError(f"missing indexed weight shard: {shard}")
    report = {"repo": repo, "revision": info.sha, "destination": str(destination),
              "weight_transport": args.transport, "files": verified}
    out = Path(__file__).resolve().parents[1] / "outputs/qwen3_restore/download_verified.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(verified)} files; report: {out}", flush=True)


if __name__ == "__main__":
    main()
