"""Launch a local vLLM OpenAI-compatible server for texture annotation.

The annotator (``scripts/annotate_textures.py``) talks HTTP to this server, so
the VLM can live on a different GPU from the Stage 2 preprocessing job.

Example (idle A100 80GB, plan section 28):

    pip install vllm
    python scripts/serve_vlm.py --config configs/annotator.yaml --port 8000

Then in another shell:

    python scripts/annotate_textures.py \
        --manifest data/processed/minecraft_16x_finetune32/manifest.jsonl \
        --out data/build/mc_finetune_annotations/annotations.jsonl
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_yaml  # noqa: E402


def load_annotator_cfg(path):
    path = Path(path) if path else ROOT / "configs" / "annotator.yaml"
    if not path.exists():
        return {}
    return (load_yaml(path) or {}).get("annotator", {}) or {}


def build_command(args, cfg):
    model = args.model or cfg.get("model")
    if not model:
        raise SystemExit("no model specified (use --model or configs/annotator.yaml)")
    served = args.served_model_name or cfg.get("served_model_name") or ""
    revision = args.revision or cfg.get("revision") or ""

    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--trust-remote-code",
        # Plan section 23 sends two images per request (single + tiled view).
        "--limit-mm-per-prompt", args.limit_mm_per_prompt,
    ]
    if served:
        command += ["--served-model-name", served]
    if revision:
        command += ["--revision", revision]
    if args.enforce_eager:
        command += ["--enforce-eager"]
    if args.max_num_seqs:
        command += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.max_num_batched_tokens:
        command += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.mtp > 0 and "--speculative-config" not in (args.extra or []):
        # Qwen3.5/3.8 ship a built-in MTP head; speculative decoding attacks the
        # decode bottleneck (~2x faster on A100 in our benchmarks).
        command += ["--speculative-config",
                    json.dumps({"method": "mtp", "num_speculative_tokens": args.mtp})]
    if args.extra:
        command += list(args.extra)
    return command


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "annotator.yaml"))
    ap.add_argument("--model", default="")
    ap.add_argument("--served-model-name", default="")
    ap.add_argument("--revision", default="")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="cap context so VRAM is not wasted (plan section 28)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--limit-mm-per-prompt", default='{"image": 2}')
    ap.add_argument("--mtp", type=int, default=0,
                    help="enable MTP speculative decoding with N draft tokens (0 = off)")
    ap.add_argument("--max-num-seqs", type=int, default=0,
                    help="max concurrent sequences (0 = vLLM default)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=0,
                    help="max tokens scheduled per step (0 = vLLM default)")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra flags forwarded to vLLM (put last)")
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify vLLM is importable")
    args = ap.parse_args()

    cfg = load_annotator_cfg(args.config)
    command = build_command(args, cfg)

    if args.check:
        try:
            import vllm  # noqa: F401
        except Exception as exc:
            raise SystemExit(f"vLLM is not importable: {exc}\nInstall with: pip install vllm")
        print(f"vLLM available; command:\n  {' '.join(command)}")
        return
    if args.print_only:
        print(" ".join(command))
        return

    if shutil.which(sys.executable) is None and not os.path.exists(sys.executable):
        raise SystemExit(f"python executable not found: {sys.executable}")
    print("launching:\n  " + " ".join(command), flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
