"""Sanity-check the multi-GPU embedding pool before a long precompute run.

Must be a real file (not stdin/-c): sentence-transformers starts workers with
multiprocessing ``spawn``, which re-imports the parent's ``__main__`` file.

    python scripts/preflight_embed.py /home/iflab/models/Qwen3-VL-Embedding-8B cuda:0,cuda:1
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.embed_text import encode_texts  # noqa: E402


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-Embedding-8B"
    devices = [d.strip() for d in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["cuda:0"]) if d.strip()]
    dim = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    texts = ["mossy deepslate bricks", "lava lace", "star blade", "iron filament",
             "weathered copper roof", "void heart emblem"]
    emb = encode_texts(
        texts, encoder_type="qwen3vl", model_name=model,
        instruction="Represent the user's input.", text_dim=dim,
        device=devices[0], devices=devices, dtype="float16", batch_size=4,
    )
    import numpy as np
    arr = np.asarray(emb)
    norms = np.linalg.norm(arr.reshape(len(texts), -1), axis=-1)
    print(f"preflight ok shape={arr.shape} dim={dim} devices={devices} "
          f"norms_min={norms.min():.4f} norms_max={norms.max():.4f}")


if __name__ == "__main__":
    main()
