"""Offline: encode offers + candidates with a multilingual Sentence-Transformer (CPU)
and cache the L2-normalized vectors to ``outputs/embeddings/<model>/``.

One-time, no GPU. The vectors (~67 MB for candidates) let the comparison run cosine
without ever loading the model again.

Usage:
    pip install -r requirements-bert.txt
    python -m scripts.build_embeddings
    python -m scripts.build_embeddings --model sentence-transformers/distiluse-base-multilingual-cased-v2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_bundle
from src.semantic import CACHE_DIR, DEFAULT_MODEL, _safe_name, encode_texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model id")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading datasets…")
    bundle = load_bundle()
    out = CACHE_DIR / _safe_name(args.model)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Encoding {len(bundle.offers):,} offers with {args.model} (CPU)…")
    off = encode_texts(bundle.offers["offer_text"], args.model, args.batch_size)
    np.save(out / "offers.npy", off)

    print(f"Encoding {len(bundle.candidates):,} candidates (CPU)… this is the long part")
    cand = encode_texts(bundle.candidates["candidate_text"], args.model, args.batch_size)
    np.save(out / "candidates.npy", cand)

    print(f"Saved offers{off.shape} + candidates{cand.shape} -> {out}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
