"""Compare TF-IDF vs BERT vs a TF-IDF+BERT hybrid on the ACPE ground truth.

CPU-only and offline: it reuses the cached embeddings from
``scripts/build_embeddings.py`` (no model reloaded) and scores a random sample of
ground-truth candidates so it runs in seconds.

Prereq:
    pip install -r requirements-bert.txt
    python -m scripts.build_embeddings          # one-time, encodes offers+candidates

Usage:
    python -m scripts.compare_engines                    # 4000-candidate sample
    python -m scripts.compare_engines --sample 41000     # (almost) full base
    python -m scripts.compare_engines --alphas 0.3,0.5,0.7

The TF-IDF row is the *shipped* engine (with the soft semantic guard); BERT is pure
cosine over embeddings; the hybrid blends the raw professional TF-IDF cosine with the
BERT cosine as ``alpha*tfidf + (1-alpha)*bert``. Decision rule: only consider
replacing TF-IDF if BERT or the hybrid clearly beats it on P@5 / NDCG@5.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics.pairwise import linear_kernel

from src.data_loader import CANDIDATE_ID, OFFER_ID, load_bundle
from src.matching import MatchingEngine
from src.metrics import ndcg_at_k
from src.semantic import DEFAULT_MODEL, SemanticEngine

KS = (5, 10)


def _blank_agg() -> dict:
    return {f"{m}@{k}": [] for m in ("precision", "recall", "ndcg") for k in KS}


def _accumulate(agg: dict, ranked_ids: list, relevant: set) -> None:
    for k in KS:
        hits = len(relevant & set(ranked_ids[:k]))
        agg[f"precision@{k}"].append(hits / k)
        agg[f"recall@{k}"].append(hits / len(relevant))
        agg[f"ndcg@{k}"].append(ndcg_at_k(ranked_ids, relevant, k))


def _means(agg: dict) -> dict:
    return {m: float(np.mean(v)) if v else 0.0 for m, v in agg.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--alphas", default="0.3,0.5,0.7", help="hybrid TF-IDF weights, comma-separated")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    print("Loading datasets + engines…")
    bundle = load_bundle()
    tf = MatchingEngine(bundle)
    se = SemanticEngine.from_cache(bundle, args.model)

    cand = bundle.candidates.reset_index(drop=True)
    gt = bundle.ground_truth
    offer_ids = tf.offers[OFFER_ID].to_numpy()

    eligible = [i for i, cid in enumerate(cand[CANDIDATE_ID]) if gt.get(cid)]
    rng = np.random.default_rng(args.seed)
    if args.sample and args.sample < len(eligible):
        eligible = rng.choice(eligible, size=args.sample, replace=False).tolist()
    print(f"Scoring {len(eligible):,} ground-truth candidates "
          f"(TF-IDF, BERT, hybrid alpha in {alphas})...")

    agg_tf, agg_be = _blank_agg(), _blank_agg()
    agg_hy = {a: _blank_agg() for a in alphas}

    for pos in eligible:
        row = cand.iloc[pos]
        relevant = set(gt[row[CANDIDATE_ID]])

        # TF-IDF (shipped engine, with soft guard)
        tf_ids = [r.offer_id for r in tf.score_candidate_row(row, top_k=10)]
        _accumulate(agg_tf, tf_ids, relevant)

        # BERT (cosine over embeddings)
        bsim = se.cand_emb[pos] @ se.offer_emb.T
        _accumulate(agg_be, offer_ids[np.argsort(-bsim)[:10]].tolist(), relevant)

        # Hybrid (raw professional TF-IDF cosine blended with BERT cosine)
        tfs = linear_kernel(
            tf.vectorizer.transform([row["candidate_text"]]), tf.offer_matrix
        ).ravel()
        for a in alphas:
            blend = a * tfs + (1.0 - a) * bsim
            _accumulate(agg_hy[a], offer_ids[np.argsort(-blend)[:10]].tolist(), relevant)

    rows = [("TF-IDF (shipped)", _means(agg_tf)), (f"BERT [{args.model.split('/')[-1]}]", _means(agg_be))]
    rows += [(f"Hybrid alpha={a:g}", _means(agg_hy[a])) for a in alphas]

    header = f"{'Engine':<34} " + "  ".join(f"{m:>9}" for m in ("P@5", "P@10", "R@5", "R@10", "NDCG@5", "NDCG@10"))
    print("\n" + header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<34} " + "  ".join(
            f"{m[k]:9.4f}" for k in
            ("precision@5", "precision@10", "recall@5", "recall@10", "ndcg@5", "ndcg@10")
        ))
    best = max(rows, key=lambda r: r[1]["ndcg@5"])
    print(f"\nBest NDCG@5: {best[0]} ({best[1]['ndcg@5']:.4f}). "
          "Keep TF-IDF unless a challenger clearly wins (and justifies the RAM/explainability cost).")


if __name__ == "__main__":
    main()
