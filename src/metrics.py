"""Ranking metrics against the ground-truth appariement file.

The guide specifies Precision@K, Recall@K and NDCG@K for K in {5, 10}. Each
candidate has at most 3 relevant offers, so Precision@5 is structurally capped at
3/5 = 0.60 and Precision@10 at 3/10 = 0.30 — read the numbers with that ceiling
in mind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import DataBundle
from .matching import MatchingEngine


def _dcg(hits: list[int]) -> float:
    return sum(h / np.log2(i + 2) for i, h in enumerate(hits))


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    hits = [1 if oid in relevant else 0 for oid in ranked_ids[:k]]
    idcg = _dcg([1] * min(len(relevant), k))
    return _dcg(hits) / idcg if idcg else 0.0


def evaluate(
    engine: MatchingEngine,
    bundle: DataBundle,
    ks: tuple[int, ...] = (5, 10),
    sample: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Compute mean Precision/Recall/NDCG@K over candidates that have ground truth.

    ``sample`` limits evaluation to a random subset (useful for quick iteration on
    the full 41k candidate base).
    """
    gt = bundle.ground_truth
    cand = bundle.candidates
    cand = cand[cand["candidate_id"].isin(gt)]
    cand = cand[cand["candidate_id"].map(lambda c: len(gt[c]) > 0)]
    if sample and sample < len(cand):
        cand = cand.sample(sample, random_state=seed)

    top_k = max(ks)
    recs = engine.recommend_all(top_k=top_k)
    recs = recs[recs["candidate_id"].isin(set(cand["candidate_id"]))]
    ranked = recs.sort_values(["candidate_id", "rank"]).groupby("candidate_id")["job_id"].apply(list)

    agg = {f"{m}@{k}": [] for m in ("precision", "recall", "ndcg") for k in ks}
    for cid, ids in ranked.items():
        relevant = set(gt[cid])
        if not relevant:
            continue
        for k in ks:
            topk = ids[:k]
            hits = len(relevant & set(topk))
            agg[f"precision@{k}"].append(hits / k)
            agg[f"recall@{k}"].append(hits / len(relevant))
            agg[f"ndcg@{k}"].append(ndcg_at_k(ids, relevant, k))
    result = {m: float(np.mean(v)) if v else 0.0 for m, v in agg.items()}
    result["n_evaluated"] = float(len(ranked))
    return result


def evaluate_from_recs(recs: pd.DataFrame, bundle: DataBundle, ks=(5, 10)) -> dict[str, float]:
    """Score a pre-computed recommendations DataFrame (candidate_id, rank, job_id, score)."""
    gt = bundle.ground_truth
    ranked = recs.sort_values(["candidate_id", "rank"]).groupby("candidate_id")["job_id"].apply(list)
    agg = {f"{m}@{k}": [] for m in ("precision", "recall", "ndcg") for k in ks}
    for cid, ids in ranked.items():
        relevant = set(gt.get(cid, []))
        if not relevant:
            continue
        for k in ks:
            hits = len(relevant & set(ids[:k]))
            agg[f"precision@{k}"].append(hits / k)
            agg[f"recall@{k}"].append(hits / len(relevant))
            agg[f"ndcg@{k}"].append(ndcg_at_k(ids, relevant, k))
    out = {m: float(np.mean(v)) if v else 0.0 for m, v in agg.items()}
    out["n_evaluated"] = float(sum(1 for _ in ranked.items()))
    return out
