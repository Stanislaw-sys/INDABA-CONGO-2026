"""Generate Top-K recommendations for every candidate and score them.

Outputs (into ``outputs/``):
    recommendations_top10.csv   candidate_id, rank, job_id, score   (submission format)
    recommendations_top5.csv    same, truncated to rank <= 5
    metrics.json                Precision/Recall/NDCG @5 and @10 on all ground-truth candidates

Usage:
    python -m scripts.generate_recommendations              # full run
    python -m scripts.generate_recommendations --sample 5000  # quick metrics on a sample
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_bundle
from src.matching import MatchingEngine
from src.metrics import evaluate_from_recs

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="limit metrics to N random candidates")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    print("Loading datasets…")
    bundle = load_bundle()
    print(f"  {len(bundle.candidates):,} candidates, {len(bundle.offers):,} offers, "
          f"{len(bundle.ground_truth):,} ground-truth rows")

    print("Building matching engine…")
    engine = MatchingEngine(bundle)

    print(f"Scoring Top-{args.top_k} for all candidates…")
    recs = engine.recommend_all(top_k=args.top_k)
    recs.to_csv(OUT / "recommendations_top10.csv", index=False)
    recs[recs["rank"] <= 5].to_csv(OUT / "recommendations_top5.csv", index=False)
    print(f"  wrote {len(recs):,} rows -> outputs/recommendations_top10.csv / _top5.csv")

    eval_recs = recs
    if args.sample:
        import numpy as np
        keep = set(np.random.default_rng(42).choice(
            recs["candidate_id"].unique(),
            size=min(args.sample, recs["candidate_id"].nunique()),
            replace=False,
        ))
        eval_recs = recs[recs["candidate_id"].isin(keep)]

    print("Evaluating against ground truth…")
    metrics = evaluate_from_recs(eval_recs, bundle)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Committed snapshot the deployed app reads (outputs/ is gitignored).
    if not args.sample:
        (OUT.parent / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
