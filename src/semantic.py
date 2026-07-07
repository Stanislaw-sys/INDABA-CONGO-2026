"""Optional semantic (BERT / Sentence-Transformers) matching backend — CPU-only.

This is an **experiment** that runs *alongside* the production TF-IDF
:class:`~src.matching.MatchingEngine`. It is deliberately isolated so the deployed
Streamlit app and the graded engine are never touched:

* ``app.py`` does not import this module;
* it lives behind its own dependency file (``requirements-bert.txt``);
* it reuses the exact same :class:`~src.data_loader.DataBundle` text
  (``offer_text`` / ``candidate_text``) that TF-IDF uses, so the comparison is fair.

Goal: measure whether dense multilingual embeddings beat (or usefully complement)
TF-IDF on the ACPE ground truth — **without a GPU**:

* offers and candidates are encoded **once, offline** with a small multilingual
  Sentence-Transformer, and the L2-normalized vectors are cached to disk
  (see ``scripts/build_embeddings.py``);
* scoring is then plain cosine over the cached vectors — **no model held in
  memory** — so evaluation stays light and deterministic (inference has no dropout);
* :meth:`SemanticEngine.recommend_all` returns the same
  ``candidate_id, rank, job_id, score`` frame as the TF-IDF engine, so
  :func:`src.metrics.evaluate_from_recs` scores it unchanged.

Install with ``pip install -r requirements-bert.txt`` (torch CPU wheel is enough).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import CANDIDATE_ID, OFFER_ID, DataBundle

# Small, CPU-friendly, multilingual (incl. French) sentence embedder (~470 MB, 384-d).
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Cached vectors live under outputs/ (gitignored) — they are a local artifact.
CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "embeddings"


def _safe_name(model_name: str) -> str:
    """Filesystem-safe folder name for a model id."""
    return model_name.replace("/", "__")


def encode_texts(texts, model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> np.ndarray:
    """Encode texts to L2-normalized float32 embeddings on CPU (deterministic).

    Imported lazily so the rest of the codebase never needs sentence-transformers.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(
        [str(t) for t in texts],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # so cosine == dot product
        show_progress_bar=True,
    )
    return emb.astype(np.float32)


class SemanticEngine:
    """Cosine matcher over precomputed sentence embeddings (mirrors MatchingEngine)."""

    def __init__(self, bundle: DataBundle, offer_emb: np.ndarray, cand_emb: np.ndarray):
        self.bundle = bundle
        self.offers = bundle.offers
        self.candidates = bundle.candidates
        self.offer_emb = offer_emb    # (n_offers, d), L2-normalized
        self.cand_emb = cand_emb      # (n_candidates, d), L2-normalized

    @classmethod
    def from_cache(
        cls, bundle: DataBundle, model_name: str = DEFAULT_MODEL, cache_dir=CACHE_DIR
    ) -> "SemanticEngine":
        """Load embeddings built by ``scripts/build_embeddings.py``.

        The row order of the cached vectors must match the current bundle (same
        cleaning/dedup), so we assert on the counts to fail fast on a stale cache.
        """
        base = Path(cache_dir) / _safe_name(model_name)
        offer_emb = np.load(base / "offers.npy")
        cand_emb = np.load(base / "candidates.npy")
        if offer_emb.shape[0] != len(bundle.offers) or cand_emb.shape[0] != len(bundle.candidates):
            raise ValueError(
                "Cached embeddings do not match the current bundle "
                f"(offers {offer_emb.shape[0]} vs {len(bundle.offers)}, "
                f"candidates {cand_emb.shape[0]} vs {len(bundle.candidates)}). "
                "Rebuild with `python -m scripts.build_embeddings`."
            )
        return cls(bundle, offer_emb, cand_emb)

    def candidate_offer_sims(self, start: int, size: int) -> np.ndarray:
        """Cosine (== dot product for normalized vectors) of a candidate chunk vs all offers."""
        return self.cand_emb[start : start + size] @ self.offer_emb.T

    def recommend_all(self, top_k: int = 10, batch_size: int = 2000) -> pd.DataFrame:
        """Top-K offers per candidate as rows of candidate_id, rank, job_id, score."""
        offer_ids = self.offers[OFFER_ID].to_numpy()
        cand_ids = self.candidates[CANDIDATE_ID].to_numpy()
        rows = []
        for start in range(0, len(self.candidates), batch_size):
            sims = self.candidate_offer_sims(start, batch_size)
            for j in range(sims.shape[0]):
                order = np.argsort(-sims[j])[:top_k]
                cid = cand_ids[start + j]
                for rank, pos in enumerate(order, start=1):
                    rows.append((cid, rank, offer_ids[pos], round(float(sims[j][pos]), 6)))
        return pd.DataFrame(rows, columns=["candidate_id", "rank", "job_id", "score"])
