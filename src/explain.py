"""Explanations for a match, plus skill-gap analysis (bonus 2).

Explainability is explicitly rewarded by the jury, so every recommendation can be
broken down into:
  * the score components (text vs métier rule) — carried on :class:`MatchResult`;
  * the concrete overlapping terms that drove the TF-IDF cosine (here);
  * the skills/qualifications the offer asks for that the candidate does not list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .matching import MatchingEngine
from .utils import norm, tokens

# Light French/generic stopword set — enough to keep skill-gap output meaningful.
_STOP = {
    "les", "des", "une", "aux", "avec", "pour", "dans", "sur", "par", "est", "ans",
    "experience", "niveau", "bac", "and", "the", "profil", "poste", "type", "non",
    "declare", "divers", "autre", "autres", "emploi", "secteur", "activite", "metier",
    "qualification", "visee", "vise", "de", "du", "en", "et", "la", "le", "un",
}


def top_matching_terms(engine: MatchingEngine, candidate_text: str, offer_id: str, top_n: int = 8):
    """Terms contributing most to the candidate↔offer TF-IDF cosine, high to low.

    The cosine is a sum over terms of ``cand_tfidf[t] * offer_tfidf[t]``; we return
    the terms with the largest such products.
    """
    vec = engine.vectorizer
    pos = engine.bundle.offer_pos.get(offer_id)
    if pos is None:
        return []
    cand_vec = vec.transform([candidate_text]).toarray().ravel()
    offer_vec = engine.offer_matrix[pos].toarray().ravel()
    contrib = cand_vec * offer_vec
    if not contrib.any():
        return []
    names = np.array(vec.get_feature_names_out())
    idx = np.argsort(-contrib)[:top_n]
    return [(names[i], float(contrib[i])) for i in idx if contrib[i] > 0]


def _requirement_tokens(offer: pd.Series) -> set[str]:
    """Skill/requirement tokens for an offer.

    Prefers the rich extension text (Compétences / Profil / Description); falls back
    to the title + sector for offers without enrichment.
    """
    ext = offer.get("offer_ext_text", "")
    source = ext if ext else " ".join(
        norm(offer.get(c)) for c in ("Intitule", "Poste", "Secteur activité")
    )
    return {t for t in tokens(source) if t not in _STOP}


def skill_gap(offer: pd.Series, candidate: pd.Series) -> dict:
    """Requirements present in the offer but absent from the candidate profile."""
    required = _requirement_tokens(offer)
    have = tokens(candidate.get("candidate_text", "")) | tokens(
        candidate.get("Diplome", "")
    ) | tokens(candidate.get("niveau_etude", ""))
    missing = sorted(required - have - _STOP)
    matched = sorted(required & have)
    coverage = len(matched) / len(required) if required else 0.0
    return {
        "required": sorted(required),
        "matched": matched,
        "missing": missing,
        "coverage": round(coverage, 3),
        "has_detailed_requirements": bool(offer.get("offer_ext_text", "")),
    }
