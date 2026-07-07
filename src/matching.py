"""Hybrid matching engine: field-specific TF-IDF similarity + rule-based boosts.

Design choices (see the hackathon guide — matching quality is 30% of the grade
and explainability is explicitly rewarded):

* **Multi-field spaces.** Matching is split into two *independent* sub-spaces so
  that unrelated signals cannot bleed into one another:

    - **Professional space** — the core of the engine. Candidate profile text
      (métier visé, qualification, filière, secteur métier) is compared to offer
      *professional* text (intitulé, poste, secteur + description/profil/compétences
      where available) by TF-IDF cosine. This — and only this — drives the
      candidate↔offer ranking.
    - **Geographic space** — a *separate* TF-IDF over the offers' official ``Lieu``
      field. Used for location-aware offer search only. It is **never** blended into
      the candidate↔offer score: candidates have no real city, and the *simulated*
      city (``SIMULATED_CITY_COL``) is an analytics/exploration artifact that must
      never pollute the pure engine.

  Keeping location out of the professional vector prevents *semantic dilution* —
  e.g. a "chauffeur" offer scoring against a "statisticien" profile just because
  both share a high-frequency, non-professional token.

* **Soft semantic guard (penalisation, not hard drop).** A geographic-only or
  purely diffuse bag-of-words match can occasionally push a professionally
  unrelated offer to the very top. We regularise *only that failure mode*: an
  offer is penalised **iff** (i) its pre-guard compatibility is already **high**
  (≥ ``GUARD_HIGH_COMPAT`` %) **and** (ii) it shares **zero** core professional
  keyword (intitulé + poste + secteur) with the candidate. Such an offer has its
  text-similarity component multiplied by ``SOFT_PENALTY`` (< 1). This is a
  targeted, conditional regulariser: it demotes absurd cross-title matches at the
  top of the list while leaving the many legitimate near-matches (which the ground
  truth rewards) completely untouched — so Precision@K / NDCG@K are preserved.

* **Rule boost** on the *métier* ↔ *intitulé* token overlap. This rewards a direct
  job-title hit on top of the diffuse bag-of-words signal, which lifts precision.

* The score components are kept separate on every result so the UI can explain
  *why* an offer was recommended, and are blended into a single ranking score.

* The displayed "compatibility %" is a fixed logistic calibration of the raw
  blended score so a strong match reads ~85–95% (cosmetic only — it is strictly
  monotonic in the raw score and never changes the ranking). A human-facing
  minimum (``MIN_COMPATIBILITY``) lets the UI hide weak matches.

The engine is backend-agnostic: :class:`MatchingEngine` uses TF-IDF; an optional
:class:`SemanticEngine` (sentence-transformers) can be dropped in for the bonus
semantic search without touching the rest of the app.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .data_loader import CANDIDATE_ID, OFFER_ID, SIMULATED_CITY_COL, DataBundle
from .utils import norm, tokens

# Blend weights for the ranking score (professional space only).
W_TEXT = 0.80
W_METIER = 0.20

# Soft semantic guard. When a candidate shares NO core professional keyword with an
# offer yet that offer's pre-guard compatibility is already high, we treat the score
# as a likely dilution artifact and scale its text-similarity component by this
# factor. Chosen so genuine near-matches (which the ground truth rewards) are never
# touched — only conspicuous, professionally unrelated top hits are demoted.
SOFT_PENALTY = 0.30
# The guard only engages above this calibrated-compatibility level (a "high" match
# with zero professional overlap is the hijack signature we regularise).
GUARD_HIGH_COMPAT = 75.0

# Additive weight of the (independent) geographic space when a locality is present
# in an *offer* search query. Applies to offer search only — never to the core
# candidate↔offer score.
W_GEO_SEARCH = 0.10

# Human-facing minimum compatibility (%) after logistic calibration. The UI hides
# recommendations below this; it does NOT affect the batch submission CSVs or the
# offline evaluation, which stay complete Top-K for the graded metrics.
MIN_COMPATIBILITY = 45.0

# Logistic calibration for the human-facing percentage: pct = 100 / (1 + e^-k(raw-x0)).
_CAL_K = 8.0
_CAL_X0 = 0.15


def calibrate(raw: float) -> float:
    """Map a raw blended score (~0..0.6) to a 0..100 compatibility percentage.

    A raw score of exactly 0 means there is *no* lexical overlap at all (no matching
    token between the two texts). Such a non-match is hard-mapped to **0.0 %** — the
    logistic floor would otherwise leak ~23 % for a genuine zero-similarity result.
    The function stays monotonic non-decreasing in ``raw``.
    """
    if raw <= 0.0:
        return 0.0
    return round(100.0 / (1.0 + math.exp(-_CAL_K * (raw - _CAL_X0))), 1)


def _inv_calibrate(pct: float) -> float:
    """Raw score corresponding to a calibrated percentage (inverse of calibrate)."""
    return _CAL_X0 - math.log(100.0 / pct - 1.0) / _CAL_K


# Raw-score threshold equivalent to GUARD_HIGH_COMPAT — precomputed so the guard is
# a cheap array comparison (no per-offer exp) inside the batch loop.
_GUARD_HIGH_RAW = _inv_calibrate(GUARD_HIGH_COMPAT)

# Minimum shared-prefix length for the lexical (stemming-tolerant) core-token guard
# and query expansion used by the SEARCH tabs. Tuned to relate real French job-word
# derivations — 'statistique'↔'statisticien' ('statisti', 8), 'mecanique'↔'mecanicien'
# ('mecani', 6), 'informatique'↔'informaticien' ('informati', 9) — while rejecting
# coincidental short heads such as 'statistique' vs 'station' ('stati', only 5).
CORE_PREFIX_MIN = 6


def _lexical_related(a: str, b: str, prefix_min: int = CORE_PREFIX_MIN) -> bool:
    """Stemming-tolerant token relation for the search-side guard / expansion.

    True when two normalized tokens (each length > 3) are the same word up to a
    suffix: one is a substring of the other, or they share a prefix of at least
    ``prefix_min`` characters. Purely lexical — no external stemmer, deterministic.
    """
    if len(a) <= 3 or len(b) <= 3:
        return False
    if a in b or b in a:
        return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= prefix_min


@dataclass
class MatchResult:
    offer_id: str
    rank: int
    score: float               # raw blended score used for ranking (0..~0.6)
    compatibility: float       # calibrated percentage (0..100)
    text_sim: float            # TF-IDF cosine component (post soft-guard) — professional space
    metier_overlap: float      # métier↔intitulé token-overlap component
    offer: pd.Series = field(repr=False, default=None)


@dataclass
class CandidateHit:
    candidate_id: str
    rank: int
    score: float               # TF-IDF cosine relevance to the query
    compatibility: float       # calibrated percentage (0..100), cosmetic
    candidate: pd.Series = field(repr=False, default=None)


class MatchingEngine:
    def __init__(self, bundle: DataBundle):
        self.bundle = bundle
        self.offers = bundle.offers

        # ---- Professional space (core candidate↔offer matching) -------------
        # Standard TF-IDF (linear tf) — best ground-truth ranking metrics on these
        # short professional texts. Attribute names `vectorizer` / `offer_matrix`
        # are part of the public contract relied on by src/explain.py — keep them.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.offer_matrix = self.vectorizer.fit_transform(self.offers["offer_text"])
        # Pre-tokenize offer titles once for the métier-overlap rule.
        self._offer_title_tokens = [
            tokens(t) for t in self.offers["Intitule"].fillna("")
        ]
        # Core professional keywords of each offer (intitulé + poste + secteur) for
        # the soft semantic guard — broad enough that legitimate related roles share
        # at least one token, so only truly unrelated offers trip the guard.
        poste = self.offers.get("Poste", pd.Series([""] * len(self.offers)))
        secteur = self.offers.get("Secteur activité", pd.Series([""] * len(self.offers)))
        self._offer_core_tokens = [
            tokens(f"{it} {po} {se}")
            for it, po, se in zip(
                self.offers["Intitule"].fillna(""), poste.fillna(""), secteur.fillna("")
            )
        ]

        # ---- Geographic space (INDEPENDENT — location-aware offer search) ----
        # Kept strictly separate; never blended into the candidate↔offer score.
        geo_text = self.offers["offer_geo_text"].fillna("")
        self.geo_vectorizer = None
        self.offer_geo_matrix = None
        if geo_text.str.strip().any():
            self.geo_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self.offer_geo_matrix = self.geo_vectorizer.fit_transform(geo_text)

        # Separate TF-IDF space over candidate profiles, for candidate NL search
        # (bonus 1, recruiter side). Cheap to build (~1.5s over 41k short profiles).
        self.candidates = bundle.candidates
        self.cand_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.candidate_matrix = self.cand_vectorizer.fit_transform(
            self.candidates["candidate_text"]
        )
        # Core professional keywords of each candidate — métier visé, qualification
        # and filière — the candidate-side analogue of `_offer_core_tokens`, used by
        # the soft semantic guard in `search_candidates`.
        n_cand = len(self.candidates)
        cand_core_cols = [
            "Métier visé / Qualification visée", "qualification_metier",
            "Qualification", "Filière / Spécialité",
        ]
        cand_core_series = [
            self.candidates.get(c, pd.Series([""] * n_cand)).fillna("")
            for c in cand_core_cols
        ]
        self._candidate_core_tokens = [
            tokens(" ".join(s.iloc[i] for s in cand_core_series))
            for i in range(n_cand)
        ]
        # Column holding the Oui/Non national-mobility flag (name carries accents).
        self._mobility_col = next(
            (c for c in self.candidates.columns if "obilit" in c and "ograph" in c),
            None,
        )
        # Known localities (normalized -> display), for parsing a city out of a query.
        self._city_lookup = {norm(c): c for c in bundle.cities}

        # ---- Search-side lexical helpers (query parsing only) ---------------
        # Geographic tokens (all known city name tokens): these are the ONLY tokens
        # excluded from the professional guard so a city term can never satisfy it,
        # and are stripped from the query text vector so location is handled solely
        # by the independent geographic space (never contaminating the text score).
        self._geo_tokens = set()
        for ncity in self._city_lookup:
            self._geo_tokens |= tokens(ncity)
        # Unigram vocabularies of each professional space, for stemming-tolerant query
        # expansion (e.g. mapping an out-of-vocabulary 'statistique' → 'statisticien').
        self._offer_vocab = {t for t in self.vectorizer.vocabulary_ if " " not in t}
        self._cand_vocab = {t for t in self.cand_vectorizer.vocabulary_ if " " not in t}

    # ------------------------------------------------------------------ scoring
    def _text_sims(self, candidate_text: str) -> np.ndarray:
        vec = self.vectorizer.transform([candidate_text])
        return linear_kernel(vec, self.offer_matrix).ravel()

    def _metier_overlaps(self, metier: str) -> np.ndarray:
        mtoks = tokens(metier)
        if not mtoks:
            return np.zeros(len(self.offers))
        out = np.empty(len(self.offers))
        for i, otoks in enumerate(self._offer_title_tokens):
            inter = mtoks & otoks
            out[i] = len(inter) / len(mtoks) if inter else 0.0
        return out

    def _apply_soft_guard(
        self, text_sim: np.ndarray, metier: np.ndarray, cand_tokens: set[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Soft semantic guard — see module docstring.

        Penalise the text component of any offer that is (a) already a *high* match
        yet (b) shares no core professional keyword with the candidate. Only the
        high-scoring candidates are examined (cheap), so legitimate near-matches are
        untouched. Returns the (possibly modified) ``text_sim`` and the resulting
        blended ``raw`` score.
        """
        raw = W_TEXT * text_sim + W_METIER * metier
        hi = np.nonzero(raw >= _GUARD_HIGH_RAW)[0]
        if hi.size:
            if cand_tokens:
                core = self._offer_core_tokens
                for pos in hi:
                    if not (cand_tokens & core[pos]):
                        text_sim[pos] *= SOFT_PENALTY
            else:
                # No usable professional keywords at all → every high match is diffuse.
                text_sim[hi] *= SOFT_PENALTY
            raw = W_TEXT * text_sim + W_METIER * metier
        return text_sim, raw

    def score_candidate_row(self, candidate: pd.Series, top_k: int = 10) -> list[MatchResult]:
        text_sim = self._text_sims(candidate["candidate_text"])
        metier = self._metier_overlaps(candidate.get("Métier visé / Qualification visée", ""))
        text_sim, raw = self._apply_soft_guard(
            text_sim, metier, tokens(candidate.get("candidate_text", ""))
        )
        order = np.argsort(-raw)[:top_k]
        results = []
        for rank, pos in enumerate(order, start=1):
            results.append(
                MatchResult(
                    offer_id=self.offers.iloc[pos][OFFER_ID],
                    rank=rank,
                    score=float(raw[pos]),
                    compatibility=calibrate(float(raw[pos])),
                    text_sim=float(text_sim[pos]),
                    metier_overlap=float(metier[pos]),
                    offer=self.offers.iloc[pos],
                )
            )
        return results

    def score_by_id(self, candidate_id: str, top_k: int = 10) -> list[MatchResult]:
        rows = self.bundle.candidates
        match = rows[rows[CANDIDATE_ID] == candidate_id]
        if match.empty:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        return self.score_candidate_row(match.iloc[0], top_k=top_k)

    # -------------------------------------------------------------- batch / eval
    def recommend_all(self, top_k: int = 10, batch_size: int = 2000) -> pd.DataFrame:
        """Top-K offers for every candidate as rows of candidate_id, rank, job_id, score.

        Applies the professional-space cosine, the métier rule and the soft semantic
        guard — the same core scoring as the interactive path — with no human-facing
        threshold, so the graded Top-K metrics stay complete.
        """
        cand = self.bundle.candidates
        metier_col = cand.get("Métier visé / Qualification visée", pd.Series([""] * len(cand)))
        rows = []
        offer_ids = self.offers[OFFER_ID].to_numpy()
        for start in range(0, len(cand), batch_size):
            chunk = cand.iloc[start : start + batch_size]
            vecs = self.vectorizer.transform(chunk["candidate_text"])
            sims = linear_kernel(vecs, self.offer_matrix)  # (chunk, offers)
            texts = chunk["candidate_text"].to_numpy()
            for j, (_, c) in enumerate(chunk.iterrows()):
                metier = self._metier_overlaps(metier_col.iloc[start + j])
                _, raw = self._apply_soft_guard(sims[j], metier, tokens(texts[j]))
                order = np.argsort(-raw)[:top_k]
                for rank, pos in enumerate(order, start=1):
                    rows.append(
                        (c[CANDIDATE_ID], rank, offer_ids[pos], round(float(raw[pos]), 6))
                    )
        return pd.DataFrame(rows, columns=["candidate_id", "rank", "job_id", "score"])

    # ---------------------------------------------------- geographic sub-space
    def geographic_similarity(self, location: str) -> np.ndarray:
        """Cosine of a location string against each offer's official ``Lieu``.

        The independent geographic space. Returns a neutral all-ones vector when no
        location is given or the space is unavailable. Used for *offer* search only
        and never mixed into the candidate↔offer score.
        """
        if not location or self.geo_vectorizer is None:
            return np.ones(len(self.offers))
        vec = self.geo_vectorizer.transform([norm(location)])
        return linear_kernel(vec, self.offer_geo_matrix).ravel()

    # ---------------------------------------------- search-side query parsing
    def _professional_query_tokens(self, query_norm: str) -> set[str]:
        """Professional query tokens = query tokens minus geographic (city) tokens.

        Guarantees a city term (e.g. "brazzaville") can never satisfy the
        professional core guard — only genuine role/qualification words do.
        """
        return {t for t in tokens(query_norm) if t not in self._geo_tokens}

    def _expand_query(self, prof_text: str, vocab: set[str]) -> str:
        """Augment the professional query with lexically-related vocabulary terms.

        A query token absent from the TF-IDF vocabulary (e.g. "statistique") would
        otherwise score 0 against everyone; we append the existing vocab terms it is
        lexically related to ("statisticien", "statistiques") so the *transient*
        query vector picks them up. The global TF-IDF matrices are never modified.
        """
        extra: set[str] = set()
        for qt in {t for t in prof_text.split() if len(t) > 3}:
            if qt in vocab:
                continue  # already scores directly; no need to expand
            extra.update(term for term in vocab if _lexical_related(qt, term))
        return prof_text + (" " + " ".join(sorted(extra)) if extra else "")

    def _professional_overlap(self, qprof: set[str], core_tokens: set[str]) -> bool:
        """True if any professional query token is lexically related (substring /
        shared prefix ≥ CORE_PREFIX_MIN) to a core professional field token."""
        return any(
            _lexical_related(q, d)
            for q in qprof if len(q) > 3
            for d in core_tokens
        )

    # ------------------------------------------------------------------- search
    def search_offers(
        self, query: str, top_k: int = 10, min_compatibility: float = MIN_COMPATIBILITY
    ) -> list[MatchResult]:
        """Bonus 1: free-text natural-language search over offers.

        Professional relevance (TF-IDF cosine) is primary; if the query names a
        known locality, the independent geographic space adds a bonus to offers
        actually located there — surfacing "développeur à Brazzaville" without
        letting the city term contaminate the professional vector.

        Three safeguards make the results trustworthy:
        * the query text vector is built from the **professional part only** (city
          tokens stripped) and **expanded** so lexical variants match
          (statistique ↔ statisticien);
        * a **soft semantic guard** demotes any offer scoring high yet sharing no
          *professional* core keyword with the query — a city match never opens it;
        * **compatibility is calibrated on the professional text similarity alone**
          (the geographic bonus only reorders offers, it never inflates the %), and
          only offers reaching ``min_compatibility`` (%) are returned. An offer with
          zero token overlap has text-sim 0 → 0 % and is dropped.
        """
        q = norm(query)
        qprof = self._professional_query_tokens(q)
        # Text vector: professional part only (city handled by the geo space), with
        # stemming-tolerant expansion so 'statistique' also fires 'statisticien'.
        prof_text = " ".join(t for t in q.split() if t not in self._geo_tokens)
        text = self._text_sims(self._expand_query(prof_text, self._offer_vocab))
        # Soft semantic guard: penalise a high offer with no professional overlap
        # (a pure city query, qprof empty, penalises every high offer).
        for pos in np.nonzero(text >= _GUARD_HIGH_RAW)[0]:
            if not (qprof and self._professional_overlap(qprof, self._offer_core_tokens[pos])):
                text[pos] *= SOFT_PENALTY
        # Ranking signal = professional relevance (+ geographic bonus for a named
        # city). The bonus reorders relevant offers; it can never rescue a
        # zero-relevance offer, whose compatibility (below) stays 0 %.
        rank_score = text.copy()
        city = self.detect_city(query)
        if city:
            rank_score = rank_score + W_GEO_SEARCH * self.geographic_similarity(city)
        # Keep only offers whose *text* compatibility clears the floor, best first.
        floor = _inv_calibrate(min_compatibility)
        keep = np.nonzero(text >= floor)[0]
        keep = keep[np.argsort(-rank_score[keep])][:top_k]
        return [
            MatchResult(
                offer_id=self.offers.iloc[pos][OFFER_ID],
                rank=rank,
                score=float(rank_score[pos]),
                compatibility=calibrate(float(text[pos])),
                text_sim=float(text[pos]),
                metier_overlap=0.0,
                offer=self.offers.iloc[pos],
            )
            for rank, pos in enumerate(keep, start=1)
        ]

    # ------------------------------------------------ candidate search (bonus 1)
    def detect_city(self, query: str) -> str | None:
        """Return the display name of a known locality mentioned in the query, if any."""
        nq = norm(query)
        for ncity, display in self._city_lookup.items():
            if ncity and ncity in nq:
                return display
        return None

    def search_candidates(
        self,
        query: str,
        top_k: int = 10,
        city: str | None = None,
        national_mobility: bool = False,
        min_compatibility: float = MIN_COMPATIBILITY,
    ) -> list[CandidateHit]:
        """Bonus 1 (recruiter side): NL search over candidate profiles.

        Handles queries like « un développeur python à Brazzaville » or « comptable
        avec mobilité nationale »: the free text is matched by TF-IDF cosine over
        candidate profiles, and the *simulated* city / national-mobility flag act as
        hard filters. A city named inside the query is auto-detected when ``city`` is
        not given explicitly.

        As on the offer side, a **soft semantic guard** demotes a high-scoring
        candidate whose core métier shares no keyword with the query, and only
        candidates reaching ``min_compatibility`` (%) are returned — a candidate with
        zero token overlap has similarity 0 → 0 % and is dropped rather than shown as
        a spurious ~23 % row.

        Note: the simulated city is used here as an *exploration filter* only (a
        recruiter narrowing candidates) — it is never part of the core matching
        score computed by :meth:`score_candidate_row` / :meth:`recommend_all`.
        """
        if city is None:
            city = self.detect_city(query)

        q = norm(query)
        qprof = self._professional_query_tokens(q)
        # Query text: professional part only (city is a hard filter, not text signal),
        # expanded so lexical variants match (statistique ↔ statisticien).
        prof_text = " ".join(t for t in q.split() if t not in self._geo_tokens)
        sims = linear_kernel(
            self.cand_vectorizer.transform([self._expand_query(prof_text, self._cand_vocab)]),
            self.candidate_matrix,
        ).ravel()

        # Soft semantic guard: demote a high candidate whose core métier / qualif /
        # filière shares no professional keyword with the query (a city match, being
        # excluded from qprof, can never open the guard).
        for pos in np.nonzero(sims >= _GUARD_HIGH_RAW)[0]:
            if not (qprof and self._professional_overlap(qprof, self._candidate_core_tokens[pos])):
                sims[pos] *= SOFT_PENALTY

        mask = np.ones(len(self.candidates), dtype=bool)
        if city:
            mask &= (self.candidates[SIMULATED_CITY_COL].values == city)
        if national_mobility and self._mobility_col:
            mob = self.candidates[self._mobility_col].astype(str).map(norm).values
            mask &= (mob == "oui")

        # Keep only candidates passing the hard filters AND clearing the % floor.
        floor = _inv_calibrate(min_compatibility)
        keep = np.nonzero(mask & (sims >= floor))[0]
        keep = keep[np.argsort(-sims[keep])][:top_k]
        return [
            CandidateHit(
                candidate_id=self.candidates.iloc[pos][CANDIDATE_ID],
                rank=rank,
                score=float(sims[pos]),
                compatibility=calibrate(float(sims[pos])),
                candidate=self.candidates.iloc[pos],
            )
            for rank, pos in enumerate(keep, start=1)
        ]
