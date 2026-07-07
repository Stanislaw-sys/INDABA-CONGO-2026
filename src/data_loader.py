"""Load and clean the four ACPE Excel datasets into a single :class:`DataBundle`.

Key facts about the raw files (verified against the data) that this module hides
from the rest of the codebase:

* Column names carry French accents *and* stray trailing spaces
  (``'offre_pertinente '``, ``'Date de publication '``) — every header is stripped.
* ``Demandeurs .xlsx`` has 13 duplicate ``Matricule`` rows; the ground-truth file
  has duplicate ``id_demandeur`` rows — both are de-duplicated.
* There are two offer files. ``Offres_ACPE_Extensions.xlsx`` is **not** a separate
  set of offers: all 143 of its references already exist in ``Offres_ACPE.xlsx``.
  It only adds free-text ``Description`` / ``Profil`` / ``Compétences``. We therefore
  keep the main file as the offer table and *enrich* it with that text where present.
* Candidate location signal is weak: ``Mobilité géographique`` is a Non/Oui/Non
  déclaré flag (not a city) and ``Secteur demandé`` is ~91% "Non déclaré". The
  usable candidate signal is the métier / qualification / filière columns.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .utils import norm

# Repo root = parent of the src/ package.
ROOT = Path(__file__).resolve().parent.parent

DEMANDEURS_FILE = "Demandeurs .xlsx"          # note the trailing space in the filename
OFFERS_FILE = "Offres_ACPE.xlsx"
OFFERS_EXT_FILE = "Offres_ACPE_Extensions.xlsx"
APPARIEMENT_FILE = "Appariement_Demandeurs_Offres.xlsx"

# Canonical offer / candidate id columns (after we standardize names below).
OFFER_ID = "offer_id"
CANDIDATE_ID = "candidate_id"

# Derived (simulated) candidate location. The source demandeur file has no usable
# city (Mobilité géographique is only a Oui/Non flag), so we *simulate* a plausible
# home locality per candidate — see _simulate_candidate_locations. This is a derived,
# in-memory column: the original Excel files are never modified.
SIMULATED_CITY_COL = "ville_simulee"
# Minor cleanups on the raw offer "Lieu" values so the simulated cities read cleanly.
_CITY_FIXES = {"Lkouala": "Likouala", "Bouansa": "Bouansa"}

# Candidate columns that carry real matching signal, best first.
CANDIDATE_TEXT_COLS = [
    "Métier visé / Qualification visée",
    "qualification_metier",
    "Qualification",
    "secteur_metier",
    "Filière / Spécialité",
    "Secteur demandé",
]

# Offer columns used to build the **professional** text blob (structured file).
# NB: location is deliberately excluded here — geography lives in its own sub-space
# (OFFER_GEO_COLS) so that a shared high-frequency locality token can never dilute
# or hijack job-title matching (see src/matching.py, "multi-field spaces").
OFFER_TEXT_COLS = ["Intitule", "Poste", "Secteur activité"]
# Extra free-text columns present only in the extension file.
OFFER_EXT_COLS = ["Description", "Profil", "Compétences"]
# Offer location field(s) — the **official** geographic signal. Kept in a separate
# sub-space, never merged into offer_text, and never blended into the core
# candidate↔offer score (candidates have no real city; the simulated one is analytics
# only). Used by the geographic sub-space for location-aware offer search/filtering.
OFFER_GEO_COLS = ["Lieu"]


@dataclass
class DataBundle:
    candidates: pd.DataFrame          # cleaned demandeurs, indexed 0..N, has CANDIDATE_ID + candidate_text
    offers: pd.DataFrame              # cleaned + enriched offers, indexed 0..M, has OFFER_ID + offer_text
    ground_truth: dict[str, list[str]]  # candidate_id -> [offer_id, ...] (up to 3, filtered to known offers)
    offer_pos: dict[str, int]         # offer_id -> row position in offers
    cities: list[str] = field(default_factory=list)  # distinct simulated candidate localities

    def offer_by_id(self, offer_id: str) -> pd.Series | None:
        pos = self.offer_pos.get(offer_id)
        return None if pos is None else self.offers.iloc[pos]


def _read(path: Path, **kw) -> pd.DataFrame:
    df = pd.read_excel(path, **kw)
    df.columns = [str(c).strip() for c in df.columns]  # kill trailing/leading spaces in headers
    return df


def _build_offers(root: Path) -> pd.DataFrame:
    offers = _read(root / OFFERS_FILE)
    offers = offers.rename(columns={"Référence offre": OFFER_ID})

    ext = _read(root / OFFERS_EXT_FILE).rename(columns={"Référence": OFFER_ID})
    ext_text = {
        row[OFFER_ID]: " ".join(norm(row.get(c)) for c in OFFER_EXT_COLS)
        for _, row in ext.iterrows()
    }

    def build_text(row: pd.Series) -> str:
        base = " ".join(norm(row.get(c)) for c in OFFER_TEXT_COLS)
        return f"{base} {ext_text.get(row[OFFER_ID], '')}".strip()

    # Attach the raw enrichment text too (used by skill-gap analysis).
    offers["offer_ext_text"] = offers[OFFER_ID].map(ext_text).fillna("")
    offers["offer_text"] = offers.apply(build_text, axis=1)
    # Separate geographic text (official Lieu), kept out of offer_text on purpose.
    offers["offer_geo_text"] = offers.apply(
        lambda r: " ".join(norm(r.get(c)) for c in OFFER_GEO_COLS).strip(), axis=1
    )
    offers = offers[offers[OFFER_ID].notna()].reset_index(drop=True)
    return offers


def _build_candidates(root: Path) -> pd.DataFrame:
    cand = _read(root / DEMANDEURS_FILE).rename(columns={"Matricule": CANDIDATE_ID})
    cand = cand[cand[CANDIDATE_ID].notna()].drop_duplicates(CANDIDATE_ID).reset_index(drop=True)
    cand["candidate_text"] = cand[CANDIDATE_TEXT_COLS].apply(
        lambda r: " ".join(norm(v) for v in r), axis=1
    )
    return cand


def _hash_unit(key: str) -> float:
    """Deterministic value in [0, 1) from a string key (stable across runs/machines)."""
    digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _simulate_candidate_locations(cand: pd.DataFrame, offers: pd.DataFrame) -> pd.Series:
    """Assign each candidate a *simulated* home locality.

    The demandeur file carries no exploitable city, yet the guide's dashboard asks for
    the "répartition géographique … des candidats" and the bonus NL search targets
    queries like « un développeur python à Brazzaville ». We therefore synthesise a
    plausible location per candidate:

    * the sampling weights are the **empirical distribution of offer locations** (`Lieu`),
      so simulated candidates concentrate where the labour market actually is
      (Pointe-Noire, Brazzaville, …) rather than being uniformly random;
    * the draw is keyed on a hash of the ``Matricule``, so it is **deterministic and
      reproducible** and never touches the source Excel files.

    This is explicitly a synthetic attribute for demonstration; it is labelled as
    "simulée" everywhere it surfaces in the UI and the report.
    """
    lieu = (
        offers["Lieu"].dropna().astype(str).str.strip().str.title()
        .replace(_CITY_FIXES)
    )
    counts = lieu[lieu != ""].value_counts()
    counts = counts[counts >= 3].head(12)  # keep the meaningful localities
    cities = counts.index.tolist()
    weights = (counts / counts.sum()).cumsum().tolist()

    def pick(matricule: str) -> str:
        u = _hash_unit(matricule)
        for city, cum in zip(cities, weights):
            if u <= cum:
                return city
        return cities[-1]

    return cand[CANDIDATE_ID].map(pick)


def _build_ground_truth(root: Path, offer_pos: dict[str, int]) -> dict[str, list[str]]:
    app = _read(root / APPARIEMENT_FILE).drop_duplicates("id_demandeur")
    gt: dict[str, list[str]] = {}
    for _, row in app.iterrows():
        offers = [row.get(f"id_offre{i}") for i in (1, 2, 3)]
        gt[row["id_demandeur"]] = [o for o in offers if o in offer_pos]
    return gt


def load_bundle(root: str | Path = ROOT) -> DataBundle:
    """Load, clean, and cross-link all four datasets."""
    root = Path(root)
    offers = _build_offers(root)
    offer_pos = {oid: i for i, oid in enumerate(offers[OFFER_ID])}
    candidates = _build_candidates(root)
    candidates[SIMULATED_CITY_COL] = _simulate_candidate_locations(candidates, offers)
    ground_truth = _build_ground_truth(root, offer_pos)
    return DataBundle(
        candidates=candidates,
        offers=offers,
        ground_truth=ground_truth,
        offer_pos=offer_pos,
        cities=sorted(candidates[SIMULATED_CITY_COL].dropna().unique().tolist()),
    )
