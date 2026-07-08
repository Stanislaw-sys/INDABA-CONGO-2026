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
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .utils import is_missing, norm

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

# --- Categorical harmonization (display / analytics only) --------------------
# These standardized columns are built for the dashboard and human-facing views.
# CRITICAL: they are derived *after* ``candidate_text`` is assembled, so the raw
# columns that feed the TF-IDF vectorizer and the métier rule (see src/matching.py)
# are left byte-for-byte unchanged — harmonization must never move the graded
# metrics. Only ``niveau_etude`` is normalized in place, and it is neither a
# matching-text column nor a scoring input (it is used solely by the dashboard).
STD_METIER_COL = "metier_std"
STD_QUALIF_METIER_COL = "qualification_metier_std"
STD_SECTOR_COL = "secteur_std"
STATUT_COL = "statut_demandeur"

# Human-facing status labels (used by the dashboard "Statut des demandeurs").
STUDENT_LABEL = "Étudiant(e)"
STAGIAIRE_LABEL = "Stagiaire"
PRO_LABEL = "Professionnel / En activité"

# Canonical gender-neutral display labels, keyed on the *normalized* métier value
# (see utils.norm: lowercased, accent-stripped, punctuation-collapsed). Groups the
# masculine/feminine variants of the frequent job titles under one clean label so
# the dashboard doesn't split e.g. Logisticien / Logisticienne into two bars.
METIER_GENDER_GROUPS = {
    "logisticien": "Logisticien(ne)", "logisticienne": "Logisticien(ne)",
    "caissier": "Caissier(ère)", "caissiere": "Caissier(ère)",
    "electricien": "Électricien(ne)", "electricienne": "Électricien(ne)",
    "mecanicien": "Mécanicien(ne)", "mecanicienne": "Mécanicien(ne)",
    "informaticien": "Informaticien(ne)", "informaticienne": "Informaticien(ne)",
    "technicien": "Technicien(ne)", "technicienne": "Technicien(ne)",
    "commercial": "Commercial(e)", "commerciale": "Commercial(e)",
    "assistant": "Assistant(e)", "assistante": "Assistant(e)",
    "soudeur": "Soudeur(se)", "soudeuse": "Soudeur(se)",
    "vendeur": "Vendeur(se)", "vendeuse": "Vendeur(se)",
    "comptable": "Comptable",
}

# Redundant / near-duplicate sector labels folded under one clean canonical label,
# keyed on the *normalized* value (see utils.norm: lowercased, accent-stripped,
# punctuation-collapsed) so every accent/hyphen/ampersand spelling collapses to one
# key. Two spellings that normalize identically (e.g. the '-' and '&' variants of
# Éducation - Formation - Enseignement) share a single entry. Extend as variants surface.
SECTOR_GROUPS = {
    # Éducation & Enseignement — '… - Formation - Enseignement' and '…, Formation &
    # Enseignement' both normalize to the same key.
    "education formation enseignement": "Éducation & Enseignement",
    # Énergie, Eau & Environnement — merges the energy/water label with the
    # sanitation/waste/environment label.
    "energie environnement eau": "Énergie, Eau & Environnement",
    "assainissement gestion des dechets environnement": "Énergie, Eau & Environnement",
    # Tourisme, Hôtellerie & Loisirs.
    "tourisme loisirs voyages": "Tourisme, Hôtellerie & Loisirs",
    # Transit / Génie logistique — the transport* declensions are caught by the prefix
    # rule below; this transit-prefixed label is folded into the same pole here.
    "transit transport et logistique genie logistique":
        "Transport, Logistique & Supply Chain",
}

# Prefix-based rules: any normalized label *starting with* the key maps to the label.
# Used for open-ended families whose declensions all share a leading word, so new
# spellings are absorbed without enumerating each one. Verified against the data that
# no unrelated sector shares these prefixes — in particular HSE/QHSE labels start with
# 'responsable hsse' / 'hsse' / 'environnement' / 'assurance', never 'securite', so
# the security prefix never swallows an environment/quality offer.
SECTOR_PREFIX_GROUPS = (
    ("agriculture", "Agriculture & Agroalimentaire"),  # 3+ Agri/Agroalimentaire spellings
    ("securite", "Sécurité, Sûreté & Défense"),        # sûreté / défense / gardiennage
    ("transport", "Transport, Logistique & Supply Chain"),  # aérien / maritime / terrestre / …
)

_STUDENT_RE = re.compile(r"\betudiant")
_STAGIAIRE_RE = re.compile(r"stagiaire|\bstage\b")


def _canonical_metier(value) -> str:
    """Clean display label for a métier: student/stagiaire folding, gender grouping,
    else a whitespace-collapsed title-case of the original. Display only."""
    n = norm(value)
    if not n:
        return ""
    if _STUDENT_RE.search(n):
        return STUDENT_LABEL
    if _STAGIAIRE_RE.search(n):
        return STAGIAIRE_LABEL
    if n in METIER_GENDER_GROUPS:
        return METIER_GENDER_GROUPS[n]
    return re.sub(r"\s+", " ", str(value).strip()).title()


def _canonical_sector(value) -> str:
    """Group redundant sector labels under one clean canonical label.

    Exact normalized-key match first, then the open-ended prefix rules
    (Agriculture / Sécurité / Transport families); otherwise the original label is
    kept (whitespace-collapsed).
    """
    n = norm(value)
    if not n:
        return ""
    if n in SECTOR_GROUPS:
        return SECTOR_GROUPS[n]
    for prefix, label in SECTOR_PREFIX_GROUPS:
        if n.startswith(prefix):
            return label
    return re.sub(r"\s+", " ", str(value).strip())


def _statut(metier, qualif) -> str:
    """Classify a demandeur as Étudiant(e) / Stagiaire / Professionnel (display)."""
    blob = f"{norm(metier)} {norm(qualif)}"
    if _STUDENT_RE.search(blob):
        return STUDENT_LABEL
    if _STAGIAIRE_RE.search(blob):
        return STAGIAIRE_LABEL
    return PRO_LABEL


def _standardize_categoricals(cand: pd.DataFrame) -> None:
    """In-place: add harmonized categorical columns for display & dashboard analytics.

    MUST run after ``candidate_text`` is built. It never touches the columns that
    feed matching (``Métier visé / Qualification visée``, ``qualification_metier``,
    …) — it only *reads* them into new ``*_std`` columns. ``niveau_etude`` is the one
    field normalized in place (uppercase + strip, merging 'Aucun'/'aucun'/'AUCUN'),
    and it is not a matching input, so the graded metrics are unaffected.
    """
    metier_col = "Métier visé / Qualification visée"
    qualif_col = "qualification_metier"
    n = len(cand)

    if "niveau_etude" in cand.columns:
        cand["niveau_etude"] = cand["niveau_etude"].map(
            lambda v: pd.NA if is_missing(v) else str(v).strip().upper()
        )

    src_metier = cand[metier_col] if metier_col in cand.columns else pd.Series([""] * n)
    src_qualif = cand[qualif_col] if qualif_col in cand.columns else pd.Series([""] * n)
    cand[STD_METIER_COL] = src_metier.map(_canonical_metier)
    cand[STD_QUALIF_METIER_COL] = src_qualif.map(_canonical_metier)
    cand[STD_SECTOR_COL] = (
        cand["Secteur d'activité"].map(_canonical_sector)
        if "Secteur d'activité" in cand.columns else ""
    )
    cand[STATUT_COL] = [_statut(m, q) for m, q in zip(src_metier, src_qualif)]


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
    # Clean/grouped sector label for dashboard analytics (never fed to the vectorizer).
    offers[STD_SECTOR_COL] = (
        offers["Secteur activité"].map(_canonical_sector)
        if "Secteur activité" in offers.columns else ""
    )
    offers = offers[offers[OFFER_ID].notna()].reset_index(drop=True)
    return offers


def _build_candidates(root: Path) -> pd.DataFrame:
    cand = _read(root / DEMANDEURS_FILE).rename(columns={"Matricule": CANDIDATE_ID})
    cand = cand[cand[CANDIDATE_ID].notna()].drop_duplicates(CANDIDATE_ID).reset_index(drop=True)
    cand["candidate_text"] = cand[CANDIDATE_TEXT_COLS].apply(
        lambda r: " ".join(norm(v) for v in r), axis=1
    )
    # Harmonize categoricals for display/analytics ONLY, after freezing candidate_text.
    _standardize_categoricals(cand)
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
