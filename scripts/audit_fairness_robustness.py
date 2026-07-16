"""Audit d'équité (parité de compatibilité par sous-groupe) et de robustesse
(stabilité du classement face à des perturbations lexicales mineures).

Ce script est déclaratif et reproductible : les deux `seed` sont fixées pour que
les chiffres cités dans le rapport (section "Analyse d'Équité et de Robustesse")
soient reproduits exactement.

Usage:
    python -m scripts.audit_fairness_robustness
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import OFFER_ID, STD_SECTOR_COL, load_bundle
from src.matching import MatchingEngine

BIAS_SEED = 42
BIAS_SAMPLE = 3000
ROBUST_SEED = 7
ROBUST_SAMPLE = 400


def _typo(text: str, n: int = 1) -> str:
    """Permute n paires de caractères adjacents (simule une faute de frappe)."""
    chars = list(text)
    for _ in range(n):
        if len(chars) < 4:
            break
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _case_and_spacing(text: str) -> str:
    """Casse aléatoire + double-espacement : doit être neutralisé par utils.norm()."""
    words = text.split()
    words = [w.upper() if random.random() < 0.5 else w for w in words]
    return "  ".join(words)


def audit_fairness(engine: MatchingEngine, bundle) -> pd.DataFrame:
    cand = bundle.candidates.sample(BIAS_SAMPLE, random_state=BIAS_SEED)
    rows = []
    for _, c in cand.iterrows():
        results = engine.score_candidate_row(c, top_k=5)
        rows.append({
            "candidate_id": c["candidate_id"],
            "Genre": c.get("Genre"),
            "Age": c.get("Age"),
            "statut": c.get("statut_demandeur"),
            "top1_compat": results[0].compatibility if results else 0.0,
            "has_offer": any(r.compatibility >= 45.0 for r in results),
        })
    return pd.DataFrame(rows)


def audit_robustness(engine: MatchingEngine, bundle) -> pd.DataFrame:
    random.seed(ROBUST_SEED)
    np.random.seed(ROBUST_SEED)
    cand = bundle.candidates.sample(ROBUST_SAMPLE, random_state=ROBUST_SEED)
    # Offer sector lookup + ground truth, to characterise *what kind* of change a
    # typo induces on the Top-1 (same-sector reshuffle vs genuine drift, and whether
    # the new Top-1 is still a ground-truth-correct offer). Read-only — no random draw
    # here, so the jaccard / top1_changed figures stay bit-for-bit reproducible.
    sector = dict(zip(bundle.offers[OFFER_ID], bundle.offers.get(STD_SECTOR_COL, bundle.offers[OFFER_ID])))
    gt = bundle.ground_truth
    rows = []
    for _, c in cand.iterrows():
        base = engine.score_candidate_row(c, top_k=10)
        base_ids = {r.offer_id for r in base}

        c_typo = c.copy()
        c_typo["candidate_text"] = _typo(c["candidate_text"], n=random.choice([1, 2]))
        typo_res = engine.score_candidate_row(c_typo, top_k=10)
        typo_ids = {r.offer_id for r in typo_res}

        c_case = c.copy()
        c_case["candidate_text"] = _case_and_spacing(c["candidate_text"])
        case_res = engine.score_candidate_row(c_case, top_k=10)
        case_ids = {r.offer_id for r in case_res}

        def jaccard(a: set, b: set) -> float:
            return len(a & b) / len(a | b) if (a or b) else 1.0

        base_top1 = base[0].offer_id if base else None
        typo_top1 = typo_res[0].offer_id if typo_res else None
        changed = base_top1 != typo_top1
        rows.append({
            "jaccard_typo": jaccard(base_ids, typo_ids),
            "jaccard_case_control": jaccard(base_ids, case_ids),
            "top1_changed_typo": changed,
            # Only meaningful when the Top-1 changed; NaN otherwise.
            "top1_same_sector": (sector.get(base_top1) == sector.get(typo_top1)) if changed else np.nan,
            "top1_new_in_gt": (typo_top1 in gt.get(c["candidate_id"], [])) if changed else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("Chargement des données et du moteur…")
    bundle = load_bundle()
    engine = MatchingEngine(bundle)

    print(f"\n=== Audit d'équité (n={BIAS_SAMPLE}, seed={BIAS_SEED}) ===")
    fair = audit_fairness(engine, bundle)
    print(fair.groupby("Genre")[["top1_compat", "has_offer"]].mean().round(3))
    print(fair.groupby("statut")[["top1_compat", "has_offer"]].mean().round(3))

    print(f"\n=== Audit de robustesse (n={ROBUST_SAMPLE}, seed={ROBUST_SEED}) ===")
    robust = audit_robustness(engine, bundle)
    print(f"Jaccard top-10 moyen (contrôle casse/espacement) : "
          f"{robust.jaccard_case_control.mean():.3f}")
    print(f"Jaccard top-10 moyen (typo)                       : "
          f"{robust.jaccard_typo.mean():.3f}")
    print(f"Top-1 changé par une typo                         : "
          f"{robust.top1_changed_typo.mean() * 100:.1f} %")
    chg = robust[robust.top1_changed_typo]
    n_chg = len(chg)
    if n_chg:
        print(f"  dont même secteur d'activité                    : "
              f"{chg.top1_same_sector.sum():.0f}/{n_chg} "
              f"({chg.top1_same_sector.mean() * 100:.0f} %)")
        print(f"  dont secteur réellement différent (vraie dérive): "
              f"{(~chg.top1_same_sector.astype(bool)).sum():.0f}/{ROBUST_SAMPLE} "
              f"({(~chg.top1_same_sector.astype(bool)).sum() / ROBUST_SAMPLE * 100:.1f} % de tous)")
        print(f"  dont nouveau Top-1 encore dans la vérité terrain: "
              f"{chg.top1_new_in_gt.sum():.0f}/{n_chg} "
              f"({chg.top1_new_in_gt.mean() * 100:.0f} %)")

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fair.to_csv(out / "audit_fairness.csv", index=False)
    robust.to_csv(out / "audit_robustness.csv", index=False)
    print(f"\nDétails sauvegardés dans {out}/audit_fairness.csv et audit_robustness.csv")


if __name__ == "__main__":
    main()
