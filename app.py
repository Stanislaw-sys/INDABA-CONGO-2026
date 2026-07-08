"""ACPE — Système intelligent d'appariement demandeurs ↔ offres d'emploi.

Streamlit prototype for the IndabaX Congo 2026 hackathon. Four tools for an ACPE
advisor:
  1. Appariement — pick a candidate, see explained Top-5/Top-10 recommendations + skill gap.
  2. Recherche d'offres — natural-language search over offers (bonus 1).
  3. Recherche de candidats — NL search over candidate profiles + city (bonus 1, recruiter side).
  4. Tableau de bord — decision dashboard with the KPIs required by the guide.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import (
    PRO_LABEL,
    STAGIAIRE_LABEL,
    STATUT_COL,
    STD_METIER_COL,
    STD_SECTOR_COL,
    STUDENT_LABEL,
    SIMULATED_CITY_COL,
    load_bundle,
)
from src.explain import skill_gap, top_matching_terms
from src.matching import MIN_COMPATIBILITY, MatchingEngine
from src.utils import is_missing


def fmt(value) -> str:
    """Human-facing metadata value: missing / NaN / « Non déclaré » → « Non spécifié »."""
    return "Non spécifié" if is_missing(value) else str(value).strip()

st.set_page_config(page_title="ACPE — Appariement Emploi", page_icon="💼", layout="wide")

ROOT = Path(__file__).resolve().parent

# Couleurs du drapeau national congolais : vert · jaune · rouge.
CONGO_GREEN, CONGO_YELLOW, CONGO_RED = "#009543", "#FCD116", "#DC241F"

st.markdown(
    f"""
    <style>
      /* Bandeau tricolore national en haut de page. */
      div[data-testid="stAppViewContainer"] {{
        border-top: 6px solid {CONGO_GREEN};
        background-image: linear-gradient(90deg,
          {CONGO_GREEN} 0%, {CONGO_GREEN} 33%,
          {CONGO_YELLOW} 33%, {CONGO_YELLOW} 66%,
          {CONGO_RED} 66%, {CONGO_RED} 100%);
        background-size: 100% 6px;
        background-repeat: no-repeat;
        background-position: top;
      }}
      .acpe-quote {{
        border-left: 5px solid {CONGO_YELLOW};
        background: {CONGO_GREEN}0D;
        padding: 0.6rem 1rem; margin: 0.4rem 0 1rem 0;
        border-radius: 6px; font-style: italic; color: #14261A;
      }}
      /* Signature discrète « Développé par S2M », toujours visible en bas à droite. */
      .s2m-sign {{
        position: fixed; right: 16px; bottom: 8px; z-index: 1000;
        font-size: 0.72rem; color: #6B7280; font-style: italic;
        padding: 2px 10px; border-radius: 8px;
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(2px);
      }}
      /* Filet tricolore sous le drapeau de la barre latérale. */
      .flag-rule {{
        height: 4px; border-radius: 2px; margin: 6px 0 2px 0;
        background: linear-gradient(90deg,
          {CONGO_GREEN} 33%, {CONGO_YELLOW} 33% 66%, {CONGO_RED} 66%);
      }}

      /* Navigation latérale : options en GRAS + MAJUSCULES, sélection en vert forêt. */
      section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        font-size: 0.95rem;
      }}
      section[data-testid="stSidebar"] div[role="radiogroup"] label {{ padding: 3px 0; }}
      section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) p {{
        color: {CONGO_GREEN} !important;
      }}

      /* En-tête principal — titre centré + bloc moderne. */
      .acpe-header {{
        background: linear-gradient(135deg, {CONGO_GREEN}14, #FFFFFF 70%);
        border: 1px solid {CONGO_GREEN}26; border-bottom: 4px solid {CONGO_GREEN};
        border-radius: 14px; padding: 16px 22px; margin: 4px 0 10px 0;
        text-align: center;
      }}
      .acpe-header h1 {{ margin: 0; font-size: 1.6rem; line-height: 1.25; color: #10231A; }}
      .acpe-agence {{
        font-weight: 700; letter-spacing: 0.5px; color: {CONGO_GREEN};
        text-transform: uppercase; font-size: 0.85rem; margin: 4px 0 4px 0;
      }}
      .flag-cap {{ text-align: center; font-size: 0.8rem; color: #4B5563; margin-top: 4px; }}

      /* Bandeau de métriques en badges colorés. */
      .acpe-badges {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 8px 0 6px 0; }}
      .acpe-badge {{
        display: flex; flex-direction: column; min-width: 120px;
        padding: 8px 14px; border-radius: 10px;
        border: 1px solid #E5E7EB; border-top: 3px solid var(--accent, {CONGO_GREEN});
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
      }}
      .acpe-badge .num {{ font-weight: 800; font-size: 1.1rem; color: #10231A; line-height: 1.15; }}
      .acpe-badge .lbl {{
        font-size: 0.7rem; color: #4B5563; text-transform: uppercase; letter-spacing: 0.4px;
      }}
    </style>
    <div class="s2m-sign">Développé par S2M</div>
    """,
    unsafe_allow_html=True,
)

# Barre latérale : logo officiel + intitulé de l'agence, navigation verticale,
# puis visuel de soutien et signature en bas.
NAV_MATCH = "🎯 APPARIEMENT"
NAV_SEARCH = "🔎 RECHERCHE D'OFFRES"
NAV_CAND = "🧭 RECHERCHE DE CANDIDATS"
NAV_DASH = "📊 TABLEAU DE BORD"

with st.sidebar:
    st.image(str(ROOT / "Logo_ACPE.png"), use_container_width=True)
    st.markdown(
        "<div class=\"acpe-agence\">Agence Congolaise pour L'emploi</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Navigation**")
    nav = st.radio(
        "Navigation",
        [NAV_MATCH, NAV_SEARCH, NAV_CAND, NAV_DASH],
        label_visibility="collapsed",
    )
    st.divider()
    st.image(str(ROOT / "Image de fond de l'application.jpg"), use_container_width=True)
    st.caption("_Développé par **S2M**_")


@st.cache_resource(show_spinner="Chargement des données et du moteur d'appariement…")
def get_engine():
    bundle = load_bundle()
    return bundle, MatchingEngine(bundle)


def filter_candidates(cities: tuple = (), niveaux: tuple = ()):
    """Candidate subset for the dashboard filters (city + niveau d'étude)."""
    sub = cand_df
    if cities:
        sub = sub[sub[SIMULATED_CITY_COL].isin(cities)]
    if niveaux:
        sub = sub[sub["niveau_etude"].astype(str).isin(niveaux)]
    return sub


@st.cache_data(show_spinner="Calcul du taux moyen de compatibilité…")
def compatibility_sample(cities: tuple = (), niveaux: tuple = (),
                         sample_n: int = 600, top_k: int = 5) -> np.ndarray:
    """Compatibility % of the Top-K recommendations over a random sample of candidates.

    Sampled (not the full 41k) so the KPI stays responsive on the free tier; cached
    per filter combination. Candidates are always scored against the full offer market,
    so this reflects each (filtered) candidate's true matching rate.
    """
    sub = filter_candidates(cities, niveaux)
    if len(sub) == 0:
        return np.zeros(0)
    samp = sub.sample(n=min(sample_n, len(sub)), random_state=0)
    vals = [
        r.compatibility
        for _, c in samp.iterrows()
        for r in engine.score_candidate_row(c, top_k=top_k)
    ]
    return np.asarray(vals, dtype=float)


@st.cache_data
def load_eval_metrics() -> dict | None:
    """Graded metrics (P@K/Recall@K/NDCG@K): live outputs first, else committed snapshot."""
    for path in (ROOT / "outputs" / "metrics.json", ROOT / "eval_metrics.json"):
        if path.exists():
            return json.loads(path.read_text())
    return None


bundle, engine = get_engine()
cand_df = bundle.candidates
off_df = bundle.offers

# En-tête : titre principal centré, drapeau national directement à sa droite.
head_l, head_r = st.columns([5, 1])
with head_l:
    st.markdown(
        '<div class="acpe-header"><h1>💼 ACPE : Appariement intelligent '
        "Demandeurs ↔ Offres</h1></div>",
        unsafe_allow_html=True,
    )
with head_r:
    st.image(str(ROOT / "Flag_Congo.jpg"), use_container_width=True)
    st.markdown('<div class="flag-cap">République du Congo</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="acpe-badges">
      <div class="acpe-badge" style="--accent:{CONGO_GREEN}">
        <span class="num">{len(cand_df):,}</span><span class="lbl">Demandeurs</span></div>
      <div class="acpe-badge" style="--accent:{CONGO_RED}">
        <span class="num">{len(off_df):,}</span><span class="lbl">Offres</span></div>
      <div class="acpe-badge" style="--accent:{CONGO_YELLOW}">
        <span class="num">TF-IDF multi-champs</span><span class="lbl">Moteur hybride + garde</span></div>
      <div class="acpe-badge" style="--accent:#6B7280">
        <span class="num">IndabaX Congo 2026</span><span class="lbl">Édition</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="acpe-quote">« Le travail acharné est la clé qui ouvre toutes les '
    "portes : le talent ouvre la voie, mais c'est l'effort qui mène au succès. »</div>",
    unsafe_allow_html=True,
)

# La navigation se fait via la barre latérale (variable `nav`) : une seule
# section est rendue à la fois, ce qui allège aussi le temps de rendu.

# ---------------------------------------------------------------- Appariement
if nav == NAV_MATCH:
    col_sel, col_k = st.columns([3, 1])
    id_list = cand_df["candidate_id"].tolist()
    id_set = set(id_list)
    with col_sel:
        cid = st.text_input(
            "Demandeur (Matricule)",
            value=id_list[0],
            help="Saisissez un matricule. Exemples ci-dessous.",
        ).strip()
        st.caption("Exemples : " + " · ".join(id_list[:4]))
    with col_k:
        top_k = st.radio("Top-K", [5, 10], horizontal=True)

    if cid not in id_set:
        st.warning(f"Matricule inconnu : « {cid} ». Essayez l'un des exemples ci-dessus.")
        st.stop()

    cand = cand_df[cand_df["candidate_id"] == cid].iloc[0]

    with st.container(border=True):
        st.subheader("Profil du demandeur")
        c1, c2, c3 = st.columns(3)
        c1.metric("Métier visé", str(cand.get("Métier visé / Qualification visée") or "—"))
        c2.metric("Niveau d'étude", str(cand.get("niveau_etude") or "—"))
        c3.metric("Secteur métier", str(cand.get("secteur_metier") or "—"))
        st.write(
            f"**Qualification :** {cand.get('Qualification') or '—'}  ·  "
            f"**Diplôme :** {cand.get('Diplome') or '—'}  ·  "
            f"**Filière :** {cand.get('Filière / Spécialité') or '—'}  ·  "
            f"**Âge :** {cand.get('Age') or '—'}  ·  "
            f"**Localité (simulée) :** {cand.get(SIMULATED_CITY_COL) or '—'}"
        )

    # Seuil de compatibilité minimal : on n'expose au conseiller que les offres
    # jugées réellement compatibles (les scores faibles sont écartés).
    all_results = engine.score_by_id(cid, top_k=top_k)
    results = [r for r in all_results if r.compatibility >= MIN_COMPATIBILITY]
    st.subheader(f"Top-{top_k} offres recommandées")
    st.caption(
        f"Seuil de compatibilité minimal appliqué : **{MIN_COMPATIBILITY:.0f} %** "
        "(les offres en deçà sont masquées)."
    )

    if not results:
        st.info(
            "Aucune offre ne correspond à ce profil avec un niveau de compatibilité "
            "suffisant pour le moment."
        )
        st.stop()

    table = pd.DataFrame(
        {
            "Rang": [r.rank for r in results],
            "Offre": [fmt(r.offer.get("Intitule")) for r in results],
            "Entreprise": [fmt(r.offer.get("Entreprise")) for r in results],
            "Lieu": [fmt(r.offer.get("Lieu")) for r in results],
            "Type de contrat": [fmt(r.offer.get("Type contrat")) for r in results],
            "Compatibilité": [f"{r.compatibility:.1f} %" for r in results],
            # Job ID / Référence offre : outil opérationnel essentiel pour le
            # placement ACPE — toujours affiché.
            "Job ID": [r.offer_id for r in results],
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    # Ground-truth badge (this candidate's reference offers, if any).
    truth = set(bundle.ground_truth.get(cid, []))
    if truth:
        hit = sum(1 for r in results if r.offer_id in truth)
        st.caption(
            f"🎯 Offres de référence retrouvées dans le Top-{top_k} affiché : "
            f"**{hit}/{len(truth)}** ({', '.join(sorted(truth))})"
        )

    st.subheader("Pourquoi ces recommandations ? (explicabilité)")
    for r in results:
        with st.expander(
            f"#{r.rank} · {r.offer.get('Intitule')} — {r.compatibility:.1f} % de compatibilité"
        ):
            a, b = st.columns(2)
            a.markdown(
                f"**Décomposition du score**\n\n"
                f"- Similarité textuelle : `{r.text_sim:.3f}`\n"
                f"- Correspondance métier/intitulé : `{r.metier_overlap:.2f}`\n"
                f"- Score global (rang) : `{r.score:.3f}`"
            )
            terms = top_matching_terms(engine, cand["candidate_text"], r.offer_id)
            if terms:
                a.markdown(
                    "**Termes déterminants :** "
                    + ", ".join(f"`{t}`" for t, _ in terms)
                )
            gap = skill_gap(r.offer, cand)
            with b:
                st.markdown(f"**Couverture des exigences : {gap['coverage'] * 100:.0f} %**")
                if gap["missing"]:
                    st.markdown("**Compétences/atouts manquants (skill gap) :**")
                    st.write(", ".join(gap["missing"][:15]))
                else:
                    st.success("Aucun écart de compétence majeur détecté.")
                if not gap["has_detailed_requirements"]:
                    st.caption("Offre sans descriptif détaillé — écart estimé sur l'intitulé/secteur.")
            desc = r.offer.get("offer_ext_text")
            if isinstance(desc, str) and desc.strip():
                st.caption("Descriptif de l'offre disponible (fichier extensions).")

# --------------------------------------------------------- Recherche (bonus 1)
elif nav == NAV_SEARCH:
    st.subheader("🔎 Recherche d'offres en langage naturel")
    st.caption('Ex. : « développeur python à Brazzaville », « comptable expérimenté »')
    query = st.text_input("Votre requête", value="agent de sécurité")
    n = st.slider("Nombre de résultats", 5, 20, 10)
    if query.strip():
        hits = engine.search_offers(query, top_k=n)
        if not hits:
            st.info(
                "Aucun résultat ne correspond à cette recherche avec un niveau de "
                "pertinence suffisant (minimum 45%)."
            )
        else:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Rang": [h.rank for h in hits],
                        "Offre": [fmt(h.offer.get("Intitule")) for h in hits],
                        "Entreprise": [fmt(h.offer.get("Entreprise")) for h in hits],
                        "Lieu": [fmt(h.offer.get("Lieu")) for h in hits],
                        "Secteur": [fmt(h.offer.get("Secteur activité")) for h in hits],
                        "Compatibilité": [f"{h.compatibility:.1f} %" for h in hits],
                        "Job ID": [h.offer_id for h in hits],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption("Seuil de pertinence minimal appliqué : **45 %**.")

# ------------------------------------------ Recherche de candidats (bonus 1)
elif nav == NAV_CAND:
    st.subheader("🧭 Recherche de candidats en langage naturel")
    st.caption(
        'Vue recruteur / conseiller. Ex. : « un développeur python à Brazzaville », '
        '« comptable avec mobilité nationale ». La localité est *simulée* (voir Tableau de bord).'
    )
    cq = st.text_input("Votre requête", value="développeur informatique à Brazzaville")
    cc1, cc2, cc3 = st.columns([2, 2, 1])
    with cc1:
        city_choice = st.selectbox("Localité (simulée)", ["Toutes"] + bundle.cities)
    with cc2:
        nat_mob = st.checkbox("Mobilité nationale uniquement", value=False)
    with cc3:
        cn = st.slider("Résultats", 5, 20, 10)

    city_filter = None if city_choice == "Toutes" else city_choice
    detected = engine.detect_city(cq)
    if city_filter is None and detected:
        st.info(f"Localité détectée dans la requête : **{detected}** (filtre appliqué).")

    if cq.strip():
        chits = engine.search_candidates(
            cq, top_k=cn, city=city_filter, national_mobility=nat_mob
        )
        if not chits:
            st.info(
                "Aucun résultat ne correspond à cette recherche avec un niveau de "
                "pertinence suffisant (minimum 45%)."
            )
        else:
            mob_col = engine._mobility_col
            st.dataframe(
                pd.DataFrame(
                    {
                        "Rang": [h.rank for h in chits],
                        "Matricule": [h.candidate_id for h in chits],
                        "Métier / Qualification": [
                            fmt(h.candidate.get("qualification_metier")
                                or h.candidate.get("Métier visé / Qualification visée"))
                            for h in chits
                        ],
                        "Diplôme": [fmt(h.candidate.get("Diplome")) for h in chits],
                        "Localité (simulée)": [
                            fmt(h.candidate.get(SIMULATED_CITY_COL)) for h in chits
                        ],
                        "Mobilité nat.": [
                            fmt(h.candidate.get(mob_col)) if mob_col else "Non spécifié"
                            for h in chits
                        ],
                        "Compatibilité": [f"{h.compatibility:.1f} %" for h in chits],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f"{len(chits)} candidat(s) — classés par pertinence textuelle (TF-IDF cosinus) "
                "du profil vis-à-vis de la requête. Seuil de pertinence minimal : **45 %**."
            )

# ------------------------------------------------------------ Tableau de bord
elif nav == NAV_DASH:
    st.subheader("📊 Tableau de bord décisionnel ACPE")

    # ---------------------------------------------------------------- Filtres
    # Métiers affichés = libellés harmonisés (casse + genre regroupés) ; le score
    # d'appariement continue d'utiliser les colonnes brutes (métriques inchangées).
    metier_col = STD_METIER_COL
    # Options du filtre = libellés de secteur harmonisés (plus de fragments bruts).
    sector_opts = sorted(
        s for s in off_df[STD_SECTOR_COL].dropna().astype(str).str.strip().unique() if s
    )
    niveau_opts = sorted(
        n for n in cand_df["niveau_etude"].dropna().astype(str).str.strip().unique() if n
    )
    with st.container(border=True):
        st.markdown("**🔎 Filtres interactifs**")
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_sectors = st.multiselect("Secteur d'activité (offres)", sector_opts)
        with f2:
            sel_cities = st.multiselect("Localisation / Ville", bundle.cities)
        with f3:
            sel_niveaux = st.multiselect("Niveau d'étude (demandeurs)", niveau_opts)

    # ------------------------------------------------- Application des filtres
    off_f = off_df
    if sel_sectors:
        off_f = off_f[off_f[STD_SECTOR_COL].astype(str).str.strip().isin(sel_sectors)]
    if sel_cities:
        off_f = off_f[off_f["Lieu"].astype(str).str.strip().str.title().isin(sel_cities)]

    cand_f = filter_candidates(tuple(sel_cities), tuple(sel_niveaux))
    comp = compatibility_sample(tuple(sel_cities), tuple(sel_niveaux))
    metrics = load_eval_metrics()

    if sel_sectors or sel_cities or sel_niveaux:
        st.caption(
            f"Filtres actifs — {len(cand_f):,} demandeurs et {len(off_f):,} offres retenus. "
            "Tous les indicateurs ci-dessous sont recalculés sur ce sous-ensemble."
        )

    # ----------------------------------------------------------- Cartes KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Demandeurs", f"{len(cand_f):,}",
              delta=None if not (sel_cities or sel_niveaux) else f"{len(cand_f) - len(cand_df):,}")
    k2.metric("Offres d'emploi", f"{len(off_f):,}",
              delta=None if not (sel_sectors or sel_cities) else f"{len(off_f) - len(off_df):,}")
    k3.metric("Taux moyen de compatibilité",
              f"{comp.mean():.1f} %" if len(comp) else "—",
              help="Moyenne de compatibilité des Top-5, sur un échantillon des demandeurs filtrés.")
    k4.metric("Performance algo. (NDCG@5)",
              f"{metrics['ndcg@5']:.3f}" if metrics else "—",
              help="Qualité de classement globale du moteur d'appariement vs. vérité terrain ACPE "
                   "(indicateur global, non filtré).")

    if len(cand_f) == 0 and len(off_f) == 0:
        st.warning("Aucune donnée ne correspond à ces filtres. Élargissez la sélection.")
        st.stop()

    # -------------------------------------------- Statut des demandeurs d'emploi
    # Les profils « Étudiant(e) » / « Stagiaire » ne sont pas des métiers techniques :
    # on les isole ici pour ne pas fausser le classement des vrais métiers plus bas.
    with st.container(border=True):
        st.markdown("**🎓 Statut des demandeurs d'emploi**")
        statut = cand_f[STATUT_COL].value_counts()
        n_pro = int(statut.get(PRO_LABEL, 0))
        n_etu = int(statut.get(STUDENT_LABEL, 0))
        n_stg = int(statut.get(STAGIAIRE_LABEL, 0))
        total = max(len(cand_f), 1)
        s1, s2, s3 = st.columns(3)
        s1.metric("Professionnels / en activité", f"{n_pro:,}",
                  help=f"{n_pro / total * 100:.1f} % des demandeurs filtrés.")
        s2.metric("Étudiant(e)s", f"{n_etu:,}",
                  help=f"{n_etu / total * 100:.1f} % des demandeurs filtrés.")
        s3.metric("Stagiaires", f"{n_stg:,}",
                  help=f"{n_stg / total * 100:.1f} % des demandeurs filtrés.")
        st.caption(
            "Les étudiant(e)s et stagiaires sont **exclus** du graphique des métiers "
            "techniques ci-dessous pour ne refléter que les métiers réellement exercés."
        )

    # ----------------------------------------------------- Barres : métiers / secteurs
    c1, c2 = st.columns(2)
    with c1:
        # Hors étudiants / stagiaires (comptabilisés dans la carte « Statut » ci-dessus).
        metier_tech = cand_f[~cand_f[STATUT_COL].isin([STUDENT_LABEL, STAGIAIRE_LABEL])]
        top_met = (metier_tech[metier_col].replace("", pd.NA).dropna()
                   .value_counts().head(10).reset_index())
        top_met.columns = ["Métier visé", "Demandeurs"]
        st.plotly_chart(
            px.bar(top_met, x="Demandeurs", y="Métier visé", orientation="h",
                   title="Métiers techniques les plus demandés (hors étudiants / stagiaires)",
                   color_discrete_sequence=[CONGO_GREEN]),
            use_container_width=True,
        )
    with c2:
        top_sec = (off_f["Secteur activité"].dropna().astype(str).str.strip()
                   .replace("", pd.NA).dropna().value_counts().head(10).reset_index())
        top_sec.columns = ["Secteur", "Offres"]
        st.plotly_chart(
            px.bar(top_sec, x="Offres", y="Secteur", orientation="h",
                   title="Secteurs les plus représentés (offres)",
                   color_discrete_sequence=[CONGO_RED]),
            use_container_width=True,
        )

    # ------------------------------------ Donuts : géographie offres vs candidats
    c3, c4 = st.columns(2)
    with c3:
        loc = (off_f["Lieu"].astype(str).str.strip().str.title()
               .replace("", pd.NA).dropna().value_counts().head(8).reset_index())
        loc.columns = ["Lieu", "Offres"]
        st.plotly_chart(
            px.pie(loc, names="Lieu", values="Offres", hole=0.45,
                   title="Répartition géographique des offres"),
            use_container_width=True,
        )
    with c4:
        cloc = cand_f[SIMULATED_CITY_COL].value_counts().head(8).reset_index()
        cloc.columns = ["Localité", "Demandeurs"]
        st.plotly_chart(
            px.pie(cloc, names="Localité", values="Demandeurs", hole=0.45,
                   title="Répartition géographique des demandeurs (simulée)"),
            use_container_width=True,
        )
        st.caption(
            "Localité *simulée* par tirage déterministe (clé = Matricule) suivant la "
            "distribution géographique réelle des offres — fichiers sources non modifiés."
        )

    # ---------------------- Secteurs offrant le plus d'opportunités par contrat
    with st.container(border=True):
        st.markdown(
            "**🏭 Top des secteurs d'activité offrant le plus d'opportunités "
            "par type de contrat (CDI / CDD)**"
        )
        cross = off_f.copy()
        cross["_contrat"] = cross["Type contrat"].astype(str).str.strip().str.upper()
        cross = cross[cross["_contrat"].isin(["CDI", "CDD"])]
        cross["_secteur"] = cross[STD_SECTOR_COL].replace("", pd.NA)
        cross = cross.dropna(subset=["_secteur"])
        if len(cross):
            top_secs = cross["_secteur"].value_counts().head(8).index.tolist()
            grp = (cross[cross["_secteur"].isin(top_secs)]
                   .groupby(["_secteur", "_contrat"]).size().reset_index(name="Offres"))
            grp.columns = ["Secteur", "Type de contrat", "Offres"]
            st.plotly_chart(
                px.bar(
                    grp, x="Offres", y="Secteur", color="Type de contrat",
                    orientation="h", barmode="group",
                    title="Opportunités par secteur et type de contrat (CDI / CDD)",
                    color_discrete_map={"CDI": CONGO_GREEN, "CDD": CONGO_YELLOW},
                    category_orders={"Secteur": top_secs[::-1]},
                ),
                use_container_width=True,
            )
            st.caption(
                "Croisement des offres nettoyées (secteurs harmonisés) restreint aux "
                "contrats stables CDI et CDD."
            )
        else:
            st.info("Aucune offre en CDI ou CDD pour ce filtre.")

    # --------------------------------- Histogramme : distribution des scores IA
    c5, c6 = st.columns(2)
    with c5:
        if len(comp):
            st.plotly_chart(
                px.histogram(
                    pd.DataFrame({"Compatibilité (%)": comp}),
                    x="Compatibilité (%)", nbins=25,
                    title="Distribution des scores de compatibilité IA (Top-5)",
                    color_discrete_sequence=[CONGO_GREEN],
                ),
                use_container_width=True,
            )
        else:
            st.info("Aucun demandeur dans ce filtre — pas de distribution à afficher.")
    with c6:
        st.markdown("**Qualité de l'appariement (vs. vérité terrain)**")
        if metrics:
            perf = pd.DataFrame(
                {
                    "Métrique": ["Precision", "Recall", "NDCG"],
                    "@5": [metrics["precision@5"], metrics["recall@5"], metrics["ndcg@5"]],
                    "@10": [metrics["precision@10"], metrics["recall@10"], metrics["ndcg@10"]],
                }
            )
            st.dataframe(
                perf.style.format({"@5": "{:.3f}", "@10": "{:.3f}"}),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                "💡 Note : Ce tableau évalue l'efficacité globale de l'algorithme hybride "
                "(80% Similarité sémantique + 20% Règles métiers) utilisé dans l'onglet "
                "🎯 Appariement, mesurée par rapport à la vérité terrain de l'ACPE (global, non filtré)."
            )
            st.caption(
                f"Évalué sur {int(metrics.get('n_evaluated', 0)):,} demandeurs. "
                "P@5 est plafonné à 0,60 (≤ 3 offres pertinentes par demandeur)."
            )
        else:
            st.info(
                "Exécuter `python -m scripts.generate_recommendations` pour produire "
                "`outputs/metrics.json` (Precision@K / Recall@K / NDCG@K)."
            )
