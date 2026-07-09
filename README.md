# ACPE : Système intelligent d'appariement Demandeurs et Offres d'emploi

Prototype pour le **Hackathon IndabaX Congo 2026** (Agence Congolaise pour l'Emploi).
Le système met automatiquement en relation les demandeurs d'emploi et les offres, produit un
**score de compatibilité expliqué**, un **classement Top-5 / Top-10**, un **tableau de bord
décisionnel**, ainsi que les deux défis bonus (recherche en langage naturel et analyse d'écart
de compétences).

## Installation & lancement

```bash
pip install -r requirements.txt

# Interface (conseillers ACPE)
streamlit run app.py

# Générer les recommandations + métriques pour toute la base
python -m scripts.generate_recommendations
# → outputs/recommendations_top10.csv, recommendations_top5.csv, metrics.json
```

## Approche (choix techniques justifiés)

Le moteur est **hybride TF-IDF + règles métier**, structuré en **espaces multi-champs** et
**régularisé par une garde sémantique douce**, conçu pour l'**explicabilité** exigée par le jury et
la **fiabilité opérationnelle** attendue sur le terrain.

1. **Espaces multi-champs (anti-dilution).** La vectorisation est séparée en deux sous-espaces
   *indépendants* : un **espace professionnel** (métier visé, qualification, filière, secteur métier
   côté demandeur ; intitulé, poste, secteur + description/profil/compétences côté offre) qui **seul**
   produit le score, et un **espace géographique** (champ officiel `Lieu` des offres) réservé à la
   recherche localisée. On évite ainsi la *dilution sémantique* : un token non métier fréquent (par
   ex. « Brazzaville ») ne peut plus rapprocher un « chauffeur » d'un « statisticien ». La localité
   *simulée* des candidats n'entre **jamais** dans le score (artefact analytique uniquement).
2. **Similarité textuelle (cœur).** TF-IDF (n-grammes 1–2), normalisation L2, comparaison par
   **cosinus**. Choix motivé : les taxonomies de secteurs diffèrent entre les deux jeux de données,
   donc une correspondance textuelle souple est plus robuste qu'une égalité stricte de catégories.
3. **Règle métier.** Un bonus sur le recouvrement de tokens entre le *métier visé* et l'*intitulé*
   de l'offre, qui récompense une correspondance directe de poste et améliore la précision.
4. **Garde sémantique douce (régularisation).** La composante textuelle est multipliée par un facteur
   doux (`σ = 0,30`) **uniquement** lorsqu'une offre est *déjà très bien classée* (compatibilité ≥ 75 %)
   *mais* ne partage **aucun** mot-clé métier cœur (intitulé + poste + secteur) avec le profil. Cela
   rétrograde les correspondances inter-métiers absurdes en tête de liste **sans toucher** aux
   quasi-correspondances légitimes → **métriques préservées** (cf. Évaluation) et sécurité gagnée.
5. **Score & calibration.** Score de classement = `0.80 × similarité_texte' + 0.20 × règle_métier`.
   Le pourcentage de compatibilité affiché est une calibration logistique (strictement monotone,
   purement cosmétique, elle ne change jamais le classement). Un **seuil minimal d'affichage (45 %)**
   masque côté interface les offres peu compatibles (sans affecter les CSV de soumission ni
   l'évaluation).
6. **Garde d'intersection lexicale stricte (recherche).** Sur les **onglets d'exploration** (bonus 1),
   un filtre **ET (AND)** exige qu'un mot-clé professionnel de la requête partage au moins un token
   avec les champs cœur de l'offre / du candidat, sinon **rejet sous le seuil d'affichage**,
   « Femme de ménage » ne remonte plus « Assistant de direction ». Le paramètre `substring=False`
   bloque les sous-chaînes fortuites (« menage » dans « aménagement »), et le filtre de mobilité
   nationale s'applique en **intersection stricte** (retour vide propre si aucun match). Ces barrières
   **n'affectent que la recherche**, le scoring batch (`recommend_all`) et l'appariement noté sont
   intacts.

### Pourquoi ce modèle plutôt qu'un autre ? (choix ML justifiés)
Le guide autorise règles, TF-IDF, ML supervisé ou embeddings. Notre arbitrage :

| Approche | Verdict |
|---|---|
| Règles métier seules | Trop rigides (taxonomies divergentes) — **gardées en complément** (0,20). |
| **TF-IDF + cosinus** | Robuste, rapide, **explicable terme à terme**, sans entraînement : **cœur du moteur** (0,80). |
| ML supervisé (LTR) | Vérité terrain trop maigre (3 positifs/candidat) → surapprentissage : **écarté**. |
| Embeddings (SBERT) | Trop lourd pour le free tier, gain incertain sur textes courts : **architecture prête, non activé**. |
| Garde **dure** (mise à zéro) | Sanctionne trop de quasi-correspondances → **dégrade** Precision@5/NDCG@5 : **écartée**. |
| **Garde douce conditionnelle** | N'agit que sur les détournements avérés, **métriques préservées, sécurité gagnée** : **retenue**. |

Décision : **hybride TF-IDF (0,80) + règle métier (0,20)**, en **espaces multi-champs** et
**garde sémantique douce**, meilleur compromis qualité / explicabilité / coût / fiabilité, chaque
score restant décomposable pour le conseiller. Détails et équations dans `RAPPORT.md` (§3).

### Pourquoi pas les champs géographiques / secteur demandé dans le score ?
`Mobilité géographique` (côté demandeur) est un indicateur Oui/Non/Non-déclaré (pas une ville) et
`Secteur demandé` est « Non déclaré » à ~91 %. Bâtir des règles dures dessus dégraderait le système ;
le signal exploitable provient des champs métier / qualification / filière. Le champ **`Lieu`** des
offres, lui, est fiable : il alimente l'**espace géographique indépendant** utilisé pour la recherche
d'offres localisée, mais **jamais** le score candidat ↔ offre (les candidats n'ont pas de vraie
ville). La **localité des candidats** est *simulée* de façon déterministe (clé = Matricule) suivant
la distribution réelle des offres, uniquement pour la carte du tableau de bord et la recherche de
candidats, voir `RAPPORT.md` §2.1 et §3.0.

## Données

| Fichier | Rôle |
|---|---|
| `Demandeurs .xlsx` | 41 285 demandeurs (profils) |
| `Offres_ACPE.xlsx` | 2 535 offres (table de référence) |
| `Offres_ACPE_Extensions.xlsx` | enrichissement texte (description/profil/compétences) de 143 offres |
| `Appariement_Demandeurs_Offres.xlsx` | vérité terrain : 3 offres pertinentes par demandeur |

Nettoyage automatisé (`src/data_loader.py`) : normalisation des en-têtes (accents + espaces
parasites), déduplication, traitement de « Non déclaré » comme valeur manquante, fusion de
l'enrichissement texte dans la table d'offres.

**Harmonisation catégorielle d'affichage** (colonnes `*_std`, dérivées *après* le texte de matching,
sans toucher aux champs qui alimentent les vectorizers) : `secteur_std` fusionne les variantes brutes
sous des étiquettes propres (p. ex. **« Agriculture & Agroalimentaire »** ; familles Sécurité,
Transport/Logistique, Éducation, Énergie-Eau-Environnement, Tourisme-Hôtellerie) et alimente le
**filtre sectoriel du tableau de bord** ; `metier_std` harmonise casse et genre (« Étudiant(e) »,
« Logisticien(ne) »…) ; `niveau_etude` est mis en **majuscules** ; `statut_demandeur` classe le
profil (Professionnel / Étudiant(e) / Stagiaire). Ces colonnes sont **purement analytiques**, le
scoring lit les champs bruts, **métriques inchangées**.

## Évaluation

Métriques du guide (Precision/Recall/NDCG @5 et @10) calculées contre la vérité terrain.
`Precision@5` est plafonnée à 0,60 (seulement 3 offres pertinentes par candidat).

| Métrique | Valeur (ensemble des 41 285 demandeurs) |
|---|---|
| Precision@5 / @10 | **0,427** / 0,257 |
| Recall@5 / @10 | **0,712** / **0,855** |
| NDCG@5 / @10 | **0,688** / 0,753 |

Mesures **garde activée** : la garde sémantique douce étant ciblée sur les seuls détournements, elle
**n'altère pas** ces métriques (écart nul au millième vs moteur non gardé) tout en supprimant les
correspondances inter-métiers absurdes. La **garde d'intersection lexicale stricte** (recherche) ne
touche ni `recommend_all` ni l'appariement noté : les valeurs ci-dessus restent **rigoureusement
inchangées**. _(Snapshot complet dans `eval_metrics.json` / `outputs/metrics.json`.)_

## Fonctionnalités

- **Appariement expliqué** : Top-5/Top-10 par demandeur avec décomposition du score et termes
  déterminants. Seuil minimal de compatibilité (45 %), message clair si aucune offre suffisante,
  métadonnées manquantes en « Non spécifié » et **Job ID (Référence offre) toujours visible** pour
  le placement.
- **Recherche intelligente (bonus 1)** : requête en langage naturel sur les **offres** (relevance
  métier + bonus de l'espace géographique quand une ville est citée) *et* sur les **candidats** (vue
  recruteur : filtres localité simulée + mobilité nationale, ville détectée automatiquement).
- **Analyse d'écart de compétences (bonus 2)** : compétences/exigences de l'offre absentes du profil.
- **Tableau de bord** : demandeurs/offres, **filtre sectoriel harmonisé**, secteurs & **métiers
  techniques dominants** (hors « Étudiant(e) »/« Stagiaire », isolés dans une carte KPI **« Statut des
  demandeurs d'emploi »**), **taux moyen de compatibilité** + distribution, répartition géographique
  des **offres et des candidats**, **secteurs par type de contrat (CDI/CDD)**, et **métriques
  d'évaluation** (Precision/Recall/NDCG @5/@10).
- **Interface aux couleurs nationales** (vert · jaune · rouge) et ma signature **S2M**.

## Structure

```
src/          utils · data_loader · matching · metrics · explain
app.py        interface Streamlit (4 onglets, couleurs nationales)
scripts/      generate_recommendations.py (CSV de soumission + métriques)
outputs/      résultats générés (non versionnés)
```
