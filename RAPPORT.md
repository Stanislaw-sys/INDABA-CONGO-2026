# Rapport technique — Système intelligent d'appariement Demandeurs ↔ Offres d'emploi

**Hackathon IndabaX Congo 2026 — Agence Congolaise pour l'Emploi (ACPE)**

- Dépôt GitHub : https://github.com/Stanislaw-sys/Hackathon-IndabaX-Congo-2026
- Application déployée : https://hackathon-indabax-congo-2026-ldcrc3qrgpuipq3rlqbpa5.streamlit.app/

---

## 1. Contexte et problématique

L'ACPE met en relation demandeurs d'emploi et entreprises, mais l'identification des meilleures
correspondances reste manuelle, chronophage et peu reproductible. Notre objectif : un prototype
fonctionnel qui, à partir des données de l'Agence, **calcule un score de compatibilité candidat ↔
offre, classe automatiquement les meilleures offres (Top-5 / Top-10), explique chaque
recommandation et identifie les écarts de compétences** — le tout exploitable directement par un
conseiller ACPE via une interface web.

## 2. Données et préparation

Quatre fichiers Excel fournis, réconciliés dans un pipeline unique (`src/data_loader.py`) :

| Fichier | Contenu | Volume |
|---|---|---|
| `Demandeurs .xlsx` | Profils des demandeurs | 41 285 (après dédoublonnage) |
| `Offres_ACPE.xlsx` | Table de référence des offres | 2 535 |
| `Offres_ACPE_Extensions.xlsx` | Enrichissement texte (description/profil/compétences) | 143 offres |
| `Appariement_Demandeurs_Offres.xlsx` | Vérité terrain : 3 offres pertinentes/demandeur | 41 285 |

**Nettoyage et décisions clés (toutes vérifiées sur les données) :**

- **Normalisation des en-têtes** : les colonnes contiennent des accents et des espaces parasites
  (`'offre_pertinente '`, `'Date de publication '`) — systématiquement nettoyés.
- **Déduplication** : 13 matricules et plusieurs `id_demandeur` en double sont supprimés.
- **Sentinelle de valeur manquante** : la chaîne « Non déclaré » est traitée partout comme valeur
  absente (`src/utils.py`).
- **Fusion des offres** : les 143 références du fichier « Extensions » **existent déjà** dans la
  table principale ; il ne s'agit donc pas d'un second jeu d'offres mais d'un **enrichissement
  textuel**. Nous conservons la table principale et y greffons la description/profil/compétences
  quand elles existent. *Vérification :* les 1 676 identifiants d'offres distincts de la vérité
  terrain (préfixes `JOB*`, `NONREF*`, `STG*`) se résolvent tous dans la table principale.
- **Champs écartés à dessein** : `Mobilité géographique` est un indicateur Oui/Non/Non-déclaré (pas
  une ville) et `Secteur demandé` est « Non déclaré » à ~91 %. Construire des règles dures dessus
  dégraderait le système. Le signal exploitable provient des champs **métier / qualification /
  filière / secteur métier**.

### 2.1 Localisation des candidats — variable simulée

Le fichier des demandeurs ne contient **aucune ville exploitable** (`Mobilité géographique` n'est
qu'un indicateur Oui/Non). Or le guide demande la « répartition géographique … des candidats » dans
le tableau de bord, et le bonus de recherche vise des requêtes du type « développeur python à
**Brazzaville** ». Nous **simulons** donc une localité par candidat (`src/data_loader.py`), sans
jamais modifier les fichiers sources :

- les **poids de tirage** sont la distribution géographique **réelle des offres** (`Lieu`), afin que
  les candidats simulés se concentrent là où se trouve effectivement le marché du travail
  (Pointe-Noire, Brazzaville, Bouenza, …) plutôt que d'être uniformément aléatoires ;
- le tirage est **déterministe et reproductible** (clé = hachage du `Matricule`) ;
- l'attribut est **étiqueté « simulé » partout** où il apparaît (interface et présent rapport) : il
  sert la démonstration et n'altère pas le moteur d'appariement, qui reste inchangé.

### 2.2 Harmonisation catégorielle (affichage et pilotage)

Les libellés catégoriels bruts sont fragmentés (variantes de casse, de genre, d'orthographe
sectorielle). Nous dérivons donc, **après** figement du texte de matching et **sans** modifier les
colonnes qui alimentent les vectorizers, des **colonnes harmonisées d'affichage** (`*_std`) :

- **`secteur_std`** — fusion des variantes brutes sous des étiquettes unifiées propres : p. ex.
  « Agriculture/Agroalimentaire » et « Agriculture & Industrie Agroalimentaire » → **« Agriculture &
  Agroalimentaire »** ; familles également regroupées pour Sécurité, Transport/Logistique, Éducation,
  Énergie-Eau-Environnement et Tourisme-Hôtellerie. Le **premier filtre interactif** du tableau de
  bord s'appuie désormais sur cette colonne consolidée (fin des fragments bruts dans la liste).
- **`metier_std`** / **`qualification_metier_std`** — casse harmonisée et regroupement du genre
  (« Etudiant »/« Etudiante » → « Étudiant(e) », « Logisticien(ne) », « Caissier(ère) »…).
- **`niveau_etude`** nettoyé en **majuscules** (« Aucun »/« aucun »/« AUCUN » → « AUCUN »).
- **`statut_demandeur`** — Professionnel / Étudiant(e) / Stagiaire, exploité par le tableau de bord.

> Ces colonnes sont **purement analytiques et d'affichage** : le cœur du scoring lit les champs
> bruts, de sorte que les métriques (§5) restent rigoureusement inchangées.

## 3. Moteur d'appariement (méthodologie)

Nous avons retenu une approche **hybride TF-IDF + règle métier**, structurée en **espaces
multi-champs** et protégée par une **garde sémantique douce**. Elle est motivée par deux exigences :
la **qualité de l'appariement** (30 % de la note) et l'**explicabilité** (soulignée par le jury),
auxquelles s'ajoute une contrainte de **fiabilité opérationnelle** : ne jamais présenter au
conseiller une correspondance manifestement absurde.

### 3.0 Espaces multi-champs (professionnel vs géographique)

Pour éviter la **dilution sémantique** — le risque qu'un token fréquent et non métier (par ex. une
localité comme « Brazzaville ») rapproche artificiellement deux emplois sans rapport (un
« chauffeur » et un « statisticien ») — la vectorisation est **séparée en deux sous-espaces
indépendants** :

- **Espace professionnel (cœur du moteur).** Seuls les champs métier sont vectorisés : côté
  demandeur (métier visé, qualification métier, qualification, secteur métier, filière) et côté
  offre (intitulé, poste, secteur + description/profil/compétences si disponibles). **Ce seul
  espace produit le score candidat ↔ offre.**
- **Espace géographique (indépendant).** Une vectorisation TF-IDF distincte, bâtie sur le champ
  officiel `Lieu` des offres, sert uniquement à la **recherche d'offres localisée** (bonus 1). Elle
  n'est **jamais** fusionnée dans le score d'appariement.

> **Garde-fou de pureté du moteur.** La localité *simulée* des candidats (§2.1) n'entre **jamais**
> dans le calcul du score : elle reste un artefact analytique réservé à la carte du tableau de bord
> et au filtre d'exploration des candidats. Le moteur d'appariement demeure ainsi strictement fondé
> sur des données réelles.

### 3.1 Composantes du score

1. **Similarité textuelle (cœur du moteur).** Le profil du demandeur est vectorisé en **TF-IDF avec
   n-grammes de longueur 1 à 2**, chaque offre de même dans l'espace professionnel. La compatibilité
   est la **similarité cosinus** entre les deux vecteurs, avec **normalisation L2** — celle-ci borne
   naturellement l'effet de la longueur, de sorte qu'une description enrichie longue et un intitulé
   court se comparent équitablement.
   *Justification :* les taxonomies de secteurs diffèrent entre les deux côtés. Une correspondance
   textuelle souple est donc bien plus robuste qu'une égalité stricte de catégories, et les
   bigrammes capturent « agent de sécurité » ou « génie logistique » comme des unités.

2. **Règle métier.** Un bonus proportionnel au **recouvrement de tokens entre le métier visé et
   l'intitulé de l'offre**. Cette règle récompense une correspondance directe de poste, ce qui
   améliore la précision au sommet du classement.

3. **Garde sémantique douce (régularisation conditionnelle).** Avant le classement, on inspecte
   l'**intersection des mots-clés professionnels cœur** (intitulé + poste + secteur de l'offre)
   avec les tokens du profil. On applique une **pénalité multiplicative douce** `σ = 0,30` à la
   composante textuelle **si et seulement si** deux conditions sont réunies :
   *(i)* l'intersection est **vide** (aucun mot-clé métier commun) **et** *(ii)* la compatibilité
   pré-garde est déjà **élevée** (≥ 75 %). Formellement, pour une offre *o* et un candidat *c* :

   ```
   sim_texte'(c,o) = sim_texte(c,o) × σ   si  |mots(c) ∩ cœur(o)| = 0  et  compat(c,o) ≥ 75 %
                   = sim_texte(c,o)        sinon
   ```

   *Justification (régularisation ciblée).* Une pénalité **dure** (mise à zéro systématique) sur
   toute intersection vide dégrade les métriques, car de nombreuses vraies correspondances récompensées
   par la vérité terrain partagent la sémantique sans partager un token exact. La garde **douce et
   conditionnelle** ne s'active au contraire que sur le **mode de défaillance précis** que l'on veut
   corriger — une offre **déjà bien classée mais sans aucun lien métier** (signature d'un détournement
   par des tokens diffus). Elle **rétrograde** ces hijacks en tête de liste **sans toucher** aux
   quasi-correspondances légitimes : sur l'ensemble des 41 285 demandeurs, la garde ne modifie qu'une
   poignée de cas par millier de candidats et **préserve exactement** Precision@K et NDCG@K (§5), tout
   en garantissant la sécurité opérationnelle demandée par un usage terrain.

4. **Score final.** `score = 0,80 × similarité_texte' + 0,20 × règle_métier`. Les deux composantes
   (dont la part textuelle *après* garde) sont conservées sur chaque résultat afin d'être affichées
   à l'utilisateur.

5. **Calibration du pourcentage.** Le score brut (typiquement 0–0,6) est transformé en
   « pourcentage de compatibilité » par une fonction logistique
   `compat = 100 / (1 + e^(-8 (score − 0,15)))`, de sorte qu'une bonne correspondance se lise
   ~85–95 %. Cette transformation est **strictement monotone** : purement cosmétique, elle ne modifie
   jamais le classement. C'est elle qui définit aussi le **seuil d'activation de la garde** (75 %) et
   le **seuil minimal d'affichage** (§4.1).

### 3.2 Seuil de compatibilité et interface épurée

Pour un usage terrain, seules les offres **réellement compatibles** sont présentées : un **seuil
minimal de compatibilité (45 %)** filtre les recommandations affichées (sans jamais affecter les
CSV de soumission ni l'évaluation, qui restent des Top-K complets, cf. §5). Si aucune offre ne
franchit ce seuil, l'interface affiche un message explicite plutôt qu'une liste vide. Les
métadonnées manquantes (par ex. `Type de contrat`) sont remplacées par « Non spécifié », tandis que
le **Job ID (Référence offre) reste toujours visible** : c'est l'identifiant opérationnel dont le
conseiller ACPE a besoin pour réaliser le placement effectif.

### 3.3 Justification des choix de modèles (choix ML)

Le guide laisse le choix de la méthode (règles, similarité textuelle, ML supervisé/non supervisé,
embeddings sémantiques). Nous avons évalué chaque famille au regard de **quatre contraintes du
problème** : (i) l'appariement pèse 30 % de la note et l'**explicabilité** est explicitement
récompensée ; (ii) la vérité terrain ne fournit que **3 offres pertinentes par candidat** (signal
d'entraînement très maigre et déséquilibré) ; (iii) seules **143/2 535 offres** ont un descriptif
détaillé — le texte est donc court et partiel ; (iv) le prototype doit **tourner sur l'hébergement
gratuit** (CPU, ~1 Go RAM) et rester reproductible.

| Approche | Avantages | Pourquoi écartée / retenue ici |
|---|---|---|
| **Règles métier seules** | Totalement transparentes | Trop rigides : les libellés de secteur des offres et des demandeurs ne coïncident pas ; couverture faible. **Retenues en complément** (règle métier↔intitulé), pas comme socle. |
| **TF-IDF + cosinus** | Robuste au vocabulaire, rapide, **explicable terme à terme**, sans entraînement | Ne capte pas les synonymes non lexicaux. **Retenu comme cœur** : meilleur rapport qualité/explicabilité/coût sur des textes courts. |
| **ML supervisé (learning-to-rank)** | Optimise directement les métriques | Vérité terrain trop maigre (3 positifs/candidat, aucun négatif explicite) → fort risque de surapprentissage ; boîte plus noire. **Écarté** comme modèle principal, gardé en perspective. |
| **Embeddings sémantiques (Sentence-Transformers)** | Capte la sémantique / les synonymes | Modèle lourd (téléchargement + RAM) mal adapté au free tier ; gain incertain sur des intitulés très courts ; explicabilité plus difficile. **Architecture prête à l'accueillir** (`SemanticEngine`), non activé par défaut. |

**Décision : un modèle hybride TF-IDF (0,80) + règle métier (0,20), en espaces multi-champs et
régularisé par une garde sémantique douce.** Il maximise la qualité d'appariement *mesurée*
(voir §5) tout en gardant chaque score **décomposable et justifiable** devant un conseiller —
critère décisif du jury —, sans dépendance lourde ni entraînement fragile. La pondération 0,80/0,20
a été retenue empiriquement (meilleur compromis Precision/NDCG sur échantillon) ; elle est isolée
dans deux constantes (`W_TEXT`, `W_METIER`) et pourrait être apprise ultérieurement par
learning-to-rank sans changer l'architecture.

*Choix d'une garde douce plutôt que dure.* Une coupure dure de la similarité (mise à zéro dès
qu'aucun token métier n'est partagé) a été testée : elle **dégrade** Precision@5 et NDCG@5 (elle
sanctionne trop de quasi-correspondances récompensées par la vérité terrain). La **garde douce et
conditionnelle** retenue (§3.1-3) n'agit que sur les détournements avérés — score déjà élevé *et*
zéro recouvrement métier — ; mesurée sur l'ensemble de la base, elle **conserve intégralement** les
métriques (§5) tout en éliminant les correspondances inter-métiers absurdes. C'est une
**régularisation** au sens propre : un a priori qui pénalise les configurations improbables sans
pénaliser les bonnes.

### 3.4 Garde d'intersection lexicale stricte (onglets de recherche)

La garde douce (§3.1-3) protège le **cœur** du moteur. Les **onglets d'exploration sémantique
visuelle** (recherche d'offres et de candidats, bonus 1) ajoutent une seconde barrière, plus
stricte, contre les **faux positifs à bas score** qui polluaient le bas du classement :

- **Filtre d'intersection lexicale (ET logique).** Une offre — resp. un candidat — n'est retenue que
  si le mot-clé professionnel de la requête (ou du métier visé) partage **au moins un token** avec
  ses champs cœur (intitulé / poste / secteur). À défaut, elle est **rejetée sous le seuil
  d'affichage (45 %)** plutôt que de combler le Top-K : une recherche **« Femme de ménage »** ne
  remonte ainsi plus un poste d'**« Assistant de direction »** (`search_offers` / `search_candidates`).
- **Anti sous-chaîne fortuite (`substring=False`).** La relation lexicale n'admet qu'un **préfixe**
  (radical / flexion), et non une sous-chaîne au milieu d'un mot : « menage » ne matche donc plus
  « aménagement ».
- **Mobilité nationale — intersection stricte.** Le filtre « mobilité nationale uniquement » se
  combine en **ET** avec les autres critères ; si l'intersection est vide, le système renvoie un
  **résultat vide propre** au lieu de lignes hors-sujet.

> Ces barrières agissent **exclusivement** sur les onglets de recherche/exploration : le scoring
> batch (`recommend_all`) et l'appariement noté demeurent intacts (§5).

## 4. Score de compatibilité et explicabilité

Pour chaque recommandation, le système restitue :

- la **décomposition du score** (part textuelle vs part règle métier) ;
- les **termes déterminants** — les mots dont le produit TF-IDF candidat × offre contribue le plus
  au cosinus (`src/explain.py::top_matching_terms`), ce qui répond directement à la demande du jury
  d'« expliquer les variables ayant contribué au score » ;
- l'**analyse d'écart de compétences** (voir bonus 2).

*Exemple réel (candidat `PPKOU2501080016340`, « Agent de transit ») :* le Top-3 renvoyé est
exactement composé de ses trois offres de référence (JOB250000904 / 1440 / 1694), avec pour termes
déterminants `de transit`, `transit`, `agent de`, `logistique`.

## 5. Recommandation Top-K et évaluation

Le système produit pour chaque demandeur les Top-5 et Top-10 offres, au format de soumission
`candidate_id, rank, job_id, score` (`scripts/generate_recommendations.py` →
`outputs/recommendations_top{5,10}.csv`).

**Évaluation** contre la vérité terrain, sur **l'ensemble des 41 285 demandeurs** :

| Métrique | @5 | @10 |
|---|---|---|
| **Precision** | **0,427** | 0,257 |
| **Recall** | **0,712** | **0,855** |
| **NDCG** | **0,688** | 0,753 |

*Lecture des résultats :* chaque candidat n'ayant que **3 offres pertinentes**, la Precision@5 est
plafonnée à 3/5 = 0,60 et la Precision@10 à 3/10 = 0,30. Notre Precision@5 de 0,427 atteint donc
**~71 % du maximum théorique**. Le Recall@10 de 0,855 signifie que, dans plus de 85 % des cas, les
offres de référence figurent dans les 10 premières recommandations — un résultat directement
exploitable pour assister les conseillers.

*Effet de la garde sémantique douce :* ces chiffres sont mesurés **garde activée**. La garde étant
ciblée sur les seuls détournements (score élevé sans lien métier), elle **n'altère pas** les
métriques par rapport au moteur non gardé (écart nul au millième sur Precision@5 / NDCG@5) : elle
apporte la sécurité opérationnelle **sans coût de performance**.

*Garde d'intersection lexicale stricte (§3.4) :* elle s'applique **uniquement** aux onglets de
recherche sémantique visuelle et **ne touche ni `recommend_all` ni l'appariement noté**. Le cœur du
modèle de scoring batch est donc **rigoureusement inchangé**, et les valeurs ci-dessus
(**P@5 = 0,427**, **NDCG@5 = 0,688**) restent identiques : ces barrières logiques ne visent que le
confort d'exploration, pas la performance mesurée.

## 6. Fonctionnalités bonus

- **Bonus 1 — Recherche intelligente (langage naturel), offres *et* candidats.**
  - *Côté offres* : la requête (« développeur informatique à Brazzaville ») est vectorisée dans
    l'espace **professionnel** et classée par similarité cosinus ; si elle mentionne une localité
    connue, l'**espace géographique indépendant** (§3.0) ajoute un bonus aux offres effectivement
    situées dans cette ville — la localité oriente donc la recherche **sans jamais contaminer** le
    vecteur métier (`src/matching.py::search_offers`).
  - *Côté candidats (vue recruteur/conseiller)* : un second espace TF-IDF est construit sur les
    profils des 41 285 demandeurs. La requête « un développeur python à Brazzaville » est classée par
    cosinus, et la **localité** (simulée, voir §2.1) ainsi que la **mobilité nationale** agissent comme
    filtres ; la ville mentionnée dans la phrase est **détectée automatiquement**
    (`src/matching.py::search_candidates`). Cela couvre les deux exemples du guide
    (« développeur Python à Brazzaville », « candidat en comptabilité avec mobilité nationale »).
- **Bonus 2 — Analyse des écarts de compétences (skill gap).** Pour chaque offre recommandée, le
  système extrait les compétences/exigences (issues du descriptif enrichi lorsqu'il existe, sinon de
  l'intitulé et du secteur), retire les mots-outils, et renvoie celles absentes du profil du
  candidat ainsi qu'un taux de couverture (`src/explain.py::skill_gap`). Cela oriente le demandeur
  vers les compétences à développer.

## 7. Tableau de bord décisionnel

L'onglet « Tableau de bord » de l'application fournit aux conseillers **tous** les indicateurs
listés par le guide : nombre de demandeurs et d'offres, secteurs les plus représentés, métiers les
plus demandés, **taux moyen de compatibilité** (moyenne des Top-5 sur échantillon, avec sa
distribution), **répartition géographique des offres *et* des candidats** (cette dernière sur la
localité simulée, §2.1), types de contrat, et **statistiques sur les recommandations générées**
(tableau Precision/Recall/NDCG @5/@10 vs vérité terrain, cf. `eval_metrics.json`).

Depuis l'harmonisation catégorielle (§2.2), le **filtre sectoriel** propose des étiquettes
consolidées ; le graphique des **métiers techniques les plus demandés exclut** les profils
**« Étudiant(e) »** et **« Stagiaire »** — désormais isolés dans une **carte KPI dédiée « Statut des
demandeurs d'emploi »**, afin de ne pas fausser le classement des vrais métiers ; enfin une vue
croise les **secteurs offrant le plus d'opportunités par type de contrat (CDI / CDD)**.

## 8. Architecture technique et reproductibilité

Séparation nette entre le cœur data/ML (`src/`) et l'interface (`app.py`) ; le moteur est réutilisé
à l'identique par l'application et par les scripts batch.

```
src/utils.py        normalisation texte (accents, sentinelle « Non déclaré »)
src/data_loader.py  chargement + nettoyage → DataBundle (texte pro + texte géo séparés)
src/matching.py     MatchingEngine (espaces multi-champs, garde douce, TF-IDF + règle, Top-K, recherche)
src/metrics.py      Precision / Recall / NDCG @K
src/explain.py      explications + skill gap
app.py              interface Streamlit (4 onglets, seuil + valeurs « Non spécifié », couleurs nationales)
scripts/            génération des recommandations + métriques
```

Reproductibilité : `pip install -r requirements.txt`, puis `streamlit run app.py` (interface) ou
`python -m scripts.generate_recommendations` (soumission complète + `metrics.json`). Le calcul
complet sur 41 285 demandeurs s'exécute en ~85 secondes sur une machine standard. Aucune dépendance
lourde : le système fonctionne intégralement sur scikit-learn (pas de GPU requis).

## 9. Limites et perspectives

- **Descriptifs partiels** : seules 143/2 535 offres disposent d'un texte détaillé ; enrichir la
  collecte améliorerait la finesse de l'appariement et du skill gap.
- **Localisation** : la ville des candidats étant *simulée* (§2.1), un filtrage géographique
  *réel* nécessiterait de collecter la localité au moment de l'inscription ; le critère de proximité
  s'intégrerait alors sans changement d'architecture (le pipeline gère déjà la colonne).
- **Garde sémantique lexicale** : la garde douce (§3.1-3) repose sur un recouvrement de tokens
  *exacts* ; elle peut, à la marge, pénaliser légèrement une quasi-correspondance dont le vocabulaire
  diffère (« caisse » vs « caissier »). L'effet est négligeable (pénalité douce 0,30, quelques cas par
  millier, métriques inchangées) et disparaîtrait avec un espace sémantique (ci-dessous).
- **Embeddings sémantiques** : l'architecture est prévue pour accueillir un backend
  Sentence-Transformers (déjà anticipé dans `requirements.txt`), qui pourrait capter des synonymes
  métier au-delà du recouvrement lexical — et rendre la garde robuste aux variantes lexicales.
- **Apprentissage supervisé** : la vérité terrain pourrait servir à apprendre les poids de
  combinaison (learning-to-rank) plutôt que de les fixer à 0,80 / 0,20.

## 10. Conclusion

Le prototype répond à l'ensemble des livrables du hackathon — score de compatibilité expliqué,
classement Top-5/Top-10 évalué (Precision/Recall/NDCG), tableau de bord, et les deux défis bonus —
au sein d'une solution **explicable, reproductible et légère**, directement mobilisable par les
conseillers de l'ACPE pour accélérer et fiabiliser la mise en relation entre demandeurs et offres.
