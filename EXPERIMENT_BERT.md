# Expérience — Embeddings BERT vs TF-IDF (comparaison hors-ligne, CPU)

Cette expérience mesure si des **embeddings sémantiques multilingues (BERT /
Sentence-Transformers)** battent, ou complètent utilement, le moteur **TF-IDF** de
production sur la vérité terrain de l'ACPE. Elle est **isolée** : ni l'application
Streamlit déployée ni le moteur noté ne l'importent, et `requirements.txt` n'est pas
modifié.

> **Décision par défaut :** garder TF-IDF en production (léger, explicable, ~1 Go de
> RAM sur le *free tier*, P@5 ≈ 0,427). N'envisager un remplacement (ou un hybride)
> **que si** BERT/hybride dépasse nettement TF-IDF sur P@5 / NDCG@5 ci-dessous.

## Résultats mesurés (échantillon 4 000, `paraphrase-multilingual-MiniLM-L12-v2`, CPU)

| Moteur | P@5 | P@10 | R@5 | R@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| **TF-IDF (déployé)** | **0,423** | **0,256** | **0,704** | **0,852** | **0,680** | **0,748** |
| BERT (MiniLM multilingue) | 0,185 | 0,125 | 0,308 | 0,415 | 0,279 | 0,328 |
| Hybride α=0,3 | 0,319 | 0,195 | 0,532 | 0,651 | 0,532 | 0,586 |
| Hybride α=0,5 | 0,354 | 0,216 | 0,590 | 0,720 | 0,595 | 0,655 |
| Hybride α=0,7 | 0,374 | 0,227 | 0,623 | 0,757 | 0,622 | 0,683 |

**Conclusion : TF-IDF l'emporte nettement.** BERT seul fait *moins de la moitié* (P@5 0,185
vs 0,423) et, dans l'hybride, **chaque point de poids donné à BERT dégrade le score** —
plus α (part TF-IDF) est élevé, meilleur est le résultat. Autrement dit, l'apport
sémantique de BERT est ici *négatif*.

*Pourquoi :* les intitulés/qualifications font 2–4 mots (peu de contexte pour un
encodeur de phrases), la vérité terrain est construite sur un alignement lexical
métier↔intitulé que les n-grammes TF-IDF captent précisément, et un MiniLM multilingue
(non spécialisé français) rapproche des rôles sémantiquement voisins mais faux. Un modèle
français dédié (type CamemBERT) pourrait réduire l'écart, mais celui-ci est trop grand
(0,185 vs 0,423) pour être renversé, et n'aiderait ni la RAM ni l'explicabilité.

**Décision confirmée : TF-IDF reste le moteur de production.** Cette comparaison sert de
justification chiffrée du choix de modèle (critère « utilisation du ML » du jury).

## Pourquoi c'est pertinent (et ses limites)

- **Gain possible :** la synonymie non lexicale (*développeur* ↔ *programmeur*,
  *statisticien* ↔ *data analyst*) que TF-IDF ne capte pas.
- **Limites :** les intitulés sont **très courts** (peu de contexte pour BERT), seules
  **143 / 2 535** offres ont un descriptif, et l'**explicabilité** (récompensée par le
  jury) est plus difficile avec des vecteurs denses qu'avec les « termes déterminants »
  du TF-IDF.
- **Sans GPU :** l'encodage se fait **une seule fois hors-ligne** (quelques minutes CPU) ;
  l'évaluation n'est ensuite qu'un produit scalaire sur des vecteurs `.npy` mis en cache
  (aucun modèle en mémoire). Déterministe.

## Reproduire (3 étapes)

```bash
# 1. Dépendances de l'expérience (dans le même venv) — CPU, pas de GPU
pip install -r requirements-bert.txt

# 2. Encoder offres + candidats une seule fois (cache dans outputs/embeddings/, ~67 Mo)
python -m scripts.build_embeddings
#    modèle par défaut : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 3. Comparer TF-IDF vs BERT vs hybride sur la vérité terrain
python -m scripts.compare_engines                 # échantillon de 4000 candidats
python -m scripts.compare_engines --sample 41000  # base (quasi) complète
python -m scripts.compare_engines --alphas 0.3,0.5,0.7
```

Le script `compare_engines.py` affiche un tableau P@5/P@10/R@5/R@10/NDCG@5/NDCG@10 pour :
- **TF-IDF (shipped)** : le moteur déployé (avec la garde sémantique douce) ;
- **BERT** : cosinus pur sur embeddings ;
- **Hybride α** : `α · cosinus_TF-IDF + (1−α) · cosinus_BERT`.

## Déploiement (BERT/hybride)

Ne **pas** charger le modèle dans l'app *free tier* (RAM ~1 Go). À la place :
précalculer les embeddings hors-ligne, committer les `.npy`, et faire uniquement le
cosinus dans l'app (aucun `torch`/`transformers` en production). La recherche en langage
naturel *en direct* nécessite le modèle pour encoder la requête — la laisser sur TF-IDF,
ou réserver la variante BERT à un usage local / à un hébergement doté de plus de RAM.

## Fichiers

```
src/semantic.py               SemanticEngine (cosinus sur embeddings cachés) + encode_texts
scripts/build_embeddings.py   encodage hors-ligne -> outputs/embeddings/<modèle>/*.npy
scripts/compare_engines.py    tableau comparatif TF-IDF / BERT / hybride vs vérité terrain
requirements-bert.txt         dépendances de l'expérience (hors app déployée)
```
