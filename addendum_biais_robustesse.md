# Section additionnelle : Analyse d'Équité et de Robustesse

> À insérer après la section 4 (Ingénierie de l'Explicabilité et Équité) du rapport, ou en annexe si l'espace est contraint. Rédigé pour s'insérer dans le style du rapport existant. Adapte les numéros de section à ta pagination réelle.

## 4bis. Analyse Complémentaire : Équité et Robustesse

Au-delà de l'exclusion architecturale de l'âge et du genre de la fonction de score (§4), nous avons conduit deux audits empiriques pour quantifier, plutôt que simplement affirmer, l'équité et la stabilité du moteur.

### 4bis.1 Audit d'équité (parité de compatibilité)

**Protocole** : échantillon aléatoire de 3 000 demandeurs (`seed=42`), score de compatibilité du Top-1 calculé pour chacun, comparé par sous-groupe.

| Sous-groupe | Compatibilité moyenne (Top-1) | Écart | Significativité |
|---|---|---|---|
| Femme | 93,58 % | — | — |
| Homme | 95,66 % | +2,09 pts | p = 0,0001 |
| Femme (Professionnels seuls) | — | — | — |
| Homme (Professionnels seuls) | — | +0,67 pt | p = 0,0032 |
| Étudiant(e) | 49,79 % | −47,4 pts vs Professionnel | — |
| Professionnel / En activité | 97,23 % | — | — |

**Interprétation** : l'écart brut Femme/Homme (2,09 points) se réduit à 0,67 point, quasi négligeable en pratique, une fois le statut du demandeur contrôlé. La cause de l'écart brut est compositionnelle et non algorithmique : dans notre échantillon, les femmes sont en moyenne plus jeunes (28,3 ans contre 31,8) et proportionnellement plus souvent étudiantes (7,4 % contre 3,4 %), deux facteurs corrélés au score indépendamment du genre, qui n'entre d'ailleurs jamais dans `candidate_text` (voir §4).

Le véritable écart structurel identifié est **statutaire** : les profils Étudiant(e) affichent une compatibilité moyenne inférieure de 47 points aux Professionnels, et un taux de couverture (au moins une offre au-dessus du seuil d'affichage) de 95,5 % contre 100 %. Ce n'est pas un artefact de biais démographique mais une conséquence directe de la richesse textuelle du profil : un(e) étudiant(e) sans expérience professionnelle déclarée produit un `candidate_text` structurellement plus pauvre en tokens professionnels spécifiques. Sur 41 285 demandeurs, cela concerne 1 804 profils (4,4 %), une population que l'ACPE devrait accompagner avec un protocole dédié (mise en avant de stages / contrats d'apprentissage, alignement sur les filières d'étude plutôt que sur un métier visé souvent non renseigné) plutôt que via le canal de recommandation standard.

### 4bis.2 Audit de robustesse (stabilité lexicale)

**Protocole** : échantillon de 400 demandeurs (`seed=7`). Deux perturbations testées sur `candidate_text`, comparées au Top-10 non perturbé via l'indice de Jaccard :
1. **Contrôle** : variation de casse et d'espacement (ne doit rien changer, la normalisation `utils.norm()` devant l'absorber intégralement).
2. **Test** : permutation de deux caractères adjacents (1 à 2 occurrences), simulant une faute de frappe réaliste.

| Perturbation | Jaccard top-10 moyen | % candidats inchangés | % avec Top-1 modifié |
|---|---|---|---|
| Casse / espacement (contrôle) | 1,000 | 100 % | 0 % |
| Faute de frappe (1-2 car.) | 0,786 | 42,0 % | 29,8 % |

**Interprétation** : le contrôle valide que le pipeline de normalisation absorbe parfaitement le bruit non lexical (casse, espacement), comportement attendu et vérifié. La perturbation typographique révèle en revanche une limite inhérente à tout matching par tokens exacts : une faute de frappe peut invalider la contribution TF-IDF du terme affecté, provoquant un changement de recommandation Top-1 dans 29,8 % des cas testés. Le score de stabilité (Jaccard = 0,786) reste correct en absolu, la majorité du classement Top-10 est préservée, mais documente une limite honnête plutôt qu'une garantie non testée.

**Nature des changements (le Top-1 « change » rarement pour le pire).** Le taux de 29,8 % surestime la gravité réelle si on l'interprète comme une « dérive ». En caractérisant chacun des 119 Top-1 modifiés (échantillon de 400) :
- **78 % (93/119) restent dans le même secteur d'activité** : le système hésite entre deux offres du même métier (p. ex. « Juriste » ↔ « Assistant.e juriste » qui échangent leur rang), un réordonnancement qu'un conseiller ACPE ne percevrait pas comme une erreur ;
- **seuls 6,5 % de l'ensemble des candidats (26/400) basculent vers un secteur réellement différent** : la seule vraie dérive au sens strict ;
- **45 % (54/119) des nouveaux Top-1 figurent toujours dans la vérité terrain** : statistiquement, près de la moitié des changements ne sont pas des erreurs mais une réorganisation entre deux bonnes réponses.

Autrement dit, la sensibilité typographique se traduit en pratique par un réordonnancement intra-secteur bénin bien plus que par une recommandation aberrante.

**Piste d'amélioration identifiée (non implémentée, hors périmètre du prototype)** : un mécanisme de correction orthographique légère en amont de la vectorisation (distance de Levenshtein bornée sur les tokens hors-vocabulaire) réduirait cette sensibilité sans coût algorithmique significatif, dans la même philosophie de robustesse lexicale déjà appliquée à la recherche en langage naturel (§3.2, `_lexical_related`).

### 4bis.3 Synthèse

Ces deux audits illustrent une démarche d'amélioration continue plutôt qu'une garantie a priori : l'équité algorithmique par exclusion des variables sensibles (§4) est confirmée empiriquement pour le genre une fois les facteurs de composition contrôlés, tandis que l'écart statutaire Étudiant(e)/Professionnel et la sensibilité typographique sont des limites documentées, quantifiées, et accompagnées de pistes de correction concrètes, plutôt que découvertes a posteriori.
