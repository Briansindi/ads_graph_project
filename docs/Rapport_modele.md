
# Rapport — Simulation de recherche publicitaire sur graphe pondéré

## 1. Contexte et Modèle de Graphe
- Nœuds : entités (utilisateurs/produits/annonces) avec 50 caractéristiques réelles dans \\([0,1]\\).
- Groupes de traits : démographie (0–9), intérêts (10–29), comportement (30–44), budget (45–49).
- Arêtes : stratégie **kNN approximative** (k=8) par échantillonnage — capte la proximité sémantique.

## 2. Distance pondérée
\\[ d_Y(u,v) = \sum_{i=1}^{50} y_i \, (u_i - v_i)^2 \\]

## 3. Algorithmes de recherche
- **Naïf** : balayage de tous les nœuds (temps \\(O(|V|\\cdot 50)\\)).
- **BFS restreint** : parcours via arêtes si le graphe reflète la similarité locale (entre \\(O(|V|+|E|)\\) et \\(O(|E|)\\)).
- **PCA + ANN (heuristique)** : projection en 10D puis *ball cover* pour filtrer les candidats et test exact.

## 4. Complexité (asymptotique)
- Naïf : \\(O(|V|\\cdot d)\\), mémoire \\(O(|V|)\\).
- BFS : \\(O(|V|+|E|)\\) pour l’exploration + coût des tests sur visités.
- PCA (SVD) : \\(O(|V|\\cdot d^2)\\) — effectué **une fois**. Filtrage ANN ~ \\(O(|V|\\cdot k)\\).

## 5. Expérimentations (à compléter)
- Varier \\(|V|\\in\\{500,1000,2000,5000\\}\\), stratégies et rayons \\(X\\).
- Mesurer temps d’exécution, taille du résultat, et (si oracle) précision/rappel de l’heuristique vs naïf.
- Reproduire via `demo.py` avec `--seed` fixé.

## 6. Discussion et limites
- Graphe kNN approx. : rapide mais localement perfectible ; on peut augmenter *k*.
- PCA linéaire : ne capte pas forcément des structures non linéaires.
- Pas d’index ANN sophistiqué (FAISS/HNSW) pour rester sans dépendances.

## 7. Extensions (bonus)
- Chemin de similarité maximale (coût d’arête = distance pondérée) : réduction depuis *Shortest Path* avec coût dépendant des traits ⇒ heuristiques type Dijkstra/A* avec bornes informées ; preuve d’intractabilité exacte via arguments de dimension élevée et combinatoire (à détailler).

## 8. Reproductibilité
- Code Python 3.10+.
- Dépendances optionnelles : NumPy.

*Annexes :* logs JSON, figures de temps d’exécution, paramétrage des poids \\(Y\\).
