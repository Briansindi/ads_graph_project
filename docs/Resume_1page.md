
# Résumé (1 page)

dossier ads_graph: contient le code source principal

demo.py: contient le script principal pour la simulation

graph.py: modélisation du graph publicitaire

metrics.py: définit la métrique dY(u,v)

search.py: contient les 2 stratégies (naïve / BFS)

heuristics.py: Implémentation PCA + ANN

Nous modélisons la recherche publicitaire comme une **requête de voisinage pondéré** dans un graphe de profils (50 traits). La distance
\\( d_Y(u,v)=\sum_i y_i (u_i-v_i)^2 \\) permet d’orienter la similarité selon un vecteur de poids \\(Y\\) (ex. accentuer le budget ou les centres d’intérêt).
Nous générons un graphe **kNN approximatif** (k=8) pour capter la proximité locale.

Nous comparons :
1. **Naïf** : exact, simple, mais \\(O(|V|\\cdot d)\\).
2. **BFS restreint** : rapide si la structure des arêtes reflète la similarité.
3. **PCA + ANN** (si NumPy) : projection 10D + *ball cover* pour réduire les candidats, puis vérification exacte.

**Résultats attendus** : pour \\(|V|\\le 5000\\), le naïf reste une baseline robuste ; la stratégie **graph** est plus rapide quand le graphe est informatif ; **PCA+ANN**
réduit fortement le coût sans perte majeure pour des rayons modérés.

**Limites** : PCA linéaire, pas d’index ANN avancé, dépendance des performances à la qualité du graphe kNN.  
**Applications** : ciblage d’annonces, systèmes de recommandation, re-quêtes “clients similaires”.
