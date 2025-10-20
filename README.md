
# Simulation de recherche publicitaire — Implémentation Python

Ce dépôt contient une implémentation **clé en main** du projet de Master : *Simulation de recherche publicitaire sur graphe pondéré*.

## ⚙️ Fonctionnalités
- Génération d’un graphe entre **500 et 5000 nœuds** (paramétrable), chaque nœud ayant **50** caractéristiques.
- Distance pondérée : \\( d_Y(u,v)=\sum_i y_i (u_i-v_i)^2 \\).
- Trois stratégies de recherche dans un rayon `X` :
  1. **Naïve** (exhaustive)
  2. **Parcours restreint par arêtes** (BFS sur graphe)
  3. **Heuristique PCA + ANN** (*si NumPy dispo*, préfiltrage en basse dimension + vérification exacte).
- Script CLI `demo.py` pour reproduire des tests et comparer les temps.

## 🚀 Démarrage rapide
```bash
cd ads_graph_project
python3 -m pip install -r requirements.txt  # optionnel (vide par défaut)
python3 demo.py --nodes 1000 --strategy naive --radius 2.5 --weights uniform
python3 demo.py --nodes 2000 --strategy graph --radius 3.0 --weights budget
python3 demo.py --nodes 3000 --strategy pca-ann --radius 2.0 --weights interests
```

## 🧪 Tests
```bash
python3 -m pytest -q
```

## 📁 Structure
```
ads_graph_project/
  ads_graph/
    __init__.py
    graph.py
    metrics.py
    search.py
    heuristics.py
  tests/
    test_metrics_and_search.py
  demo.py
  README.md
  docs/
    Rapport_modele.md
    Resume_1page.md
```

## 🎛️ Poids `Y` prédéfinis
- `uniform` : tous les traits pèsent 1.
- `budget` : accent sur les indices 45–49 (budget).
- `interests` : accent sur les indices 10–29 (centres d’intérêt).
- Vous pouvez aussi fournir 50 flottants séparés par des virgules.

## 📊 Mesures
Le CLI retourne un JSON avec le temps d’exécution, le nombre de résultats et les **10 meilleurs** nœuds dans le rayon.

## 🧠 Heuristique
- **PCA** (si NumPy disponible) pour projeter en 10D puis **ball cover** pour réduire les candidats.
- Vérification exacte dans l’espace original avec la distance pondérée.

## 📌 Compatibilité
Pas de dépendances obligatoires. **NumPy** est utilisé si présent pour accélérer PCA/ANN. 
