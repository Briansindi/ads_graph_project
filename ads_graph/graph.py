from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable, Optional

try:
    import numpy as np  # optional
except Exception:  # pragma: no cover
    np = None

@dataclass
class Node:
    id: int
    features: Tuple[float, ...]

@dataclass
class AdsGraph:
    """Simple adjacency-list graph whose nodes carry feature vectors."""
    dim: int
    nodes: Dict[int, Node] = field(default_factory=dict)
    edges: Dict[int, List[int]] = field(default_factory=dict)

    def add_node(self, node_id: int, features: Iterable[float]) -> None:
        feats = tuple(float(x) for x in features)
        if len(feats) != self.dim:
            raise ValueError(f"Feature length {len(feats)} != dim {self.dim}")
        self.nodes[node_id] = Node(node_id, feats)
        self.edges.setdefault(node_id, [])

    def add_edge(self, u: int, v: int, undirected: bool = True) -> None:
        if u not in self.nodes or v not in self.nodes:
            raise KeyError("Nodes must be added before edges.")
        self.edges.setdefault(u, []).append(v)
        if undirected:
            self.edges.setdefault(v, []).append(u)

    def neighbors(self, u: int) -> Iterable[int]:
        return self.edges.get(u, [])

    @staticmethod
    def random_graph(n: int = 1000, dim: int = 50, seed: Optional[int] = None,
                     edge_strategy: str = "knn", k: int = 8) -> "AdsGraph":
        """Generate a random graph with n nodes and `dim`-dim features.
        edge_strategy: 'random' or 'knn' (approximate k-nearest using sampling)
        """
        rng = random.Random(seed)
        g = AdsGraph(dim=dim)

        # Feature generator: simulate ad targeting profile (0..1 range)
        # Flexible groups: ~20% demo, ~40% interests, ~30% behavior, rest budget.
        def gen_features(i: int):
            base = [rng.random() for _ in range(dim)]

            n_demo = max(1, int(0.2 * dim))
            n_interest = max(1, int(0.4 * dim))
            n_behavior = max(1, int(0.3 * dim))
            used = n_demo + n_interest + n_behavior
            n_budget = max(0, dim - used)

            b0 = 0
            b1 = b0 + n_demo
            b2 = b1 + n_interest
            b3 = b2 + n_behavior
            b4 = dim  # end

            demo_bias = rng.uniform(-0.1, 0.1)
            interest_bias = rng.uniform(-0.05, 0.15)
            beh_bias = rng.uniform(-0.1, 0.1)
            bud_bias = rng.uniform(0.0, 0.3)

            for j in range(b0, b1):
                base[j] = min(1.0, max(0.0, base[j] + demo_bias))
            for j in range(b1, b2):
                base[j] = min(1.0, max(0.0, base[j] + interest_bias))
            for j in range(b2, b3):
                base[j] = min(1.0, max(0.0, base[j] + beh_bias))
            for j in range(b3, b4):
                base[j] = min(1.0, max(0.0, base[j] + bud_bias))
            return base

        feats = [gen_features(i) for i in range(n)]
        for i in range(n):
            g.add_node(i, feats[i])

        if edge_strategy == "random":
            # connect each node to k random others
            for u in range(n):
                partners = set()
                while len(partners) < k:
                    v = rng.randrange(n)
                    if v != u:
                        partners.add(v)
                for v in partners:
                    g.add_edge(u, v, undirected=True)
        else:  # knn (approximate via sampling for speed)
            sample = min(200, n)  # limit per-node candidate list
            def dist2(a, b):
                return sum((x - y) ** 2 for x, y in zip(a, b))
            for u in range(n):
                cand_idx = rng.sample(range(n), sample) if sample < n else list(range(n))
                cand_idx = [v for v in cand_idx if v != u]
                cand_idx.sort(key=lambda v: dist2(feats[u], feats[v]))
                for v in cand_idx[:k]:
                    g.add_edge(u, v, undirected=True)
        return g
