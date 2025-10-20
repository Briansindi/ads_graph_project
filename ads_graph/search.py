
from __future__ import annotations
from collections import deque
from typing import Dict, List, Tuple, Iterable, Sequence, Callable, Optional
from .metrics import weighted_distance
from .graph import AdsGraph

def radius_search_naive(g: AdsGraph, start: int, y: Iterable[float], X: float) -> List[Tuple[int, float]]:
    """Exhaustive search over all nodes. Returns list of (node_id, distance) with d_Y <= X."""
    y = list(y)
    res: List[Tuple[int, float]] = []
    A = g.nodes[start].features
    for vid, node in g.nodes.items():
        d = weighted_distance(A, node.features, y)
        if d <= X:
            res.append((vid, d))
    res.sort(key=lambda x: x[1])
    return res

def radius_search_graph_bfs(g: AdsGraph, start: int, y: Iterable[float], X: float) -> List[Tuple[int, float]]:
    """Restrict to nodes reachable by following edges outward (best if graph already encodes proximity)."""
    y = list(y)
    res: List[Tuple[int, float]] = []
    A = g.nodes[start].features
    seen = set([start])
    q = deque([start])
    while q:
        u = q.popleft()
        node = g.nodes[u]
        d = weighted_distance(A, node.features, y)
        if d <= X:
            res.append((u, d))
            for v in g.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    res.sort(key=lambda x: x[1])
    return res
