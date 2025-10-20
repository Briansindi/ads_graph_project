
from __future__ import annotations
from typing import Iterable, Tuple, Sequence

def weighted_distance(u: Sequence[float], v: Sequence[float], y: Iterable[float]) -> float:
    """d_Y(u,v) = sum_i y_i * (u_i - v_i)^2"""
    total = 0.0
    for ui, vi, yi in zip(u, v, y):
        total += float(yi) * (float(ui) - float(vi)) ** 2
    return total
