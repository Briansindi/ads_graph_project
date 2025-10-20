
from __future__ import annotations
from typing import List, Tuple, Iterable, Sequence, Optional
try:
    import numpy as np  # optional
except Exception:  # pragma: no cover
    np = None

def pca_project(data: Sequence[Sequence[float]], k: int = 10) -> Tuple[Optional["np.ndarray"], Optional["np.ndarray"], Optional["np.ndarray"]]:
    """Return (X_centered_proj, mean, components). Uses NumPy if available; otherwise returns (None, None, None)."""
    if np is None:
        return None, None, None
    X = np.asarray(data, dtype=float)
    mu = X.mean(0)
    Xc = X - mu
    # economical SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]
    Z = Xc @ comps.T
    return Z, mu, comps

def kmeans(X: "np.ndarray", k: int = 20, iters: int = 20, seed: int = 0) -> Tuple["np.ndarray", "np.ndarray"]:
    """Simple NumPy KMeans for clustering as a pre-filter (heuristic)."""
    if np is None:
        raise RuntimeError("NumPy required for kmeans")
    rng = np.random.default_rng(seed)
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        # assign
        dist2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        labels = dist2.argmin(1)
        # update
        for j in range(k):
            pts = X[labels == j]
            if len(pts):
                C[j] = pts.mean(0)
    return C, labels

def ann_ball_cover(X: "np.ndarray", radius: float) -> List[List[int]]:
    """Very simple ball-cover structure for approximate radius queries in projected (PCA) space.
    Returns list of buckets (list of indices per ball center)."""
    if np is None:
        raise RuntimeError("NumPy required for ann_ball_cover")
    n, d = X.shape
    used = np.zeros(n, dtype=bool)
    buckets: List[List[int]] = []
    for i in range(n):
        if used[i]: 
            continue
        # create new ball around i
        center = X[i]
        mask = ((X - center) ** 2).sum(1) <= radius * radius
        idxs = np.nonzero(mask)[0].tolist()
        used[mask] = True
        buckets.append(idxs)
    return buckets
