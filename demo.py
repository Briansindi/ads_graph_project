
#!/usr/bin/env python3
"""
Demo CLI for the advertising graph search project.

Usage examples:
  python demo.py --nodes 1000 --dim 50 --seed 42 --strategy naive --radius 2.5 --weights uniform
  python demo.py --nodes 2000 --strategy graph --radius 3.0 --weights budget
"""
import argparse, time, json, math, random
from ads_graph.graph import AdsGraph
from ads_graph.metrics import weighted_distance
from ads_graph.search import radius_search_naive, radius_search_graph_bfs
from ads_graph.heuristics import pca_project, kmeans, ann_ball_cover

def build_weights(dim: int, mode: str):
    if mode == "uniform":
        return [1.0] * dim
    if mode == "budget":
        w = [0.5]*dim
        for j in range(45, 50):
            w[j] = 3.0
        return w
    if mode == "interests":
        w = [0.5]*dim
        for j in range(10, 30):
            w[j] = 2.0
        return w
    # custom: comma-separated floats
    try:
        parts = [float(x) for x in mode.split(",")]
        assert len(parts) == dim
        return parts
    except Exception:
        raise SystemExit("Invalid --weights. Use 'uniform', 'budget', 'interests' or 50 comma-separated floats.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=1000)
    p.add_argument("--dim", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strategy", choices=["naive", "graph", "pca-ann"], default="naive")
    p.add_argument("--radius", type=float, default=2.5)
    p.add_argument("--weights", type=str, default="uniform")
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args()

    g = AdsGraph.random_graph(n=args.nodes, dim=args.dim, seed=args.seed, edge_strategy="knn", k=8)
    Y = build_weights(args.dim, args.weights)

    t0 = time.time()
    if args.strategy == "naive":
        res = radius_search_naive(g, args.start, Y, args.radius)
    elif args.strategy == "graph":
        res = radius_search_graph_bfs(g, args.start, Y, args.radius)
    else:
        # PCA + simple ANN prefilter to shrink candidate set, then exact check
        try:
            import numpy as np
        except Exception:
            print("NumPy not available; falling back to naive.")
            res = radius_search_naive(g, args.start, Y, args.radius)
        else:
            X = [g.nodes[i].features for i in range(args.nodes)]
            Z, mu, comps = pca_project(X, k=min(10, args.dim))
            # If PCA failed, fallback
            if Z is None:
                res = radius_search_naive(g, args.start, Y, args.radius)
            else:
                # Build ball cover in low-dim space w.r.t Euclidean radius ~ heuristic from args.radius
                # Map weighted radius to Euclidean guess: simple proportional heuristic
                ball_r = max(0.3, args.radius / args.dim)
                buckets = ann_ball_cover(Z, radius=ball_r)
                # pick bucket containing start
                import numpy as np
                d2 = ((Z - Z[args.start])**2).sum(1)
                # choose closest center bucket by scanning
                # rebuild centers from first idx of each bucket
                centers = np.array([Z[b[0]] for b in buckets])
                idx_center = ((centers - Z[args.start])**2).sum(1).argmin()
                cand = set(buckets[idx_center])
                # also include neighbors of start in the graph (diversify)
                for v in g.neighbors(args.start):
                    cand.add(v)
                # exact check on candidates only
                A = g.nodes[args.start].features
                res = []
                for vid in cand:
                    d = weighted_distance(A, g.nodes[vid].features, Y)
                    if d <= args.radius:
                        res.append((vid, d))
                res.sort(key=lambda x: x[1])
    t1 = time.time()

    print(json.dumps({
        "strategy": args.strategy,
        "nodes": args.nodes,
        "dim": args.dim,
        "radius": args.radius,
        "start": args.start,
        "result_count": len(res),
        "top10": res[:10],
        "elapsed_sec": round(t1 - t0, 4)
    }, indent=2))

if __name__ == "__main__":
    main()
