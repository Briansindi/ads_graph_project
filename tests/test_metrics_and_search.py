
import math
from ads_graph.graph import AdsGraph
from ads_graph.metrics import weighted_distance
from ads_graph.search import radius_search_naive, radius_search_graph_bfs

def test_distance_basic():
    u = [0, 0, 0]
    v = [1, 2, 3]
    y = [1, 1, 1]
    assert weighted_distance(u, v, y) == 1 + 4 + 9

def test_radius_search_small():
    g = AdsGraph.random_graph(n=50, dim=5, seed=1, k=4)
    Y = [1]*5
    res_naive = radius_search_naive(g, 0, Y, X=1.5)
    res_graph = radius_search_graph_bfs(g, 0, Y, X=1.5)
    # starting node must always be included (dist 0)
    assert res_naive[0][0] == 0 and abs(res_naive[0][1]) < 1e-12
    # graph-restricted results are a subset of naive (or equal) when graph encodes proximity
    ids_naive = {i for i,_ in res_naive}
    ids_graph = {i for i,_ in res_graph}
    assert ids_graph.issubset(ids_naive)
