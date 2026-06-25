"""Constructive heuristics for LA-SEco-ECVRPTW (pickup, capacity + time windows)."""
import numpy as np


def nearest_feasible(inst, sc, default_speed=1, demand=None, tt=None):
    """
    Greedy nearest-feasible multi-route constructor.
    Uses mean demand and mean medium-speed travel time for feasibility checks.
    Returns (routes, speeds).
    """
    N = inst.N; depot = inst.depot
    demand = demand if demand is not None else sc.q.mean(0)     # [N] expected
    tt = tt if tt is not None else sc.tt.mean(0)                # [N,N,3] expected
    unserved = set(range(N)) - {depot}
    routes = []; speeds = []
    while unserved:
        route = []; load = 0.0; t = inst.tw[depot, 0]; cur = depot
        while True:
            best, best_d = None, np.inf
            for j in sorted(unserved):
                if load + demand[j] > inst.cap:
                    continue
                arr = t + tt[cur, j, default_speed]
                start = max(arr, inst.tw[j, 0])
                if start > inst.tw[j, 1] + 1e-6:        # would be late
                    continue
                if inst.D[cur, j] < best_d:
                    best_d, best = inst.D[cur, j], j
            if best is None:
                break
            route.append(best); load += demand[best]
            arr = t + tt[cur, best, default_speed]
            t = max(arr, inst.tw[best, 0]) + inst.service[best]
            cur = best; unserved.discard(best)
        if not route:                                   # capacity/TW too tight: force singletons
            j = sorted(unserved)[0]; route = [j]; unserved.discard(j)
        routes.append(route)
        speeds.append([default_speed] * (len(route) + 1))
    return routes, speeds
