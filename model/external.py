"""
External algorithmic baseline: Google OR-Tools routing (guided local search) builds
the routes on planning demand and nominal times; planned speeds are then refined by
a first-improvement speed descent under the SAME scenario evaluator used by every
other method. This provides a route-construction baseline that does not share any
code with HGLS.
"""
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from . import ecvrptw as E

SCALE_D = 100        # km -> int
SCALE_Q = 1000       # m3 -> int
SCALE_T = 10         # min -> int


def ortools_routes(inst, demand_plan, tt_nom, n_veh=None, time_limit_s=10,
                   speed_level=1, seed=0):
    """Solve a CVRPTW on planning demand with OR-Tools; return routes."""
    N = inst.N
    n_veh = n_veh or max(2, int(np.ceil(demand_plan.sum() / inst.cap * 1.5)) + 2)
    man = pywrapcp.RoutingIndexManager(N, n_veh, inst.depot)
    rt = pywrapcp.RoutingModel(man)

    dmat = (inst.D * SCALE_D).astype(int)
    di = rt.RegisterTransitMatrix(dmat.tolist())
    rt.SetArcCostEvaluatorOfAllVehicles(di)

    dem = (demand_plan * SCALE_Q).astype(int)
    qi = rt.RegisterUnaryTransitVector(dem.tolist())
    rt.AddDimensionWithVehicleCapacity(qi, 0, [int(inst.cap * SCALE_Q)] * n_veh,
                                       True, "cap")

    tmat = (tt_nom[:, :, speed_level] * SCALE_T).astype(int)
    srv = (inst.service * SCALE_T).astype(int)
    ti = rt.RegisterTransitMatrix((tmat + srv[:, None]).tolist())
    horizon = int(inst.tw[inst.depot, 1] * SCALE_T)
    rt.AddDimension(ti, horizon, horizon, False, "time")
    tdim = rt.GetDimensionOrDie("time")
    for node in range(N):
        if node == inst.depot:
            continue
        idx = man.NodeToIndex(node)
        tdim.CumulVar(idx).SetRange(int(inst.tw[node, 0] * SCALE_T),
                                    int(inst.tw[node, 1] * SCALE_T) + horizon)

    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    prm.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    prm.time_limit.FromSeconds(int(time_limit_s))
    sol = rt.SolveWithParameters(prm)
    if sol is None:
        return None
    routes = []
    for v in range(n_veh):
        idx = rt.Start(v); route = []
        while not rt.IsEnd(idx):
            node = man.IndexToNode(idx)
            if node != inst.depot:
                route.append(node)
            idx = sol.Value(rt.NextVar(idx))
        if route:
            routes.append(route)
    return routes


def speed_descent(inst, sc, routes, speeds, max_pass=3):
    """First-improvement descent over per-arc speed levels (same evaluator)."""
    best = E.evaluate(inst, sc, routes, speeds)["fitness"]
    for _ in range(max_pass):
        improved = False
        for r in range(len(routes)):
            for k in range(len(speeds[r])):
                orig = speeds[r][k]
                for lv in (0, 1, 2):
                    if lv == orig:
                        continue
                    speeds[r][k] = lv
                    f = E.evaluate(inst, sc, routes, speeds)["fitness"]
                    if f < best - 1e-9:
                        best = f; orig = lv; improved = True
                    else:
                        speeds[r][k] = orig
        if not improved:
            break
    return routes, speeds


def solve(inst, sc_tr, q_plan=None, time_limit_s=10, seed=0):
    """Full external baseline: OR-Tools routes + speed descent. Returns (routes, speeds)."""
    qp = q_plan if q_plan is not None else np.quantile(sc_tr.q, 0.75, axis=0)
    routes = ortools_routes(inst, qp, sc_tr.tt.mean(0), time_limit_s=time_limit_s,
                            seed=seed)
    if routes is None:
        return None, None
    speeds = [[1] * (len(r) + 1) for r in routes]
    return speed_descent(inst, sc_tr, routes, speeds)
