"""
Exact solution of small instances by depth-first branch and bound over route
sequences, at a fixed (medium) planned speed level. The leaf objective is the
SAME scenario evaluator used by HGLS, so there is no model mismatch: a zero gap
means the heuristic reproduces the optimum of exactly the objective it searches.

Symmetry breaking: the set of routes is unordered, so routes are enumerated with
strictly increasing first customers (each route's first customer exceeds the
previous route's first customer); within-route direction is NOT canonicalised
because load accumulation makes costs direction-dependent. Bounding: the partial
cost (closed routes plus the open route so far, simulated per scenario with the
same recourse rules) plus an admissible completion bound (each unserved customer
costs at least its cheapest inbound empty-load arc fuel) must stay below the
incumbent.
"""
import time
import numpy as np
from . import ecvrptw as E

SPEED = 1   # medium level throughout


def subinstance(inst, sc, nodes):
    """Restrict an instance and scenario set to depot + the given customers."""
    ids = [inst.depot] + list(nodes)
    D = inst.D[np.ix_(ids, ids)]
    sub = E.Instance(inst.name + f"_sub{len(nodes)}", D,
                     inst.demand_base[ids], inst.cap, inst.tw[ids],
                     inst.service[ids], inst.sp_kmh, depot=0)
    ssc = E.Scenarios(sc.q[:, ids], sc.tt[:, ids][:, :, ids], sc.feas[:, ids][:, :, ids])
    return sub, ssc


class _State:
    __slots__ = ("load", "t", "cost")


def exact_solve(inst, sc, time_limit=900.0, verbose=False):
    """Return (best_routes, best_cost, explored, optimal_flag)."""
    N, S = inst.N, sc.S
    depot = inst.depot
    cust = [i for i in range(N) if i != depot]
    shift_end = inst.tw[depot, 1]
    q = sc.q

    # admissible per-customer completion bound: cheapest inbound empty-load fuel
    lb_in = np.zeros(N)
    for j in cust:
        dmin = min(inst.D[i, j] for i in range(N) if i != j)
        lb_in[j] = dmin * E.RHO0 * E.PHI[SPEED]

    best = {"cost": np.inf, "routes": None}
    t0 = time.perf_counter()
    explored = [0]; timed_out = [False]

    def leaf(routes):
        m = E.evaluate(inst, sc, routes, [[SPEED] * (len(r) + 1) for r in routes])
        explored[0] += 1
        if m["E_cost"] < best["cost"] - 1e-12:
            best["cost"] = m["E_cost"]; best["routes"] = [list(r) for r in routes]

    def step(st, i, j):
        """Simulate travelling i->j and serving j from state st; returns new state."""
        ns = _State()
        lf = np.clip(st.load / inst.cap, 0, 1)
        cost = st.cost + E._arc_fuel(inst.D[i, j], lf, SPEED)
        t = st.t + sc.tt[:, i, j, SPEED]
        cost = cost + E.C_INFEAS_SPEED * (1 - sc.feas[:, i, j, SPEED])
        load = st.load
        trig = (load + q[:, j] > inst.cap + 1e-9)
        if trig.any():
            lf2 = np.clip(load / inst.cap, 0, 1)
            extra_f = (E._arc_fuel(inst.D[j, depot], lf2, SPEED)
                       + E._arc_fuel(inst.D[depot, j], 0.0, SPEED)) + E.C_RET
            extra_t = (sc.tt[:, j, depot, SPEED] + sc.tt[:, depot, j, SPEED]
                       + E.UNLOAD_MIN)
            cost = cost + np.where(trig, extra_f, 0.0)
            t = t + np.where(trig, extra_t, 0.0)
            load = np.where(trig, 0.0, load)
        start = np.maximum(t, inst.tw[j, 0])
        dfr = (start > shift_end + 1e-9)
        cost = cost + np.where(dfr, E.C_MISS,
                               E.C_LATE * np.maximum(0.0, start - inst.tw[j, 1]))
        ns.t = np.where(dfr, t, start + inst.service[j])
        ns.load = np.where(dfr, load, load + q[:, j])
        ns.cost = cost
        return ns

    def close(st, i):
        lf = np.clip(st.load / inst.cap, 0, 1)
        return st.cost + E._arc_fuel(inst.D[i, depot], lf, SPEED)

    def _fresh():
        st0 = _State(); st0.load = np.zeros(S)
        st0.t = np.full(S, float(inst.tw[depot, 0])); st0.cost = np.zeros(S)
        return st0

    def dfs(routes, open_route, st, cur, remaining, closed_cost, last_first):
        if time.perf_counter() - t0 > time_limit:
            timed_out[0] = True; return
        if not remaining:
            leaf(routes + [open_route] if open_route else routes)
            return
        # bound: accrued cost so far (no closing leg, which is not yet committed)
        # plus each unserved customer's cheapest inbound empty-load fuel
        part = closed_cost + (float(np.mean(st.cost)) if open_route else 0.0)
        if part + sum(lb_in[j] for j in remaining) >= best["cost"] - 1e-12:
            return
        if open_route:
            # extend the open route with any remaining customer
            for j in sorted(remaining):
                ns = step(st, cur, j)
                dfs(routes, open_route + [j], ns, j, remaining - {j},
                    closed_cost, last_first)
            # or close it and open a new route whose first customer keeps the
            # canonical increasing-first-customer order
            newc = closed_cost + float(np.mean(close(st, cur)))
            for f in sorted(remaining):
                if f <= last_first:
                    continue
                ns = step(_fresh(), depot, f)
                dfs(routes + [open_route], [f], ns, f, remaining - {f}, newc, f)
        else:
            for f in sorted(remaining):
                if f <= last_first:
                    continue
                ns = step(_fresh(), depot, f)
                dfs(routes, [f], ns, f, remaining - {f}, closed_cost, f)

    dfs([], [], None, depot, set(cust), 0.0, -1)
    return best["routes"], best["cost"], explored[0], not timed_out[0]
