"""
Scenario-aware Hybrid Guided Local Search (HGLS) for LA-SEco-ECVRPTW.

Optimises the scenario-aware recourse+CVaR fitness from ecvrptw.evaluate via adaptive
operator selection over structural, eco-speed, and risk-aware neighbourhoods, with a
GLS arc-penalty term for diversification. Search on penalised fitness; track best on
true fitness.
"""
import numpy as np
import copy
from . import ecvrptw as E


# ---------------- neighbourhood operators (each returns a new (routes,speeds)) -------
def _clone(routes, speeds):
    return [list(r) for r in routes], [list(s) for s in speeds]


def _fix_speeds(route, speeds, default=1):
    need = len(route) + 1
    if len(speeds) < need:
        speeds = speeds + [default] * (need - len(speeds))
    return speeds[:need]


def op_relocate(routes, speeds, rng):
    routes, speeds = _clone(routes, speeds)
    ne = [i for i, r in enumerate(routes) if r]
    if not ne: return routes, speeds
    a = rng.choice(ne)
    if not routes[a]: return routes, speeds
    pi = rng.integers(len(routes[a])); node = routes[a].pop(pi)
    speeds[a] = _fix_speeds(routes[a], speeds[a])
    b = rng.integers(len(routes) + 1)
    if b == len(routes):
        routes.append([node]); speeds.append([1, 1])
    else:
        pj = rng.integers(len(routes[b]) + 1); routes[b].insert(pj, node)
        speeds[b] = _fix_speeds(routes[b], speeds[b])
    return routes, speeds


def op_swap(routes, speeds, rng):
    routes, speeds = _clone(routes, speeds)
    ne = [i for i, r in enumerate(routes) if r]
    if len(ne) < 1: return routes, speeds
    a, b = rng.choice(ne), rng.choice(ne)
    if not routes[a] or not routes[b]: return routes, speeds
    i, j = rng.integers(len(routes[a])), rng.integers(len(routes[b]))
    routes[a][i], routes[b][j] = routes[b][j], routes[a][i]
    return routes, speeds


def op_2opt(routes, speeds, rng):
    routes, speeds = _clone(routes, speeds)
    ne = [i for i, r in enumerate(routes) if len(r) >= 4]
    if not ne: return routes, speeds
    a = rng.choice(ne); r = routes[a]
    i, j = sorted(rng.choice(len(r), 2, replace=False))
    r[i:j+1] = r[i:j+1][::-1]
    return routes, speeds


def op_oropt(routes, speeds, rng):
    routes, speeds = _clone(routes, speeds)
    ne = [i for i, r in enumerate(routes) if len(r) >= 2]
    if not ne: return routes, speeds
    a = rng.choice(ne); r = routes[a]
    L = rng.integers(1, min(3, len(r)) + 1)
    s = rng.integers(len(r) - L + 1); seg = r[s:s+L]; del r[s:s+L]
    speeds[a] = _fix_speeds(r, speeds[a])
    b = rng.integers(len(routes))
    pj = rng.integers(len(routes[b]) + 1)
    routes[b][pj:pj] = seg; speeds[b] = _fix_speeds(routes[b], speeds[b])
    return routes, speeds


def op_cross(routes, speeds, rng):
    routes, speeds = _clone(routes, speeds)
    ne = [i for i, r in enumerate(routes) if len(r) >= 2]
    if len(ne) < 2: return routes, speeds
    a, b = rng.choice(ne, 2, replace=False)
    ca, cb = rng.integers(1, len(routes[a])), rng.integers(1, len(routes[b]))
    ta, tb = routes[a][ca:], routes[b][cb:]
    routes[a] = routes[a][:ca] + tb; routes[b] = routes[b][:cb] + ta
    speeds[a] = _fix_speeds(routes[a], speeds[a]); speeds[b] = _fix_speeds(routes[b], speeds[b])
    return routes, speeds


def op_speedflip(routes, speeds, rng):
    routes, speeds = _clone(routes, speeds)
    ne = [i for i, r in enumerate(routes) if r]
    if not ne: return routes, speeds
    a = rng.choice(ne); k = rng.integers(len(speeds[a]))
    speeds[a][k] = int(rng.integers(3))
    return routes, speeds


def op_speedblock(routes, speeds, rng):
    """Lower speed on a block (buffer / eco-speed on slack arcs)."""
    routes, speeds = _clone(routes, speeds)
    ne = [i for i, r in enumerate(routes) if r]
    if not ne: return routes, speeds
    a = rng.choice(ne); n = len(speeds[a])
    i = rng.integers(n); j = min(n, i + int(rng.integers(1, 4)))
    lvl = int(rng.integers(2))            # low or med (slower => greener)
    for k in range(i, j): speeds[a][k] = lvl
    return routes, speeds


OPS = [op_relocate, op_swap, op_2opt, op_oropt, op_cross, op_speedflip, op_speedblock]
OP_NAMES = [o.__name__ for o in OPS]


# ---------------- GLS penalty on arcs ------------------------------------------------
def _arcs(routes, depot=0):
    out = []
    for r in routes:
        if not r: continue
        path = [depot] + list(r) + [depot]
        out += list(zip(path[:-1], path[1:]))
    return out


def _cap_ok(routes, cap_demand, cap):
    """True if every route's planning demand fits capacity (hard chance-constraint)."""
    for r in routes:
        if r and cap_demand[r].sum() > cap + 1e-9:
            return False
    return True


def hgls(inst: E.Instance, sc: E.Scenarios, routes, speeds, max_iter=2000, no_improve=300,
         beta=0.5, alpha=0.9, lam=0.1, seed=0, eval_kwargs=None, verbose=False,
         cap_demand=None):
    """cap_demand: optional [N] planning demand enforced as a hard per-route capacity
    constraint throughout the search (contextual chance constraint). If None, capacity
    is handled softly via recourse in the objective."""
    rng = np.random.default_rng(seed)
    eval_kwargs = eval_kwargs or {}
    penalty = np.zeros((inst.N, inst.N))
    enforce_cap = cap_demand is not None

    def true_fit(R, Sp):
        return E.evaluate(inst, sc, R, Sp, beta=beta, alpha=alpha, **eval_kwargs)["fitness"]

    def pen_fit(R, Sp):
        p = sum(penalty[i, j] for i, j in _arcs(R, inst.depot))
        return true_fit(R, Sp) + lam * p

    cur_r, cur_s = _clone(routes, speeds)
    cur_pf = pen_fit(cur_r, cur_s)
    best_r, best_s = _clone(cur_r, cur_s); best_tf = true_fit(best_r, best_s)
    scores = np.ones(len(OPS)); counts = np.ones(len(OPS))
    stale = 0

    for it in range(max_iter):
        w = scores / counts; w = w / w.sum()
        oi = rng.choice(len(OPS), p=w)
        nr, ns = OPS[oi](cur_r, cur_s, rng)
        if enforce_cap and not _cap_ok(nr, cap_demand, inst.cap):
            counts[oi] += 1; stale += 1            # reject capacity-infeasible move
            continue
        npf = pen_fit(nr, ns)
        gain = cur_pf - npf
        if gain > 1e-9:
            cur_r, cur_s, cur_pf = nr, ns, npf
            scores[oi] += gain; counts[oi] += 1
            tf = true_fit(cur_r, cur_s)
            if tf < best_tf - 1e-9:
                best_r, best_s = _clone(cur_r, cur_s); best_tf = tf; stale = 0
            else:
                stale += 1
        else:
            counts[oi] += 1; stale += 1
            # GLS: penalise the most "costly" arc in current solution
            arcs = _arcs(cur_r, inst.depot)
            if arcs:
                util = [(inst.D[i, j] / (1 + penalty[i, j]), (i, j)) for i, j in arcs]
                _, (pi, pj) = max(util)
                penalty[pi, pj] += 1
            cur_pf = pen_fit(cur_r, cur_s)     # refresh with new penalty
        if stale >= no_improve:
            # restart from best with penalty reset (diversify)
            cur_r, cur_s = _clone(best_r, best_s); penalty *= 0.5
            cur_pf = pen_fit(cur_r, cur_s); stale = 0
        if verbose and it % 500 == 0:
            print(f"  it{it} best_tf={best_tf:.2f}")
    return best_r, best_s, best_tf
