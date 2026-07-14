"""
Redesigned experiment harness (equal wall-clock, shared constructor, nested data).

Every regime plans on the training scenarios, may use the calibration scenarios
for conformal / DRO / safeguard decisions, and is scored once on the locked test
scenarios. All regimes share the same constructor (OR-Tools on the regime's
planning demand, greedy fallback) and the same wall-clock refinement budget, so
differences reflect the planning model, not solver tuning.

Regimes
  DET   deterministic mean-value plan, refined against the mean view
  PTO   median point-forecast plan, refined against the median view
  SAA   scenario plan with simulated recourse (expected cost)
  CVAR  scenario plan, expected cost + 0.5 * CVaR(0.9)
  Q90C  fixed 0.90 quantile protection + conformal multiplier, hard capacity
  RO    conservative plan protected at the scenario maximum, hard capacity
  DFR   proposed: global learned service levels + conformal + DRO + safeguard
  ORT   external OR-Tools solver + speed descent (no HGLS refinement)
"""
import os, json
import numpy as np
from . import ecvrptw as E, construct as C, hgls as H, robust as R, policy as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORT_S = 5            # constructor budget (seconds)
REFINE_S = 15.0      # HGLS refinement budget (seconds, wall clock)
BIG = 10**9

METHODS = ["DET", "PTO", "SAA", "CVAR", "Q90C", "RO", "DFR", "ORT"]


def _mean_view(sc, stat="mean"):
    """Single-scenario deterministic view of a scenario set."""
    if stat == "mean":
        q = sc.q.mean(0, keepdims=True); tt = sc.tt.mean(0, keepdims=True)
    else:
        q = np.quantile(sc.q, 0.5, 0, keepdims=True)
        tt = np.quantile(sc.tt, 0.5, 0, keepdims=True)
    feas = (sc.feas.mean(0, keepdims=True) >= 0.5).astype(np.int8)
    return E.Scenarios(q, tt, feas)


def _refine(inst, sc_obj, r0, s0, seed, cap=None, fitness_fn=None, beta=0.0,
            trace=None, budget=REFINE_S):
    return H.hgls(inst, sc_obj, r0, s0, max_iter=BIG, no_improve=BIG, beta=beta,
                  seed=seed, cap_demand=cap, fitness_fn=fitness_fn,
                  time_limit=budget, trace=trace)


def run_regime(method, inst, sc_tr, sc_ca, seed=0, theta=None, trace=None,
               budget=REFINE_S):
    """Build the plan for one regime. Returns (routes, speeds, info)."""
    info = {}
    if method == "DET":
        qp = sc_tr.q.mean(0)
        r0, s0 = P.construct(inst, sc_tr, qp, ortools_s=ORT_S, seed=seed)
        view = _mean_view(sc_tr, "mean")
        r, s, _ = _refine(inst, view, r0, s0, seed, trace=trace, budget=budget)
    elif method == "PTO":
        qp = np.quantile(sc_tr.q, 0.5, axis=0)
        r0, s0 = P.construct(inst, sc_tr, qp, ortools_s=ORT_S, seed=seed)
        view = _mean_view(sc_tr, "median")
        r, s, _ = _refine(inst, view, r0, s0, seed, trace=trace, budget=budget)
    elif method == "SAA":
        qp = np.quantile(sc_tr.q, 0.75, axis=0)
        r0, s0 = P.construct(inst, sc_tr, qp, ortools_s=ORT_S, seed=seed)
        r, s, _ = _refine(inst, sc_tr, r0, s0, seed, trace=trace, budget=budget)
    elif method == "CVAR":
        qp = np.quantile(sc_tr.q, 0.75, axis=0)
        r0, s0 = P.construct(inst, sc_tr, qp, ortools_s=ORT_S, seed=seed)
        r, s, _ = _refine(inst, sc_tr, r0, s0, seed, beta=0.5, trace=trace,
                          budget=budget)
    elif method == "Q90C":
        qp0 = np.quantile(sc_tr.q, 0.90, axis=0)
        r0, s0 = P.construct(inst, sc_tr, qp0, ortools_s=ORT_S, seed=seed)
        r_ref, s_ref, _ = _refine(inst, sc_tr, r0, s0, seed, budget=budget / 3)
        kap = R.conformal_kappa(inst, sc_ca, r_ref, eps=0.20)
        qp = np.minimum(qp0 * kap, inst.cap)
        info["kappa"] = kap
        r0, s0 = P.construct(inst, sc_tr, qp, ortools_s=ORT_S, seed=seed)
        r, s, _ = _refine(inst, sc_tr, r0, s0, seed, cap=qp, trace=trace,
                          budget=budget)
    elif method == "RO":
        qp = np.minimum(sc_tr.q.max(0), inst.cap)
        r0, s0 = P.construct(inst, sc_tr, qp, ortools_s=ORT_S, seed=seed)
        r, s, _ = _refine(inst, sc_tr, r0, s0, seed, cap=qp, trace=trace,
                          budget=budget)
    elif method == "DFR":
        assert theta is not None, "DFR requires a trained policy"
        r, s, info = P.deploy(inst, sc_tr, sc_ca, theta, seed=seed,
                              time_limit=budget, ortools_s=ORT_S)
    elif method == "ORT":
        from . import external as XT
        r, s = XT.solve(inst, sc_tr, time_limit_s=int(ORT_S + budget), seed=seed)
        if r is None:
            r0, s0 = C.nearest_feasible(inst, sc_tr)
            r, s = r0, s0
    else:
        raise ValueError(method)
    return r, s, info


def score(inst, sc_te, routes, speeds):
    m = E.evaluate(inst, sc_te, routes, speeds, beta=0.0)
    m["n_veh"] = len([x for x in routes if x])
    return m
