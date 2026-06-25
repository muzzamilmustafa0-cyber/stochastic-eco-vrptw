"""
Baseline methods and the out-of-sample protocol.

Scenarios are split into planning (train) and evaluation (test) sets. Every method
plans on train and is scored by the same scenario-aware evaluator on test.

  D-HGLS    deterministic mean-value plan
  PTO-HGLS  point (median) forecast plan
  Q-HGLS    quantile plan (q90 safety stock)
  RO-Eco    conservative plan (max demand)
  S-HGLS    scenario plan, expected cost + recourse
  LA-SHGLS  scenario plan with CVaR weighting
"""
import numpy as np
import copy
from . import ecvrptw as E, construct as C, hgls as H


def split_scenarios(sc, train_frac=0.67, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(sc.S); cut = int(sc.S*train_frac)
    tr, te = idx[:cut], idx[cut:]
    mk = lambda I: E.Scenarios(sc.q[I], sc.tt[I], sc.feas[I])
    return mk(tr), mk(te)


def _det_scenarios(q_plan, tt_mean, feas_mean):
    """Single-scenario set representing a deterministic planning view."""
    return E.Scenarios(q_plan[None, :], tt_mean[None, :, :, :],
                       (feas_mean >= 0.5).astype(np.int8)[None, :, :, :])


def _quantile_demand(sc_tr, qlevel):
    return np.quantile(sc_tr.q, qlevel, axis=0)


def run_method(name, inst, sc_tr, sc_te, budget=1500, seed=0):
    tt_mean = sc_tr.tt.mean(0); feas_mean = sc_tr.feas.mean(0)
    beta = 0.0; eval_kw = {}; plan_sc = sc_tr

    if name in ("D-HGLS", "PTO-HGLS"):
        qp = sc_tr.q.mean(0)                       # point/mean plan
        plan_sc = _det_scenarios(qp, tt_mean, feas_mean)
    elif name == "Q-HGLS":
        qp = _quantile_demand(sc_tr, 0.90)         # safety stock
        plan_sc = _det_scenarios(qp, tt_mean, feas_mean)
    elif name == "RO-Eco":
        qp = sc_tr.q.max(0)                         # worst-case demand
        plan_sc = _det_scenarios(qp, tt_mean, feas_mean)
    elif name == "S-HGLS":
        beta = 0.0                                  # expected cost + recourse, no CVaR
    elif name == "LA-SHGLS":
        beta = 0.5                                  # + CVaR tail control
    else:
        raise ValueError(name)

    # initial solution from a conservative (q75) plan for feasibility headroom
    q_init = _quantile_demand(sc_tr, 0.75)
    r0, s0 = C.nearest_feasible(inst, sc_tr, demand=q_init, tt=tt_mean)
    br, bs, _ = H.hgls(inst, plan_sc, r0, s0, max_iter=budget, no_improve=budget//3,
                       beta=beta, seed=seed)
    # OUT-OF-SAMPLE evaluation on test scenarios (always full recourse + CVaR reported)
    m = E.evaluate(inst, sc_te, br, bs, beta=0.5, alpha=0.9)
    m["n_vehicles"] = len([x for x in br if x])
    m["method"] = name
    return m, (br, bs)


BASELINES = ["D-HGLS", "PTO-HGLS", "Q-HGLS", "RO-Eco", "S-HGLS", "LA-SHGLS"]


def run_instance(name, budget=1500, seed=0, methods=None):
    inst, sc = E.load(name)
    sc_tr, sc_te = split_scenarios(sc, seed=seed)
    rows = []
    for mth in (methods or BASELINES):
        m, _ = run_method(mth, inst, sc_tr, sc_te, budget=budget, seed=seed)
        rows.append(m)
    return inst, rows
