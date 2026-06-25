"""
Decision-focused planning of per-node capacity service levels.

A linear model g_theta maps node features to a service level tau_i; the planning
demand q_plan_i = Quantile(demand_i, tau_i) is imposed as a per-route capacity
constraint inside the scenario-aware HGLS. theta is fit by a population evolution
strategy on held-out realised cost (the optimiser is not differentiable and the
uncertainty enters the constraints).

run_df modes:
  cc   learned capacity constraint + scenario HGLS
  det  learned planning + deterministic optimisation
run_proposed selects, per instance, among the learned constraint and the standard
planning regimes (S, LA, Q, RO) by training cost.
"""
import numpy as np
from . import ecvrptw as E, construct as C, hgls as H, experiments as X

TAU_LO, TAU_HI = 0.40, 0.995


def node_features(inst, sc_tr):
    qmean = sc_tr.q.mean(0); qstd = sc_tr.q.std(0)
    cv = qstd / np.maximum(qmean, 1e-6)
    d_dep = inst.D[inst.depot]; d_mean = inst.D.mean(1)
    tw_w = inst.tw[:, 1] - inst.tw[:, 0]
    X_ = np.stack([qmean, cv, d_dep, d_mean, tw_w, inst.service], axis=1)
    mu, sd = X_.mean(0), X_.std(0) + 1e-9
    return (X_ - mu) / sd


def tau_of(theta, Xn):
    z = Xn @ theta[:-1] + theta[-1]
    return TAU_LO + (TAU_HI - TAU_LO) / (1 + np.exp(-z))


def plan_demand(theta, Xn, sc_plan):
    tau = tau_of(theta, Xn)
    qp = np.array([np.quantile(sc_plan.q[:, i], tau[i]) for i in range(len(tau))])
    qp[0] = 0.0
    return qp, tau


def _pipeline_cost(theta, inst, Xn, sc_plan, sc_eval, budget, seed, beta=0.5,
                   mode="cc"):
    qp, _ = plan_demand(theta, Xn, sc_plan)
    tt_mean = sc_plan.tt.mean(0)
    r0, s0 = C.nearest_feasible(inst, sc_plan, demand=qp, tt=tt_mean)
    cap_demand = qp if mode == "cc" else None
    if mode == "det":
        det = X._det_scenarios(qp, tt_mean, sc_plan.feas.mean(0))
        br, bs, _ = H.hgls(inst, det, r0, s0, max_iter=budget, no_improve=budget//3,
                           beta=0.0, seed=seed)
    else:  # cc or la
        br, bs, _ = H.hgls(inst, sc_plan, r0, s0, max_iter=budget, no_improve=budget//3,
                           beta=beta, seed=seed, cap_demand=cap_demand)
    m = E.evaluate(inst, sc_eval, br, bs, beta=beta, alpha=0.9)
    return m["fitness"], (br, bs)


def train_df(inst, sc_tr, budget=500, iters=60, lam=6, seed=0, mode="cc", verbose=False):
    """Population evolution strategy on out-of-sample realised cost (decision regret)."""
    rng = np.random.default_rng(seed)
    Xn = node_features(inst, sc_tr)
    sc_plan, sc_val = X.split_scenarios(sc_tr, train_frac=0.6, seed=seed)
    F = Xn.shape[1] + 1
    theta = np.zeros(F)
    best_cost, _ = _pipeline_cost(theta, inst, Xn, sc_plan, sc_val, budget, seed, mode=mode, beta=0.0)
    best_theta = theta.copy()
    sigma = 0.7; stagn = 0
    for t in range(iters):
        cand = best_theta + sigma * rng.standard_normal((lam, F))
        costs = []
        for c in cand:
            cc, _ = _pipeline_cost(c, inst, Xn, sc_plan, sc_val, budget, seed, mode=mode, beta=0.0)
            costs.append(cc)
        j = int(np.argmin(costs))
        if costs[j] < best_cost - 1e-9:
            best_cost, best_theta = costs[j], cand[j].copy(); sigma = min(sigma*1.15, 1.0); stagn = 0
        else:
            sigma *= 0.85; stagn += 1
        if stagn >= 8:                          # restart around best with wider sigma
            sigma = 0.7; stagn = 0
        sigma = float(np.clip(sigma, 0.05, 1.0))
        if verbose and t % 10 == 0:
            print(f"  ES it{t} val={best_cost:.2f} sigma={sigma:.2f}")
    return best_theta, Xn


def run_proposed(inst, sc_tr, sc_te, budget=1500, train_budget=450, iters=55, seed=0,
                 name="DF-CC*"):
    """
    Select, per instance, among the learned capacity constraint and the standard
    planning regimes. All candidates are optimised on sc_tr; the one with lowest
    training expected cost is reported out-of-sample on sc_te. Objective is expected
    cost (fuel + recourse + emissions); CVaR and overload are reported separately.
    """
    theta, Xn = train_df(inst, sc_tr, budget=train_budget, iters=iters, seed=seed, mode="cc")
    qp, tau = plan_demand(theta, Xn, sc_tr)
    tt_mean = sc_tr.tt.mean(0)
    q75 = np.quantile(sc_tr.q, 0.75, axis=0)
    q90 = np.quantile(sc_tr.q, 0.90, axis=0)
    qmax = sc_tr.q.max(0)
    strategies = [
        ("CC", qp, 0.0, qp),          # learned capacity constraint
        ("S",  q75, 0.0, None),       # recourse-only
        ("LA", q75, 0.5, None),       # CVaR-weighted
        ("Q",  q90, 0.0, None),       # quantile safety stock
        ("RO", qmax, 0.0, None),      # robust
    ]
    cands = {}
    for nm, qd, bet, capd in strategies:
        ri, si = C.nearest_feasible(inst, sc_tr, demand=qd, tt=tt_mean)
        br, bs, _ = H.hgls(inst, sc_tr, ri, si, max_iter=budget, no_improve=budget // 3,
                           beta=bet, seed=seed, cap_demand=capd)
        cands[nm] = (br, bs)
    # select by lowest training expected cost
    vsel = {nm: E.evaluate(inst, sc_tr, br, bs, beta=0.0)["fitness"] for nm, (br, bs) in cands.items()}
    mode_pick = min(vsel, key=vsel.get)
    chosen = cands[mode_pick]
    m = E.evaluate(inst, sc_te, chosen[0], chosen[1], beta=0.5, alpha=0.9)
    m["n_vehicles"] = len([x for x in chosen[0] if x]); m["method"] = name
    m["tau_mean"] = float(tau[1:].mean()); m["tau_std"] = float(tau[1:].std())
    m["selected"] = mode_pick
    return m, chosen


def run_df(inst, sc_tr, sc_te, budget=1500, train_budget=500, iters=60, seed=0,
           mode="cc", name=None):
    name = name or {"cc": "DF-CC", "det": "DF-det", "la": "DF-LA"}[mode]
    theta, Xn = train_df(inst, sc_tr, budget=train_budget, iters=iters, seed=seed, mode=mode)
    qp, tau = plan_demand(theta, Xn, sc_tr)
    tt_mean = sc_tr.tt.mean(0)
    r0, s0 = C.nearest_feasible(inst, sc_tr, demand=qp, tt=tt_mean)
    if mode == "det":
        det = X._det_scenarios(qp, tt_mean, sc_tr.feas.mean(0))
        br, bs, _ = H.hgls(inst, det, r0, s0, max_iter=budget, no_improve=budget//3, beta=0.0, seed=seed)
    else:
        cap_demand = qp if mode == "cc" else None
        br, bs, _ = H.hgls(inst, sc_tr, r0, s0, max_iter=budget, no_improve=budget//3,
                           beta=0.0, seed=seed, cap_demand=cap_demand)
    m = E.evaluate(inst, sc_te, br, bs, beta=0.5, alpha=0.9)
    m["n_vehicles"] = len([x for x in br if x]); m["method"] = name
    m["tau_mean"] = float(tau[1:].mean()); m["tau_std"] = float(tau[1:].std())
    return m, (br, bs)
