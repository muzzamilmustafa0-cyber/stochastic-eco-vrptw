"""
Global decision-focused service-level policy (roadmap section 13).

One linear-logistic policy theta maps standardised node features to a capacity
service level tau_i in [TAU_LO, TAU_HI]. The policy is trained ONCE across a set
of training (family, replicate) instances by minimising realised routing cost on
their calibration scenarios (decision regret), and is then deployed unchanged on
held-out families. Per-instance retraining is deliberately not used: that would
be hyperparameter tuning, not transferable learning.

Deployment pipeline for one instance:
  1. tau_i = policy(features);  q_plan_i = Quantile_train(demand_i, tau_i)
  2. one conformal pass on calibration scenarios inflates q_plan by kappa
  3. robust HGLS (DRO-blended fitness, hard capacity at q_plan)
  4. safeguard: the simpler recourse-only plan is also built; the learned plan is
     deployed only if it beats it on the calibration scenarios by at least
     SAFEGUARD_DELTA (relative), otherwise the recourse-only plan is used.
"""
import numpy as np
from . import ecvrptw as E, construct as C, hgls as H, robust as R

TAU_LO, TAU_HI = 0.50, 0.98
SAFEGUARD_DELTA = 0.0075          # predeclared 0.75 % validation-improvement threshold
N_FEAT = 7                        # 6 features + bias


def features(inst, sc_tr):
    qm = sc_tr.q.mean(0); qs = sc_tr.q.std(0)
    cv = qs / np.maximum(qm, 1e-9)
    X = np.stack([qm, cv, inst.D[inst.depot], inst.D.mean(1),
                  inst.tw[:, 1] - inst.tw[:, 0], inst.service], 1)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd


def tau_of(theta, X):
    z = X @ theta[:-1] + theta[-1]
    return TAU_LO + (TAU_HI - TAU_LO) / (1 + np.exp(-z))


def plan_demand(theta, inst, sc_tr, kappa=1.0):
    tau = tau_of(theta, features(inst, sc_tr))
    qp = np.array([np.quantile(sc_tr.q[:, i], tau[i]) for i in range(inst.N)])
    qp[inst.depot] = 0.0
    return qp * kappa, tau


def construct(inst, sc_tr, qp, ortools_s=0, seed=0):
    """Shared constructor: OR-Tools when a time budget is given, else greedy."""
    if ortools_s > 0:
        try:
            from . import external as XT
            r = XT.ortools_routes(inst, qp, sc_tr.tt.mean(0),
                                  time_limit_s=ortools_s, seed=seed)
            if r:
                return r, [[1] * (len(x) + 1) for x in r]
        except Exception:
            pass
    return C.nearest_feasible(inst, sc_tr, demand=qp, tt=sc_tr.tt.mean(0))


def _build_plan(inst, sc_tr, qp, budget_iter, seed, rho=None, lam=R.LAM_DEFAULT,
                time_limit=None, trace=None, ortools_s=0):
    r0, s0 = construct(inst, sc_tr, qp, ortools_s=ortools_s, seed=seed)
    fit = None
    if rho is not None:
        fit = R.robust_fitness_fn(inst, sc_tr, rho, lam)
    return H.hgls(inst, sc_tr, r0, s0, max_iter=budget_iter,
                  no_improve=max(budget_iter // 3, 100), beta=0.0, seed=seed,
                  cap_demand=qp, fitness_fn=fit, time_limit=time_limit, trace=trace)


def eval_policy(theta, train_insts, budget_iter=400, seed=0):
    """Decision loss of theta: mean relative realised cost on calibration scenarios
    across the training instances (relative to each instance's own scale)."""
    tot = 0.0
    for (inst, sc_tr, sc_ca) in train_insts:
        qp, _ = plan_demand(theta, inst, sc_tr)
        br, bs, _ = _build_plan(inst, sc_tr, qp, budget_iter, seed)
        m = E.evaluate(inst, sc_ca, br, bs)
        scale = max(float(np.mean(sc_ca.q.sum(1))), 1e-9)   # instance size proxy
        tot += m["E_cost"] / scale
    return tot / len(train_insts)


def train_global(train_insts, iters=30, pop=6, sigma0=0.7, seed=0,
                 budget_iter=300, verbose=True):
    """(mu+lambda) evolution strategy on the pooled decision loss."""
    rng = np.random.default_rng(seed)
    theta = np.zeros(N_FEAT)
    best = eval_policy(theta, train_insts, budget_iter, seed)
    sigma = sigma0; hist = [best]
    for t in range(iters):
        cand = theta + sigma * rng.standard_normal((pop, N_FEAT))
        vals = [eval_policy(c, train_insts, budget_iter, seed) for c in cand]
        j = int(np.argmin(vals))
        if vals[j] < best - 1e-12:
            theta, best = cand[j].copy(), vals[j]; sigma = min(sigma * 1.1, 1.0)
        else:
            sigma = max(sigma * 0.85, 0.05)
        hist.append(best)
        if verbose:
            print(f"  policy ES it{t+1}/{iters} loss={best:.5f} sigma={sigma:.2f}",
                  flush=True)
    return theta, hist


def deploy(inst, sc_tr, sc_ca, theta, seed=0, eps=0.20, lam=R.LAM_DEFAULT,
           time_limit=15.0, ortools_s=5, budget_iter=10**9):
    """Full deployment with conformal pass, DRO, and safeguarded selection.
    Both the learned plan and the recourse-only comparator get the same
    constructor and the same wall-clock refinement budget. Returns
    (routes, speeds, info)."""
    # reference plan for rho calibration and conformal scores (short budget)
    qp0, tau = plan_demand(theta, inst, sc_tr)
    r_ref, s_ref, _ = _build_plan(inst, sc_tr, qp0, budget_iter, seed,
                                  time_limit=max(time_limit / 3, 3.0),
                                  ortools_s=ortools_s)
    z_tr = E.evaluate(inst, sc_tr, r_ref, s_ref, return_z=True)["z"]
    z_ca = E.evaluate(inst, sc_ca, r_ref, s_ref, return_z=True)["z"]
    rho = R.calibrate_rho(z_tr, z_ca)
    kappa = R.conformal_kappa(inst, sc_ca, r_ref, eps=eps)
    # learned robust plan
    qp, _ = plan_demand(theta, inst, sc_tr, kappa=kappa)
    qp = np.minimum(qp, inst.cap)                     # a node cannot exceed a truck
    r_df, s_df, _ = _build_plan(inst, sc_tr, qp, budget_iter, seed, rho=rho,
                                lam=lam, time_limit=time_limit, ortools_s=ortools_s)
    # recourse-only comparator (same constructor, same budget)
    q75 = np.quantile(sc_tr.q, 0.75, axis=0)
    r_sa0, s_sa0 = construct(inst, sc_tr, q75, ortools_s=ortools_s, seed=seed)
    r_sa, s_sa, _ = H.hgls(inst, sc_tr, r_sa0, s_sa0, max_iter=budget_iter,
                           no_improve=10**9, beta=0.0, seed=seed,
                           time_limit=time_limit)
    # safeguarded selection on CALIBRATION scenarios
    c_df = E.evaluate(inst, sc_ca, r_df, s_df)["E_cost"]
    c_sa = E.evaluate(inst, sc_ca, r_sa, s_sa)["E_cost"]
    use_df = c_df < c_sa * (1 - SAFEGUARD_DELTA)
    routes, speeds = (r_df, s_df) if use_df else (r_sa, s_sa)
    info = {"rho": rho, "kappa": kappa, "tau_mean": float(tau[1:].mean()),
            "tau_std": float(tau[1:].std()), "selected": "DF" if use_df else "SAA",
            "calib_df": c_df, "calib_saa": c_sa}
    return routes, speeds, info
