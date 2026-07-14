"""
Robust evaluation layer: discrete distributionally robust reweighting over
scenarios, conformal calibration of capacity protection, and a finite-sample
reliability screen.

DRO. The ambiguity set is a chi-square divergence ball of radius rho around the
empirical scenario distribution. For scenario costs z the worst-case expectation
has the closed form  sup_Q E_Q[z] = mean(z) + sqrt(rho * Var(z)).  The search
objective blends nominal and worst case:

    F(x) = (1 - lam) * mean(z) + lam * [ mean(z) + sqrt(rho * Var(z)) ]
         = mean(z) + lam * sqrt(rho * Var(z))

The radius has a finite-sample component chi2(1, conf)/S and a shift component
measured on the calibration scenarios for a reference plan, so instances whose
calibration data disagree more with the training scenarios receive a larger ball.

Conformal capacity protection. Given planned routes, the nonconformity score of a
route under a calibration scenario is realised route demand divided by planned
route demand. The (1 - eps) empirical quantile of the pooled scores gives a
protection multiplier kappa; planning demands are inflated by kappa and the plan
is rebuilt once. This calibrates the planning quantile to out-of-sample route
reliability instead of trusting the learned marginal quantiles.

Reliability screen. A candidate plan satisfies the route-failure chance
constraint at level eps only if the Wilson upper confidence bound of its observed
trigger rate over the training scenarios is at most eps plus a small allowance.
"""
import numpy as np
from scipy import stats
from . import ecvrptw as E

CHI2_CONF = 0.90
LAM_DEFAULT = 0.25          # blend weight, validated range 0.15-0.35 (roadmap)


def dro_stats(z, rho, lam=LAM_DEFAULT):
    m, v = float(np.mean(z)), float(np.var(z))
    worst = m + np.sqrt(max(rho, 0.0) * v)
    return m + lam * (worst - m), worst


def calibrate_rho(z_train, z_calib):
    """Finite-sample radius + observed train->calibration shift for a reference plan."""
    rho_fs = stats.chi2.ppf(CHI2_CONF, df=1) / max(len(z_train), 1)
    sd = float(np.std(z_train)) + 1e-9
    shift = max(0.0, (float(np.mean(z_calib)) - float(np.mean(z_train))) / sd)
    return float(rho_fs + shift**2)


def conformal_kappa(inst, sc_calib, routes, eps=0.20):
    """Protection multiplier from pooled route-level nonconformity scores."""
    scores = []
    for route in routes:
        if not route:
            continue
        planned = max(float(np.quantile(sc_calib.q[:, route].sum(1), 0.5)), 1e-9)
        realised = sc_calib.q[:, route].sum(1)
        scores.append(realised / planned)
    if not scores:
        return 1.0
    pooled = np.concatenate(scores)
    return float(np.quantile(pooled, 1.0 - eps))


def wilson_upper(k, n, conf=0.90):
    """Wilson score upper bound for a binomial proportion."""
    if n == 0:
        return 1.0
    z = stats.norm.ppf(conf)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    rad = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float((centre + rad) / denom)


def reliability_ok(trigger_count, n_scen, eps, allowance=0.05, conf=0.90):
    return wilson_upper(trigger_count, n_scen, conf) <= eps + allowance


def robust_fitness_fn(inst, sc_train, rho, lam=LAM_DEFAULT, beta=0.0,
                      cap_demand=None):
    """Fitness closure for HGLS: nominal + DRO blend on scenario costs."""
    def fit(routes, speeds):
        m = E.evaluate(inst, sc_train, routes, speeds, beta=beta, return_z=True)
        f, _ = dro_stats(m["z"], rho, lam)
        if beta > 0:
            f += beta * m["CVaR_cost"]
        return f
    return fit
