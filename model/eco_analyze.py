"""Statistics for the eco-speed ablation, at the same standard as analyze2.

Each configuration is run over several seeds because wall-clock budgets make a
single run non-reproducible. The statistical unit is the instance: seed medians
are taken first, then instances are compared pairwise against the optimized
regime with a Wilcoxon signed-rank test, Holm correction, a bootstrap interval
on the median paired difference, and a rank-biserial effect size.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results2")
REF = "opt_full"
ORDER = ["fixed_low", "fixed_med", "fixed_high", "opt_nofeas"]


def boot_ci_median(x, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = rng.choice(x, (n, len(x)), replace=True)
    return tuple(np.percentile(np.median(b, axis=1), [2.5, 97.5]))


def rank_biserial(diff):
    d = np.asarray(diff, float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    out = np.empty(m)
    run = 0.0
    for k, i in enumerate(idx):
        run = max(run, (m - k) * pvals[i])
        out[i] = min(run, 1.0)
    return out


def main():
    d = pd.read_csv(os.path.join(RES, "eco_ablation.csv"))
    seeds = d.seed.nunique()
    print(f"runs {len(d)} | families {d.family.nunique()} | "
          f"instances {d.groupby(['family','rep']).ngroups} | seeds {seeds}")

    # per-instance seed medians
    med = (d.groupby(["family", "rep", "policy"])["E_cost"]
             .median().unstack("policy"))
    spread = (d.groupby(["family", "rep", "policy"])["E_cost"]
                .agg(lambda x: 100 * (x.max() - x.min()) / x.mean()))
    print(f"\nsame-seed-set spread across seeds (% of mean): "
          f"median {spread.median():.1f}, max {spread.max():.1f}")

    rows, pv = [], []
    for pol in ORDER:
        diff = (med[pol] - med[REF]).dropna()
        rel = (100 * (med[pol] - med[REF]) / med[REF]).dropna()
        try:
            p = stats.wilcoxon(diff)[1]
        except ValueError:
            p = 1.0
        lo, hi = boot_ci_median(rel.values)
        rows.append(dict(policy=pol,
                         median_rel=float(np.median(rel)),
                         mean_rel=float(rel.mean()),
                         ci_lo=lo, ci_hi=hi,
                         wins=int((diff < 0).sum()),
                         losses=int((diff > 0).sum()),
                         n=int(len(diff)),
                         rb=rank_biserial(diff.values)))
        pv.append(p)
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(np.array(pv))
    res["significant"] = res.p_holm < 0.05

    print("\n=== paired vs optimized regime (negative = cheaper than optimized) ===")
    print(res.round(3).to_string(index=False))

    print("\n=== per-instance seed medians ===")
    cols = [c for c in ["fixed_low", "fixed_med", "fixed_high",
                        "opt_nofeas", "opt_full"] if c in med.columns]
    print(med[cols].round(1).to_string())

    ag = (d.groupby("policy")[["E_fuel", "E_emission", "E_late_min",
                               "P_defer", "arc_shortfall_share",
                               "pct_low", "pct_med", "pct_high"]]
            .median())
    print("\n=== median secondary metrics ===")
    print(ag.round(3).to_string())

    res.to_csv(os.path.join(RES, "eco_stats.csv"), index=False)
    med.to_csv(os.path.join(RES, "eco_medians.csv"))
    print("\nwrote eco_stats.csv, eco_medians.csv")


if __name__ == "__main__":
    main()
