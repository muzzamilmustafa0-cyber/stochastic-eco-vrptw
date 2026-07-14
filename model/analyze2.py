"""
Statistical analysis of the final campaign (roadmap section 24).

Unit of analysis: the instance (family x replicate), 55 paired observations.
For each comparison against the proposed regime we report the mean and median
paired percentage difference, a 95 percent bootstrap confidence interval on the
median, the Wilcoxon signed-rank p-value with Holm correction over the
predeclared family of comparisons, the matched-pairs rank-biserial correlation
and Cliff's delta as effect sizes, win/tie/loss counts at a 0.3 percent
practical-significance threshold, and the Friedman test across regimes.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results2")
PROPOSED = "DFR"
PRACTICAL = 0.003          # 0.3 % relative difference = practical tie band


def _instance_table(df, metric="E_cost"):
    """(family, rep) x method matrix, averaging any repeated seeds."""
    return (df.groupby(["family", "rep", "method"])[metric].mean()
            .unstack("method"))


def rank_biserial(diff):
    """Matched-pairs rank-biserial correlation from signed ranks."""
    d = diff[np.abs(diff) > 1e-12]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum(); r_neg = ranks[d < 0].sum()
    return float((r_pos - r_neg) / (r_pos + r_neg))


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a); lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def boot_ci_median(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def pairwise(df, metric="E_cost", baselines=None):
    piv = _instance_table(df, metric).dropna()
    baselines = baselines or [m for m in piv.columns if m != PROPOSED]
    rows, pvals = [], []
    for m in baselines:
        rel = (piv[m] - piv[PROPOSED]) / piv[m] * 100     # + = proposed better
        wins = int((rel > PRACTICAL * 100).sum())
        loss = int((rel < -PRACTICAL * 100).sum())
        ties = len(rel) - wins - loss
        try:
            _, p = stats.wilcoxon(piv[m], piv[PROPOSED])
        except ValueError:
            p = 1.0
        lo, hi = boot_ci_median(rel.values)
        rows.append(dict(baseline=m, n=len(rel),
                         mean_gain_pct=round(float(rel.mean()), 2),
                         median_gain_pct=round(float(rel.median()), 2),
                         ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                         wins=wins, ties=ties, losses=loss,
                         rank_biserial=round(rank_biserial(rel.values), 3),
                         cliffs_delta=round(cliffs_delta(
                             piv[m].values, piv[PROPOSED].values), 3),
                         wilcoxon_p=p))
        pvals.append(p)
    # Holm correction
    order = np.argsort(pvals); k = len(pvals); adj = [0.0] * k; prev = 0.0
    for rank, idx in enumerate(order):
        v = min(1.0, (k - rank) * pvals[idx]); v = max(v, prev)
        adj[idx] = v; prev = v
    for r, a in zip(rows, adj):
        r["holm_p"] = round(a, 5); r["wilcoxon_p"] = round(r["wilcoxon_p"], 6)
    return pd.DataFrame(rows)


def friedman(df, metric="E_cost"):
    piv = _instance_table(df, metric).dropna()
    arrs = [piv[c].values for c in piv.columns]
    st, p = stats.friedmanchisquare(*arrs)
    ranks = piv.rank(axis=1).mean().sort_values()
    return {"chi2": round(float(st), 2), "p": float(p),
            "mean_ranks": {k: round(float(v), 2) for k, v in ranks.items()}}


def summarize(df):
    keys = ["E_cost", "E_fuel", "E_emission", "CVaR_cost", "P_trigger",
            "E_returns", "P_late", "P_defer", "n_veh", "secs"]
    return df.groupby("method")[keys].mean().round(3)


def shift_analysis():
    """Normalized regret degradation by shift kind and magnitude."""
    sh = pd.read_csv(os.path.join(RES, "shifts.csv"))
    base = pd.read_csv(os.path.join(RES, "main.csv"))
    base = base[base.method.isin(sh.method.unique())]
    b = base.set_index(["family", "rep", "method"])["E_cost"]
    sh["base"] = sh.apply(lambda r: b.get((r.family, r.rep, r.method), np.nan), 1)
    sh["degr_pct"] = (sh["E_cost"] - sh["base"]) / sh["base"] * 100
    return (sh.groupby(["kind", "mag", "method"])["degr_pct"].mean()
            .unstack("method").round(2))


def main():
    df = pd.read_csv(os.path.join(RES, "main.csv"))
    out = {}
    print("=== summary (means over 55 instances) ===")
    print(summarize(df).to_string())
    print("\n=== paired vs proposed (E_cost) ===")
    pw = pairwise(df, "E_cost")
    print(pw.to_string(index=False))
    pw.to_csv(os.path.join(RES, "pairwise_Ecost.csv"), index=False)
    for metric in ["CVaR_cost", "P_trigger"]:
        pairwise(df, metric).to_csv(
            os.path.join(RES, f"pairwise_{metric}.csv"), index=False)
    fr = friedman(df)
    print("\nFriedman:", fr)
    out["friedman_Ecost"] = fr
    if os.path.exists(os.path.join(RES, "shifts.csv")):
        sa = shift_analysis()
        sa.to_csv(os.path.join(RES, "shift_degradation.csv"))
        print("\n=== shift degradation (mean % vs own nominal) ===")
        print(sa.to_string())
    json.dump(out, open(os.path.join(RES, "stats2.json"), "w"), indent=2)
    summarize(df).to_csv(os.path.join(RES, "summary_by_method.csv"))


if __name__ == "__main__":
    main()
