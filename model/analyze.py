"""
Aggregate campaign results: per-method means, relative gaps, paired Wilcoxon
signed-rank tests (Holm-corrected), Friedman test, and a Markdown report.
Reads results/results.csv.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
PROPOSED = "DF-CC*"
METHOD_ORDER = ["D-HGLS", "PTO-HGLS", "Q-HGLS", "RO-Eco", "S-HGLS", "LA-SHGLS",
                "DF-det", "DF-CC", "DF-CC*"]
PRIMARY = "E_cost"


def load():
    return pd.read_csv(os.path.join(RES, "results.csv"))


def per_instance_means(df, metric):
    # average over seeds -> one value per (instance, method)
    return df.groupby(["instance", "method"])[metric].mean().unstack("method")


def summary(df):
    rows = []
    for m in METHOD_ORDER:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        rows.append({
            "method": m,
            "E_cost": sub["E_cost"].mean(), "E_fuel": sub["E_fuel"].mean(),
            "E_emission": sub["E_emission"].mean(), "CVaR_cost": sub["CVaR_cost"].mean(),
            "E_recourse": sub["E_recourse"].mean(), "P_overload": sub["P_overload"].mean(),
            "P_late": sub["P_late"].mean(), "n_vehicles": sub["n_vehicles"].mean(),
            "secs": sub["secs"].mean(),
        })
    return pd.DataFrame(rows).round(3)


def pairwise_vs_proposed(df, metric=PRIMARY):
    piv = per_instance_means(df, metric)
    prop = piv[PROPOSED]
    out = []; pvals = []; names = []
    for m in METHOD_ORDER:
        if m == PROPOSED or m not in piv:
            continue
        a, b = piv[m].dropna(), prop.loc[piv[m].dropna().index]
        common = piv[[m, PROPOSED]].dropna()
        x, y = common[m].values, common[PROPOSED].values
        relgap = float(np.median((x - y) / y * 100))         # baseline vs proposed
        wins = int((y < x).sum()); ties = int((y == x).sum()); losses = int((y > x).sum())
        try:
            stat, p = stats.wilcoxon(x, y, alternative="greater")  # baseline > proposed?
        except ValueError:
            stat, p = np.nan, 1.0
        out.append({"baseline": m, "n": len(common), "median_relgap_%": round(relgap, 2),
                    "proposed_wins": wins, "ties": ties, "losses": losses,
                    "wilcoxon_p": p})
        pvals.append(p); names.append(m)
    # Holm correction
    order = np.argsort(pvals); adj = np.empty(len(pvals)); k = len(pvals)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (k - rank) * pvals[idx]); val = max(val, prev); adj[idx] = val; prev = val
    res = pd.DataFrame(out)
    res["holm_p"] = adj
    return res.round(5)


def friedman(df, metric=PRIMARY):
    piv = per_instance_means(df, metric).dropna()
    cols = [m for m in METHOD_ORDER if m in piv.columns]
    arrs = [piv[c].values for c in cols]
    stat, p = stats.friedmanchisquare(*arrs)
    ranks = piv[cols].rank(axis=1).mean().sort_values()
    return {"friedman_chi2": float(stat), "p": float(p), "df": len(cols)-1,
            "mean_ranks": {k: round(float(v), 3) for k, v in ranks.items()}}


def main():
    df = load()
    s = summary(df); s.to_csv(os.path.join(RES, "summary_by_method.csv"), index=False)
    rep = {}
    for metric in ["E_cost", "CVaR_cost", "P_overload"]:
        pw = pairwise_vs_proposed(df, metric)
        pw.to_csv(os.path.join(RES, f"pairwise_{metric}.csv"), index=False)
        rep[metric] = {"friedman": friedman(df, metric),
                       "pairwise_vs_proposed": pw.to_dict("records")}
    json.dump(rep, open(os.path.join(RES, "stats.json"), "w"), indent=2)

    with open(os.path.join(RES, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# Experimental Results\n\n")
        f.write(f"Instances: {df['instance'].nunique()} | seeds: {sorted(df['seed'].unique())} "
                f"| methods: {df['method'].nunique()} | proposed: **{PROPOSED}**\n\n")
        f.write("## Mean performance by method (lower is better)\n\n")
        f.write(s.to_markdown(index=False) + "\n\n")
        for metric in ["E_cost", "CVaR_cost", "P_overload"]:
            f.write(f"## {metric}: paired tests vs {PROPOSED}\n\n")
            f.write(pd.read_csv(os.path.join(RES, f"pairwise_{metric}.csv")).to_markdown(index=False) + "\n\n")
            fr = friedman(df, metric)
            f.write(f"Friedman chi2={fr['friedman_chi2']:.2f} (df={fr['df']}), p={fr['p']:.2e}; "
                    f"mean ranks: {fr['mean_ranks']}\n\n")
    print(s.to_string(index=False))
    print("\nWrote summary_by_method.csv, pairwise_*.csv, stats.json, REPORT.md")


if __name__ == "__main__":
    main()
