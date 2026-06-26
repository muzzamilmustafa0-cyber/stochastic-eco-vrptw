"""
Additional result figures: ML prediction-layer validation, cost-risk trade-off,
recourse-weight sensitivity, and the learned planning behaviour. Retrains the small
prediction models for plotting and reads results/*.csv for the rest.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc
from . import predictors as P

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results"); FIG = os.path.join(RES, "figures")
PROC = os.path.join(HERE, "data_pipeline", "processed")
RAW = os.path.join(HERE, "data_pipeline", "raw")
os.makedirs(FIG, exist_ok=True)
PROP = "DF-CC*"


# ---------------------------------------------------------------- demand pipeline
def _austin_daily():
    a = pd.read_csv(os.path.join(RAW, "austin_waste_loads.csv"))
    a["load_weight"] = pd.to_numeric(a["load_weight"], errors="coerce")
    a = a.dropna(subset=["load_weight"]); a = a[a["load_weight"] > 0]
    wt = a["load_type"].str.upper().fillna("")
    a = a[wt.str.contains("GARBAGE|RECYCL|ORGANIC|COMPOST|TRASH")].copy()
    a["dt"] = pd.to_datetime(a["report_date"], errors="coerce")
    a = a.dropna(subset=["dt"])
    d = a.groupby(["route_number", a["dt"].dt.date]).agg(
        load=("load_weight", "sum"), lt=("load_type", "first")).reset_index()
    d.columns = ["route", "date", "load", "lt"]
    d["date"] = pd.to_datetime(d["date"])
    d["dow"] = d["date"].dt.dayofweek; d["month"] = d["date"].dt.month
    d["doy"] = d["date"].dt.dayofyear
    d["route_code"] = d["route"].astype("category").cat.codes
    d = d.sort_values(["route", "date"])
    d["hist"] = d.groupby("route")["load"].transform(lambda s: s.shift(1).expanding().mean())
    d = d.dropna(subset=["hist"])
    d = d[d["date"] < "2020-03-01"]                 # stationary pre-pandemic window
    vc = d["route"].value_counts(); d = d[d["route"].isin(vc[vc >= 60].index)]
    d["log_load"] = np.log1p(d["load"]); d["log_hist"] = np.log1p(d["hist"])
    return d


# -------------------------------------------------------- (1) ML validation panel
def fig_ml_validation():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    # (a) demand quantile calibration
    d = _austin_daily()
    feats = ["dow", "month", "doy", "route_code", "log_hist"]
    cut = d["date"].quantile(0.8)
    tr, te = d[d["date"] <= cut], d[d["date"] > cut]
    te = te[te["route_code"].isin(tr["route_code"].unique())]
    alphas = [0.1, 0.25, 0.5, 0.75, 0.9]; emp = []
    for q in alphas:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=400,
                              learning_rate=0.05, num_leaves=63, verbose=-1)
        m.fit(tr[feats], tr["log_load"])
        pred = np.expm1(m.predict(te[feats]))
        emp.append(float((te["load"].values <= pred).mean()))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax[0].plot(alphas, emp, "o-", color="#2980b9", label="model")
    ax[0].set_xlabel("nominal quantile"); ax[0].set_ylabel("empirical coverage")
    ax[0].set_title("(a) Demand quantile calibration (Austin)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # (b) travel-time predicted vs actual (held-out OD pairs)
    od = P._load_od_with_features()
    tfeats = ["dist_km", "hour", "plat", "plon", "dlat", "dlon"]
    pairs = od[["PULocationID", "DOLocationID"]].drop_duplicates().sample(frac=0.2, random_state=0)
    tk = pairs.set_index(["PULocationID", "DOLocationID"]).index
    is_te = od.set_index(["PULocationID", "DOLocationID"]).index.isin(tk)
    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=500,
                          learning_rate=0.05, num_leaves=63, verbose=-1)
    m.fit(od.loc[~is_te, tfeats], od.loc[~is_te, "p50"])
    yt = od.loc[is_te, "p50"].values; yp = m.predict(od.loc[is_te, tfeats])
    mae = np.mean(np.abs(yt - yp))
    hb = ax[1].hexbin(yt, yp, gridsize=40, cmap="Blues", mincnt=1)
    lim = [0, np.percentile(yt, 99)]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("actual median travel time (min)")
    ax[1].set_ylabel("predicted (min)")
    ax[1].set_title(f"(b) Travel-time model (MAE {mae:.2f} min)")

    # (c) eco-speed feasibility ROC per level
    od["ach_kmh"] = od["mean_mph"] * 1.60934
    levels = {"low": np.percentile(od["ach_kmh"], 30),
              "med": np.percentile(od["ach_kmh"], 60),
              "high": np.percentile(od["ach_kmh"], 85)}
    colors = {"low": "#27ae60", "med": "#e67e22", "high": "#c0392b"}
    for nm, thr in levels.items():
        y = (od["ach_kmh"] >= thr).astype(int)
        if y[~is_te].nunique() < 2:
            continue
        c = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1)
        c.fit(od.loc[~is_te, tfeats], y[~is_te])
        pr = c.predict_proba(od.loc[is_te, tfeats])[:, 1]
        fpr, tpr, _ = roc_curve(y[is_te], pr)
        ax[2].plot(fpr, tpr, color=colors[nm], label=f"{nm} (AUC {auc(fpr, tpr):.3f})")
    ax[2].plot([0, 1], [0, 1], "k--", lw=1)
    ax[2].set_xlabel("false positive rate"); ax[2].set_ylabel("true positive rate")
    ax[2].set_title("(c) Eco-speed feasibility ROC"); ax[2].legend(loc="lower right")
    ax[2].grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ml_validation.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------- (2) cost-risk trade-off
def fig_cost_risk():
    d = pd.read_csv(os.path.join(RES, "results.csv"))
    order = ["D-HGLS", "PTO-HGLS", "Q-HGLS", "RO-Eco", "S-HGLS", "LA-SHGLS",
             "DF-det", "DF-CC", "DF-CC*"]
    g = d.groupby("method")[["E_cost", "CVaR_cost", "P_overload"]].mean()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for axi, (yk, yl) in zip(ax, [("CVaR_cost", "CVaR of cost"), ("P_overload", "overload probability")]):
        for m in order:
            if m not in g.index:
                continue
            is_p = (m == PROP)
            axi.scatter(g.loc[m, "E_cost"], g.loc[m, yk],
                        s=140 if is_p else 80, c="#c0392b" if is_p else "#34495e",
                        marker="*" if is_p else "o", zorder=3 if is_p else 2,
                        edgecolors="k", linewidths=0.5)
            axi.annotate(m, (g.loc[m, "E_cost"], g.loc[m, yk]), fontsize=8,
                         xytext=(4, 4), textcoords="offset points")
        axi.set_xlabel("expected cost"); axi.set_ylabel(yl); axi.grid(alpha=0.3)
    ax[0].set_title("Cost vs tail risk (lower-left is better)")
    ax[1].set_title("Cost vs overload probability")
    fig.suptitle(f"Cost-risk trade-off across methods ({PROP} = star)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cost_risk.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------- (3) sensitivity figure
def fig_sensitivity():
    p = os.path.join(RES, "sensitivity.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    # vary C_OVER at base C_LATE=0.05
    sub = d[d["c_late"] == 0.05]
    g = sub.groupby(["c_over", "method"])["E_cost"].mean().unstack("method")
    base = sub.groupby(["c_over", "instance", "method"])["E_cost"].mean()
    for m, c in [("S-HGLS", "#34495e"), ("LA-SHGLS", "#2980b9"), ("DF-CC*", "#c0392b")]:
        # relative to per-(over,instance) so scale-free
        piv = sub.groupby(["c_over", "instance", "method"])["E_cost"].mean().unstack("method")
        rel = ((piv["S-HGLS"] - piv[m]) / piv["S-HGLS"] * 100).groupby("c_over").mean()
        ax[0].plot(rel.index, rel.values, "o-", color=c, label=m)
    ax[0].set_xlabel("overload weight C_over"); ax[0].set_ylabel("% cost reduction vs S-HGLS")
    ax[0].set_title("Sensitivity to overload weight"); ax[0].legend(); ax[0].grid(alpha=0.3)
    sub2 = d[d["c_over"] == 2.0]
    for m, c in [("S-HGLS", "#34495e"), ("LA-SHGLS", "#2980b9"), ("DF-CC*", "#c0392b")]:
        piv = sub2.groupby(["c_late", "instance", "method"])["E_cost"].mean().unstack("method")
        rel = ((piv["S-HGLS"] - piv[m]) / piv["S-HGLS"] * 100).groupby("c_late").mean()
        ax[1].plot(rel.index, rel.values, "o-", color=c, label=m)
    ax[1].set_xlabel("lateness weight C_late"); ax[1].set_ylabel("% cost reduction vs S-HGLS")
    ax[1].set_title("Sensitivity to lateness weight"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(f"{PROP} advantage is robust to recourse weights")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_sensitivity.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------- (4) learned planning
def fig_learned_planning():
    d = pd.read_csv(os.path.join(RES, "results.csv"))
    p = d[d["method"] == "DF-CC*"].dropna(subset=["tau_mean"])
    if p.empty:
        return
    g = p.groupby("instance")["tau_mean"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(range(len(g)), g.values, color="#8e44ad")
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index, rotation=45, ha="right")
    ax.set_ylabel("mean learned service level (tau)")
    ax.set_title("Learned per-node capacity service level by instance")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_learned_planning.png"), dpi=300)
    plt.close(fig)


def main():
    fig_ml_validation(); fig_cost_risk(); fig_sensitivity(); fig_learned_planning()
    print("wrote ml_validation, cost_risk, sensitivity, learned_planning figures")


if __name__ == "__main__":
    main()
