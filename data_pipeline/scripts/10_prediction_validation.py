"""
10 - Prediction-layer validation per the readiness roadmap (probabilistic metrics,
blocked splits, calibration before and after conformal adjustment).

Demand (Austin, blocked by date): pinball loss per quantile, 80 percent interval
coverage and width before/after conformal widening, CRPS approximated from a
quantile ensemble, reliability curve data by context (weekday/weekend).

Travel time (NYC TLC, blocked by OD pair): pinball, interval coverage/width,
CRPS approximation.

Achievable-speed feasibility: probabilities derived from a quantile model of the
achievable speed (not a classifier on a thresholded label); evaluated by Brier
score and reliability curve on held-out OD pairs. Discrimination (AUC) is
reported only as a secondary diagnostic.

Output: processed/prediction_validation.json (+ reliability curve arrays)
"""
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw"); PROC = os.path.join(HERE, "processed")
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def pinball(y, p, q):
    e = y - p
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def crps_from_quantiles(y, preds):
    """CRPS approximated by averaging pinball losses over the quantile grid."""
    return float(2 * np.mean([pinball(y, preds[q], q) for q in QS]))


def reliability(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


# ------------------------------------------------------------------ demand ----
def demand_block():
    a = pd.read_csv(os.path.join(RAW, "austin_waste_loads.csv"))
    a["load_weight"] = pd.to_numeric(a["load_weight"], errors="coerce")
    a = a.dropna(subset=["load_weight"]); a = a[a["load_weight"] > 0]
    wt = a["load_type"].str.upper().fillna("")
    a = a[wt.str.contains("GARBAGE|RECYCL|ORGANIC|COMPOST|TRASH")].copy()
    a["dt"] = pd.to_datetime(a["report_date"], errors="coerce")
    a = a.dropna(subset=["dt"])
    d = a.groupby(["route_number", a["dt"].dt.date]).agg(
        load=("load_weight", "sum")).reset_index()
    d.columns = ["route", "date", "load"]
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] < "2020-03-01"]
    vc = d["route"].value_counts(); d = d[d["route"].isin(vc[vc >= 60].index)]
    d["dow"] = d["date"].dt.dayofweek; d["month"] = d["date"].dt.month
    d["doy"] = d["date"].dt.dayofyear
    d["rc"] = d["route"].astype("category").cat.codes
    d = d.sort_values(["route", "date"])
    d["hist"] = d.groupby("route")["load"].transform(
        lambda s: s.shift(1).expanding().mean())
    d = d.dropna(subset=["hist"])
    d["lh"] = np.log1p(d["hist"]); d["y"] = np.log1p(d["load"])
    feats = ["dow", "month", "doy", "rc", "lh"]
    # blocked chronological split: fit / calibration / test
    t1, t2 = d["date"].quantile(0.6), d["date"].quantile(0.8)
    fit, cal, te = d[d["date"] <= t1], d[(d["date"] > t1) & (d["date"] <= t2)], \
        d[d["date"] > t2]
    te = te[te["rc"].isin(fit["rc"].unique())]
    models, pr_cal, pr_te = {}, {}, {}
    for q in QS:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=400,
                              learning_rate=0.05, num_leaves=63, verbose=-1)
        m.fit(fit[feats], fit["y"])
        pr_cal[q] = np.expm1(m.predict(cal[feats]))
        pr_te[q] = np.expm1(m.predict(te[feats]))
    y_cal, y_te = cal["load"].values, te["load"].values
    # conformal widening of the [.10, .90] interval on the calibration block
    sc = np.maximum(pr_cal[0.10] - y_cal, y_cal - pr_cal[0.90])
    d_conf = float(np.quantile(sc, 0.80))
    out = {
        "n_fit": len(fit), "n_cal": len(cal), "n_test": len(te),
        "pinball": {str(q): round(pinball(y_te, pr_te[q], q), 1) for q in QS},
        "crps": round(crps_from_quantiles(y_te, pr_te), 1),
        "cover80_raw": round(reliability(y_te, pr_te[0.10], pr_te[0.90]), 3),
        "cover80_conformal": round(reliability(y_te, pr_te[0.10] - d_conf,
                                               pr_te[0.90] + d_conf), 3),
        "width80_raw": round(float(np.mean(pr_te[0.90] - pr_te[0.10])), 1),
        "width80_conformal": round(float(np.mean(pr_te[0.90] - pr_te[0.10]
                                                 + 2 * d_conf)), 1),
        "wape_p50": round(float(np.sum(np.abs(y_te - pr_te[0.50]))
                                / np.sum(y_te)), 3),
    }
    # reliability by context (weekday vs weekend)
    wk = te["dow"] < 5
    out["cover80_conf_weekday"] = round(reliability(
        y_te[wk], pr_te[0.10][wk] - d_conf, pr_te[0.90][wk] + d_conf), 3)
    out["cover80_conf_weekend"] = round(reliability(
        y_te[~wk], pr_te[0.10][~wk] - d_conf, pr_te[0.90][~wk] + d_conf), 3)
    # quantile reliability curve (nominal vs empirical, conformal-adjusted median band)
    out["quantile_reliability"] = {
        str(q): round(float(np.mean(y_te <= pr_te[q])), 3) for q in QS}
    return out


# --------------------------------------------------------------- travel time --
def travel_block():
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from model import predictors as P
    od = P._load_od_with_features()
    feats = ["dist_km", "hour", "plat", "plon", "dlat", "dlon"]
    rng = np.random.default_rng(0)
    pairs = od[["PULocationID", "DOLocationID"]].drop_duplicates()
    te_pairs = pairs.sample(frac=0.2, random_state=0)
    cal_pairs = pairs.drop(te_pairs.index).sample(frac=0.15, random_state=1)
    key = od.set_index(["PULocationID", "DOLocationID"]).index
    is_te = key.isin(te_pairs.set_index(["PULocationID", "DOLocationID"]).index)
    is_cal = key.isin(cal_pairs.set_index(["PULocationID", "DOLocationID"]).index)
    fit = od[~is_te & ~is_cal]; cal = od[is_cal]; te = od[is_te]
    pr_cal, pr_te = {}, {}
    for q in QS:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=400,
                              learning_rate=0.05, num_leaves=63, verbose=-1)
        m.fit(fit[feats], fit["p50"])
        pr_cal[q] = m.predict(cal[feats]); pr_te[q] = m.predict(te[feats])
    y_cal, y_te = cal["p50"].values, te["p50"].values
    sc = np.maximum(pr_cal[0.10] - y_cal, y_cal - pr_cal[0.90])
    d_conf = float(np.quantile(sc, 0.80))
    return {
        "n_fit": len(fit), "n_cal": len(cal), "n_test": len(te),
        "pinball_p50": round(pinball(y_te, pr_te[0.50], 0.5), 3),
        "crps": round(crps_from_quantiles(y_te, pr_te), 3),
        "mae_min": round(float(np.mean(np.abs(y_te - pr_te[0.50]))), 3),
        "cover80_raw": round(reliability(y_te, pr_te[0.10], pr_te[0.90]), 3),
        "cover80_conformal": round(reliability(y_te, pr_te[0.10] - d_conf,
                                               pr_te[0.90] + d_conf), 3),
        "width80_raw": round(float(np.mean(pr_te[0.90] - pr_te[0.10])), 2),
        "width80_conformal": round(float(np.mean(pr_te[0.90] - pr_te[0.10]
                                                 + 2 * d_conf)), 2),
    }


# --------------------------------------------------- speed feasibility --------
def feasibility_block():
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from model import predictors as P
    od = P._load_od_with_features()
    od["ach"] = od["mean_mph"] * 1.60934
    feats = ["dist_km", "hour", "plat", "plon", "dlat", "dlon"]
    pairs = od[["PULocationID", "DOLocationID"]].drop_duplicates()
    te_pairs = pairs.sample(frac=0.2, random_state=0)
    is_te = od.set_index(["PULocationID", "DOLocationID"]).index.isin(
        te_pairs.set_index(["PULocationID", "DOLocationID"]).index)
    fit, te = od[~is_te], od[is_te]
    # quantile model of the achievable speed itself
    mods = {}
    for q in QS:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=300,
                              learning_rate=0.05, num_leaves=63, verbose=-1)
        m.fit(fit[feats], fit["ach"]); mods[q] = m
    pr = {q: mods[q].predict(te[feats]) for q in QS}
    levels = {"low": float(np.percentile(od["ach"], 30)),
              "med": float(np.percentile(od["ach"], 60)),
              "high": float(np.percentile(od["ach"], 85))}
    out = {"levels_kmh": {k: round(v, 1) for k, v in levels.items()}, "brier": {},
           "reliability_curves": {}}
    grid = np.array(QS)
    for name, thr in levels.items():
        # pi = P(ach >= thr) interpolated from the predicted quantile curve
        pi = np.zeros(len(te))
        qmat = np.stack([pr[q] for q in QS], 1)      # increasing in q
        for i in range(len(te)):
            pi[i] = 1.0 - float(np.interp(thr, qmat[i], grid,
                                          left=0.0, right=1.0))
        ylab = (te["ach"].values >= thr).astype(float)
        out["brier"][name] = round(float(np.mean((pi - ylab) ** 2)), 4)
        bins = np.linspace(0, 1, 11); mid, obs, cnt = [], [], []
        for b in range(10):
            m = (pi >= bins[b]) & (pi < bins[b + 1])
            if m.sum() >= 30:
                mid.append(round(float(pi[m].mean()), 3))
                obs.append(round(float(ylab[m].mean()), 3))
                cnt.append(int(m.sum()))
        out["reliability_curves"][name] = {"pred": mid, "obs": obs, "n": cnt}
        # secondary discrimination diagnostic
        from sklearn.metrics import roc_auc_score
        if len(np.unique(ylab)) > 1:
            out.setdefault("auc_secondary", {})[name] = round(
                float(roc_auc_score(ylab, pi)), 3)
    return out


if __name__ == "__main__":
    out = {"demand": demand_block(), "travel_time": travel_block(),
           "feasibility": feasibility_block()}
    json.dump(out, open(os.path.join(PROC, "prediction_validation.json"), "w"),
              indent=2)
    print(json.dumps(out, indent=2)[:2200])
