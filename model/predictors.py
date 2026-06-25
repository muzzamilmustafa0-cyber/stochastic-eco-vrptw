"""
ML prediction layer (real-data, validated):
  - travel-time model: predict median + q10/q90 arc travel time from context (real TLC)
  - eco-speed feasibility classifier: P(speed level feasible | context) (real TLC)
(demand quantile model is trained separately in data_pipeline/scripts/07.)

All models are trained + validated on REAL NYC TLC data with held-out OD pairs, and
reports are written to data_pipeline/processed/ for transparent reporting in the paper.
"""
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_absolute_error

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data_pipeline", "processed")


def _load_od_with_features():
    od = pd.read_parquet(os.path.join(PROC, "nyc_od_traveltime.parquet"))
    zc = pd.read_csv(os.path.join(PROC, "nyc_zone_centroids.csv")).set_index("LocationID")
    def feat(df, col):
        return df[col].map(zc["lat"]), df[col].map(zc["lon"])
    plat, plon = feat(od, "PULocationID"); dlat, dlon = feat(od, "DOLocationID")
    R = 6371.0
    p1, p2 = np.radians(plat.values), np.radians(dlat.values)
    dphi = np.radians(dlat.values-plat.values); dl = np.radians(dlon.values-plon.values)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    od = od.copy()
    od["dist_km"] = 2*R*np.arcsin(np.sqrt(a))
    od["plat"], od["plon"], od["dlat"], od["dlon"] = plat.values, plon.values, dlat.values, dlon.values
    od = od.dropna(subset=["dist_km", "plat", "dlat"])
    od = od[od["dist_km"] > 0]
    return od


def train_travel_time(seed=0):
    od = _load_od_with_features()
    # hold out 20% of OD *pairs* (generalisation to unseen arcs)
    rng = np.random.default_rng(seed)
    pairs = od[["PULocationID", "DOLocationID"]].drop_duplicates()
    test_pairs = pairs.sample(frac=0.2, random_state=seed)
    key = od.set_index(["PULocationID", "DOLocationID"]).index
    tkey = test_pairs.set_index(["PULocationID", "DOLocationID"]).index
    is_test = key.isin(tkey)
    feats = ["dist_km", "hour", "plat", "plon", "dlat", "dlon"]
    tr, te = od[~is_test], od[is_test]
    out = {}
    qmods = {}
    for q, tgt in [(0.5, "p50"), (0.1, "p10"), (0.9, "p90")]:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=500,
                              learning_rate=0.05, num_leaves=63, verbose=-1)
        m.fit(tr[feats], tr[tgt]); qmods[q] = m
    p50 = qmods[0.5].predict(te[feats])
    mae = float(mean_absolute_error(te["p50"], p50))
    mape = float(np.mean(np.abs(te["p50"]-p50)/np.maximum(te["p50"], 1e-6)))
    cov = float(np.mean((te["p50"] >= qmods[0.1].predict(te[feats])) &
                        (te["p50"] <= qmods[0.9].predict(te[feats]))))
    out = {"n_train": int(len(tr)), "n_test": int(len(te)),
           "MAE_min": round(mae, 2), "MAPE": round(mape, 3),
           "PI80_coverage_of_median": round(cov, 3), "features": feats,
           "note": "Real NYC TLC OD x hour; quantile LightGBM; held-out OD pairs."}
    json.dump(out, open(os.path.join(PROC, "travel_time_model_report.json"), "w"), indent=2)
    return out


def train_feasibility(seed=0):
    od = _load_od_with_features()
    # achievable km/h and per-level feasibility labels from real speeds
    od["ach_kmh"] = od["mean_mph"] * 1.60934
    levels = {"low": np.percentile(od["ach_kmh"], 30),
              "med": np.percentile(od["ach_kmh"], 60),
              "high": np.percentile(od["ach_kmh"], 85)}
    rng = np.random.default_rng(seed)
    pairs = od[["PULocationID", "DOLocationID"]].drop_duplicates()
    test_pairs = pairs.sample(frac=0.2, random_state=seed)
    tkey = test_pairs.set_index(["PULocationID", "DOLocationID"]).index
    is_test = od.set_index(["PULocationID", "DOLocationID"]).index.isin(tkey)
    feats = ["dist_km", "hour", "plat", "plon", "dlat", "dlon"]
    report = {"levels_kmh": {k: round(float(v), 1) for k, v in levels.items()}, "auc": {}}
    for name, thr in levels.items():
        y = (od["ach_kmh"] >= thr).astype(int)
        tr, te = ~is_test, is_test
        if y[tr].nunique() < 2 or y[te].nunique() < 2:
            report["auc"][name] = None; continue
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1)
        m.fit(od.loc[tr, feats], y[tr])
        proba = m.predict_proba(od.loc[te, feats])[:, 1]
        report["auc"][name] = round(float(roc_auc_score(y[te], proba)), 3)
    report["features"] = feats
    report["note"] = "Real NYC TLC; predict P(eco-speed feasible | context); held-out OD pairs."
    json.dump(report, open(os.path.join(PROC, "feasibility_model_report.json"), "w"), indent=2)
    return report


if __name__ == "__main__":
    print("travel-time:", json.dumps(train_travel_time(), indent=2))
    print("feasibility:", json.dumps(train_feasibility(), indent=2))
