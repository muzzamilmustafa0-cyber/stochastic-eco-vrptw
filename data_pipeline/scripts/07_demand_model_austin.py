"""
07 - REAL demand quantile model trained on Austin daily route loads.

Demonstrates that the demand-prediction layer is LEARNED + VALIDATED on real data
(not parametric). Quantile LightGBM (pinball loss = doc Eq.4) predicts q10/q50/q90 of
daily route load from real context features. Temporal holdout (train past, test future).

Outputs:
  processed/austin_demand_model_report.json : holdout pinball, P50 MAPE, 80% PI coverage
  processed/austin_demand_quantile_preds.csv: sample predictions
"""
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw"); PROC = os.path.join(HERE, "processed")

a = pd.read_csv(os.path.join(RAW, "austin_waste_loads.csv"))
a["load_weight"] = pd.to_numeric(a["load_weight"], errors="coerce")
a = a.dropna(subset=["load_weight"]); a = a[a["load_weight"] > 0]
wt = a["load_type"].str.upper().fillna("")
a = a[wt.str.contains("GARBAGE|RECYCL|ORGANIC|COMPOST|TRASH")].copy()
a["dt"] = pd.to_datetime(a["report_date"], errors="coerce")
a = a.dropna(subset=["dt"])
# daily route load (target) + real context features
daily = (a.groupby(["route_number", a["dt"].dt.date])
         .agg(load=("load_weight", "sum"), load_type=("load_type", "first")).reset_index())
daily.columns = ["route", "date", "load", "load_type"]
daily["date"] = pd.to_datetime(daily["date"])
daily["dow"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["doy"] = daily["date"].dt.dayofyear
daily["route_code"] = daily["route"].astype("category").cat.codes
daily["ltype_code"] = daily["load_type"].astype("category").cat.codes
# lag feature: route's expanding mean up to previous day (real temporal signal)
daily = daily.sort_values(["route", "date"])
daily["route_hist_mean"] = (daily.groupby("route")["load"]
                            .transform(lambda s: s.shift(1).expanding().mean()))
daily["route_hist_std"] = (daily.groupby("route")["load"]
                           .transform(lambda s: s.shift(1).expanding().std()))
daily = daily.dropna(subset=["route_hist_mean", "route_hist_std"])
# keep routes with enough history (stable, predictable demand units)
vc = daily["route"].value_counts(); keep = vc[vc >= 60].index
daily = daily[daily["route"].isin(keep)].copy()
daily["log_load"] = np.log1p(daily["load"])
daily["log_hist"] = np.log1p(daily["route_hist_mean"])

feats = ["dow", "month", "doy", "route_code", "ltype_code", "log_hist", "route_hist_std"]
cats = ["route_code", "ltype_code"]
for c in cats:
    daily[c] = daily[c].astype("category")

# stationary (pre-pandemic) subset for a fair coverage measure; COVID period (2020+)
# is reported separately because the demand distribution shifts sharply.
import os as _os
STATIONARY = _os.environ.get("STATIONARY", "1") == "1"
if STATIONARY:
    daily = daily[daily["date"] < "2020-03-01"].copy()

cut = daily["date"].quantile(0.8)
trn = daily[daily["date"] <= cut]
te = daily[(daily["date"] > cut) & daily["route_code"].isin(trn["route_code"].unique())]
# split train -> fit + conformal-calibration slice (last 15% of train by date)
ccut = trn["date"].quantile(0.85)
trf = trn[trn["date"] <= ccut]; cal = trn[trn["date"] > ccut]
print(f"fit {len(trf)} / calib {len(cal)} / test {len(te)} (split @ {cut.date()})")
Xtrf, ytrf = trf[feats], trf["log_load"]
yte = te["load"].values

def pinball(y, p, q):
    e = y - p
    return float(np.mean(np.maximum(q*e, (q-1)*e)))

models = {}
for q in [0.1, 0.5, 0.9]:
    m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=600,
                          learning_rate=0.05, num_leaves=63, min_child_samples=50,
                          subsample=0.8, colsample_bytree=0.8, verbose=-1)
    m.fit(Xtrf, ytrf, categorical_feature=cats)
    models[q] = m

def predict(X):
    return {q: np.expm1(models[q].predict(X)) for q in models}

# conformalized quantile regression (CQR): widen [q10,q90] to hit 80% coverage
pc = predict(cal[feats]); yc = cal["load"].values
scores = np.maximum(pc[0.1] - yc, yc - pc[0.9])
d = float(np.quantile(scores, 0.80))             # conformal correction (kg)
Xte = te[feats]
preds = predict(Xte)
preds[0.1] = preds[0.1] - d; preds[0.9] = preds[0.9] + d
pb = {q: pinball(yte, preds[q], q) for q in [0.1, 0.5, 0.9]}

p50 = preds[0.5]
wape = float(np.sum(np.abs(yte - p50)) / np.sum(yte))           # robust scale-free
mdape = float(np.median(np.abs(yte - p50) / np.maximum(yte, 1e-6)))
mae = float(np.mean(np.abs(yte - p50)))
cover80 = float(np.mean((yte >= preds[0.1]) & (yte <= preds[0.9])))
report = {
    "n_fit": int(len(trf)), "n_calib": int(len(cal)), "n_test": int(len(te)),
    "n_routes": int(len(keep)), "temporal_split_date": str(cut.date()),
    "conformal_correction_kg": round(d, 1),
    "pinball_q10": round(pb[0.1], 1), "pinball_q50": round(pb[0.5], 1), "pinball_q90": round(pb[0.9], 1),
    "P50_MAE_kg": round(mae, 1), "P50_WAPE": round(wape, 3), "P50_MdAPE": round(mdape, 3),
    "PI80_coverage": round(cover80, 3), "PI80_target": 0.80,
    "features": feats,
    "note": "Real Austin daily route loads; log-target quantile LightGBM; temporal holdout; routes>=60 obs.",
}
json.dump(report, open(os.path.join(PROC, "austin_demand_model_report.json"), "w"), indent=2)
out = te[["route", "date", "load"]].copy()
out["q10"], out["q50"], out["q90"] = preds[0.1], p50, preds[0.9]
out.head(500).to_csv(os.path.join(PROC, "austin_demand_quantile_preds.csv"), index=False)
print(json.dumps(report, indent=2))
