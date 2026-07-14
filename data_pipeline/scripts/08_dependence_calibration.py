"""
08 - Calibrate joint (cross-node / cross-arc) dependence from real data.

Demand: decompose Austin log route-day loads into route effect + common day effect
+ idiosyncratic residual. The day-effect variance share gives the intra-day
correlation of log demand across collection units on the same operating day.

Traffic: from NYC TLC trips, compute the city median achievable speed per
(date, hour); the across-date dispersion of log median speed at fixed hour gives
the common day-to-day traffic factor that shifts all arcs together.

Output: processed/dependence_calibration.json
"""
import os, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw"); PROC = os.path.join(HERE, "processed")

# ---------------- demand: Austin day-effect decomposition --------------------
a = pd.read_csv(os.path.join(RAW, "austin_waste_loads.csv"))
a["load_weight"] = pd.to_numeric(a["load_weight"], errors="coerce")
a = a.dropna(subset=["load_weight"]); a = a[a["load_weight"] > 0]
wt = a["load_type"].str.upper().fillna("")
a = a[wt.str.contains("GARBAGE|RECYCL|ORGANIC|COMPOST|TRASH")].copy()
a["date"] = pd.to_datetime(a["report_date"], errors="coerce").dt.date
daily = a.groupby(["route_number", "date"])["load_weight"].sum().reset_index()
vc = daily["route_number"].value_counts()
daily = daily[daily["route_number"].isin(vc[vc >= 60].index)].copy()
daily["y"] = np.log(daily["load_weight"])
# remove route fixed effect
daily["y_c"] = daily["y"] - daily.groupby("route_number")["y"].transform("mean")
# day effect = cross-route mean of centred logs on the same day (>= 20 routes/day)
dsz = daily.groupby("date")["y_c"].transform("size")
sub = daily[dsz >= 20].copy()
day_eff = sub.groupby("date")["y_c"].mean()
sub["d"] = sub["date"].map(day_eff)
sub["e"] = sub["y_c"] - sub["d"]
var_d, var_e = float(day_eff.var()), float(sub["e"].var())
rho_demand = var_d / (var_d + var_e)

# ---------------- traffic: TLC common day factor ------------------------------
frames = []
for f in sorted(glob.glob(os.path.join(RAW, "nyc_yellow_*.parquet"))):
    df = pd.read_parquet(f, columns=["tpep_pickup_datetime", "tpep_dropoff_datetime",
                                     "trip_distance"])
    dur = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()/60
    ok = (dur.between(1, 120)) & (df["trip_distance"].between(0.2, 30))
    df = df[ok].copy(); dur = dur[ok]
    df["mph"] = df["trip_distance"] / (dur / 60.0)
    df = df[df["mph"].between(1, 70)]
    df["date"] = df["tpep_pickup_datetime"].dt.date
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    frames.append(df[["date", "hour", "mph"]])
t = pd.concat(frames, ignore_index=True)
med = t.groupby(["date", "hour"])["mph"].median().reset_index()
med = med[med.groupby("hour")["mph"].transform("size") >= 30]
med["lm"] = np.log(med["mph"])
med["lm_c"] = med["lm"] - med.groupby("hour")["lm"].transform("mean")
sigma_traffic = float(med["lm_c"].std())

out = {
    "rho_demand_intraday": round(rho_demand, 4),
    "sigma_day_effect_log": round(float(np.sqrt(var_d)), 4),
    "sigma_idiosyncratic_log": round(float(np.sqrt(var_e)), 4),
    "n_route_days": int(len(sub)), "n_days": int(day_eff.size),
    "sigma_traffic_common_log": round(sigma_traffic, 4),
    "n_date_hour_cells": int(len(med)),
    "note": ("rho_demand = share of log-demand variance common to all units on the "
             "same day (Austin). sigma_traffic = across-date std of log median "
             "achievable speed at fixed hour (NYC TLC)."),
}
json.dump(out, open(os.path.join(PROC, "dependence_calibration.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
